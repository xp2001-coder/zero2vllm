"""Generate marked course sections and check this repository's Markdown.

Standard library only; no network requests or GPU imports. Unmarked prose is
kept verbatim. This is a checker for the Markdown conventions used here, not a
general Markdown parser or a verification of upstream feature support.
"""

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
import tomllib
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
STATUS = {"planned": "📋", "partial": "🧩", "complete": "✅"}
EXCLUDED = {".git", ".venv", "__pycache__", "runs", "models", "build", "dist"}
LOCAL_ONLY = {"vllm", "work"}


def project_files(root: Path):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (path.is_file() and not EXCLUDED.intersection(relative.parts)
                and relative.parts[0] not in LOCAL_ONLY
                and relative.parts[:2] != ("configs", "local")):
            yield path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_catalog(catalog, inventory):
    require(catalog["schema_version"] == inventory["schema_version"] == 1,
            "Unsupported course schema version.")
    ids, numbers, group_ids = set(), set(), set()
    require(bool(catalog["parts"]), "The course must contain parts.")
    for key, source in catalog["sources"].items():
        require(bool(source["title"]), f"Empty source title: {key}")
        url = urlsplit(source["url"])
        require(url.scheme in ("https", "http") and bool(url.netloc), f"Invalid source URL: {key}")
    for part in catalog["parts"]:
        number = part["number"]
        require(type(number) is int and number >= 0 and number not in numbers,
                f"Invalid or duplicate part: {number}")
        numbers.add(number)
        require(part["title"] and part["prerequisites"] and part["chapters"], f"Empty part: {number}")
        require(all(key in catalog["sources"] for key in part["sources"]),
                f"Unknown source in Part {number}")
        for chapter in part["chapters"]:
            cid = chapter["id"]
            require(re.fullmatch(rf"{number}\.[1-9][0-9]*", cid) and cid not in ids,
                    f"Invalid or duplicate chapter: {cid}")
            ids.add(cid)
            require(all(isinstance(chapter[key], str) and chapter[key].strip()
                        for key in ("title", "topics", "exercise", "hardware")), f"Empty chapter field: {cid}")
            require(chapter["status"] in STATUS, f"Unknown status: {cid}")
            if chapter.get("reading"):
                require(re.fullmatch(r"docs/(chapters|labs)/[a-z0-9-]+\.md", chapter["reading"]),
                        f"Invalid reading path: {cid}")
    for group in inventory["groups"]:
        gid = group["id"]
        require(gid not in group_ids, f"Duplicate feature group: {gid}")
        group_ids.add(gid)
        require(group["chapters"] and all(cid in ids for cid in group["chapters"]),
                f"Unknown or missing chapter mapping: {gid}")
        require(group["names"] and len(set(group["names"])) == len(group["names"]),
                f"Empty or duplicate feature names: {gid}")


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def table(headers, rows):
    return "\n".join("| " + " | ".join(cell(v) for v in row) + " |"
                     for row in [headers, ["---"] * len(headers), *rows])


def validate_knowledge(knowledge, catalog):
    require(knowledge["schema_version"] == 1, "Unsupported knowledge schema version")
    chapters = {c["id"] for p in catalog["parts"] for c in p["chapters"]}
    ids = set()
    for source in knowledge["sources"]:
        sid = source["id"]
        require(sid not in ids and re.fullmatch(r"(vllm|sglang)-(\d{2}|notes|manual)", sid),
                f"Invalid or duplicate knowledge source: {sid}")
        ids.add(sid)
        require(source["chapters"] and all(c in chapters for c in source["chapters"]),
                f"Unknown knowledge chapter mapping: {sid}")
        require(source["path"].startswith("vllm/") and ".." not in Path(source["path"]).parts,
                f"Invalid knowledge path: {sid}")
        require(re.fullmatch(r"[0-9a-f]{64}", source["text_sha256"]), f"Invalid text hash: {sid}")
        require(source["use"] and source["review_note"], f"Missing editorial decision: {sid}")


def english_outline(catalog, translation):
    require(translation["schema_version"] == 1 and translation["language"] == "en",
            "Unsupported English curriculum schema.")
    parts = catalog["parts"]
    chapter_ids = {c["id"] for p in parts for c in p["chapters"]}
    source_keys = {key for p in parts for key in p["sources"]}
    require(set(translation["parts"]) == {str(p["number"]) for p in parts},
            "English part translations do not match the catalog.")
    require(set(translation["chapters"]) == chapter_ids,
            "English chapter translations do not match the catalog.")
    require(set(translation["sources"]) == source_keys,
            "English source translations do not match the catalog.")
    for cid, chapter in translation["chapters"].items():
        require(all(isinstance(chapter.get(key), str) and chapter[key].strip()
                    for key in ("title", "topics")), f"Empty English chapter field: {cid}")
    for key, title in translation["sources"].items():
        require(isinstance(title, str) and title.strip(), f"Empty English source: {key}")
    for part in parts:
        translated = translation["parts"][str(part["number"])]
        require(isinstance(translated.get("title"), str) and translated["title"].strip(),
                f"Empty English part title: {part['number']}")
        require(bool(part.get("readme_note")) == bool(translated.get("readme_note")),
                f"Missing or extra English part note: {part['number']}")
    outline = [
        f"This edition plans **{len(parts)} parts and {len(chapter_ids)} chapters**, alongside two completed CPU labs. Linked chapter titles lead to existing topic guides. See the [detailed curriculum](docs/curriculum.md) for prerequisites, exercises, and hardware requirements. Chapter materials are currently primarily in Chinese.",
        "**Status: ✅ complete with local validation; 🧩 partial material or code available; 📋 planned, chapter text pending.** Upstream availability does not mean a feature has been tested in this project.",
        "[✅ CPU starter lab: checking correctness with and without KV cache](docs/labs/01-kv-cache.md)",
        "[✅ CPU quantization lab: INT4 packing, group scales, and Linear](docs/labs/02-int4-linear.md)",
        table(["Part", "Scope", "Chapters"],
              [[f"Part {p['number']}", translation["parts"][str(p["number"])]["title"],
                len(p["chapters"])] for p in parts]),
    ]
    for part in parts:
        translated = translation["parts"][str(part["number"])]
        rows = []
        for chapter in part["chapters"]:
            text = translation["chapters"][chapter["id"]]
            title = (f"[{text['title']}]({chapter['reading']})"
                     if chapter.get("reading") else text["title"])
            rows.append([f"{chapter['id']} {title}", text["topics"], STATUS[chapter["status"]]])
        outline += [
            f"<details>\n<summary>Part {part['number']} · {translated['title']} ({len(rows)} chapters)</summary>",
            table(["Chapter", "Topics to cover", "Status"], rows),
        ]
        if translated.get("readme_note"):
            outline.append(translated["readme_note"])
        links = " · ".join(f"[{translation['sources'][key]}]({catalog['sources'][key]['url']})"
                           for key in part["sources"])
        outline += [f"Sources and further reading: {links}.", "</details>"]
    return "\n\n".join(outline)


def rendered_sections(catalog, inventory, knowledge, translation):
    validate_catalog(catalog, inventory)
    validate_knowledge(knowledge, catalog)
    parts, sources = catalog["parts"], catalog["sources"]
    count = sum(len(p["chapters"]) for p in parts)
    summary = f"{len(parts)} 篇、{count} 个正式章节规划"
    readme = [f"本版共 **{summary}**，另有两个已完成的 CPU 实验。已编写的专题导读可点击章节名称进入；前置知识、练习产出、设备条件见[详细课程目录](docs/curriculum.md)。",
              "**状态：✅ 已完成并有本地验证；🧩 有部分材料或代码；📋 已规划、正文待编写。** 功能已被上游收录也不等于本项目已经实测。",
              "[✅ CPU 引导实验：KV Cache 前后是否计算一致](docs/labs/01-kv-cache.md)",
              "[✅ CPU 量化实验：INT4 打包、分组缩放与 Linear](docs/labs/02-int4-linear.md)",
              table(["篇", "专题范围", "章节数"],
                    [[f"Part {p['number']}", p["title"], len(p["chapters"])] for p in parts])]
    detailed = []
    for part in parts:
        chapters = part["chapters"]
        title = f"Part {part['number']} · {part['title']}"
        links = " · ".join(f"[{sources[key]['title']}]({sources[key]['url']})" for key in part["sources"])
        readme += [f"<details>\n<summary>{title}（{len(chapters)} 章）</summary>",
                   table(["章节", "必须覆盖的知识点", "状态"],
                         [[f"{c['id']} " + (f"[{c['title']}]({c['reading']})" if c.get("reading") else c["title"]),
                           c["topics"], STATUS[c["status"]]] for c in chapters])]
        if part.get("readme_note"):
            readme.append(part["readme_note"])
        readme += [f"来源与阅读入口：{links}。", "</details>"]
        detailed += [f"### {title}", f"前置：{part['prerequisites']}。"]
        if part.get("teaching_focus"):
            detailed.append(part["teaching_focus"])
        detailed.append(table(["编号", "章节", "验收练习或产出", "设备", "状态"],
                              [[c["id"], (f"[{c['title']}](../{c['reading']})" if c.get("reading") else c["title"]),
                                c["exercise"], c["hardware"], STATUS[c["status"]]]
                               for c in chapters]))
        if part.get("curriculum_note"):
            detailed.append(part["curriculum_note"])
        detailed.append(f"源码与资料：{links}。")
    groups = inventory["groups"]
    entries = sum(len(g["names"]) for g in groups)
    appendix = [f"下面逐项列出本次核对的 **{len(groups)} 组、{entries} 个命名条目**。计数包含别名、默认选项、占位项和模型映射，**不是 {entries} 个独立算法**。机器可读清单见 [feature-inventory.json](feature-inventory.json)。",
                "清单用途是避免遗漏后端家族和新配置；不作为复制粘贴的启动菜单。模型专用项、平台专用项、弃用项、示例连接器均保留说明。"]
    for group in groups:
        appendix += [f"### {group['title']}",
                     f"章节：{'、'.join(group['chapters'])}。来源：[官方源码]({group['source']})。",
                     group["note"],
                     "\n".join("- " + "、".join(f"`{name}`" for name in group["names"][i:i + 6]) + "。"
                               for i in range(0, len(group["names"]), 6))]
    return {
        "README.md": {"course-outline": "\n\n".join(readme)},
        "README_en.md": {"course-outline": english_outline(catalog, translation)},
        "docs/curriculum.md": {
            "course-summary": f"本版：{summary}。与 [README](../README.md#课程大纲) 和 [结构化目录](curriculum-catalog.json) 同步。",
            "course-chapters": "\n\n".join(detailed)},
        "docs/references.md": {"course-sources": table(["索引", "官方来源 / 项目"],
            [[f"`{key}`", f"[{s['title']}]({s['url']})"] for key, s in sources.items()])},
        "docs/vllm-feature-map.md": {"feature-inventory": "\n\n".join(appendix)},
        "docs/knowledge-base.md": {"knowledge-map": table(
            ["来源 ID", "本地资料文件", "对应章节", "吸收重点"],
            [[f"`{s['id']}`", f"`{s['path']}`", "、".join(s["chapters"]), s["use"]]
             for s in knowledge["sources"]])},
    }


def replace_section(text, name, body):
    start, end = f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"
    require(text.count(start) == text.count(end) == 1, f"Missing or repeated markers: {name}")
    before, rest = text.split(start, 1)
    old_body, after = rest.split(end, 1)
    return before + start + "\n\n" + body.strip() + "\n\n" + end + after


def outside_code(content):
    lines, fence = [], None
    for line in content.splitlines():
        match = re.match(r"^\s{0,3}(`{3,}|~{3,})(.*)$", line)
        if match:
            mark, suffix = match.groups()
            if fence is None:
                fence = mark
            elif mark[0] == fence[0] and len(mark) >= len(fence) and not suffix.strip():
                fence = None
            continue
        if fence is None:
            lines.append(line)
    require(fence is None, "Unclosed Markdown code fence")
    return "\n".join(lines)


def headings(content):
    slugs, used = set(), set()
    for title in re.findall(r"^#{1,6} (.+)$", outside_code(content), flags=re.M):
        title = re.sub(r"[`*_]", "", title)
        base = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        slug, repeat = base, 0
        while slug in used:
            repeat += 1
            slug = f"{base}-{repeat}"
        used.add(slug)
        slugs.add(slug)
    return slugs


class HTMLLinks(HTMLParser):
    """Collect links used by the README heroes, including decoded entities."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.targets = []

    def handle_starttag(self, tag, attrs):
        attribute = {"a": "href", "img": "src"}.get(tag)
        if attribute:
            for key, value in attrs:
                if key == attribute and value:
                    self.targets.append(value)


def check_documents(root, replacements=None):
    replacements = replacements or {}
    links, fragments, rows, markdown, parsed = 0, 0, 0, 0, []
    for path in project_files(root):
        relative = path.relative_to(root).as_posix()
        if path.suffix in (".json", ".toml"):
            parser = json.loads if path.suffix == ".json" else tomllib.loads
            parser(path.read_text(encoding="utf-8"))
            parsed.append(relative)
        if path.suffix != ".md":
            continue
        markdown += 1
        content = outside_code(replacements.get(relative, path.read_text(encoding="utf-8")))
        width = None
        for line in content.splitlines():
            if line.startswith("|"):
                current = len(re.split(r"(?<!\\)\|", line.strip().strip("|")))
                require(width is None or current == width, f"Inconsistent table columns: {relative}")
                width, rows = current, rows + 1
            else:
                width = None
        require(content.count("<details>") == content.count("</details>"), f"Unclosed details: {relative}")
        html_links = HTMLLinks()
        html_links.feed(content)
        targets = re.findall(r"\[[^\]\n]*\]\(([^)\n]+)\)", content) + html_links.targets
        for target in targets:
            url = urlsplit(target.strip("<>"))
            if url.scheme or url.netloc:
                continue
            resolved = (path.parent / unquote(url.path)).resolve() if url.path else path
            require(resolved.is_relative_to(root) and resolved.exists(), f"Broken local link: {relative}: {target}")
            links += 1
            if url.fragment:
                key = resolved.relative_to(root).as_posix()
                linked = replacements.get(key, resolved.read_text(encoding="utf-8"))
                require(unquote(url.fragment) in headings(linked), f"Broken heading link: {relative}: {target}")
                fragments += 1
    return {"markdown_files": markdown, "local_links": links, "local_fragments": fragments,
            "table_rows": rows, "parsed_json_and_toml": parsed}


def maintain(root=ROOT, write=False):
    root = root.resolve()
    catalog = json.loads((root / "docs/curriculum-catalog.json").read_text(encoding="utf-8"))
    inventory = json.loads((root / "docs/feature-inventory.json").read_text(encoding="utf-8"))
    knowledge = json.loads((root / "docs/knowledge-sources.json").read_text(encoding="utf-8"))
    translation = json.loads((root / "docs/curriculum-en.json").read_text(encoding="utf-8"))
    replacements, changed = {}, []
    # Validate every marker and link before writing any generated document.
    for filename, sections in rendered_sections(catalog, inventory, knowledge, translation).items():
        old = (root / filename).read_text(encoding="utf-8")
        new = old
        for name, body in sections.items():
            new = replace_section(new, name, body)
        replacements[filename] = new
        if old != new:
            changed.append(filename)
    checks = check_documents(root, replacements if write else None)
    if write:
        for filename in changed:
            (root / filename).write_text(replacements[filename], encoding="utf-8", newline="\n")
    return {"passed": write or not changed, "mode": "write" if write else "check",
            "changed_files": changed, "parts": len(catalog["parts"]),
            "chapters": sum(len(p["chapters"]) for p in catalog["parts"]),
            "inventory_groups": len(inventory["groups"]),
            "inventory_named_entries": sum(len(g["names"]) for g in inventory["groups"]),
            "knowledge_sources": len(knowledge["sources"]),
            "chapters_with_reading": sum(bool(c.get("reading")) for p in catalog["parts"] for c in p["chapters"]),
            "readme_languages": ["zh", "en"],
            **checks, "scope": "Local course consistency, bilingual outline structure, Markdown/HTML local links, headings/tables and JSON/TOML syntax. No network or GPU execution; translation meaning requires human review."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="Check without changing documents (default).")
    modes.add_argument("--write", action="store_true", help="Update marked sections from the course, feature and knowledge catalogs.")
    parser.add_argument("--report", type=Path, help="Optionally save the check result as JSON.")
    args = parser.parse_args(argv)
    try:
        report = maintain(write=args.write)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = {"passed": False, "error": str(error)}
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    if not report["passed"]:
        print("Course checks failed. Fix catalog/link errors or run --write to synchronize generated sections.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
