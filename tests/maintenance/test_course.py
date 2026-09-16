import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.docs.course import (ROOT, check_documents, maintain, project_files,
                                 validate_catalog)
from scripts.docs.catalog import (CATALOG, TRANSLATION, load_catalog, load_translation,
                                  save_catalog, save_translation)


class CourseDocsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "课程 副本"
        self.root.mkdir()
        for source in project_files(ROOT):
            target = self.root / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    def write(self, name, text):
        if name == CATALOG:
            save_catalog(self.root, json.loads(text))
            return
        if name == TRANSLATION:
            save_translation(self.root, json.loads(text))
            return
        (self.root / name).write_text(text, encoding="utf-8")

    def read(self, name):
        if name == CATALOG:
            return json.dumps(load_catalog(self.root), ensure_ascii=False)
        if name == TRANSLATION:
            return json.dumps(load_translation(self.root), ensure_ascii=False)
        return (self.root / name).read_text(encoding="utf-8")

    def test_check_detects_generated_drift_without_writing(self):
        old = self.read("README.md").replace("本版共 **", "手工修改的目录 **")
        self.write("README.md", old)
        report = maintain(self.root)
        self.assertFalse(report["passed"])
        self.assertIn("README.md", report["changed_files"])
        self.assertEqual(self.read("README.md"), old)

    def test_write_preserves_manual_prose_and_is_idempotent(self):
        note = "维护者补充说明：这是生成区段之外的正文。\n\n"
        self.write("README.md", note + self.read("README.md").replace("本版共 **", "过期统计 **"))
        self.assertTrue(maintain(self.root, write=True)["passed"])
        self.assertTrue(self.read("README.md").startswith(note))
        self.assertTrue(maintain(self.root)["passed"])
        self.assertEqual(maintain(self.root, write=True)["changed_files"], [])

    def test_bad_marker_prevents_writing_other_documents(self):
        old = self.read("README.md").replace("本版共 **", "过期统计 **")
        self.write("README.md", old)
        self.write("docs/references/README.md", self.read("docs/references/README.md").replace("<!-- END course-sources -->", ""))
        with self.assertRaisesRegex(ValueError, "markers"):
            maintain(self.root, write=True)
        self.assertEqual(self.read("README.md"), old)

    def test_invalid_catalog_links_and_duplicates_are_rejected(self):
        original = json.loads(self.read("metadata/course/catalog.json"))
        original_inventory = json.loads(self.read("metadata/upstream/features.json"))
        for defect in ("duplicate chapter", "unknown status", "unknown source", "unknown chapter mapping"):
            catalog, inventory = copy.deepcopy(original), copy.deepcopy(original_inventory)
            if defect == "duplicate chapter":
                catalog["parts"][0]["chapters"].append(catalog["parts"][0]["chapters"][0])
            elif defect == "unknown status":
                catalog["parts"][0]["chapters"][0]["status"] = "maybe"
            elif defect == "unknown source":
                catalog["parts"][0]["sources"].append("missing-source")
            else:
                inventory["groups"][0]["chapters"].append("99.99")
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                validate_catalog(catalog, inventory)

    def test_missing_files_and_heading_fragments_are_rejected(self):
        for target in ("../missing.md", "../README.md#missing-heading"):
            self.write("docs/broken.md", f"[链接]({target})\n")
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "Broken"):
                check_documents(self.root)

    def test_catalog_change_updates_all_affected_documents(self):
        catalog = json.loads(self.read("metadata/course/catalog.json"))
        chapter = catalog["parts"][0]["chapters"][0]
        chapter["title"] = "新的章节名称"
        chapter["exercise"] = "新的验收产出"
        catalog["sources"]["quickstart"]["title"] = "新的来源名称"
        self.write("metadata/course/catalog.json", json.dumps(catalog, ensure_ascii=False))
        report = maintain(self.root, write=True)
        self.assertTrue(report["passed"])
        for name in ("README.md", "docs/chapters/README.md", "docs/references/README.md"):
            self.assertIn(name, report["changed_files"])
            self.assertIn("新的来源名称", self.read(name))
        self.assertIn("新的章节名称", self.read("README.md"))
        self.assertIn("新的验收产出", self.read("docs/chapters/README.md"))

    def test_local_library_and_scratch_are_optional_and_excluded(self):
        for directory in ("vllm", "work"):
            folder = self.root / directory
            folder.mkdir()
            (folder / "notes.md").write_text("[local](missing-private-file.md)", encoding="utf-8")
        self.assertTrue(maintain(self.root)["passed"])
        self.assertFalse(any(p.relative_to(self.root).parts[0] in {"vllm", "work"}
                             for p in project_files(self.root)))

    def test_private_reference_prevents_document_write(self):
        lesson = load_catalog(self.root)['parts'][0]['chapters'][0]['reading']
        self.write(lesson, self.read(lesson) + '\n参考本地知识库 `vllm-06`。\n')
        old = self.read("README.md").replace("本版共 **", "待更新 **")
        self.write("README.md", old)
        with self.assertRaisesRegex(ValueError, "Private editorial reference"):
            maintain(self.root, write=True)
        self.assertEqual(self.read("README.md"), old)

    def test_public_reading_pages_reject_internal_source_labels(self):
        lesson = load_catalog(self.root)['parts'][0]['chapters'][0]['reading']
        for filename in ('README.md', 'README_en.md', lesson, 'docs/references/topic-guides.md'):
            original = self.read(filename)
            for citation in ('本地知识库', '`vllm-08`', '`sglang-notes`', 'local knowledge base'):
                with self.subTest(filename=filename, citation=citation):
                    self.write(filename, original + '\n' + citation + '\n')
                    with self.assertRaisesRegex(ValueError, 'Private editorial reference'):
                        check_documents(self.root)
            self.write(filename, original)

    def test_public_reading_links_cannot_depend_on_private_files(self):
        private = self.root / '.local' / 'private notes.md'
        private.parent.mkdir()
        private.write_text('editorial notes', encoding='utf-8')
        original = self.read('README.md')
        for link in ('[notes](.local/private%20notes.md)',
                     '<a href=".local/private%20notes.md">notes</a>',
                     '[notes](file:///C:/notes.md)',
                     '[notes](C:/Users/author/notes.md)',
                     r'[notes](C:\Users\author\notes.md)'):
            with self.subTest(link=link):
                self.write('README.md', original + '\n' + link + '\n')
                with self.assertRaisesRegex(ValueError, 'Private editorial reference'):
                    check_documents(self.root)

    def test_private_archive_is_not_required_or_parsed(self):
        self.assertFalse((self.root / 'metadata/knowledge').exists())
        private = self.root / '.local' / 'archive' / 'editorial' / 'sources.json'
        private.parent.mkdir(parents=True)
        private.write_text('private incomplete JSON', encoding='utf-8')
        self.assertTrue(maintain(self.root)['passed'])
        self.assertNotIn('knowledge_sources', maintain(self.root))

    def test_public_sources_repo_links_and_code_examples_are_allowed(self):
        note = ('\n[paper](https://arxiv.org/abs/1706.03762)\n'
                '[source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)\n'
                '[lesson](docs/chapters/part-02-engine/01-pytorch-model/README.md)\n'
                '[repository](.)\n'
                'Output: `.local/runs/example.json`.\n'
                '```text\nvllm-06 is a literal in a code example, not a reference.\n```\n')
        self.write('README.md', self.read('README.md') + note)
        self.assertTrue(maintain(self.root)['passed'])

    def test_missing_reading_body_is_rejected(self):
        catalog = json.loads(self.read("metadata/course/catalog.json"))
        catalog["parts"][0]["chapters"][0]["reading"] = "docs/chapters/part-00-orientation/01-missing/README.md"
        self.write("metadata/course/catalog.json", json.dumps(catalog, ensure_ascii=False))
        with self.assertRaisesRegex(ValueError, "Broken local link"):
            maintain(self.root, write=True)

    def test_english_drift_is_detected_and_manual_text_is_preserved(self):
        old = "English editorial note outside generated sections.\n\n" + self.read("README_en.md")
        self.write("README_en.md", old.replace("This edition plans **", "Outdated outline **"))
        report = maintain(self.root)
        self.assertFalse(report["passed"])
        self.assertEqual(report["changed_files"], ["README_en.md"])
        self.assertIn("Outdated outline **", self.read("README_en.md"))
        self.assertTrue(maintain(self.root, write=True)["passed"])
        self.assertEqual(self.read("README_en.md"), old)
        self.assertEqual(maintain(self.root, write=True)["changed_files"], [])

    def test_both_languages_share_status_reading_and_source_urls(self):
        catalog = json.loads(self.read("metadata/course/catalog.json"))
        chapter = catalog["parts"][0]["chapters"][0]
        chapter["status"] = "complete"
        chapter["content"] = "lesson"
        catalog["sources"]["quickstart"]["url"] = "https://example.com/revised-quickstart"
        self.write("metadata/course/catalog.json", json.dumps(catalog, ensure_ascii=False))
        self.assertTrue(maintain(self.root, write=True)["passed"])
        for filename in ("README.md", "README_en.md"):
            content = self.read(filename)
            row = next(line for line in content.splitlines() if line.startswith("| 0.1 "))
            self.assertIn(f"]({chapter['reading']})", row)
            self.assertTrue(row.endswith("| ✅ |"))
            self.assertIn("https://example.com/revised-quickstart", content)

    def test_incomplete_english_translation_prevents_any_write(self):
        original = json.loads(self.read("metadata/translations/en/index.json"))
        old = self.read("README.md").replace("本版共 **", "待生成 **")
        self.write("README.md", old)
        for defect in ("chapter", "part", "source", "empty topics", "part note"):
            translation = copy.deepcopy(original)
            if defect == "chapter":
                del translation["chapters"]["9.2"]
            elif defect == "part":
                del translation["parts"]["9"]
            elif defect == "source":
                del translation["sources"]["quickstart"]
            elif defect == "empty topics":
                translation["chapters"]["9.2"]["topics"] = " "
            else:
                del translation["parts"]["9"]["readme_note"]
            self.write("metadata/translations/en/index.json", json.dumps(translation))
            with self.subTest(defect=defect), self.assertRaisesRegex(ValueError, "English"):
                maintain(self.root, write=True)
            self.assertEqual(self.read("README.md"), old)

    def test_html_links_images_entities_and_code_fences(self):
        self.write("docs/notes&help.md", "# Help\n")
        valid = ('<a href="notes&amp;help.md#help">Help</a>\n'
                 '<img src="assets/zero2vllm-banner.svg" alt="Banner">\n'
                 '```html\n<img src="missing-example.svg">\n```\n')
        self.write("docs/html-check.md", valid)
        check_documents(self.root)
        for tag in ('<img src="assets/missing-banner.svg">',
                    '<a href="../README_missing.md">English</a>',
                    '<a href="notes&amp;help.md#missing-heading">Help</a>'):
            self.write("docs/html-check.md", valid + tag)
            with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, "Broken"):
                check_documents(self.root)

    def test_hierarchy_has_unique_lesson_paths_and_part_indexes(self):
        catalog = json.loads(self.read('metadata/course/catalog.json'))
        paths = [c['reading'] for p in catalog['parts'] for c in p['chapters']]
        self.assertEqual(len(paths), len(set(paths)))
        for part in catalog['parts']:
            content = self.read(part['directory'] + '/README.md')
            self.assertIn('<!-- BEGIN part-index -->', content)
            for chapter in part['chapters']:
                self.assertTrue((self.root / chapter['reading']).is_file())

    def test_scaffold_cannot_be_marked_complete_or_escape_its_part(self):
        original = json.loads(self.read('metadata/course/catalog.json'))
        inventory = json.loads(self.read('metadata/upstream/features.json'))
        for field, value in [('status', 'complete'), ('reading', '../README.md'),
                             ('reading', original['parts'][1]['chapters'][0]['reading'])]:
            catalog = copy.deepcopy(original)
            catalog['parts'][0]['chapters'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_catalog(catalog, inventory)

    def test_part_index_and_scaffold_follow_catalog_without_overwriting_prose(self):
        catalog = json.loads(self.read('metadata/course/catalog.json'))
        chapter = catalog['parts'][2]['chapters'][1]
        chapter['exercise'] = '独立验收产出'
        chapter['topics'] = '新增专题一、新增专题二'
        note = '\n作者手写备注，必须保留。\n'
        self.write(chapter['reading'], self.read(chapter['reading']) + note)
        self.write('metadata/course/catalog.json', json.dumps(catalog, ensure_ascii=False))
        maintain(self.root, write=True)
        self.assertIn('独立验收产出', self.read(catalog['parts'][2]['directory']+'/README.md'))
        self.assertIn('新增专题二', self.read(chapter['reading']))
        self.assertTrue(self.read(chapter['reading']).endswith(note))


class DocumentRootTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="zero2vllm_document_links_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "docs").mkdir()
        (self.root / "README.md").write_text(
            "[Guide](docs/guide.md#guide)\n", encoding="utf-8")
        (self.root / "docs/guide.md").write_text(
            '# Guide\n<a href="../README.md">Home</a>\n', encoding="utf-8")

    def assert_valid_links(self, root):
        report = check_documents(root)
        self.assertEqual(report["markdown_files"], 2)
        self.assertEqual(report["local_links"], 2)
        self.assertEqual(report["local_fragments"], 1)

    def test_root_with_parent_component_preserves_links_and_anchors(self):
        self.assert_valid_links(self.root / "docs" / "..")

    @unittest.skipUnless(os.name == "nt", "Windows short-path alias")
    def test_windows_short_root_preserves_links_and_anchors(self):
        import ctypes
        from ctypes import wintypes

        get_short_path = ctypes.WinDLL("kernel32", use_last_error=True).GetShortPathNameW
        get_short_path.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        get_short_path.restype = wintypes.DWORD
        buffer = ctypes.create_unicode_buffer(32768)
        length = get_short_path(str(self.root), buffer, len(buffer))
        self.assertGreater(length, 0, ctypes.get_last_error())
        self.assertLess(length, len(buffer))
        alias = Path(buffer.value)
        if alias == alias.resolve():
            self.skipTest("8.3 aliases are disabled on this volume")
        self.assert_valid_links(alias)

    def test_existing_file_outside_document_root_is_rejected(self):
        # README exists, but is outside the selected docs root.
        with self.assertRaisesRegex(ValueError, "Broken local link.*README"):
            check_documents(self.root / "docs")


if __name__ == "__main__":
    unittest.main()
