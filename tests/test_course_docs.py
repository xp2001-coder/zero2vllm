import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.course_docs import (ROOT, check_documents, maintain, project_files,
                                 validate_catalog)


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
        (self.root / name).write_text(text, encoding="utf-8")

    def read(self, name):
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
        self.write("docs/references.md", self.read("docs/references.md").replace("<!-- END course-sources -->", ""))
        with self.assertRaisesRegex(ValueError, "markers"):
            maintain(self.root, write=True)
        self.assertEqual(self.read("README.md"), old)

    def test_invalid_catalog_links_and_duplicates_are_rejected(self):
        original = json.loads(self.read("docs/curriculum-catalog.json"))
        original_inventory = json.loads(self.read("docs/feature-inventory.json"))
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
        catalog = json.loads(self.read("docs/curriculum-catalog.json"))
        chapter = catalog["parts"][0]["chapters"][0]
        chapter["title"] = "新的章节名称"
        chapter["exercise"] = "新的验收产出"
        catalog["sources"]["quickstart"]["title"] = "新的来源名称"
        self.write("docs/curriculum-catalog.json", json.dumps(catalog, ensure_ascii=False))
        report = maintain(self.root, write=True)
        self.assertTrue(report["passed"])
        for name in ("README.md", "docs/curriculum.md", "docs/references.md"):
            self.assertIn(name, report["changed_files"])
            self.assertIn("新的来源名称", self.read(name))
        self.assertIn("新的章节名称", self.read("README.md"))
        self.assertIn("新的验收产出", self.read("docs/curriculum.md"))

    def test_local_library_and_scratch_are_optional_and_excluded(self):
        for directory in ("vllm", "work"):
            folder = self.root / directory
            folder.mkdir()
            (folder / "notes.md").write_text("[local](missing-private-file.md)", encoding="utf-8")
        self.assertTrue(maintain(self.root)["passed"])
        self.assertFalse(any(p.relative_to(self.root).parts[0] in {"vllm", "work"}
                             for p in project_files(self.root)))

    def test_invalid_knowledge_mapping_prevents_document_write(self):
        knowledge = json.loads(self.read("docs/knowledge-sources.json"))
        knowledge["sources"][0]["chapters"].append("99.99")
        self.write("docs/knowledge-sources.json", json.dumps(knowledge, ensure_ascii=False))
        old = self.read("README.md").replace("本版共 **", "待更新 **")
        self.write("README.md", old)
        with self.assertRaisesRegex(ValueError, "knowledge chapter mapping"):
            maintain(self.root, write=True)
        self.assertEqual(self.read("README.md"), old)

    def test_missing_reading_body_is_rejected(self):
        catalog = json.loads(self.read("docs/curriculum-catalog.json"))
        catalog["parts"][0]["chapters"][0]["reading"] = "docs/chapters/missing.md"
        self.write("docs/curriculum-catalog.json", json.dumps(catalog, ensure_ascii=False))
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
        catalog = json.loads(self.read("docs/curriculum-catalog.json"))
        chapter = catalog["parts"][0]["chapters"][0]
        chapter["status"] = "complete"
        chapter["reading"] = "docs/labs/01-kv-cache.md"
        catalog["sources"]["quickstart"]["url"] = "https://example.com/revised-quickstart"
        self.write("docs/curriculum-catalog.json", json.dumps(catalog, ensure_ascii=False))
        self.assertTrue(maintain(self.root, write=True)["passed"])
        for filename in ("README.md", "README_en.md"):
            content = self.read(filename)
            row = next(line for line in content.splitlines() if line.startswith("| 0.1 "))
            self.assertIn("](docs/labs/01-kv-cache.md)", row)
            self.assertTrue(row.endswith("| ✅ |"))
            self.assertIn("https://example.com/revised-quickstart", content)

    def test_incomplete_english_translation_prevents_any_write(self):
        original = json.loads(self.read("docs/curriculum-en.json"))
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
            self.write("docs/curriculum-en.json", json.dumps(translation))
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


if __name__ == "__main__":
    unittest.main()
