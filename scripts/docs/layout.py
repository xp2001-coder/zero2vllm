"""Check repository ownership, empty directories and experiment artifact links."""
import argparse
import json
import os
from pathlib import Path

from scripts.common import ROOT
from scripts.docs.catalog import load_catalog

MANAGED = ('docs', 'metadata', 'configs', 'scripts', 'tests', 'reports', 'zero2vllm')
OBSOLETE = ('docs/course', 'docs/data', 'docs/labs', 'docs/templates', 'examples', 'runs', 'work',
            'reports/part1', 'reports/part2')
ROOT_FILES = {
    'docs': {'README.md'}, 'metadata': {'README.md'}, 'configs': {'README.md'},
    'reports': {'README.md'}, 'scripts': {'README.md', '__init__.py', 'common.py'},
    'tests': {'README.md', '__init__.py'},
}


def audit(root=ROOT):
    root = root.resolve()
    errors, empty, artifacts = [], [], 0
    for old in OBSOLETE:
        if (root/old).exists():
            errors.append(f'Obsolete directory remains: {old}')
    for name, allowed in ROOT_FILES.items():
        if not (root/name/'README.md').is_file():
            errors.append(f'Missing directory guide: {name}/README.md')
        for path in (root/name).glob('*'):
            if path.is_file() and path.name not in allowed:
                errors.append(f'Unclassified root file: {path.relative_to(root).as_posix()}')
    for name in MANAGED:
        for base, dirs, files in os.walk(root/name):
            dirs[:] = [d for d in dirs if d != '__pycache__']
            if not dirs and not files:
                empty.append(Path(base).relative_to(root).as_posix())
    errors += [f'Empty directory: {name}' for name in empty]
    catalog = load_catalog(root)
    for part in catalog['parts']:
        index = root/part['directory']/'README.md'
        if not index.is_file(): errors.append(f'Missing part index: {index}')
        for lesson in part['chapters']:
            if not (root/lesson['reading']).is_file():
                errors.append(f'Missing lesson: {lesson["id"]}')
    # Check published summary output links after moving results into lessons.
    for summary in (root/'docs/chapters').rglob('summary.json'):
        if 'results' not in summary.parts: continue
        data = json.loads(summary.read_text(encoding='utf-8'))
        items = list(data.get('cases', []))
        items += [data[k] for k in ('tests', 'course_check') if isinstance(data.get(k), dict)]
        for item in items:
            if not item.get('output'): continue
            target = (summary.parent/item['output']).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                errors.append(f'Broken result link: {summary.relative_to(root)}: {item["output"]}')
            artifacts += 1
    return {'passed': not errors, 'errors': errors, 'empty_directories': empty,
            'parts': len(catalog['parts']), 'lessons': sum(len(p['chapters']) for p in catalog['parts']),
            'published_artifact_links': artifacts,
            'local_workspace_excluded': '.local/; archives and current runs do not enter public course checks'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = audit()
    text = json.dumps(report, ensure_ascii=False, indent=2)+'\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding='utf-8', newline='\n')
    print(text, end='')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
