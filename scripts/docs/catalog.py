"""Load per-part course data and English translations without external packages."""
import json
from pathlib import Path

CATALOG = 'metadata/course/catalog.json'
TRANSLATION = 'metadata/translations/en/index.json'


def read_json(path):
    return path.read_text(encoding='utf-8')


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')


def resolve_part(root, index, value):
    if not isinstance(value, str):
        raise ValueError('Course indexes must contain part file paths.')
    path = (index.parent / value).resolve()
    if not path.is_relative_to(index.parent.resolve()) or not path.is_file():
        raise ValueError(f'Invalid or missing part data file: {value}')
    return path


def load_catalog(root):
    index = root / CATALOG
    catalog = json.loads(read_json(index))
    paths = [resolve_part(root, index, name) for name in catalog['part_files']]
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate part data file.')
    catalog['parts'] = [json.loads(read_json(p)) for p in paths]
    return catalog


def save_catalog(root, catalog):
    """Update existing per-part files and index (used by editor tooling/tests)."""
    index = root / CATALOG
    value = dict(catalog)
    parts = value.pop('parts')
    if len(parts) != len(value['part_files']):
        raise ValueError('Part count must match the course index.')
    for name, part in zip(value['part_files'], parts):
        dump(resolve_part(root, index, name), part)
    dump(index, value)


def load_translation(root):
    index = root / TRANSLATION
    translation = json.loads(read_json(index))
    translation['parts'], translation['chapters'] = {}, {}
    paths = [resolve_part(root, index, name) for name in translation['part_files']]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate English part data file.')
    for path in paths:
        item = json.loads(read_json(path))
        number = str(item['number'])
        if number in translation['parts']:
            raise ValueError('Duplicate English part number.')
        translation['parts'][number] = item['part']
        for cid, chapter in item['chapters'].items():
            if cid in translation['chapters']:
                raise ValueError('Duplicate English chapter.')
            translation['chapters'][cid] = chapter
    return translation


def save_translation(root, translation):
    index = root / TRANSLATION
    value = dict(translation)
    parts, chapters = value.pop('parts'), value.pop('chapters')
    for name in value['part_files']:
        path = resolve_part(root, index, name)
        original = json.loads(read_json(path))
        number = original['number']
        # Missing keys are retained as explicit invalid data for validation tests.
        item = {'number': number, 'part': parts.get(str(number), {}),
                'chapters': {k:v for k,v in chapters.items() if k.split('.')[0] == str(number)}}
        dump(path, item)
    dump(index, value)
