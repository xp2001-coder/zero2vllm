"""Read pinned vLLM source without importing or executing it (Python 3.11+)."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.request import urlopen

from scripts.common import ROOT, source_fingerprint

LESSON = ROOT / 'docs/chapters/part-03-vllm/01-a-request-s-journey-through-vllm'
MANIFEST = LESSON / 'data/source-manifest.json'
DEFAULT_SOURCE = ROOT / '.local/sources/vllm-v0.8.5'
DEFAULT_OUTPUT = ROOT / '.local/runs/part-03-vllm/01-request-journey'


def digest(payload):
    """Permit Git's Windows line endings, but no substantive source changes."""
    return hashlib.sha256(payload.replace(b'\r\n', b'\n')).hexdigest()


def source_path(root, name):
    if '\\' in name or ':' in name or '..' in Path(name).parts:
        raise ValueError(f'Invalid manifest path: {name}')
    root = root.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f'Invalid manifest path: {name}')
    return path


def read_manifest(path=MANIFEST):
    data = json.loads(path.read_text(encoding='utf-8'))
    if (data['schema_version'] != 1
            or data['repository'] != 'https://github.com/vllm-project/vllm'
            or not re.fullmatch(r'[0-9a-f]{40}', data['commit'])):
        raise ValueError('Expected an immutable vLLM GitHub source manifest')
    for name, info in data['files'].items():
        source_path(Path.cwd(), name)
        if not re.fullmatch(r'[0-9a-f]{64}', info['sha256_lf']):
            raise ValueError(f'Invalid SHA256: {name}')
    ids = [item['id'] for item in data['checkpoints']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate checkpoint ID')
    for item in data['checkpoints'] + data.get('structures', []):
        if item['path'] not in data['files']:
            raise ValueError(f'Unknown checkpoint file: {item["path"]}')
    return data


def fetch_sources(root, manifest):
    """Only called with --fetch. Never overwrite an existing modified file."""
    downloaded = []
    for name, info in manifest['files'].items():
        path = source_path(root, name)
        if path.exists():
            if digest(path.read_bytes()) != info['sha256_lf']:
                raise ValueError(f'Source hash mismatch; preserved existing file: {name}')
            continue
        url = f'https://raw.githubusercontent.com/vllm-project/vllm/{manifest["commit"]}/{name}'
        with urlopen(url, timeout=30) as response:
            payload = response.read()
        if digest(payload) != info['sha256_lf']:
            raise ValueError(f'Download hash mismatch (not saved): {name}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        downloaded.append(name)
    return downloaded


def symbols(tree, prefix='', required=None):
    """Include lexical scopes: A.run and A.run.helper are different symbols."""
    result = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = prefix + node.name
            if name in result and (required is None or name in required):
                raise ValueError(f'Ambiguous symbol: {name}')
            result[name] = node
            result.update(symbols(node, name + '.', required))
        else:
            result.update(symbols(node, prefix, required))
    return result


def call_sites(node):
    """Calls in this body; nested functions and decorators are separate scopes."""
    result = []

    def visit(child):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(child, ast.Call):
            result.append({'target': ast.unparse(child.func), 'line': child.lineno})
        for descendant in ast.iter_child_nodes(child):
            visit(descendant)

    for statement in node.body:
        visit(statement)
    return sorted(result, key=lambda item: item['line'])


def audit(root, manifest, *, only_required_symbols=False):
    errors, files, trees, checkpoints, structures = [], [], {}, [], []
    indexed_paths = {item['path'] for item in manifest['checkpoints'] + manifest.get('structures', [])}
    for name, info in manifest['files'].items():
        path = source_path(root, name)
        if not path.is_file():
            errors.append(f'Missing source file: {name}; first run with --fetch')
            continue
        payload = path.read_bytes()
        actual = digest(payload)
        files.append({'path': name, 'sha256_lf': actual, 'matched': actual == info['sha256_lf']})
        if actual != info['sha256_lf']:
            errors.append(f'Source hash mismatch: {name}')
            continue
        if name in indexed_paths:
            try:
                # Files can legitimately contain unrelated property setters or
                # typing overloads. Opt-in readers still reject duplicate
                # requested symbols; unrequested definitions are not evidence.
                required = {i['symbol'] for i in manifest['checkpoints'] + manifest.get('structures', [])
                            if i['path'] == name} if only_required_symbols else None
                trees[name] = symbols(ast.parse(payload.decode('utf-8'), filename=name), required=required)
            except (SyntaxError, UnicodeError, ValueError) as error:
                errors.append(f'Cannot parse {name}: {error}')
    for item in manifest['checkpoints']:
        node = trees.get(item['path'], {}).get(item['symbol'])
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            errors.append(f'Missing function: {item["symbol"]}')
            continue
        sites = call_sites(node)
        found = {site['target'] for site in sites}
        missing = sorted(set(item['calls']) - found)
        if missing:
            errors.append(f'{item["symbol"]}: missing call expressions {missing}')
        checkpoints.append({**item, 'line': node.lineno, 'end_line': node.end_lineno,
                            'url': f'{manifest["repository"]}/blob/{manifest["commit"]}/{item["path"]}#L{node.lineno}',
                            'matched': not missing,
                            'call_sites': [site for site in sites if site['target'] in item['calls']]})
    for item in manifest.get('structures', []):
        node = trees.get(item['path'], {}).get(item['symbol'])
        fields = [n.target.id for n in node.body
                  if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)] if isinstance(node, ast.ClassDef) else []
        matched = bool(fields) and set(item['fields']).issubset(fields)
        if not matched:
            errors.append(f'Missing declared fields: {item["symbol"]}')
        structures.append({**item, 'matched': matched,
                           'line': node.lineno if node else None, 'declared_fields': fields})
    return {'passed': not errors, 'errors': errors, 'files': files,
            'checkpoints': checkpoints, 'structures': structures,
            'verification_level': 'static_source_inspection',
            'executed_upstream_code': False, 'gpu_validated': False,
            'repository': manifest['repository'], 'tag': manifest['tag'], 'commit': manifest['commit']}


def render_map(report):
    lines = ['# 3.1 固定版本源码定位表', '',
             f'版本：{report["tag"]}；提交：`{report["commit"]}`。', '',
             '这是源码的静态定位记录。函数中的调用表达式不等于运行时调用轨迹；分支、继承和消息分派仍需按正文阅读。', '',
             '| 阶段 | 阅读目的 | 符号与位置 |', '| --- | --- | --- |']
    for item in report['checkpoints']:
        lines.append(f'| {item["stage"]} | {item["title"]} | [`{item["symbol"]}`]({item["url"]}) |')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--fetch', action='store_true', help='Explicitly download missing pinned source files')
    parser.add_argument('--focus', choices=['all', 'input', 'core', 'output', 'cancel'], default='all',
                        help='Filter terminal navigation; all sources are still checked')
    args = parser.parse_args(argv)
    try:
        manifest = read_manifest()
        downloaded = fetch_sources(args.source_dir, manifest) if args.fetch else []
        report = audit(args.source_dir, manifest)
    except (OSError, ValueError) as error:
        print(f'Source inspection failed: {error}', file=sys.stderr)
        return 1
    report.update({'recorded_at': datetime.now(timezone.utc).isoformat(), 'downloaded': downloaded,
                   'course_source': source_fingerprint()})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    (args.output_dir/'source-map.md').write_text(render_map(report), encoding='utf-8', newline='\n')
    for item in report['checkpoints']:
        if args.focus in ('all', item['stage']):
            print(f'{item["stage"]:6} {item["id"]:16} {item["path"]}:{item["line"]}  {item["symbol"]}')
    print(f'Static source inspection: {"PASS" if report["passed"] else "FAIL"}; '
          f'{len(report["files"])} files, {len(report["checkpoints"])} functions, {len(report["structures"])} structures')
    for error in report['errors']:
        print(error, file=sys.stderr)
    print('No vLLM import, model execution or GPU validation was performed.')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
