"""Repository paths and reproducibility metadata shared by maintenance tools."""
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / '.local'


def source_fingerprint():
    paths = sorted({*ROOT.glob('zero2vllm/**/*.py'), *ROOT.glob('tests/**/*.py'),
                    *ROOT.glob('scripts/**/*.py'), *ROOT.glob('docs/chapters/**/examples/*.py'),
                    *ROOT.glob('docs/chapters/**/data/*.json')})
    sha = hashlib.sha256()
    for path in paths:
        sha.update(path.relative_to(ROOT).as_posix().encode('utf-8') + b'\0')
        sha.update(path.read_bytes().replace(b'\r\n', b'\n') + b'\0')
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True, check=True).stdout)
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {'git_commit': commit, 'working_tree_dirty': dirty, 'source_sha256': sha.hexdigest(),
            'recipe': 'Sorted recursive package/tests/scripts Python files, chapter examples and input JSON; relative UTF-8 path + NUL + LF-normalized bytes + NUL.'}
