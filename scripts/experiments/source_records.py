"""Optional source inspection shared by lessons 3.4 and 3.5."""
from scripts.experiments.lesson0301 import audit, fetch_sources, read_manifest


def inspect_sources(manifests, root, fetch=False):
    records = []
    for path in manifests:
        manifest = read_manifest(path)
        directory = root / ('vllm-' + manifest['tag'])
        downloaded = fetch_sources(directory, manifest) if fetch else []
        report = audit(directory, manifest, only_required_symbols=True)
        report['downloaded'] = downloaded
        report['errors'] = [e.replace('first run with --fetch', 'first run with --fetch-source') for e in report['errors']]
        records.append(report)
    return records
