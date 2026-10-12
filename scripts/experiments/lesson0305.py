"""Lesson 3.5: trace a bounded configuration policy without loading vLLM."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

from scripts.common import ROOT, source_fingerprint
from scripts.experiments.source_records import inspect_sources
from zero2vllm.reference.config_resolution import engine_choice, resolve_v1, verify_scheduler

LESSON = ROOT/'docs/chapters/part-03-vllm/05-resolving-configuration-and-capabilities'


def case(name, checks, **details):
    return dict(name=name, checks=checks, passed=all(checks.values()), **details)


def failure(call):
    try: call()
    except ValueError as error: return str(error)
    return ''


def experiments():
    precedence = resolve_v1({'max_num_batched_tokens': 4096}, {'max_num_batched_tokens': 3072})
    disabled = resolve_v1({'enable_prefix_caching': True}, {'enable_prefix_caching': False})
    chunk = resolve_v1(cli_options={'enable_chunked_prefill': False})
    defaults = {f'{gib}_{usage}': resolve_v1(usage=usage, memory_gib=gib)['values'] for gib in (40, 80) for usage in ('server', 'offline')}
    choices = {f'{env}_{status}': engine_choice(env, status) for env, status in [(None, 'supported'), (None, 'unsupported'), (None, 'experimental'), ('1', 'experimental'), ('0', 'supported')]}
    budget_error = failure(lambda: resolve_v1(cli_options={'max_num_batched_tokens': 8, 'max_num_seqs': 16}))
    length_error = failure(lambda: resolve_v1(cli_options={'max_model_len': 8192}, model_limit=4096))
    chunk_valid = resolve_v1(cli_options={'max_num_batched_tokens': 512, 'max_num_seqs': 16}, model_limit=4096)
    eager = resolve_v1(cli_options={'enforce_eager': True})
    late = resolve_v1(cli_options={'block_size': 7})
    return [
        case('cli_over_file', {'final_3072': precedence['values']['max_num_batched_tokens'] == 3072,
                               'origin_cli': precedence['origins']['max_num_batched_tokens'] == 'cli'}, resolution=precedence),
        case('false_is_explicit', {'false_retained': disabled['values']['enable_prefix_caching'] is False,
                                   'unset_defaults_true': resolve_v1()['values']['enable_prefix_caching'] is True}, resolution=disabled),
        case('v1_forces_chunking', {'false_becomes_true': chunk['values']['enable_chunked_prefill'] is True,
                                    'reason_recorded': chunk['origins']['enable_chunked_prefill'].startswith('v1_policy')}, resolution=chunk),
        case('usage_and_memory_defaults', {'server_40': defaults['40_server']['max_num_batched_tokens'] == 2048,
                                          'offline_40': defaults['40_offline']['max_num_batched_tokens'] == 8192,
                                          'server_80': defaults['80_server']['max_num_batched_tokens'] == 8192,
                                          'offline_80': defaults['80_offline']['max_num_batched_tokens'] == 16384,
                                          'seqs': [defaults['40_server']['max_num_seqs'], defaults['80_server']['max_num_seqs']] == [256, 1024]}, defaults=defaults),
        case('engine_intent', {'auto_supported': choices['None_supported']['engine'] == 'V1',
                               'auto_unsupported': choices['None_unsupported']['engine'] == 'V0',
                               'auto_experimental': choices['None_experimental']['engine'] == 'V0',
                               'explicit_experimental': choices['1_experimental']['engine'] == 'V1',
                               'explicit_zero': choices['0_supported']['engine'] == 'V0'}, decisions=choices),
        case('forced_unsupported_errors', {'rejects': bool(failure(lambda: engine_choice('1', 'unsupported')))}),
        case('cross_field_validation', {'rejects_budget_lt_seqs': bool(budget_error)}, expected_error=budget_error),
        case('model_limit', {'rejects_over_fixture_limit': bool(length_error)}, expected_error=length_error),
        case('chunk_budget', {'small_budget_valid_for_chunked': chunk_valid['values']['max_num_batched_tokens'] == 512,
                              'unchunked_rejects': bool(failure(lambda: verify_scheduler(512, 16, 4096, False)))}, resolution=chunk_valid),
        case('eager_intent', {'disabled_intent': eager['graph_intent'] == 'disabled',
                             'default_not_claimed_active': resolve_v1()['graph_intent'] == 'eligible_for_configuration_not_proven_active'}, resolution=eager),
        case('validation_stages', {'block_size_preserved': late['values']['block_size'] == 7,
                                   'backend_check_still_pending': 'backend dimensions/dtype' in late['runtime_checks_pending']}, resolution=late,
             note='Value 7 is not validated for FlashAttention; acceptance by this stage does not make it usable.'),
        case('unverified_platform', {'refuses_cuda_defaults_as_maca_evidence': bool(failure(lambda: resolve_v1(platform='maca')))}),
    ]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inspect-source', action='store_true')
    p.add_argument('--fetch-source', action='store_true')
    p.add_argument('--source-root', type=Path, default=ROOT/'.local/sources')
    p.add_argument('--output-dir', type=Path, default=ROOT/'.local/runs/part-03-vllm/05-config-resolution')
    a = p.parse_args(argv)
    report = dict(recorded_at=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                  course_source=source_fingerprint(), verification_level='cpu_config_policy_subset',
                  upstream_executed=False, gpu_validated=False, cases=[], passed=False)
    try:
        report['cases'] = experiments()
        report['passed'] = all(c['passed'] for c in report['cases'])
        if a.inspect_source or a.fetch_source:
            report['source_inspection'] = inspect_sources([LESSON/'data/source-manifest.json'], a.source_root, a.fetch_source)
            report['passed'] &= all(r['passed'] for r in report['source_inspection'])
    except (ValueError, OSError) as error:
        report.update(passed=False, error=str(error))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    for c in report['cases']: print(c['name'], 'PASS' if c['passed'] else 'FAIL')
    if report.get('error'): print(report['error'])
    for r in report.get('source_inspection', []): print(r['tag'], 'static inspection', r['passed'], r['errors'])
    print('Synthetic configuration context; no vLLM import, device detection or GPU validation.')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
