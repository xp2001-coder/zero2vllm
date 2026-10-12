"""Lesson 2.3 lifecycle traces; default is standard-library CPU with actual KV lists."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

from scripts.common import ROOT, source_fingerprint
from zero2vllm.engine.lifecycle import LifecycleEngine, Status, TERMINAL
from zero2vllm.engine.runners import TinyCachedRunner, LlamaFullPrefixRunner
from zero2vllm.reference.attention import TinyAttentionLM, generate


class InstrumentedTinyRunner(TinyCachedRunner):
    """Lab-only fault/cancellation hook runs after actual KV mutation."""
    def __init__(self):
        super().__init__(TinyAttentionLM(seed=7, max_context=32))
        self.contexts = []
        self.after_forward = None

    def new_context(self):
        context = super().new_context()
        self.contexts.append(context)
        return context

    def advance(self, *args):
        token = super().advance(*args)
        if self.after_forward:
            self.after_forward()
        return token


def drain(engine, name):
    while engine.view(name).status == Status.RUNNING:
        engine.step(name)


def capture(engine, names, stage):
    return {'stage': stage, 'used_tokens': engine.used_tokens,
            'available_tokens': engine.available_tokens,
            'requests': [engine.view(n).to_dict() for n in names]}


def finish_case(name, engine, names, timeline, assertions):
    assertions = {**assertions, 'budget_returned': engine.used_tokens == 0,
                  'all_terminal': all(engine.view(n).status in TERMINAL for n in names),
                  'no_live_cache': all(engine.view(n).cached_positions == 0 for n in names),
                  'finalized_once': all(sum(e['kind'] == 'finalized' for e in engine.events(n)) == 1 for n in names)}
    if isinstance(engine.runner, InstrumentedTinyRunner):
        assertions['all_contexts_closed'] = all(c.closed and c.kv is None for c in engine.runner.contexts)
    return {'name': name, 'cache_kind': engine.runner.cache_kind,
            'capacity_tokens': engine.capacity_tokens, 'timeline': timeline,
            'requests': [engine.view(n).to_dict() for n in names],
            'events': {n: engine.events(n) for n in names},
            'assertions': assertions, 'passed': all(assertions.values())}


def run_tiny_cases():
    cases = []
    prompt, count = [1, 2, 3], 4
    for mode in ('length', 'eos', 'zero_output', 'cancel_waiting', 'cancel_running',
                 'cancel_in_forward', 'model_error', 'initialization_error'):
        runner = InstrumentedTinyRunner()
        expected = generate(runner.model, prompt, count, use_cache=False).token_ids
        n = 0 if mode == 'zero_output' else count
        with LifecycleEngine(runner, capacity_tokens=8) as engine:
            engine.submit('a', prompt, n, expected[0] if mode == 'eos' else None)
            timeline = [capture(engine, ['a'], 'submitted')]
            if mode == 'initialization_error':
                def fail_init():
                    raise MemoryError('injected context creation failure')
                runner.new_context = fail_init
            if mode == 'cancel_waiting':
                engine.cancel('a')
            elif engine.view('a').status == Status.WAITING:
                engine.admit('a')
                timeline.append(capture(engine, ['a'], 'admission_attempted'))
            if mode == 'cancel_in_forward':
                def cancel_in_forward():
                    engine.cancel('a')
                    timeline.append(capture(engine, ['a'], 'cancel_latched_during_forward'))
                runner.after_forward = cancel_in_forward
            if mode == 'model_error':
                def fail_forward():
                    raise RuntimeError('injected after KV append')
                runner.after_forward = fail_forward
            if mode == 'cancel_running':
                engine.step('a')
                engine.step('a')
                timeline.append(capture(engine, ['a'], 'two_tokens_committed'))
                engine.cancel('a')
            drain(engine, 'a')
            timeline.append(capture(engine, ['a'], 'terminal'))
            view = engine.view('a')
            status = Status.CANCELLED if mode.startswith('cancel') else Status.FAILED if mode.endswith('error') else Status.FINISHED
            expected_ids = expected if mode == 'length' else expected[:1] if mode == 'eos' else expected[:2] if mode == 'cancel_running' else []
            assertions = {'expected_status': view.status == status, 'expected_ids': list(view.output_ids) == expected_ids}
            if mode == 'cancel_in_forward':
                during = next(t for t in timeline if t['stage'] == 'cancel_latched_during_forward')['requests'][0]
                assertions['resources_retained_in_forward'] = during['reserved_tokens'] == 6 and during['cached_positions'] == 3
            cases.append(finish_case(mode, engine, ['a'], timeline, assertions))

    runner = InstrumentedTinyRunner()
    with LifecycleEngine(runner, capacity_tokens=6) as engine:
        for name in ('a', 'b'):
            engine.submit(name, prompt, count)
        engine.admit('a')
        deferred = not engine.admit('b')
        timeline = [capture(engine, ['a', 'b'], 'b_waits_for_budget')]
        engine.step('a')
        engine.cancel('a')
        timeline.append(capture(engine, ['a', 'b'], 'a_cancelled'))
        admitted = engine.admit('b')
        drain(engine, 'b')
        timeline.append(capture(engine, ['a', 'b'], 'b_finished'))
        expected = generate(runner.model, prompt, count, use_cache=False).token_ids
        cases.append(finish_case('capacity_wait_resume', engine, ['a', 'b'], timeline,
                                {'deferred_then_admitted': deferred and admitted,
                                 'b_matches_full_recompute': list(engine.view('b').output_ids) == expected}))

    runner = InstrumentedTinyRunner()
    with LifecycleEngine(runner, capacity_tokens=16) as engine:
        engine.submit('a', prompt, count)
        engine.submit('b', [4, 2], 3)
        engine.admit('a')
        engine.admit('b')
        distinct = runner.contexts[0].kv is not runner.contexts[1].kv
        timeline = []
        for i in range(4):
            engine.step('a')
            engine.step('b')
            timeline.append(capture(engine, ['a', 'b'], f'manual_round_{i+1}'))
        assertions = {'distinct_cache_objects': distinct}
        for name, inputs, n in [('a', prompt, count), ('b', [4, 2], 3)]:
            assertions[f'{name}_matches_full_recompute'] = list(engine.view(name).output_ids) == generate(runner.model, inputs, n, use_cache=False).token_ids
        cases.append(finish_case('interleaved_isolation', engine, ['a', 'b'], timeline, assertions))

    runner = InstrumentedTinyRunner()
    engine = LifecycleEngine(runner, capacity_tokens=8)
    for name in ('a', 'b'):
        engine.submit(name, prompt, count)
    engine.admit('a')
    engine.step('a')
    timeline = [capture(engine, ['a', 'b'], 'before_close')]
    engine.close()
    engine.close()
    timeline.append(capture(engine, ['a', 'b'], 'after_close'))
    cases.append(finish_case('engine_close', engine, ['a', 'b'], timeline,
                            {'both_cancelled': all(engine.view(n).status == Status.CANCELLED for n in ('a', 'b'))}))
    return cases


def run_real_cases():
    import torch
    from transformers import AutoTokenizer
    from scripts.experiments.lesson0202 import prepare_snapshot, RECIPE
    from zero2vllm.modeling.checkpoint import load_checkpoint, read_json
    from zero2vllm.modeling.llama import greedy_generate
    torch.set_num_threads(2)
    directory = ROOT/'.local/models/smollm2-135m'
    receipt = prepare_snapshot(directory, read_json(RECIPE))
    model, _ = load_checkpoint(directory)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    runner = LlamaFullPrefixRunner(model)
    prompts = {'a': 'Once upon a time', 'b': 'The capital of France is'}
    ids = {name: tokenizer(text, add_special_tokens=False)['input_ids'] for name, text in prompts.items()}
    expected = {name: greedy_generate(model, torch.tensor([tokens]), 4, tokenizer.eos_token_id) for name, tokens in ids.items()}
    cases = []
    with LifecycleEngine(runner, capacity_tokens=32) as engine:
        for name in prompts:
            engine.submit(name, ids[name], 4, tokenizer.eos_token_id)
            engine.admit(name)
        engine.step('a')
        engine.step('b')
        timeline = [capture(engine, list(prompts), 'first_round')]
        engine.cancel('a')
        drain(engine, 'b')
        timeline.append(capture(engine, list(prompts), 'a_cancelled_b_finished'))
        cases.append(finish_case('real_cancel_isolation', engine, list(prompts), timeline,
                                {'a_prefix_preserved': list(engine.view('a').output_ids) == expected['a'][:1],
                                 'b_matches_0202_loop': list(engine.view('b').output_ids) == expected['b'],
                                 'same_shared_model': runner.model is model}))
    with LifecycleEngine(runner, capacity_tokens=32) as engine:
        engine.submit('fresh', ids['a'], 4, tokenizer.eos_token_id)
        engine.admit('fresh')
        drain(engine, 'fresh')
        cases.append(finish_case('real_reuse_after_cancel', engine, ['fresh'], [],
                                {'matches_0202_loop': list(engine.view('fresh').output_ids) == expected['a']}))
    return {'verification': 'executed', 'reference': 'lesson0202 greedy_generate; same model, separate request loop',
            'checkpoint': receipt, 'prompts': prompts, 'expected_ids': expected,
            'torch': torch.__version__, 'dtype': 'float32', 'threads': 2,
            'cache_kind': runner.cache_kind, 'cases': cases, 'passed': all(c['passed'] for c in cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-checkpoint', action='store_true', help='Reuse the existing, verified 2.2 snapshot offline.')
    parser.add_argument('--output-dir', type=Path, default=Path('.local/runs/part-02-engine/03-request-and-cache-lifecycles'))
    args = parser.parse_args()
    cases = run_tiny_cases()
    real = run_real_cases() if args.real_checkpoint else {'verification': 'not_run', 'passed': None}
    report = {'experiment': 'part2-03-request-lifecycle', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'source': source_fingerprint(), 'environment': {'python': platform.python_version(), 'system': platform.platform(), 'device': 'cpu'},
              'mode': 'synchronous single-thread; manually ordered steps',
              'tiny_model': {'vocab_size': 16, 'width': 8, 'max_context': 32, 'seed': 7},
              'budget_unit': 'logical maximum input token positions; not bytes or physical blocks',
              'cases': cases, 'real_checkpoint': real,
              'gpu_status': {'cuda-a100': 'not_run', 'maca-c500': 'not_run'},
              'passed': all(c['passed'] for c in cases) and real['passed'] is not False}
    output = ROOT/args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for case in cases + real.get('cases', []):
        print(f'{"PASS" if case["passed"] else "FAIL"} {case["name"]}')
    print(f'Real checkpoint: {real["verification"]}; summary: {output / "summary.json"}')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
