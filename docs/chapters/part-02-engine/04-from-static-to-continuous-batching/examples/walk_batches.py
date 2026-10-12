"""Submit at iteration boundaries, then let the scheduler choose every step."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from zero2vllm.engine.lifecycle import LifecycleEngine
from zero2vllm.engine.runners import TinyCachedRunner
from zero2vllm.engine.scheduler import BatchScheduler


def main():
    outputs = {}
    for policy in ('static', 'continuous'):
        print(f'\n{policy}')
        with LifecycleEngine(TinyCachedRunner(), capacity_tokens=32) as engine:
            scheduler = BatchScheduler(engine, policy=policy, max_running=2, max_tokens_per_round=8)
            scheduler.submit('a', [1, 2, 3], 6)
            scheduler.submit('b', [4, 2], 2)
            while scheduler.has_pending:
                if scheduler.round_index == 1:
                    scheduler.submit('c', [5, 6], 2)
                if scheduler.round_index == 2:
                    scheduler.submit('d', [3, 1, 2], 2)
                trace = scheduler.tick()
                work = ', '.join(f'{i.request_id}:{i.phase}/{i.input_tokens}' for i in trace.planned)
                print(f'r={trace.round_index} work=[{work}] wait={trace.waiting} returned={trace.terminal}')
            outputs[policy] = {r['request_id']: r['output_ids'] for r in scheduler.records()}
            print(f'rounds={scheduler.round_index}; reserved={engine.used_tokens}')
            assert scheduler.round_index == (8 if policy == 'static' else 6)
            assert engine.used_tokens == 0
    assert outputs['static'] == outputs['continuous']
    print('\nPASS: same outputs, different admission boundaries.')


if __name__ == '__main__':
    main()
