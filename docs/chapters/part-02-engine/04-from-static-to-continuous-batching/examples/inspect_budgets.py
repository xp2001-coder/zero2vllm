"""Three limits: active slots, per-round input tokens, and retained capacity."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from zero2vllm.engine.lifecycle import LifecycleEngine
from zero2vllm.engine.runners import TinyCachedRunner
from zero2vllm.engine.scheduler import BatchScheduler


def main():
    with LifecycleEngine(TinyCachedRunner(), capacity_tokens=32) as engine:
        scheduler = BatchScheduler(engine, max_running=2, max_tokens_per_round=3)
        scheduler.submit('a', [1, 2, 3], 4)
        scheduler.submit('b', [4, 2], 3)
        for _ in range(2):
            t = scheduler.tick()
            cost = sum(i.input_tokens for i in t.planned)
            print(f'r={t.round_index}: active={t.running}, planned={t.planned}, skipped={t.skipped_budget}')
            print(f'  iteration={cost}/3; retained={t.reserved_after_execution}/32')
        # Leaving this scope cancels remaining work through the existing lifecycle.
    assert engine.used_tokens == 0

    with LifecycleEngine(TinyCachedRunner(), capacity_tokens=8) as engine:
        scheduler = BatchScheduler(engine, max_running=3, max_tokens_per_round=8)
        for name, prompt, count in [('a', [1, 2, 3], 4), ('b', [2, 3, 4], 3), ('c', [1], 1)]:
            scheduler.submit(name, prompt, count)
        t = scheduler.tick()
        print(f'\nFIFO: wait={t.waiting}; reason={t.admission_blocked}; available={engine.available_tokens}')
        assert t.waiting == ('b', 'c') and engine.available_tokens == 2
        scheduler.cancel('b')
        t = scheduler.tick()
        print(f'Cancel waiting head b: admitted={t.admitted}')
        assert t.admitted == ('c',)
    assert engine.used_tokens == 0
    print('\nPASS: iteration budget and retained capacity are separate ledgers.')


if __name__ == '__main__':
    main()
