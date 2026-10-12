"""Five requested positions need eight slots when a page holds four."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from zero2vllm.engine.lifecycle import LifecycleEngine
from zero2vllm.engine.paged_runner import PagedTinyRunner
from zero2vllm.engine.scheduler import BatchScheduler


def main():
    runner = PagedTinyRunner(num_blocks=3, block_size=4)
    with LifecycleEngine(runner) as engine:
        scheduler = BatchScheduler(engine, max_running=2)
        for name in ('a', 'b'):
            scheduler.submit(name, [1, 2, 3], 3)
            print(name, 'logical=', engine.view(name).required_tokens,
                  'rounded=', engine.reservation_for(name))
        while scheduler.has_pending:
            trace = scheduler.tick()
            print(f'r={trace.round_index} run={trace.executed} wait={trace.waiting} '
                  f'reserved={engine.used_tokens} free_blocks={runner.store.free_blocks}')
            assert engine.used_tokens == (runner.store.num_blocks-runner.store.free_blocks)*runner.store.block_size
        assert scheduler.round_index == 6
    assert runner.store.free_blocks == 3
    print('PASS: whole-block admission and token ledger agree.')


if __name__ == '__main__':
    main()
