"""Two requests share one model; cancellation returns admission capacity."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from zero2vllm.engine.lifecycle import LifecycleEngine, Status
from zero2vllm.engine.runners import TinyCachedRunner


def show(engine, stage):
    print(f'\n{stage}: used={engine.used_tokens}, available={engine.available_tokens}')
    for name in ('a', 'b'):
        view = engine.view(name)
        print(f'  {name}: {view.status.value:9} output={list(view.output_ids)} '
              f'KV={view.cached_positions} reserved={view.reserved_tokens}')


def main():
    with LifecycleEngine(TinyCachedRunner(), capacity_tokens=6) as engine:
        engine.submit('a', [1, 2, 3], 4)
        engine.submit('b', [4, 2, 1], 4)
        show(engine, '1. submitted')
        assert engine.admit('a') and not engine.admit('b')
        engine.step('a')
        show(engine, '2. a computed one token; b is waiting')
        engine.cancel('a')
        show(engine, '3. a cancelled')
        assert engine.admit('b')
        while engine.view('b').status == Status.RUNNING:
            engine.step('b')
        show(engine, '4. b finished')
        assert engine.used_tokens == 0
    print('\nPASS: cancellation returned capacity; b completed.')


if __name__ == '__main__':
    main()
