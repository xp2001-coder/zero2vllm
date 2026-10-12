"""A same-thread hook demonstrates cancellation ordering, not real concurrency."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from zero2vllm.engine.lifecycle import LifecycleEngine, Status
from zero2vllm.engine.runners import TinyCachedRunner


class CancelAfterCompute(TinyCachedRunner):
    callback = None

    def advance(self, *args):
        token = super().advance(*args)
        self.callback()
        return token


def main():
    runner = CancelAfterCompute()
    with LifecycleEngine(runner, capacity_tokens=8) as engine:
        engine.submit('a', [1, 2, 3], 4)
        engine.admit('a')
        def request_cancel():
            engine.cancel('a')
            during = engine.view('a')
            assert during.status == Status.RUNNING and during.cached_positions == 3
            assert during.reserved_tokens == 6
            print('During forward: cancellation latched; KV=3, reserved=6.')
        runner.callback = request_cancel
        after = engine.step('a')
        assert after.status == Status.CANCELLED and after.output_ids == ()
        assert after.cached_positions == 0 and engine.used_tokens == 0
        for event in engine.events('a'):
            print(event['sequence'], event['kind'], event['status'])
    print('PASS: no token committed after cancellation; release happened after return.')


if __name__ == '__main__':
    main()
