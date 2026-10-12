"""Run the same prompt twice and compare actual computation, not elapsed time."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from zero2vllm.engine.prefix_runner import PrefixTinyRunner
from zero2vllm.reference.attention import TinyAttentionLM, generate

prompt, count = [1, 2, 3, 4, 5, 6], 3
runner = PrefixTinyRunner(num_blocks=8, block_size=2)
reference = generate(TinyAttentionLM(max_context=32), prompt, count, use_cache=False)
for name in ('cold', 'warm'):
    context = runner.new_context()
    outputs = []
    try:
        context.reserve(len(prompt)+count-1)
        for _ in range(count):
            outputs.append(runner.advance(prompt, outputs, context))
        print(name, 'hits=', context.hit_positions,
              'projected=', context.projected_positions,
              'attention_pairs=', context.attention_pairs,
              'output=', outputs)
        assert outputs == reference.token_ids
    finally:
        context.close()
    pool = runner.store.pool_snapshot()
    print('after release: active_refs=', pool['active_block_references'],
          'idle_cached=', pool['idle_cached_blocks'])
    assert pool['active_block_references'] == 0
assert context.hit_positions == 4 and context.projected_positions == 4
print('PASS: identical outputs; four projected positions saved on the warm request.')
