"""Two live requests share full blocks while keeping their different tails private."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from zero2vllm.engine.prefix_runner import PrefixTinyRunner
from zero2vllm.reference.attention import TinyAttentionLM, generate

runner = PrefixTinyRunner(num_blocks=12, block_size=2)
a, b = runner.new_context(), runner.new_context()
pa, pb = [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 9, 10]
try:
    a.reserve(8)
    b.reserve(8)
    runner.advance(pa, [], a)
    outputs = [runner.advance(pb, [], b)]
    sa, sb = runner.store.snapshot(a.handle), runner.store.snapshot(b.handle)
    print('A table:', sa['block_table'])
    print('B table:', sb['block_table'])
    print('B refs before A closes:', sb['block_refs'])
    assert sa['block_table'][:2] == sb['block_table'][:2]
    assert sb['block_refs'][:2] == [2, 2]
    a.close()
    print('B refs after A closes:', runner.store.snapshot(b.handle)['block_refs'])
    for _ in range(2):
        outputs.append(runner.advance(pb, outputs, b))
    assert outputs == generate(TinyAttentionLM(max_context=32), pb, 3, use_cache=False).token_ids
    print('B output:', outputs)
finally:
    a.close()
    b.close()
pool = runner.store.pool_snapshot()
print('final active refs:', pool['active_block_references'])
print('retained idle blocks:', pool['idle_cached_blocks'])
assert pool['active_block_references'] == 0
print('PASS: releasing A did not invalidate B; completed prefixes remain reusable.')
