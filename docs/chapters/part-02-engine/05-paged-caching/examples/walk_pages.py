"""A's logical positions remain consecutive while its physical pages are not."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
import torch
from zero2vllm.cache.paged import PagedKVStore


def main():
    store = PagedKVStore(3, 2, head_dim=2)
    a, b = store.create(), store.create()
    try:
        store.reserve(a, 2)
        store.reserve(b, 2)  # B occupies block 1 between A's two blocks.
        keys = torch.tensor([[[1., 0.]], [[0., 1.]], [[1., 1.]]], dtype=torch.float64)
        values = torch.tensor([[[10., 0.]], [[20., 10.]], [[30., 20.]]], dtype=torch.float64)
        store.append(a, keys[:2], values[:2])
        store.reserve(a, 3)
        store.append(a, keys[2:], values[2:])
        print('A block table:', store.table(a))
        print('A slots:', store.slot_mapping(a, range(3)))
        print('A values in logical order:', store.gather(a)[1].squeeze(1).tolist())
        result = store.attention(a, torch.tensor([[1., 0.]], dtype=torch.float64))
        print('Attention:', result.tolist())
        print('Snapshot:', store.snapshot(a))
        assert store.table(a) == (0, 2)
        assert torch.equal(store.gather(a)[1], values)
        try:
            store.gather(a, upto=4)
        except ValueError as error:
            print('Expected read rejection:', error)
        else:
            raise AssertionError('Reading beyond valid history must fail.')
    finally:
        store.release(a)
        store.release(b)
    assert store.free_blocks == 3
    print('PASS: mapping, attention and cleanup.')


if __name__ == '__main__':
    main()
