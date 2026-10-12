"""CPU tensor KV pages with explicit ownership, reservation and valid length.

Layout: separate K/V tensors [physical_block, offset, kv_head, head_dim].
One store represents one layer. Reads gather in logical order for correctness;
this is not a fused PagedAttention kernel. Released bytes are deliberately left
unchanged, making visibility bugs observable without relying on zero-fill.
"""
from dataclasses import dataclass
import math

import torch

from zero2vllm.reference.blocks import BlockPool


@dataclass(frozen=True)
class CacheHandle:
    key: str


@dataclass
class _Entry:
    handle: CacheHandle
    reserved: int = 0
    written: int = 0


class PagedKVStore:
    def __init__(self, num_blocks, block_size, *, kv_heads=1, head_dim=8,
                 dtype=torch.float64):
        self._pool = BlockPool(num_blocks, block_size)
        if any(type(n) is not int or n < 1 for n in (kv_heads, head_dim)):
            raise ValueError('KV dimensions must be positive integers.')
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('This CPU lesson supports float32 or float64 storage.')
        self.num_blocks, self.block_size = num_blocks, block_size
        self.kv_heads, self.head_dim, self.dtype = kv_heads, head_dim, dtype
        shape = (num_blocks, block_size, kv_heads, head_dim)
        self._keys = torch.full(shape, float('nan'), dtype=dtype, device='cpu')
        self._values = torch.full_like(self._keys, float('nan'))
        self._entries: dict[str, _Entry] = {}
        self._serial = 0

    @property
    def capacity_tokens(self):
        return self.num_blocks * self.block_size

    @property
    def free_blocks(self):
        return self._pool.free_blocks

    @property
    def storage_bytes(self):
        return 2 * self._keys.numel() * self._keys.element_size()

    def create(self):
        self._serial += 1
        handle = CacheHandle(f'cache-{self._serial}')
        self._pool.reserve(handle.key, 0)
        self._entries[handle.key] = _Entry(handle)
        return handle

    def _entry(self, handle):
        entry = self._entries.get(handle.key) if isinstance(handle, CacheHandle) else None
        # Object identity prevents copied, stale or foreign handles being reused.
        if entry is None or entry.handle is not handle:
            raise ValueError('Cache handle is foreign, copied or already released.')
        return entry

    def reserve(self, handle, max_positions):
        entry = self._entry(handle)
        table = self._pool.reserve(handle.key, max_positions)
        entry.reserved = max_positions  # Commit only after the allocator succeeds.
        return table

    def table(self, handle):
        self._entry(handle)
        return self._pool.table(handle.key)

    def length(self, handle):
        return self._entry(handle).written

    def locate(self, handle, position):
        self._entry(handle)
        return self._pool.locate(handle.key, position)

    def slot_mapping(self, handle, positions):
        """Addresses within the reservation; existence does not imply valid data."""
        self._entry(handle)
        slots = []
        for position in positions:
            block, offset = self.locate(handle, position)
            slots.append(block*self.block_size + offset)
        return tuple(slots)

    def _validate_tensor(self, tensor, trailing):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError('K/V and query inputs must be torch tensors.')
        if tensor.device.type != 'cpu' or tensor.dtype != self.dtype:
            raise ValueError('Tensor dtype/device must match the CPU store.')
        if tensor.ndim != len(trailing)+1 or tuple(tensor.shape[1:]) != trailing or tensor.shape[0] == 0:
            raise ValueError('Invalid tensor shape or empty input.')
        if tensor.requires_grad or not bool(torch.isfinite(tensor).all()):
            raise ValueError('Inference tensors must be finite and not require gradients.')

    def append(self, handle, keys, values):
        entry = self._entry(handle)
        self._validate_tensor(keys, (self.kv_heads, self.head_dim))
        self._validate_tensor(values, (self.kv_heads, self.head_dim))
        if keys.shape != values.shape:
            raise ValueError('Key/value shapes differ.')
        end = entry.written + keys.shape[0]
        if end > entry.reserved:
            raise ValueError('Append exceeds the reserved logical length; reserve first.')
        slots = torch.tensor(self.slot_mapping(handle, range(entry.written, end)), dtype=torch.long)
        # A copy failure can dirty unwritten slots, but cannot expose them: the
        # valid prefix is committed only after BOTH writes. Retrying overwrites it.
        with torch.no_grad():
            self._keys.view(-1, self.kv_heads, self.head_dim).index_copy_(0, slots, keys)
            self._values.view(-1, self.kv_heads, self.head_dim).index_copy_(0, slots, values)
        entry.written = end
        return tuple(slots.tolist())

    def gather(self, handle, upto=None):
        """Return copies of a valid logical prefix; never read padding/tail slots."""
        entry = self._entry(handle)
        if upto is None:
            upto = entry.written
        if type(upto) is not int or not 0 <= upto <= entry.written:
            raise ValueError('Read length must stay within the written prefix.')
        slots = torch.tensor(self.slot_mapping(handle, range(upto)), dtype=torch.long)
        return tuple(t.view(-1, self.kv_heads, self.head_dim).index_select(0, slots)
                     for t in (self._keys, self._values))

    def attention(self, handle, query, *, upto=None):
        """One query position, optionally GQA. Caller supplies its causal prefix."""
        self._entry(handle)
        self._validate_tensor(query, (self.head_dim,))
        if query.shape[0] % self.kv_heads:
            raise ValueError('Query head count must be a multiple of KV heads.')
        keys, values = self.gather(handle, upto)
        if keys.shape[0] == 0:
            raise ValueError('Attention needs at least one written key/value.')
        groups = query.shape[0] // self.kv_heads
        keys = keys.repeat_interleave(groups, dim=1)
        values = values.repeat_interleave(groups, dim=1)
        scores = torch.einsum('hd,thd->ht', query, keys) / math.sqrt(self.head_dim)
        probabilities = scores.softmax(dim=-1)
        return torch.einsum('ht,thd->hd', probabilities, values)

    def snapshot(self, handle):
        entry = self._entry(handle)
        table = self.table(handle)
        return {'handle': handle.key, 'block_table': table,
                'reserved_positions': entry.reserved, 'written_positions': entry.written,
                'allocated_slots': len(table)*self.block_size,
                'rounding_tail_slots': len(table)*self.block_size-entry.reserved,
                'reserved_unwritten_positions': entry.reserved-entry.written,
                'written_slot_mapping': self.slot_mapping(handle, range(entry.written))}

    def physical_snapshot(self):
        """Diagnostic copies of the whole pool, including invalid/stale bytes."""
        return self._keys.clone(), self._values.clone()

    def release(self, handle):
        self._entry(handle)
        self._pool.release(handle.key)
        del self._entries[handle.key]


class PagedContext:
    def __init__(self, store):
        self.store = store
        self.handle = store.create()
        self.closed = False

    @property
    def cached_positions(self):
        return 0 if self.closed else self.store.length(self.handle)

    def reserve(self, max_positions):
        if self.closed:
            raise ValueError('Context is closed.')
        self.store.reserve(self.handle, max_positions)

    def close(self):
        if not self.closed:
            self.store.release(self.handle)
            self.closed = True
