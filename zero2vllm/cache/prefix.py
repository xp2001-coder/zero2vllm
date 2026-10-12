"""Single-layer CPU prefix KV: immutable full blocks, exact guarded hash lookup.

Active requests pin blocks; zero-reference full blocks stay cached until pressure.
Partial tails are private, so this lesson needs no copy-on-write. Metadata scans
and full-prefix collision guards favor inspectability over production complexity.
"""
from dataclasses import dataclass, field
import hashlib
import json
import math

import torch


@dataclass(frozen=True)
class PrefixHandle:
    serial: int


@dataclass
class _Block:
    refs: int = 0
    digest: str | None = None
    signature: tuple | None = None
    released_at: int = 0


@dataclass
class _Request:
    handle: PrefixHandle
    namespace: str
    prompt: tuple[int, ...]
    reserved: int
    table: list[int]
    tokens: list[int] = field(default_factory=list)
    hit_positions: int = 0


class PrefixKVStore:
    def __init__(self, num_blocks=16, block_size=2, *, head_dim=8,
                 dtype=torch.float64, digest_fn=None):
        if any(type(n) is not int or n < 1 for n in (num_blocks, block_size, head_dim)):
            raise ValueError('Block count, block size and head dimension must be positive integers.')
        if dtype not in (torch.float64, torch.float32):
            raise ValueError('This CPU lesson supports float64 and float32.')
        self.num_blocks, self.block_size, self.head_dim, self.dtype = num_blocks, block_size, head_dim, dtype
        self._keys = torch.full((num_blocks, block_size, 1, head_dim), float('nan'), dtype=dtype, device='cpu')
        self._values = torch.full_like(self._keys, float('nan'))
        self._blocks = [_Block() for _ in range(num_blocks)]
        self._index: dict[str, list[int]] = {}
        self._requests: dict[int, _Request] = {}
        self._serial = self._clock = 0
        self._digest = digest_fn or (lambda data: hashlib.sha256(data).hexdigest())
        self.events: list[dict] = []

    @property
    def capacity_tokens(self):
        return self.num_blocks * self.block_size

    def _entry(self, handle):
        entry = self._requests.get(handle.serial) if isinstance(handle, PrefixHandle) else None
        if entry is None or entry.handle is not handle:
            raise ValueError('Prefix handle is foreign, copied or already released.')
        return entry

    def _descriptors(self, tokens, namespace):
        parent = None
        for end in range(self.block_size, len(tokens)+1, self.block_size):
            chunk = tuple(tokens[end-self.block_size:end])
            payload = json.dumps([namespace, parent, chunk], ensure_ascii=False,
                                 separators=(',', ':')).encode('utf-8')
            digest = self._digest(payload)
            if not isinstance(digest, str) or not digest:
                raise ValueError('Digest function must return a nonempty string.')
            # The exact prefix guard makes even deliberately forced collisions safe.
            yield digest, (namespace, tuple(tokens[:end]))
            parent = digest

    def _find(self, digest, signature):
        return next((i for i in self._index.get(digest, ())
                     if self._blocks[i].signature == signature), None)

    def _evict(self, block_id):
        block = self._blocks[block_id]
        if block.refs:
            raise RuntimeError('Cannot evict an active block.')
        if block.digest is not None:
            bucket = self._index[block.digest]
            bucket.remove(block_id)
            if not bucket:
                del self._index[block.digest]
            self.events.append({'kind': 'evicted', 'block': block_id})
        block.digest = block.signature = None

    def acquire(self, prompt, reserved_positions, *, namespace='default', reuse=True):
        prompt = tuple(prompt)
        if not prompt or any(type(t) is not int or t < 0 for t in prompt):
            raise ValueError('Prompt must contain nonnegative integer token IDs.')
        if type(reserved_positions) is not int or reserved_positions < len(prompt):
            raise ValueError('Reservation must cover the entire nonempty prompt.')
        if not isinstance(namespace, str) or not namespace:
            raise ValueError('Namespace must be a nonempty string.')
        if type(reuse) is not bool:
            raise ValueError('reuse must be a bool.')
        # KV alone does not contain the final prompt logits. Leave >= 1 input.
        hit = []
        if reuse:
            for digest, signature in self._descriptors(prompt[:-1], namespace):
                found = self._find(digest, signature)
                if found is None:
                    break
                hit.append(found)
        needed = (reserved_positions+self.block_size-1)//self.block_size - len(hit)
        excluded = set(hit)
        candidates = [i for i, block in enumerate(self._blocks) if not block.refs and i not in excluded]
        # Prefer empty slots, then LRU among cached zero-reference blocks.
        candidates.sort(key=lambda i: (self._blocks[i].digest is not None,
                                      self._blocks[i].released_at if self._blocks[i].digest else i, i))
        if needed > len(candidates):
            raise MemoryError('Insufficient unpinned blocks; acquisition changed no state.')
        allocated = candidates[:needed]
        # All expected validation/capacity failures are above this commit point.
        for i in hit:
            self._blocks[i].refs += 1
        for i in allocated:
            self._evict(i)
            self._blocks[i].refs = 1
        self._serial += 1
        handle = PrefixHandle(self._serial)
        matched = len(hit)*self.block_size
        self._requests[handle.serial] = _Request(handle, namespace, prompt, reserved_positions,
                                                  hit+allocated, list(prompt[:matched]), matched)
        self.events.append({'kind': 'acquired', 'request': handle.serial,
                            'hit_blocks': list(hit), 'new_blocks': list(allocated)})
        return handle

    def _tensor(self, value, shape):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError('Unexpected CPU tensor shape.')
        if value.dtype != self.dtype or value.device.type != 'cpu' or value.requires_grad:
            raise ValueError('Tensor dtype/device must match; gradients are not supported.')
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Tensor values must be finite.')

    def _slots(self, entry, start, end):
        return [entry.table[p//self.block_size]*self.block_size+p%self.block_size for p in range(start, end)]

    def append(self, handle, token_ids, keys, values):
        entry = self._entry(handle)
        token_ids = list(token_ids)
        if not token_ids or any(type(t) is not int or t < 0 for t in token_ids):
            raise ValueError('Append requires nonnegative integer token IDs.')
        start, end = len(entry.tokens), len(entry.tokens)+len(token_ids)
        if end > entry.reserved:
            raise ValueError('Append exceeds the reserved logical length.')
        prompt_count = max(0, min(end, len(entry.prompt))-start)
        if tuple(token_ids[:prompt_count]) != entry.prompt[start:start+prompt_count]:
            raise ValueError('Appended tokens do not match the declared prompt.')
        for value in (keys, values):
            self._tensor(value, (len(token_ids), 1, self.head_dim))
        written_blocks = {entry.table[p//self.block_size] for p in range(start, end)}
        if any(self._blocks[i].refs != 1 or self._blocks[i].digest is not None for i in written_blocks):
            raise RuntimeError('A shared or published block is immutable.')
        slots = torch.tensor(self._slots(entry, start, end), dtype=torch.long, device='cpu')
        # Commit the valid prefix only after both writes; dirty invalid tails stay hidden.
        with torch.no_grad():
            self._keys.view(-1, 1, self.head_dim).index_copy_(0, slots, keys)
            self._values.view(-1, 1, self.head_dim).index_copy_(0, slots, values)
        entry.tokens.extend(token_ids)
        for logical, (digest, signature) in enumerate(self._descriptors(entry.tokens, entry.namespace)):
            i = entry.table[logical]
            block = self._blocks[i]
            if block.digest is None and self._find(digest, signature) is None:
                block.digest, block.signature = digest, signature
                self._index.setdefault(digest, []).append(i)
                self.events.append({'kind': 'published', 'block': i, 'positions': len(signature[1])})
        return tuple(slots.tolist())

    def gather(self, handle, upto=None):
        entry = self._entry(handle)
        upto = len(entry.tokens) if upto is None else upto
        if type(upto) is not int or not 0 <= upto <= len(entry.tokens):
            raise ValueError('Read length must stay within the written prefix.')
        slots = torch.tensor(self._slots(entry, 0, upto), dtype=torch.long, device='cpu')
        return tuple(t.view(-1, 1, self.head_dim).index_select(0, slots) for t in (self._keys, self._values))

    def attention(self, handle, query, *, upto=None):
        self._tensor(query, (1, self.head_dim))
        keys, values = self.gather(handle, upto)
        if not len(keys):
            raise ValueError('Attention requires a nonempty valid prefix.')
        scores = torch.einsum('hd,thd->ht', query, keys)/math.sqrt(self.head_dim)
        return torch.einsum('ht,thd->hd', scores.softmax(-1), values)

    def snapshot(self, handle):
        entry = self._entry(handle)
        return {'request': handle.serial, 'namespace': entry.namespace, 'block_table': list(entry.table),
                'reserved_positions': entry.reserved, 'written_positions': len(entry.tokens),
                'hit_positions': entry.hit_positions, 'tokens': list(entry.tokens),
                'block_refs': [self._blocks[i].refs for i in entry.table]}

    def pool_snapshot(self):
        return {'capacity_blocks': self.num_blocks, 'active_requests': len(self._requests),
                'active_unique_blocks': sum(b.refs > 0 for b in self._blocks),
                'active_block_references': sum(b.refs for b in self._blocks),
                'idle_cached_blocks': sum(b.refs == 0 and b.digest is not None for b in self._blocks),
                'empty_blocks': sum(b.refs == 0 and b.digest is None for b in self._blocks),
                'blocks': [{'id': i, 'refs': b.refs, 'digest': b.digest,
                            'signature': b.signature, 'released_at': b.released_at}
                           for i, b in enumerate(self._blocks)]}

    def release(self, handle):
        entry = self._entry(handle)
        # Release later blocks first, so tied request releases prefer keeping roots.
        for i in reversed(entry.table):
            block = self._blocks[i]
            block.refs -= 1
            if not block.refs:
                self._clock += 1
                block.released_at = self._clock
        del self._requests[handle.serial]
        self.events.append({'kind': 'released', 'request': handle.serial})

    def reset(self):
        if self._requests:
            raise RuntimeError('Cannot reset the cache while requests hold handles.')
        for i in range(self.num_blocks):
            self._evict(i)
        self.events.append({'kind': 'reset'})
