"""Count-only teaching model for lesson 3.2, NOT a vLLM implementation.

Models one synchronous, text-only, full-attention path without speculative
tokens. Prefix identity is supplied explicitly in the separate BlockLedger
exercise; the scheduler itself runs with prefix caching disabled. No tensors,
hashes, sampling, timing, GPU allocation or upstream code are executed.
"""
from collections import Counter, deque
from dataclasses import dataclass


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'Expected an integer >= {minimum}')
    return value


def reusable_prefix_tokens(known_tokens, cached_full_blocks, block_size):
    """Full-attention teaching limit with one token left for logits.

    cached_full_blocks must describe a verified consecutive prefix. Identity
    lookup, prompt-logprob and EAGLE exceptions belong to upstream source.
    """
    integer(known_tokens, 1)
    integer(cached_full_blocks)
    integer(block_size, 1)
    return min(cached_full_blocks, (known_tokens - 1) // block_size) * block_size


class BlockLedger:
    """Physical block IDs, references and eviction candidates; block 0 reserved.

    Cached means an explicitly seeded, reusable full block in this exercise.
    reserve() does not compute content or register newly computed cache entries.
    Failure is atomic here; upstream sliding-window cleanup is out of scope.
    """
    def __init__(self, num_blocks, block_size=4):
        self.num_blocks = integer(num_blocks, 2)
        self.block_size = integer(block_size, 1)
        self.free = deque(range(1, num_blocks))
        self.refs = [0] * num_blocks
        self.tables = {}
        self.cached = set()

    def seed_cached(self, block_ids):
        ids = list(block_ids)
        if len(ids) != len(set(ids)) or any(b not in self.free for b in ids):
            raise ValueError('Seed only distinct unowned non-null blocks')
        self.cached.update(ids)

    def reserve(self, request_id, tokens, hits=()):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError('A nonempty request ID is required')
        integer(tokens, 1)
        hits = list(hits)
        old = self.tables.get(request_id, [])
        required = (tokens + self.block_size - 1) // self.block_size
        if (len(set(hits)) != len(hits) or any(b not in self.cached for b in hits)
                or (hits and old) or required < len(old) + len(hits)):
            raise ValueError('Invalid prefix or shrinking reservation')
        new_count = required - len(old) - len(hits)
        evictable_hits = sum(self.refs[b] == 0 for b in hits)
        if new_count > len(self.free) - evictable_hits:
            return None
        for block in hits:
            if self.refs[block] == 0:
                self.free.remove(block)
            self.refs[block] += 1
        fresh = []
        for _ in range(new_count):
            block = self.free.popleft()
            self.cached.discard(block)
            self.refs[block] = 1
            fresh.append(block)
        self.tables[request_id] = old + hits + fresh
        self.check()
        return fresh

    def release(self, request_id, *, reverse=True):
        # Reverse release illustrates the cache-enabled tail-first ordering.
        table = self.tables.pop(request_id, [])
        for block in reversed(table) if reverse else table:
            self.refs[block] -= 1
            if self.refs[block] == 0:
                self.free.append(block)
        self.check()

    def check(self):
        owners = Counter(b for table in self.tables.values() for b in table)
        assert len(set(self.free)) == len(self.free)
        assert 0 not in self.free and all(0 not in t for t in self.tables.values())
        assert set(self.free) == {b for b in range(1, self.num_blocks) if not self.refs[b]}
        assert all(self.refs[b] == owners[b] for b in range(1, self.num_blocks))
        assert all(len(t) == len(set(t)) for t in self.tables.values())
        assert self.cached <= set(range(1, self.num_blocks))

    def snapshot(self):
        return {'free_ids': list(self.free), 'cached_ids': sorted(self.cached),
                'tables': {k: list(v) for k, v in self.tables.items()},
                'references': list(self.refs)}


@dataclass
class CountRequest:
    request_id: str
    known: int
    computed: int = 0
    status: str = 'WAITING'


class CountScheduler:
    """Running-first scheduling, queue-head admission, recompute preemption.

    Requests contain counts only. complete() accepts teacher-supplied output
    counts (0 or 1), never predicts token IDs. Prefix caching is disabled here.
    """
    def __init__(self, *, token_budget=4, max_seqs=4, num_blocks=12,
                 block_size=4, threshold=0, max_model_len=64):
        self.budget = integer(token_budget, 1)
        self.max_seqs = integer(max_seqs, 1)
        self.threshold = integer(threshold)
        self.max_model_len = integer(max_model_len, 1)
        self.pool = BlockLedger(num_blocks, block_size)
        self.requests = {}
        self.running = []
        self.waiting = deque()
        self.pending = None

    def add(self, request, *, running=False):
        if self.pending is not None:
            raise ValueError('Complete the pending round before adding requests')
        integer(request.known, 1)
        integer(request.computed)
        if (not isinstance(request.request_id, str) or not request.request_id
                or request.request_id in self.requests
                or not 0 <= request.computed < request.known <= self.max_model_len
                or (not running and request.computed != 0)):
            raise ValueError('Invalid or duplicate request')
        if running:
            if len(self.running) >= self.max_seqs or request.computed == 0:
                raise ValueError('Invalid prepared running request')
            if self.pool.reserve(request.request_id, request.computed) is None:
                raise ValueError('Prepared state does not fit')
        request.status = 'RUNNING' if running else 'WAITING'
        self.requests[request.request_id] = request
        (self.running if running else self.waiting).append(request.request_id)

    def snapshot(self):
        return {'running': list(self.running), 'waiting': list(self.waiting),
                'requests': {k: {'known': r.known, 'computed': r.computed,
                                  'status': r.status} for k, r in self.requests.items()},
                'blocks': self.pool.snapshot()}

    def step(self):
        if self.pending is not None:
            raise ValueError('Complete the previous round first')
        before = self.snapshot()
        budget, index = self.budget, 0
        scheduled, preempted = [], []
        reason = 'queues_examined'

        def count(r):
            return min(r.known - r.computed, budget,
                       self.threshold or self.max_model_len,
                       self.max_model_len - r.computed)

        def record(r, n, fresh, kind):
            scheduled.append({'request': r.request_id, 'kind': kind,
                              'known_before': r.known, 'computed_before': r.computed,
                              'scheduled': n, 'new_block_ids': fresh,
                              'block_table': list(self.pool.tables[r.request_id])})

        while index < len(self.running) and budget:
            r = self.requests[self.running[index]]
            n = count(r)
            if not n:
                index += 1
                continue
            fresh = self.pool.reserve(r.request_id, r.computed + n)
            while fresh is None:
                victim = self.requests[self.running.pop()]
                self.pool.release(victim.request_id, reverse=False)
                victim.computed, victim.status = 0, 'PREEMPTED'
                self.waiting.appendleft(victim.request_id)
                preempted.append(victim.request_id)
                if victim is r:
                    break
                fresh = self.pool.reserve(r.request_id, r.computed + n)
            if fresh is None:
                break
            record(r, n, fresh, 'running')
            budget -= n
            index += 1

        if preempted:
            reason = 'preemption_blocks_admission'
        else:
            while self.waiting and budget:
                if len(self.running) >= self.max_seqs:
                    reason = 'max_seqs'
                    break
                r = self.requests[self.waiting[0]]
                n = count(r)
                fresh = self.pool.reserve(r.request_id, r.computed + n)
                if fresh is None:
                    reason = 'waiting_head_no_blocks'
                    break
                self.waiting.popleft()
                kind = 'resumed' if r.status == 'PREEMPTED' else 'new'
                self.running.append(r.request_id)
                r.status = 'RUNNING'
                record(r, n, fresh, kind)
                budget -= n
        for row in scheduled:
            self.requests[row['request']].computed += row['scheduled']
        self.pending = {row['request'] for row in scheduled}
        self.pool.check()
        return {'before': before, 'scheduled': scheduled, 'preempted': preempted,
                'unused_token_budget': budget,
                'stop_reason': 'token_budget' if not budget else reason,
                'after_schedule': self.snapshot()}

    def complete(self, outputs=None, finished=()):
        """Apply synthetic runner feedback; validate the whole message first."""
        if self.pending is None:
            raise ValueError('No pending round')
        outputs, finished = dict(outputs or {}), set(finished)
        if not set(outputs) <= self.pending or not finished <= self.pending:
            raise ValueError('Feedback must belong to scheduled requests')
        for rid in self.pending:
            r, n = self.requests[rid], outputs.get(rid, 0)
            integer(n)
            if n > 1 or (n and r.computed != r.known):
                raise ValueError('Partial prefill cannot produce output')
            if r.computed == r.known and not n and rid not in finished:
                raise ValueError('A caught-up request needs output or finish')
            if r.computed == r.known and r.known + n >= self.max_model_len and rid not in finished:
                raise ValueError('Finish requests reaching the model length limit')
        for rid in self.pending:
            r = self.requests[rid]
            r.known += outputs.get(rid, 0)
            if rid in finished:
                self.running.remove(rid)
                self.pool.release(rid, reverse=False)
                r.status = 'FINISHED'
        self.pending = None
        self.pool.check()
