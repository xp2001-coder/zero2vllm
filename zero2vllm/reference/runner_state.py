"""Lesson 3.4 CPU models of ownership and dispatch, not vLLM implementations."""
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class Row:
    request_id: str
    token_ids: tuple[int, ...]
    computed: int
    generation: int


@dataclass(frozen=True)
class Step:
    request_ids: tuple[str, ...]
    indices: tuple[int, ...]
    generations: tuple[int, ...]


class StablePool:
    """Use lowest free row for legible examples; upstream uses a free list.

    Generation guards are an extra teaching safeguard, NOT an upstream field.
    All methods are synchronous; this does not implement an async engine.
    """
    def __init__(self, capacity=3):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('capacity must be a positive integer')
        self.rows = [None] * capacity
        self.epochs = [0] * capacity
        self.index = {}

    def add(self, request_id, token_ids, computed=0):
        tokens = tuple(token_ids)
        if (not isinstance(request_id, str) or not request_id or request_id in self.index
                or not tokens or any(type(t) is not int or t < 0 for t in tokens)
                or type(computed) is not int or not 0 <= computed <= len(tokens)):
            raise ValueError('invalid or duplicate request')
        try:
            slot = self.rows.index(None)
        except ValueError:
            raise ValueError('no free state row') from None
        self.epochs[slot] += 1
        self.rows[slot] = Row(request_id, tokens, computed, self.epochs[slot])
        self.index[request_id] = slot
        return slot

    def remove(self, request_id):
        slot = self.index.pop(request_id, None)
        if slot is not None:
            self.rows[slot] = None

    def view(self, order):
        order = tuple(order)
        if len(set(order)) != len(order) or any(r not in self.index for r in order):
            raise ValueError('unknown or duplicate scheduled request')
        indices = tuple(self.index[r] for r in order)
        return Step(order, indices, tuple(self.rows[i].generation for i in indices))

    def gather(self, step):
        if not (len(step.request_ids) == len(step.indices) == len(step.generations)):
            raise ValueError('invalid step lengths')
        if len(set(step.indices)) != len(step.indices):
            raise ValueError('duplicate state row')
        rows = []
        for rid, i, epoch in zip(step.request_ids, step.indices, step.generations):
            if type(i) is not int or not 0 <= i < len(self.rows):
                raise ValueError('invalid state row')
            row = self.rows[i]
            if row is None or (row.request_id, row.generation) != (rid, epoch):
                raise ValueError('stale step: state row has been released or reused')
            rows.append(row)
        return tuple(rows)

    def advance(self, step, counts):
        rows, counts = self.gather(step), tuple(counts)
        if len(counts) != len(rows) or any(type(q) is not int or q < 0 or
                r.computed + q > len(r.token_ids) for r, q in zip(rows, counts)):
            raise ValueError('invalid progress update')
        # Validate the whole batch BEFORE writing any row.
        for i, row, q in zip(step.indices, rows, counts):
            self.rows[i] = replace(row, computed=row.computed + q)


class StagedTable:
    """List-backed disjoint scatter writes; no GPU, UVA, DMA or concurrency.

    Overlapping writes are rejected, not assigned an invented GPU write order.
    """
    def __init__(self, rows=3, columns=6):
        if any(type(v) is not int or v < 1 for v in (rows, columns)):
            raise ValueError('positive table dimensions required')
        self.table = [[0] * columns for _ in range(rows)]
        self.pending = []
        self.destinations = set()

    def stage(self, row, start, values):
        values = tuple(values)
        if (type(row) is not int or not 0 <= row < len(self.table)
                or type(start) is not int or not 0 <= start <= len(self.table[0])
                or start + len(values) > len(self.table[0])
                or any(type(v) is not int or not -(2**31) <= v < 2**31 for v in values)):
            raise ValueError('invalid int32 scatter update')
        cells = {(row, start + i) for i in range(len(values))}
        if self.destinations & cells:
            raise ValueError('overlapping pending writes have no defined order')
        if values:
            self.pending.append((row, start, values))
            self.destinations.update(cells)

    def packed(self):
        indices, starts, ends, contents = [], [], [], []
        for row, start, values in self.pending:
            indices.append(row)
            starts.append(start)
            contents.extend(values)
            ends.append(len(contents))
        return dict(indices=indices, starts=starts, cumulative_ends=ends, contents=contents,
                    payload_bytes=4 * (len(contents) + 3 * len(indices)),
                    full_table_bytes=4 * len(self.table) * len(self.table[0]))

    def apply(self):
        payload = self.packed()
        for row, start, values in self.pending:
            self.table[row][start:start+len(values)] = values
        self.pending.clear()
        self.destinations.clear()
        return payload


def copy_race_demo(snapshot):
    """Deterministic ownership demonstration, NOT an actual async transfer."""
    source = [3, 7]
    submitted = list(source) if snapshot else source
    source[0] = 9  # CPU starts the next step before the pretend transfer reads.
    return {'source_now': source, 'received': list(submitted), 'expected_previous_step': [3, 7]}


def expand_mapping(indices, counts):
    indices, counts = tuple(indices), tuple(counts)
    if len(indices) != len(counts) or any(type(i) is not int or i < 0 for i in indices) or any(type(n) is not int or n < 1 for n in counts):
        raise ValueError('one positive logits count per state index required')
    return [i for i, n in zip(indices, counts) for _ in range(n)], [j for n in counts for j in range(n)]


def graph_dispatch(mode, query_lengths, sizes=(1, 2, 4, 8), captured=()):
    """Single-card, ordinary decode policy illustration; no graph execution.

    Sizes are assumed prevalidated. Piecewise readiness belongs to its wrappers;
    only full-graph registry readiness is represented in `captured` here.
    """
    modes = {'NONE': ('NONE', 'NONE'), 'PIECEWISE': ('PIECEWISE', 'PIECEWISE'),
             'FULL': ('FULL', 'FULL'), 'FULL_DECODE_ONLY': ('FULL', 'NONE'),
             'FULL_AND_PIECEWISE': ('FULL', 'PIECEWISE')}
    qs, sizes, captured = tuple(query_lengths), tuple(sizes), tuple(captured)
    if mode not in modes or not qs or any(type(n) is not int or n < 1 for n in qs + sizes + captured):
        raise ValueError('invalid graph policy inputs')
    total = sum(qs)
    bucket = next((s for s in sorted(set(sizes)) if s >= total), None)
    selected = modes[mode][0 if all(q == 1 for q in qs) else 1]
    reason = 'configured policy'
    if selected == 'NONE': bucket, reason = None, 'mode disables graph for this batch'
    elif bucket is None: selected, reason = 'NONE', 'no shape bucket'
    elif selected == 'FULL' and bucket not in captured:
        selected, bucket, reason = 'NONE', None, 'full graph not captured'
    return dict(mode=selected, bucket=bucket, num_actual_tokens=total, reason=reason)


def selected_logprobs(logits, k):
    """Stable full-vocabulary normalizer, but only retain selected logprobs."""
    logits = tuple(logits)
    if not logits or any(not math.isfinite(x) for x in logits) or type(k) is not int or not 1 <= k <= len(logits):
        raise ValueError('finite logits and valid k required')
    maximum = max(logits)
    shifted_logsum = math.log(sum(math.exp(x - maximum) for x in logits))
    indices = sorted(range(len(logits)), key=lambda i: (-logits[i], i))[:k]
    return {'indices': indices, 'logprobs': [(logits[i] - maximum) - shifted_logsum for i in indices]}


def gumbel_choice(logits, uniforms):
    """Fixed uniforms demonstrate Gumbel-max algebra, not upstream RNG parity."""
    logits, uniforms = tuple(logits), tuple(uniforms)
    if (not logits or len(logits) != len(uniforms) or any(not math.isfinite(x) for x in logits)
            or any(not 0 < u < 1 for u in uniforms)):
        raise ValueError('finite logits and uniforms strictly between zero and one required')
    return max(range(len(logits)), key=lambda i: logits[i] - math.log(-math.log(uniforms[i])))
