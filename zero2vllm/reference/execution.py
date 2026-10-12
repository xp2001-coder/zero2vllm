"""Lesson 3.3: immutable, CPU-only input packing; no upstream vLLM import."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestSlice:
    request_id: str
    token_ids: tuple[int, ...]
    computed: int
    scheduled: int
    block_table: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, 'token_ids', tuple(self.token_ids))
        object.__setattr__(self, 'block_table', tuple(self.block_table))


@dataclass(frozen=True)
class PackedInputs:
    requests: tuple[RequestSlice, ...]
    input_ids: tuple[int, ...]
    positions: tuple[int, ...]
    slot_mapping: tuple[int, ...]
    query_start_loc: tuple[int, ...]
    seq_lens: tuple[int, ...]
    logits_indices: tuple[int, ...]
    num_actual_tokens: int
    num_input_tokens: int

    @property
    def req_id_to_index(self):
        return {r.request_id: i for i, r in enumerate(self.requests)}


def pack_inputs(requests, order, *, block_size=4, num_blocks=10,
                vocab_size=32, pad_to=None, padding_token=0):
    """Pack a chosen execution order, with explicit non-shared block tables.

    `computed` is the BEFORE-execution count carried in the scheduling message,
    not the scheduler's already advanced live count. Padding is a teaching
    buffer experiment, not CUDA Graph capture. Block 0 is reserved as in 3.2.
    """
    if any(type(v) is not int or v <= 0 for v in (block_size, num_blocks, vocab_size)):
        raise ValueError('Positive integer sizes required')
    requests, order = tuple(requests), tuple(order)
    ids = [r.request_id for r in requests]
    if (not requests or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids) or len(order) != len(ids) or set(order) != set(ids)):
        raise ValueError('Require a nonempty, unique request set and exact order permutation')
    blocks = set()
    for r in requests:
        if (type(r.computed) is not int or type(r.scheduled) is not int
                or r.computed < 0 or r.scheduled <= 0
                or r.computed + r.scheduled > len(r.token_ids)):
            raise ValueError('Invalid computed/scheduled range')
        if any(type(t) is not int or not 0 <= t < vocab_size for t in r.token_ids):
            raise ValueError('Token outside vocabulary')
        needed = (r.computed + r.scheduled + block_size - 1) // block_size
        if len(r.block_table) < needed:
            raise ValueError('Block table does not cover this step')
        if any(type(b) is not int or not 0 < b < num_blocks for b in r.block_table):
            raise ValueError('Block ID outside pool or reserved null block')
        if len(set(r.block_table)) != len(r.block_table) or blocks.intersection(r.block_table):
            raise ValueError('This fixture requires disjoint non-aliased block tables')
        blocks.update(r.block_table)
    by_id = {r.request_id: r for r in requests}
    ordered = tuple(by_id[i] for i in order)
    tokens, positions, slots, boundaries, lengths = [], [], [], [0], []
    for r in ordered:
        end = r.computed + r.scheduled
        tokens.extend(r.token_ids[r.computed:end])
        positions.extend(range(r.computed, end))
        slots.extend(r.block_table[p // block_size]*block_size + p % block_size
                     for p in range(r.computed, end))
        boundaries.append(len(tokens))
        lengths.append(end)
    actual = len(tokens)
    padded = actual if pad_to is None else pad_to
    if type(padded) is not int or padded < actual:
        raise ValueError('Padded buffer must cover actual tokens')
    if type(padding_token) is not int or not 0 <= padding_token < vocab_size:
        raise ValueError('Padding token must be a valid embedding index')
    tokens.extend([padding_token]*(padded-actual))
    positions.extend([0]*(padded-actual))
    return PackedInputs(ordered, tuple(tokens), tuple(positions), tuple(slots),
                        tuple(boundaries), tuple(lengths),
                        tuple(b-1 for b in boundaries[1:]), actual, padded)


def route_candidates(batch, candidate_ids):
    """Retain one greedy candidate only for requests that caught up.

    The upstream non-speculative path samples partial prefills and discards
    their candidates; this lesson illustrates that boundary without RNG state.
    """
    if len(candidate_ids) != len(batch.requests):
        raise ValueError('One candidate per scheduled request required')
    if any(type(t) is not int or t < 0 for t in candidate_ids):
        raise ValueError('Candidate IDs must be nonnegative integers')
    return {r.request_id: ([candidate_ids[i]] if r.computed+r.scheduled == len(r.token_ids) else [])
            for i, r in enumerate(batch.requests)}
