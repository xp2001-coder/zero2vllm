"""Iteration admission and token planning over the lesson 2.3 lifecycle engine.

The reference executor runs selected requests serially with independent real KV.
Rounds are logical boundaries, not equal-duration GPU batches. No chunked prefill.
Use a fresh engine exclusively through this scheduler; inspect views/events freely.
"""
from collections import deque
from dataclasses import asdict, dataclass, field

from zero2vllm.engine.lifecycle import LifecycleEngine, Status, TERMINAL


@dataclass(frozen=True)
class WorkItem:
    request_id: str
    phase: str
    input_tokens: int


@dataclass(frozen=True)
class RoundTrace:
    round_index: int
    cohort_id: int
    admitted: tuple[str, ...]
    waiting: tuple[str, ...]
    running: tuple[str, ...]
    planned: tuple[WorkItem, ...]
    skipped_budget: tuple[str, ...]
    admission_blocked: str | None
    reserved_before_execution: int
    executed: tuple[str, ...]
    terminal: tuple[str, ...]
    reserved_after_execution: int

    def to_dict(self):
        return asdict(self)


@dataclass
class _Timing:
    arrival: int
    admitted: int | None = None
    terminal: int | None = None
    token_boundaries: list[int] = field(default_factory=list)


class BatchScheduler:
    """FIFO admission, rotating per-round selection, and frozen execution plans.

    static: admit a cohort only when the previous cohort is entirely terminal.
    continuous: fill available active slots at every round boundary.
    Both use the same limits and selection rule. A whole prompt must fit one
    round. Rotation gives every active request first choice within max_running
    rounds; it does not remove FIFO head-of-line blocking in the waiting queue.
    """
    def __init__(self, engine: LifecycleEngine, *, policy='continuous',
                 max_running=2, max_tokens_per_round=8):
        if policy not in ('static', 'continuous'):
            raise ValueError('policy must be static or continuous.')
        for name, value in [('max_running', max_running),
                            ('max_tokens_per_round', max_tokens_per_round)]:
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if getattr(engine.runner, 'supports_cached_steps', False) is not True:
            raise ValueError('Scheduler requires P-then-1 cached steps; full-prefix runners have a different cost.')
        self.engine = engine
        self.policy = policy
        self.max_running = max_running
        self.max_tokens_per_round = max_tokens_per_round
        self.round_index = 0
        self.cohort_id = 0
        self._waiting = deque()
        self._active = deque()
        self._timings: dict[str, _Timing] = {}
        self._in_tick = False

    def _guard(self):
        if self._in_tick:
            raise RuntimeError('Scheduler operations must occur between rounds.')

    def submit(self, request_id, prompt, max_new_tokens, eos_token_id=None):
        self._guard()
        prompt = tuple(prompt)
        # Reject an impossible whole prefill before creating a waiting request.
        if max_new_tokens != 0 and len(prompt) > self.max_tokens_per_round:
            raise ValueError('Prompt exceeds one round; increase max_tokens_per_round. Chunking is not implemented.')
        view = self.engine.submit(request_id, prompt, max_new_tokens, eos_token_id)
        self._timings[request_id] = _Timing(arrival=self.round_index)
        if view.status in TERMINAL:
            self._timings[request_id].terminal = self.round_index
        else:
            self._waiting.append(request_id)
        return view

    def cancel(self, request_id):
        self._guard()
        timing = self._timings[request_id]
        changed = self.engine.cancel(request_id)
        if changed:
            timing.terminal = self.round_index
        return changed

    @property
    def has_pending(self):
        return any(self.engine.view(n).status not in TERMINAL for n in self._timings)

    def records(self):
        records = []
        for name, timing in self._timings.items():
            boundaries = list(timing.token_boundaries)
            records.append({**self.engine.view(name).to_dict(),
                            'arrival_boundary': timing.arrival,
                            'admission_boundary': timing.admitted,
                            'terminal_boundary': timing.terminal,
                            'token_boundaries': boundaries,
                            'first_token_delay_rounds': boundaries[0]-timing.arrival if boundaries else None,
                            'completion_delay_rounds': timing.terminal-timing.arrival if timing.terminal is not None else None,
                            'inter_token_gaps_rounds': [b-a for a, b in zip(boundaries, boundaries[1:])]})
        return records

    def tick(self):
        self._guard()
        self._in_tick = True
        try:
            return self._tick()
        finally:
            self._in_tick = False

    def _tick(self):
        r = self.round_index
        # Terminal members no longer occupy scheduler slots. Context cleanup was
        # already performed by LifecycleEngine, including boundary cancellation.
        self._waiting = deque(n for n in self._waiting if self.engine.view(n).status == Status.WAITING)
        self._active = deque(n for n in self._active if self.engine.view(n).status == Status.RUNNING)
        admitted, blocked = [], None
        if self.policy == 'static' and self._active:
            blocked = 'cohort_barrier' if self._waiting else None
        else:
            if self.policy == 'static' and self._waiting:
                self.cohort_id += 1
            while self._waiting and len(self._active) < self.max_running:
                name = self._waiting[0]
                if self.engine.reservation_for(name) > self.engine.available_tokens:
                    blocked = 'fifo_capacity'
                    break
                self._waiting.popleft()
                if self.engine.admit(name):
                    self._active.append(name)
                    admitted.append(name)
                    self._timings[name].admitted = r
                else:  # Context initialization failed and lifecycle rolled back.
                    self._timings[name].terminal = r
            if self._waiting and len(self._active) == self.max_running:
                blocked = 'max_running'

        running = tuple(self._active)
        planned, skipped, remaining = [], [], self.max_tokens_per_round
        for name in running:
            view = self.engine.view(name)
            prefill = not view.output_ids
            cost = view.prompt_tokens if prefill else 1
            if cost <= remaining:
                planned.append(WorkItem(name, 'prefill' if prefill else 'decode', cost))
                remaining -= cost
            else:
                skipped.append(name)
        # Rotate once even if the head's large prefill consumed the whole budget.
        # New arrivals append at the tail, so they cannot repeatedly overtake it.
        self._active.rotate(-1)
        reserved_before = self.engine.used_tokens
        executed = []
        # Freeze membership before any forward. A finished request cannot trigger
        # same-round admission; otherwise Python loop order changes the policy.
        for item in planned:
            if self.engine.view(item.request_id).status != Status.RUNNING:
                continue  # An earlier forward's cooperative hook cancelled it.
            before = len(self.engine.view(item.request_id).output_ids)
            view = self.engine.step(item.request_id)
            executed.append(item.request_id)
            if len(view.output_ids) > before:
                self._timings[item.request_id].token_boundaries.append(r+1)
        terminal = tuple(n for n in running if self.engine.view(n).status in TERMINAL)
        for name in terminal:
            self._timings[name].terminal = r+1
        trace = RoundTrace(r, self.cohort_id, tuple(admitted), tuple(self._waiting),
                           running, tuple(planned), tuple(skipped), blocked,
                           reserved_before, tuple(executed), terminal, self.engine.used_tokens)
        self.round_index += 1
        return trace
