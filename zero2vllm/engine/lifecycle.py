"""Single-threaded request ownership, explicit admission, one token per step.

No scheduling policy, physical block allocator, HTTP server or GPU concurrency.
Cancellation during a synchronous forward is observed at its return boundary.
"""
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Protocol

from zero2vllm.reference.attention import KVCache


class Status(str, Enum):
    WAITING = 'waiting'
    RUNNING = 'running'
    FINISHED = 'finished'
    CANCELLED = 'cancelled'
    FAILED = 'failed'


TERMINAL = {Status.FINISHED, Status.CANCELLED, Status.FAILED}


@dataclass
class RequestContext:
    """Owned by one request; only the tiny runner currently stores actual KV."""
    kv: KVCache | None = None
    closed: bool = False

    def reserve(self, max_positions):
        # List/full-prefix contexts need no physical allocation at admission.
        pass

    @property
    def cached_positions(self):
        return len(self.kv.keys) if self.kv is not None else 0

    def close(self):
        # Clear actual lists as well as dropping our reference. No backend callbacks.
        if self.kv is not None:
            self.kv.keys.clear()
            self.kv.values.clear()
            self.kv = None
        self.closed = True


class ExecutionContext(Protocol):
    @property
    def cached_positions(self) -> int: ...

    def reserve(self, max_positions: int) -> None: ...

    def close(self) -> None: ...


class Runner(Protocol):
    vocab_size: int
    max_context: int
    cache_kind: str

    def new_context(self) -> ExecutionContext: ...

    def advance(self, prompt: tuple[int, ...], generated: tuple[int, ...],
                context: ExecutionContext) -> int: ...


@dataclass(frozen=True)
class RequestView:
    request_id: str
    status: Status
    reason: str | None
    prompt_tokens: int
    prompt_ids: tuple[int, ...]
    max_new_tokens: int
    eos_token_id: int | None
    required_tokens: int
    output_ids: tuple[int, ...]
    reserved_tokens: int
    cached_positions: int
    peak_cached_positions: int
    forward_calls: int
    cancel_requested: bool
    error: str | None

    def to_dict(self):
        return asdict(self)


@dataclass
class _Request:
    request_id: str
    prompt: tuple[int, ...]
    max_new_tokens: int
    eos_token_id: int | None
    required_tokens: int
    status: Status = Status.WAITING
    reason: str | None = None
    outputs: list[int] = field(default_factory=list)
    reserved_tokens: int = 0
    context: ExecutionContext | None = None
    peak_cached_positions: int = 0
    forward_calls: int = 0
    cancel_requested: bool = False
    error: str | None = None
    events: list[dict] = field(default_factory=list)


class LifecycleEngine:
    """Reserve each admitted request's maximum input length up front.

    The default budget is a logical token ledger. A paged runner declares an
    allocation unit and a dedicated pool capacity, so admission rounds up to
    complete physical blocks. This still is not a measurement of process RAM.
    submit/admit/step are deliberately separate; the caller chooses their order.
    Only cancel/view/events are allowed during advance (a cooperative test hook),
    not concurrent calls from threads. Terminal results live until forget().
    """
    def __init__(self, runner: Runner, capacity_tokens: int | None = None):
        pool_capacity = getattr(runner, 'cache_capacity_tokens', None)
        if capacity_tokens is None:
            capacity_tokens = pool_capacity if pool_capacity is not None else 64
        if type(capacity_tokens) is not int or capacity_tokens < 1:
            raise ValueError('capacity_tokens must be a positive integer.')
        if pool_capacity is not None and capacity_tokens != pool_capacity:
            raise ValueError('Engine capacity must match the dedicated paged pool.')
        self.allocation_unit = getattr(runner, 'allocation_unit', 1)
        if type(self.allocation_unit) is not int or self.allocation_unit < 1:
            raise ValueError('allocation_unit must be a positive integer.')
        self.runner = runner
        self.capacity_tokens = capacity_tokens
        self._used = 0
        self._requests: dict[str, _Request] = {}
        self._busy: str | None = None
        self._closed = False
        self._sequence = 0

    @property
    def used_tokens(self):
        return self._used

    @property
    def available_tokens(self):
        return self.capacity_tokens - self._used

    def _rounded(self, positions):
        return ((positions + self.allocation_unit - 1) // self.allocation_unit) * self.allocation_unit

    def reservation_for(self, request_id):
        """Admission cost, including whole-block rounding for paged runners."""
        return self._rounded(self._requests[request_id].required_tokens)

    def _guard(self):
        if self._closed:
            raise RuntimeError('Engine is closed.')
        if self._busy is not None:
            raise RuntimeError('Nested execution is not supported; only cancellation may be requested.')

    def view(self, request_id):
        r = self._requests[request_id]
        return RequestView(
            request_id=r.request_id, status=r.status, reason=r.reason,
            prompt_tokens=len(r.prompt), prompt_ids=r.prompt,
            max_new_tokens=r.max_new_tokens, eos_token_id=r.eos_token_id,
            required_tokens=r.required_tokens, output_ids=tuple(r.outputs),
            reserved_tokens=r.reserved_tokens,
            cached_positions=r.context.cached_positions if r.context else 0,
            peak_cached_positions=r.peak_cached_positions, forward_calls=r.forward_calls,
            cancel_requested=r.cancel_requested, error=r.error,
        )

    def events(self, request_id):
        # Callers cannot edit the engine's event records through returned dictionaries.
        return [dict(e) for e in self._requests[request_id].events]

    def _event(self, r, kind, **extra):
        self._sequence += 1
        r.events.append({'sequence': self._sequence, 'kind': kind, 'status': r.status.value,
                         'output_tokens': len(r.outputs), 'reserved_tokens': r.reserved_tokens,
                         'cached_positions': r.context.cached_positions if r.context else 0,
                         'engine_used_tokens': self._used, **extra})

    def submit(self, request_id, prompt, max_new_tokens, eos_token_id=None):
        self._guard()
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError('A nonempty request ID is required.')
        if request_id in self._requests:
            raise ValueError('Duplicate request ID; retrieve/forget the old result first.')
        prompt = tuple(prompt)
        if not prompt or any(type(t) is not int or not 0 <= t < self.runner.vocab_size for t in prompt):
            raise ValueError('Prompt must contain valid integer token IDs.')
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError('max_new_tokens must be a nonnegative integer.')
        if eos_token_id is not None and (type(eos_token_id) is not int or not 0 <= eos_token_id < self.runner.vocab_size):
            raise ValueError('Invalid EOS token ID.')
        required = len(prompt) + max(0, max_new_tokens-1) if max_new_tokens else 0
        if max(len(prompt), required) > self.runner.max_context:
            raise ValueError('Request exceeds the model input context limit.')
        if self._rounded(required) > self.capacity_tokens:
            raise ValueError('Request can never fit the total token budget.')
        r = _Request(request_id, prompt, max_new_tokens, eos_token_id, required)
        self._requests[request_id] = r
        self._event(r, 'submitted')
        if max_new_tokens == 0:
            self._finish(r, Status.FINISHED, 'length')
        return self.view(request_id)

    def admit(self, request_id):
        self._guard()
        r = self._requests[request_id]
        if r.status != Status.WAITING:
            return False
        reserved = self.reservation_for(request_id)
        if reserved > self.available_tokens:
            self._event(r, 'admission_deferred')
            return False
        self._used += reserved
        r.reserved_tokens = reserved
        try:
            r.context = self.runner.new_context()
            r.context.reserve(r.required_tokens)
        except Exception as error:
            r.error = f'{type(error).__name__}: {error}'
            self._finish(r, Status.FAILED, 'initialization_error')
            return False
        r.status = Status.RUNNING
        self._event(r, 'admitted')
        return True

    def _finish(self, r, status, reason):
        if r.status in TERMINAL:
            return
        # Only called after forward returns; never free a buffer still in use.
        if r.context is not None:
            r.peak_cached_positions = max(r.peak_cached_positions, r.context.cached_positions)
            r.context.close()
            r.context = None
        released = r.reserved_tokens
        self._used -= released
        r.reserved_tokens = 0
        r.status, r.reason = status, reason
        self._event(r, 'finalized', reason=reason, released_tokens=released)

    def cancel(self, request_id):
        r = self._requests[request_id]
        if r.status in TERMINAL or r.cancel_requested:
            return False
        r.cancel_requested = True
        self._event(r, 'cancel_requested')
        if self._busy != request_id:
            self._finish(r, Status.CANCELLED, 'cancelled')
        return True

    def step(self, request_id):
        self._guard()
        r = self._requests[request_id]
        if r.status in TERMINAL:
            return self.view(request_id)
        if r.status != Status.RUNNING:
            raise ValueError('Admit the request before stepping it.')
        r.forward_calls += 1
        self._event(r, 'forward_started')
        self._busy = request_id
        try:
            token = self.runner.advance(r.prompt, tuple(r.outputs), r.context)
            if type(token) is not int or not 0 <= token < self.runner.vocab_size:
                raise ValueError('Runner produced an invalid token ID.')
        except Exception as error:
            r.error = f'{type(error).__name__}: {error}'
            # A failed forward is reported as failure, even if cancellation arrived.
            self._finish(r, Status.FAILED, 'model_error')
            return self.view(request_id)
        finally:
            self._busy = None
        r.peak_cached_positions = max(r.peak_cached_positions, r.context.cached_positions)
        self._event(r, 'forward_returned')
        if r.cancel_requested:
            self._finish(r, Status.CANCELLED, 'cancelled')
        else:
            r.outputs.append(token)
            self._event(r, 'token_committed', token_id=token)
            if token == r.eos_token_id:
                self._finish(r, Status.FINISHED, 'eos')
            elif len(r.outputs) == r.max_new_tokens:
                self._finish(r, Status.FINISHED, 'length')
        return self.view(request_id)

    def forget(self, request_id):
        self._guard()
        if self._requests[request_id].status not in TERMINAL:
            raise ValueError('Cannot forget an active request; cancel it first.')
        del self._requests[request_id]

    def close(self):
        if self._closed:
            return
        self._guard()
        for request_id in tuple(self._requests):
            self.cancel(request_id)
        self._closed = True

    def __enter__(self):
        self._guard()
        return self

    def __exit__(self, *exc):
        self.close()
