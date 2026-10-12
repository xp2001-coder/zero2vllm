"""One event loop owns the lifecycle engine; network tasks never run forwards.

This readiness-aware driver is separate from lesson 2.4's BatchScheduler. It
skips full output queues rather than blocking the engine on a client's put().
Synchronous tiny forwards still block this loop until they return.
"""
import asyncio
from collections import deque
from dataclasses import dataclass, field
import math
import uuid

from zero2vllm.engine.lifecycle import LifecycleEngine, Status, TERMINAL
from zero2vllm.engine.runners import TinyCachedRunner


class RequestError(ValueError):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code = status, code


@dataclass
class StreamJob:
    request_id: str
    queue: asyncio.Queue
    created: float
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    terminal: dict | None = None
    terminal_taken: bool = False
    sequence: int = 0
    full_since: float | None = None
    peak_queue: int = 0
    send_completed_tokens: int = 0


class StreamService:
    def __init__(self, runner=None, *, capacity_tokens=None, max_requests=8,
                 max_running=2, queue_size=2, request_timeout=30.0,
                 slow_timeout=5.0, step_delay=0.0, history_size=32):
        for value in (max_requests, max_running, queue_size, history_size):
            if type(value) is not int or value < 1:
                raise ValueError('Counts and queue sizes must be positive integers.')
        for value in (request_timeout, slow_timeout):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Timeouts must be finite and positive.')
        if not isinstance(step_delay, (int, float)) or not math.isfinite(step_delay) or step_delay < 0:
            raise ValueError('step_delay must be finite and nonnegative.')
        self.engine = LifecycleEngine(runner if runner is not None else TinyCachedRunner(), capacity_tokens)
        self.max_requests, self.max_running, self.queue_size = max_requests, max_running, queue_size
        self.request_timeout, self.slow_timeout, self.step_delay = request_timeout, slow_timeout, step_delay
        self.jobs: dict[str, StreamJob] = {}
        self.history = deque(maxlen=history_size)
        self.wake = asyncio.Event()
        self._task = None
        self.closed = False

    async def start(self):
        if self.closed:
            raise RuntimeError('Service cannot restart after close.')
        if self._task is None:
            self._task = asyncio.create_task(self._drive(), name='zero2vllm-stream-driver')

    def open(self, payload):
        if self.closed or self._task is None or self._task.done():
            raise RequestError(503, 'unavailable', 'Service is not running.')
        if not isinstance(payload, dict) or set(payload)-{'prompt_ids', 'max_new_tokens', 'eos_token_id'}:
            raise RequestError(422, 'invalid_request', 'Expected an object with supported generation fields.')
        if not isinstance(payload.get('prompt_ids'), list) or 'max_new_tokens' not in payload:
            raise RequestError(422, 'invalid_request', 'prompt_ids must be a list; max_new_tokens is required.')
        if len(self.jobs) >= self.max_requests:
            raise RequestError(503, 'busy', 'Too many open generation streams; retry later.')
        request_id = uuid.uuid4().hex
        try:
            self.engine.submit(request_id, payload['prompt_ids'], payload['max_new_tokens'], payload.get('eos_token_id'))
        except (ValueError, TypeError) as error:
            raise RequestError(422, 'invalid_request', str(error)) from error
        job = StreamJob(request_id, asyncio.Queue(self.queue_size), asyncio.get_running_loop().time())
        self.jobs[request_id] = job
        self._put(job, 'accepted', {'queue_capacity': self.queue_size})
        self._finish_if_terminal(job)
        self.wake.set()
        return job

    def _event(self, job, kind, data):
        event = {'event': kind, 'data': {'request_id': job.request_id, 'seq': job.sequence, **data}}
        job.sequence += 1
        return event

    def _put(self, job, kind, data):
        job.queue.put_nowait(self._event(job, kind, data))
        job.peak_queue = max(job.peak_queue, job.queue.qsize())
        job.ready.set()

    def _finish_if_terminal(self, job, reason=None):
        view = self.engine.view(job.request_id)
        if view.status not in TERMINAL or job.terminal is not None:
            return
        kind = 'done' if view.status == Status.FINISHED else 'error'
        # A separate terminal slot cannot be lost or block on a full data queue.
        job.terminal = self._event(job, kind, {
            'status': view.status.value, 'reason': reason or view.reason,
            'output_tokens': len(view.output_ids), 'forward_calls': view.forward_calls})
        job.ready.set()

    def cancel(self, job, reason='cancelled'):
        if job.request_id not in self.jobs:
            return
        self.engine.cancel(job.request_id)
        self._finish_if_terminal(job, reason)
        self.wake.set()

    async def next_event(self, job):
        while True:
            if not job.queue.empty():
                event = job.queue.get_nowait()
                job.full_since = None
                self.wake.set()
                return event
            if job.terminal is not None:
                if job.terminal_taken:
                    raise StopAsyncIteration
                job.terminal_taken = True
                return job.terminal
            if job.request_id not in self.jobs:
                raise StopAsyncIteration
            job.ready.clear()
            await job.ready.wait()

    def detach(self, job, reason='stream_closed'):
        if job.request_id not in self.jobs:
            return
        self.cancel(job, reason)
        view = self.engine.view(job.request_id)
        self.history.append({**view.to_dict(), 'terminal_event': job.terminal,
                             'transport_end': reason, 'peak_queue': job.peak_queue,
                             'send_completed_tokens': job.send_completed_tokens,
                             'engine_events': self.engine.events(job.request_id)})
        self.engine.forget(job.request_id)
        del self.jobs[job.request_id]
        while not job.queue.empty():
            job.queue.get_nowait()
        job.ready.set()

    def snapshot(self):
        return {'open_streams': len(self.jobs), 'used_tokens': self.engine.used_tokens,
                'capacity_tokens': self.engine.capacity_tokens, 'max_requests': self.max_requests,
                'queue_capacity': self.queue_size,
                'running': sum(self.engine.view(n).status == Status.RUNNING for n in self.jobs),
                'queue_lengths': {n: j.queue.qsize() for n, j in self.jobs.items()}}

    async def _drive(self):
        try:
            while not self.closed:
                self.wake.clear()
                now = asyncio.get_running_loop().time()
                for job in list(self.jobs.values()):
                    if job.terminal is not None:
                        continue
                    if now-job.created >= self.request_timeout:
                        self.cancel(job, 'deadline_exceeded')
                    elif job.queue.full():
                        if job.full_since is None:
                            job.full_since = now
                        elif now-job.full_since >= self.slow_timeout:
                            self.cancel(job, 'slow_consumer')
                    else:
                        job.full_since = None
                running = sum(self.engine.view(n).status == Status.RUNNING for n in self.jobs)
                # FIFO among ready waiters; a full output queue is temporarily skipped.
                for job in list(self.jobs.values()):
                    if running >= self.max_running:
                        break
                    if job.terminal is not None or job.queue.full():
                        continue
                    view = self.engine.view(job.request_id)
                    if view.status != Status.WAITING:
                        continue
                    if self.engine.reservation_for(job.request_id) > self.engine.available_tokens:
                        break
                    if self.engine.admit(job.request_id):
                        running += 1
                    self._finish_if_terminal(job)
                progressed = False
                for job in list(self.jobs.values()):
                    if job.request_id not in self.jobs or job.terminal is not None or job.queue.full():
                        continue
                    before = self.engine.view(job.request_id)
                    if before.status != Status.RUNNING:
                        continue
                    # No await inside this forward; engine access stays on this loop.
                    after = self.engine.step(job.request_id)
                    if len(after.output_ids) > len(before.output_ids):
                        self._put(job, 'token', {'index': len(after.output_ids)-1,
                                                  'token_id': after.output_ids[-1]})
                    self._finish_if_terminal(job)
                    progressed = True
                    await asyncio.sleep(0)
                if progressed:
                    # Nonzero delay is only a teaching aid, never a performance metric.
                    await asyncio.sleep(self.step_delay)
                else:
                    try:
                        await asyncio.wait_for(self.wake.wait(), timeout=0.01)
                    except TimeoutError:
                        pass
        finally:
            for job in list(self.jobs.values()):
                self.cancel(job, 'service_stopped')

    async def close(self):
        if self.closed:
            return
        self.closed = True
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        for job in list(self.jobs.values()):
            self.detach(job, 'service_stopped')
        self.engine.close()
