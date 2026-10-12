import asyncio
import importlib.util
import json
import unittest

from zero2vllm.engine.runners import TinyCachedRunner
from zero2vllm.reference.attention import TinyAttentionLM, generate
from zero2vllm.serving.app import StreamingApp
from zero2vllm.serving.protocol import SSEDecoder, StreamValidator, encode_event
from zero2vllm.serving.service import RequestError, StreamService
from scripts.experiments.http_probe import Exchange, collect, eventually

PAYLOAD = {'prompt_ids': [1, 2, 3, 4, 5, 6], 'max_new_tokens': 3}


class ProtocolTests(unittest.TestCase):
    def test_client_checks_routing_sequence_and_terminal_count(self):
        accepted = {'event': 'accepted', 'data': {'request_id': 'a', 'seq': 0}}
        for bad in [
            {'event': 'token', 'data': {'request_id': 'b', 'seq': 1, 'index': 0, 'token_id': 12}},
            {'event': 'token', 'data': {'request_id': 'a', 'seq': 2, 'index': 0, 'token_id': 12}},
            {'event': 'token', 'data': {'request_id': 'a', 'seq': 1, 'index': 1, 'token_id': 12}},
            {'event': 'done', 'data': {'request_id': 'a', 'seq': 1, 'output_tokens': 1}},
            accepted,
        ]:
            validator = StreamValidator()
            validator.accept(accepted)
            with self.assertRaises(ValueError):
                validator.accept(bad)

    def test_client_allows_zero_output_and_rejects_events_after_done(self):
        validator = StreamValidator()
        validator.accept({'event': 'accepted', 'data': {'request_id': 'a', 'seq': 0}})
        done = {'event': 'done', 'data': {'request_id': 'a', 'seq': 1, 'output_tokens': 0}}
        validator.accept(done)
        self.assertTrue(validator.terminal)
        with self.assertRaises(ValueError):
            validator.accept(done)

    def test_arbitrary_byte_boundaries_and_utf8(self):
        events = [{'event': 'accepted', 'data': {'note': '逐字读取'}},
                  {'event': 'token', 'data': {'token_id': 12}}]
        raw = b': ping\n\n'+b''.join(map(encode_event, events))
        for step in (1, 2, 7, len(raw)):
            decoder, result = SSEDecoder(), []
            for start in range(0, len(raw), step):
                result.extend(decoder.feed(raw[start:start+step]))
            decoder.finish()
            self.assertEqual(result, events)

    def test_crlf_multiline_and_comments(self):
        decoder = SSEDecoder()
        raw = b'event: token\r\ndata: {\r\ndata: "token_id": 3}\r\n\r\n: keepalive\r\n\r\n'
        self.assertEqual(decoder.feed(raw), [{'event': 'token', 'data': {'token_id': 3}}])
        decoder.finish()

    def test_truncation_is_not_a_completed_event(self):
        decoder = SSEDecoder()
        self.assertEqual(decoder.feed(b'event: done\ndata: {}\n'), [])
        with self.assertRaises(ValueError):
            decoder.finish()

    def test_limits_and_malformed_data(self):
        for raw in (b'data: nope\n\n', b'data: "\xff"\n\n'):
            with self.assertRaises(ValueError):
                SSEDecoder().feed(raw)
        with self.assertRaises(ValueError):
            SSEDecoder(8).feed(b'data: 123456789')


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.services = []

    async def make(self, **kwargs):
        service = StreamService(**kwargs)
        self.services.append(service)
        await service.start()
        return service

    async def asyncTearDown(self):
        for service in self.services:
            await service.close()
            self.assertEqual(service.engine.used_tokens, 0)
            self.assertFalse(service.jobs)

    async def test_complete_order_and_reference_output(self):
        service = await self.make()
        job = service.open(PAYLOAD)
        events = await collect(service, job)
        self.assertEqual([e['event'] for e in events], ['accepted', 'token', 'token', 'token', 'done'])
        self.assertEqual([e['data']['seq'] for e in events], list(range(5)))
        expected = generate(TinyAttentionLM(max_context=32), PAYLOAD['prompt_ids'], 3, use_cache=False).token_ids
        self.assertEqual([e['data']['token_id'] for e in events if e['event'] == 'token'], expected)
        service.detach(job, 'response_complete')
        self.assertEqual(service.history[-1]['cached_positions'], 0)
        self.assertEqual(service.history[-1]['forward_calls'], 3)

    async def test_zero_output_and_eos(self):
        service = await self.make()
        for count, eos, expected in [(0, None, []), (8, 12, [12])]:
            job = service.open({**PAYLOAD, 'max_new_tokens': count, 'eos_token_id': eos})
            events = await collect(service, job)
            self.assertEqual([e['data']['token_id'] for e in events if e['event'] == 'token'], expected)
            self.assertEqual(events[-1]['data']['reason'], 'eos' if eos is not None else 'length')
            service.detach(job)

    async def test_full_queue_pauses_one_producer_and_peer_finishes(self):
        service = await self.make(queue_size=1)
        slow = service.open({**PAYLOAD, 'max_new_tokens': 8})
        await service.next_event(slow)  # consume accepted; allow its first token
        await eventually(slow.queue.full)
        before = service.engine.view(slow.request_id).forward_calls
        fast = service.open(PAYLOAD)
        events = await collect(service, fast)
        self.assertEqual(events[-1]['event'], 'done')
        self.assertEqual(service.engine.view(slow.request_id).forward_calls, before)
        self.assertEqual(slow.peak_queue, 1)
        service.detach(slow, 'test_cancel')
        service.detach(fast)

    async def test_slow_consumer_terminal_is_available_even_with_full_queue(self):
        service = await self.make(queue_size=1, slow_timeout=0.02)
        job = service.open(PAYLOAD)
        await eventually(lambda: job.terminal is not None)
        events = await collect(service, job)
        self.assertEqual([e['event'] for e in events], ['accepted', 'error'])
        self.assertEqual(events[-1]['data']['reason'], 'slow_consumer')
        self.assertEqual(service.engine.used_tokens, 0)

    async def test_deadline_and_waiting_cancel_do_not_allocate(self):
        service = await self.make(queue_size=1, request_timeout=0.02, slow_timeout=1)
        job = service.open(PAYLOAD)
        await eventually(lambda: job.terminal is not None)
        self.assertEqual(job.terminal['data']['reason'], 'deadline_exceeded')
        self.assertEqual(service.engine.view(job.request_id).forward_calls, 0)
        service.detach(job)
        other = service.open(PAYLOAD)
        service.cancel(other, 'client_disconnected')
        self.assertEqual(other.terminal['data']['reason'], 'client_disconnected')

    async def test_deadline_after_first_token_returns_running_cache(self):
        service = await self.make(request_timeout=0.03, step_delay=0.06)
        job = service.open({**PAYLOAD, 'max_new_tokens': 20})
        frames = await collect(service, job)
        self.assertEqual([e['event'] for e in frames], ['accepted', 'token', 'error'])
        self.assertEqual(frames[-1]['data']['reason'], 'deadline_exceeded')
        self.assertEqual(service.engine.used_tokens, 0)

    async def test_validation_busy_and_bounded_history(self):
        service = await self.make(max_requests=1, history_size=2)
        for bad in ({}, {'prompt_ids': [], 'max_new_tokens': 1},
                    {'prompt_ids': [True], 'max_new_tokens': 1},
                    {**PAYLOAD, 'max_new_tokens': -1}, {**PAYLOAD, 'unknown': 1},
                    {**PAYLOAD, 'max_new_tokens': 999}):
            with self.assertRaises(RequestError) as error:
                service.open(bad)
            self.assertEqual(error.exception.status, 422)
            self.assertFalse(service.jobs)
        for _ in range(4):
            job = service.open(PAYLOAD)
            with self.assertRaises(RequestError) as error:
                service.open(PAYLOAD)
            self.assertEqual(error.exception.status, 503)
            service.detach(job)
        self.assertEqual(len(service.history), 2)
        self.assertFalse(service.engine._requests)

    async def test_model_failure_preserves_committed_prefix_and_closes_cache(self):
        class FailSecond(TinyCachedRunner):
            def advance(self, prompt, generated, context):
                token = super().advance(prompt, generated, context)
                if generated:
                    raise RuntimeError('private diagnostic must not reach wire')
                return token
        service = await self.make(runner=FailSecond())
        job = service.open(PAYLOAD)
        events = await collect(service, job)
        self.assertEqual([e['event'] for e in events], ['accepted', 'token', 'error'])
        self.assertEqual(events[-1]['data']['reason'], 'model_error')
        self.assertNotIn('private diagnostic', json.dumps(events))
        self.assertEqual(service.engine.used_tokens, 0)

    async def test_close_stops_driver_and_forgets_requests(self):
        service = await self.make()
        service.open(PAYLOAD)
        await service.close()
        await service.close()
        self.assertTrue(service._task.done())
        with self.assertRaises(RequestError):
            service.open(PAYLOAD)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional CPU PyTorch is not installed')
    async def test_prefix_backend_shared_refs_survive_stream_cancellation(self):
        from zero2vllm.engine.prefix_runner import PrefixTinyRunner
        runner = PrefixTinyRunner(num_blocks=16)
        service = await self.make(runner=runner, queue_size=1)
        a = service.open({**PAYLOAD, 'max_new_tokens': 8})
        await service.next_event(a)
        await eventually(a.queue.full)
        b = service.open({**PAYLOAD, 'prompt_ids': [1, 2, 3, 4, 9, 10]})
        await service.next_event(b)
        await eventually(b.queue.full)
        self.assertTrue(any(v['refs'] == 2 for v in runner.store.pool_snapshot()['blocks']))
        service.detach(a, 'client_disconnected')
        result = await collect(service, b)
        self.assertEqual([e['data']['token_id'] for e in result if e['event'] == 'token'], [14, 13, 1])
        service.detach(b)
        self.assertEqual(runner.store.pool_snapshot()['active_block_references'], 0)


class ASGITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = StreamService()
        await self.service.start()
        self.app = StreamingApp(self.service)

    async def asyncTearDown(self):
        await self.service.close()

    async def test_normal_stream_and_fragmented_request_body(self):
        raw = json.dumps(PAYLOAD).encode()
        ex = Exchange(self.app, chunks=[raw[:4], raw[4:9], raw[9:]]).start()
        await ex.finish()
        self.assertEqual(ex.status, 200)
        events = SSEDecoder().feed(ex.body)
        self.assertEqual(events[-1]['event'], 'done')
        self.assertFalse(self.service.jobs)
        self.assertEqual(self.service.history[-1]['send_completed_tokens'], 3)
        self.assertFalse(ex.sent[-1]['more_body'])

    async def test_input_errors_happen_before_stream_start(self):
        cases = [(dict(raw=b'{'), 400), (dict(raw=b'\xff'), 400),
                 (dict(raw=b'{"max_new_tokens":NaN}'), 400),
                 (dict(payload=PAYLOAD, headers=[]), 415),
                 (dict(payload={**PAYLOAD, 'max_new_tokens': True}), 422)]
        for kwargs, status in cases:
            ex = Exchange(self.app, **kwargs).start()
            await ex.finish()
            self.assertEqual(ex.status, status)
            self.assertIn('error', json.loads(ex.body))
            self.assertFalse(self.service.jobs)

    async def test_body_byte_limit_and_timeout(self):
        app = StreamingApp(self.service, body_limit=10, body_timeout=0.02)
        ex = Exchange(app, chunks=[b'12345678', b'abcdefgh']).start()
        await ex.finish()
        self.assertEqual(ex.status, 413)
        ex = Exchange(app, feed_body=False).start()
        await ex.finish()
        self.assertEqual(ex.status, 408)

    async def test_disconnect_before_body_has_no_request(self):
        ex = Exchange(self.app, feed_body=False).start()
        ex.disconnect()
        await ex.finish()
        self.assertEqual(ex.sent, [])
        self.assertFalse(self.service.jobs)

    async def test_disconnect_during_blocked_send_returns_kv(self):
        blocked = asyncio.Event()
        async def stall(message):
            if message.get('body', b'').startswith(b'event: token'):
                blocked.set()
                await asyncio.Event().wait()
        ex = Exchange(self.app, {**PAYLOAD, 'max_new_tokens': 10}, send_hook=stall).start()
        await asyncio.wait_for(blocked.wait(), 1)
        ex.disconnect()
        await ex.finish()
        self.assertFalse(self.service.jobs)
        self.assertEqual(self.service.engine.used_tokens, 0)
        self.assertEqual(self.service.history[-1]['status'], 'cancelled')
        self.assertEqual(self.service.history[-1]['transport_end'], 'client_disconnected')

    async def test_handler_cancellation_cleans_children_and_engine(self):
        blocked = asyncio.Event()
        async def stall(message):
            blocked.set()
            await asyncio.Event().wait()
        ex = Exchange(self.app, PAYLOAD, send_hook=stall).start()
        await blocked.wait()
        ex.task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await ex.task
        self.assertFalse(self.service.jobs)
        self.assertEqual(self.service.engine.used_tokens, 0)
        self.assertFalse(any(t.get_name().startswith(('sse-', 'disconnect-')) for t in asyncio.all_tasks() if not t.done()))

    async def test_send_timeout_cleans_up_without_second_http_status(self):
        async def stall(message):
            if message['type'] == 'http.response.body':
                await asyncio.Event().wait()
        ex = Exchange(StreamingApp(self.service, send_timeout=0.02), PAYLOAD, send_hook=stall).start()
        await ex.finish()
        self.assertEqual([m['status'] for m in ex.sent if m['type'] == 'http.response.start'], [200])
        self.assertEqual(self.service.history[-1]['transport_end'], 'send_timeout')
        self.assertFalse(self.service.jobs)

    async def test_broken_transport_cleans_up(self):
        async def fail(message):
            raise OSError('socket closed')
        ex = Exchange(self.app, PAYLOAD, send_hook=fail).start()
        await ex.finish()
        self.assertEqual(self.service.history[-1]['transport_end'], 'connection_error')
        self.assertFalse(self.service.jobs)

    async def test_heartbeat_and_post_start_model_error(self):
        class Fail(TinyCachedRunner):
            def advance(self, *args):
                raise RuntimeError('server-only diagnostic')
        service = StreamService(Fail())
        await service.start()
        try:
            ex = Exchange(StreamingApp(service), PAYLOAD).start()
            await ex.finish()
            self.assertEqual(ex.status, 200)
            self.assertEqual(SSEDecoder().feed(ex.body)[-1]['event'], 'error')
            self.assertNotIn(b'server-only', ex.body)
        finally:
            await service.close()
        self.service.step_delay = 0.05
        ex = Exchange(StreamingApp(self.service, heartbeat=0.01), PAYLOAD).start()
        await ex.finish()
        self.assertIn(b': ping\n\n', ex.body)

    async def test_routes_busy_and_health(self):
        for path, method, status in [('/unknown', 'GET', 404), ('/generate', 'GET', 405), ('/health', 'GET', 200)]:
            ex = Exchange(self.app, path=path, method=method).start()
            await ex.finish()
            self.assertEqual(ex.status, status)
        self.service.max_requests = 1
        job = self.service.open(PAYLOAD)
        ex = Exchange(self.app, PAYLOAD).start()
        await ex.finish()
        self.assertEqual(ex.status, 503)
        self.assertIn((b'retry-after', b'1'), ex.sent[0]['headers'])
        self.service.detach(job)
