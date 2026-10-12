"""Small ASGI app: validate before response start, cancel and clean up on exit."""
import asyncio
import json
import math

from zero2vllm.serving.protocol import encode_event
from zero2vllm.serving.service import RequestError, StreamService


def reject_constant(value):
    raise ValueError(f'Nonstandard JSON constant: {value}')


class StreamingApp:
    def __init__(self, service=None, *, body_limit=8192, body_timeout=5.0,
                 send_timeout=5.0, heartbeat=1.0):
        if type(body_limit) is not int or body_limit < 1:
            raise ValueError('body_limit must be positive.')
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
               for v in (body_timeout, send_timeout, heartbeat)):
            raise ValueError('Transport timeouts must be finite and positive.')
        self.service = service if service is not None else StreamService()
        self.body_limit, self.body_timeout = body_limit, body_timeout
        self.send_timeout, self.heartbeat = send_timeout, heartbeat

    async def _send(self, send, message):
        await asyncio.wait_for(send(message), timeout=self.send_timeout)

    async def _json(self, send, status, data):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        headers = [(b'content-type', b'application/json; charset=utf-8'),
                   (b'content-length', str(len(body)).encode())]
        if status == 503:
            headers.append((b'retry-after', b'1'))
        if status == 405:
            headers.append((b'allow', b'POST'))
        await self._send(send, {'type': 'http.response.start', 'status': status, 'headers': headers})
        await self._send(send, {'type': 'http.response.body', 'body': body, 'more_body': False})

    async def _body(self, scope, receive):
        content_types = [v for k, v in scope.get('headers', []) if k.lower() == b'content-type']
        if len(content_types) != 1 or content_types[0].split(b';')[0].strip().lower() != b'application/json':
            raise RequestError(415, 'content_type', 'Use Content-Type: application/json.')
        body = bytearray()
        async with asyncio.timeout(self.body_timeout):
            while True:
                message = await receive()
                if message['type'] == 'http.disconnect':
                    raise ConnectionError('Disconnected while reading request body.')
                if message['type'] != 'http.request':
                    continue
                chunk = message.get('body', b'')
                if len(body)+len(chunk) > self.body_limit:
                    raise RequestError(413, 'body_too_large', 'Request body exceeds the byte limit.')
                body.extend(chunk)
                if not message.get('more_body', False):
                    break
        try:
            # Reject JSON's nonstandard NaN/Infinity extensions as well.
            return json.loads(body.decode('utf-8'), parse_constant=reject_constant)
        except (ValueError, UnicodeError, RecursionError) as error:
            raise RequestError(400, 'invalid_json', 'Body must contain valid UTF-8 JSON.') from error

    async def _stream(self, job, send):
        await self._send(send, {'type': 'http.response.start', 'status': 200, 'headers': [
            (b'content-type', b'text/event-stream; charset=utf-8'),
            (b'cache-control', b'no-cache'), (b'x-accel-buffering', b'no')]})
        while True:
            try:
                event = await asyncio.wait_for(self.service.next_event(job), timeout=self.heartbeat)
            except TimeoutError:
                await self._send(send, {'type': 'http.response.body', 'body': b': ping\n\n', 'more_body': True})
                continue
            await self._send(send, {'type': 'http.response.body', 'body': encode_event(event), 'more_body': True})
            if event['event'] == 'token':
                # This is transport handoff, NOT proof the client displayed a token.
                job.send_completed_tokens += 1
            if event['event'] in ('done', 'error'):
                break
        await self._send(send, {'type': 'http.response.body', 'body': b'', 'more_body': False})

    async def _disconnected(self, receive):
        while (await receive())['type'] != 'http.disconnect':
            pass

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'lifespan':
            while True:
                message = await receive()
                if message['type'] == 'lifespan.startup':
                    await self.service.start()
                    await send({'type': 'lifespan.startup.complete'})
                elif message['type'] == 'lifespan.shutdown':
                    await self.service.close()
                    await send({'type': 'lifespan.shutdown.complete'})
                    return
        if scope['type'] != 'http':
            raise ValueError('Only HTTP and lifespan scopes are supported.')
        if scope['path'] == '/health' and scope['method'] == 'GET':
            await self._json(send, 200, self.service.snapshot())
            return
        if scope['path'] != '/generate':
            await self._json(send, 404, {'error': {'code': 'not_found'}})
            return
        if scope['method'] != 'POST':
            await self._json(send, 405, {'error': {'code': 'method_not_allowed'}})
            return
        try:
            payload = await self._body(scope, receive)
            job = self.service.open(payload)
        except RequestError as error:
            await self._json(send, error.status, {'error': {'code': error.code, 'message': str(error)}})
            return
        except TimeoutError:
            await self._json(send, 408, {'error': {'code': 'body_timeout'}})
            return
        except ConnectionError:
            return
        stream = asyncio.create_task(self._stream(job, send), name='sse-'+job.request_id)
        disconnected = asyncio.create_task(self._disconnected(receive), name='disconnect-'+job.request_id)
        reason = 'handler_cancelled'
        try:
            done, _ = await asyncio.wait((stream, disconnected), return_when=asyncio.FIRST_COMPLETED)
            if disconnected in done:
                disconnected.result()
                reason = 'client_disconnected'
            else:
                stream.result()
                reason = 'response_complete'
        except TimeoutError:
            reason = 'send_timeout'
        except OSError:
            reason = 'connection_error'
        finally:
            # Return engine ownership first, then stop both transport tasks.
            self.service.cancel(job, reason)
            for task in (stream, disconnected):
                if not task.done():
                    task.cancel()
            self.service.detach(job, reason)
            await asyncio.gather(stream, disconnected, return_exceptions=True)
