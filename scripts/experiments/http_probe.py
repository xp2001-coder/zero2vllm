"""Controlled ASGI and real loopback HTTP probes for lesson 2.7."""
import asyncio
from contextlib import asynccontextmanager
import json
import socket


async def eventually(predicate, timeout=3.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


class Exchange:
    """ASGI transport with controllable receive/disconnect/send boundaries."""
    def __init__(self, app, payload=None, *, raw=None, chunks=None, headers=None,
                 path='/generate', method='POST', send_hook=None, feed_body=True):
        self.app, self.send_hook = app, send_hook
        self.scope = {'type': 'http', 'path': path, 'method': method,
                      'headers': headers if headers is not None else [(b'content-type', b'application/json')]}
        self.incoming = asyncio.Queue()
        self.sent = []
        if feed_body:
            body = raw if raw is not None else json.dumps(payload).encode()
            parts = chunks if chunks is not None else [body]
            for i, part in enumerate(parts):
                self.incoming.put_nowait({'type': 'http.request', 'body': part, 'more_body': i < len(parts)-1})
        self.task = None

    async def send(self, message):
        if self.send_hook:
            await self.send_hook(message)
        self.sent.append(message)

    def start(self):
        self.task = asyncio.create_task(self.app(self.scope, self.incoming.get, self.send))
        return self

    async def finish(self):
        await asyncio.wait_for(self.task, 3.0)
        return self

    def disconnect(self):
        self.incoming.put_nowait({'type': 'http.disconnect'})

    @property
    def status(self):
        return next(x['status'] for x in self.sent if x['type'] == 'http.response.start')

    @property
    def body(self):
        return b''.join(x.get('body', b'') for x in self.sent)


async def collect(service, job):
    result = []
    while True:
        event = await service.next_event(job)
        result.append(event)
        if event['event'] in ('done', 'error'):
            return result


@asynccontextmanager
async def local_server(app):
    """Bind port 0 before launch, avoiding check-then-bind port races."""
    import uvicorn
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.setblocking(False)
    config = uvicorn.Config(app, loop='asyncio', http='h11', ws='none', lifespan='on',
                            log_level='error', timeout_graceful_shutdown=1)
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    task.result()
                    raise RuntimeError('HTTP server exited before startup.')
                await asyncio.sleep(0.005)
        yield f'http://127.0.0.1:{port}'
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=5)
        finally:
            sock.close()
