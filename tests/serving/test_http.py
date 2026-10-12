import asyncio
import importlib.util
import unittest

from scripts.experiments.http_probe import eventually, local_server
from scripts.experiments.lesson0207 import invalid_http, read_http, tokens
from zero2vllm.serving.app import StreamingApp
from zero2vllm.serving.service import StreamService


@unittest.skipUnless(importlib.util.find_spec('uvicorn'), 'optional Uvicorn is not installed')
class HTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_real_clients_have_isolated_ordered_output(self):
        service = StreamService(step_delay=0.01)
        async with local_server(StreamingApp(service)) as url:
            results = await asyncio.gather(*[
                asyncio.to_thread(read_http, url, {'prompt_ids': prompt, 'max_new_tokens': 3})
                for prompt in ([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 9, 10])])
            self.assertEqual([tokens(v) for v in results], [[12, 3, 8], [14, 13, 1]])
            await eventually(lambda: not service.jobs)
            self.assertEqual(service.engine.used_tokens, 0)

    async def test_real_disconnect_cancels_generation_and_preserves_peer(self):
        service = StreamService(step_delay=0.03)
        async with local_server(StreamingApp(service)) as url:
            partial, full = await asyncio.gather(
                asyncio.to_thread(read_http, url, {'prompt_ids': [1, 2, 3, 4, 5, 6], 'max_new_tokens': 20}, 1),
                asyncio.to_thread(read_http, url, {'prompt_ids': [1, 2, 3, 4, 9, 10], 'max_new_tokens': 3}))
            await eventually(lambda: not service.jobs)
            record = next(r for r in service.history if r['request_id'] == partial[0]['data']['request_id'])
            self.assertEqual(record['status'], 'cancelled')
            self.assertLess(len(record['output_ids']), 20)
            self.assertEqual(tokens(full), [14, 13, 1])
            self.assertEqual(service.engine.used_tokens, 0)

    async def test_real_http_invalid_json_returns_400_before_generation(self):
        service = StreamService()
        async with local_server(StreamingApp(service)) as url:
            response = await asyncio.to_thread(invalid_http, url, b'{')
            self.assertEqual(response['status'], 400)
            self.assertFalse(service.jobs)
            self.assertEqual(len(service.history), 0)
