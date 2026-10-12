"""Inspect async iteration and explicit early-close ownership without HTTP."""
import asyncio
from contextlib import aclosing
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from zero2vllm.serving.service import StreamService


async def request_events(service, payload):
    job = service.open(payload)
    try:
        while True:
            event = await service.next_event(job)
            yield event
            if event['event'] in ('done', 'error'):
                return
    finally:
        # aclosing ensures even an early async-for break runs this immediately.
        service.detach(job, 'example_consumer_closed')


async def main():
    service = StreamService(queue_size=1)
    await service.start()
    payload = {'prompt_ids': [1, 2, 3, 4, 5, 6], 'max_new_tokens': 3}
    try:
        ids = []
        async with aclosing(request_events(service, payload)) as stream:
            async for event in stream:
                print('full:', event['event'], 'seq=', event['data']['seq'])
                if event['event'] == 'token':
                    ids.append(event['data']['token_id'])
        assert ids == [12, 3, 8]
        print('full output:', ids)
        async with aclosing(request_events(service, {**payload, 'max_new_tokens': 20})) as stream:
            async for event in stream:
                print('early:', event['event'])
                if event['event'] == 'token':
                    break
        assert not service.jobs and service.engine.used_tokens == 0
        assert service.history[-1]['status'] == 'cancelled'
        print('after early close: open_streams=0 used_tokens=0 status=cancelled')
        print('PASS: async iteration and explicit cleanup.')
    finally:
        await service.close()


if __name__ == '__main__':
    asyncio.run(main())
