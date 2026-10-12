"""Lesson 2.7: real HTTP streaming plus controlled backpressure/failure probes."""
import argparse
import asyncio
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import urllib.error
import urllib.request

from scripts.common import ROOT, source_fingerprint
from scripts.experiments.http_probe import Exchange, collect, eventually, local_server
from zero2vllm.engine.runners import TinyCachedRunner
from zero2vllm.reference.attention import TinyAttentionLM, generate
from zero2vllm.serving.app import StreamingApp
from zero2vllm.serving.client import events
from zero2vllm.serving.protocol import SSEDecoder, encode_event
from zero2vllm.serving.service import StreamService

LESSON = ROOT/'docs/chapters/part-02-engine/07-http-and-streaming-output'


def result(name, level, data, checks):
    return {'name': name, 'level': level, **data, 'assertions': checks, 'passed': all(checks.values())}


def tokens(frames):
    return [frame['data']['token_id'] for frame in frames if frame['event'] == 'token']


def read_http(url, payload, stop_after=0):
    stream, received = events(url+'/generate', payload), []
    try:
        for event in stream:
            received.append(event)
            if stop_after and len(tokens(received)) >= stop_after:
                break
    finally:
        stream.close()
    return received


def invalid_http(url, raw, content_type='application/json'):
    request = urllib.request.Request(url+'/generate', raw, {'Content-Type': content_type})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=3) as response:
            return {'status': response.status, 'body': response.read().decode()}
    except urllib.error.HTTPError as error:
        return {'status': error.code, 'body': json.loads(error.read())}


async def http_cases(data):
    service = StreamService(step_delay=0.03)
    cases, payload = [], data['normal']
    async with local_server(StreamingApp(service)) as url:
        for name, request, expected in [('http-normal', payload, data['expected_ids']),
                                        ('http-zero', {**payload, 'max_new_tokens': 0}, []),
                                        ('http-eos', {**payload, 'max_new_tokens': 8, 'eos_token_id': 12}, [12])]:
            frames = await asyncio.to_thread(read_http, url, request)
            await eventually(lambda: not service.jobs)
            cases.append(result(name, 'real_loopback_http', {'events': frames, 'record': service.history[-1]},
                                {'reference_output': tokens(frames) == expected,
                                 'one_done': sum(e['event'] == 'done' for e in frames) == 1,
                                 'ordered': [e['data']['seq'] for e in frames] == list(range(len(frames))),
                                 'released': service.engine.used_tokens == 0}))
        invalid = []
        for raw, content_type in [(b'{', 'application/json'),
                                   (json.dumps({**payload, 'prompt_ids': [99]}).encode(), 'application/json'),
                                   (b'hello', 'text/plain')]:
            invalid.append(await asyncio.to_thread(invalid_http, url, raw, content_type))
        cases.append(result('http-input-errors', 'real_loopback_http', {'responses': invalid},
                            {'statuses': [v['status'] for v in invalid] == [400, 422, 415],
                             'no_requests': not service.jobs, 'no_reservation': service.engine.used_tokens == 0}))
        other = {**payload, 'prompt_ids': data['branch_prompt']}
        concurrent = await asyncio.gather(asyncio.to_thread(read_http, url, payload),
                                          asyncio.to_thread(read_http, url, other))
        await eventually(lambda: not service.jobs)
        cases.append(result('http-two-clients', 'real_loopback_http', {'streams': concurrent},
                            {'isolated_outputs': [tokens(v) for v in concurrent] == [data['expected_ids'], data['branch_expected_ids']],
                             'different_ids': concurrent[0][0]['data']['request_id'] != concurrent[1][0]['data']['request_id'],
                             'released': service.engine.used_tokens == 0}))
        partial, survivor = await asyncio.gather(
            asyncio.to_thread(read_http, url, {**payload, 'max_new_tokens': 20}, 1),
            asyncio.to_thread(read_http, url, other))
        await eventually(lambda: not service.jobs)
        closed_id = partial[0]['data']['request_id']
        record = next(r for r in service.history if r['request_id'] == closed_id)
        cases.append(result('http-disconnect', 'real_loopback_http',
                            {'received_before_close': partial, 'closed_request': record, 'survivor': survivor},
                            {'client_read_one': len(tokens(partial)) == 1,
                             'no_done_claim': all(e['event'] != 'done' for e in partial),
                             'cancelled_before_limit': record['status'] == 'cancelled' and len(record['output_ids']) < 20,
                             'peer_correct': tokens(survivor) == data['branch_expected_ids'],
                             'released': service.engine.used_tokens == 0}))
    return cases


async def controlled_cases(data):
    payload, cases = data['normal'], []
    service = StreamService(queue_size=2)
    await service.start()
    app = StreamingApp(service)
    stalled = asyncio.Event()
    async def block_token(message):
        if message.get('body', b'').startswith(b'event: token'):
            stalled.set()
            await asyncio.Event().wait()
    slow = Exchange(app, {**payload, 'max_new_tokens': 20}, send_hook=block_token).start()
    try:
        await asyncio.wait_for(stalled.wait(), 2)
        slow_job = next(iter(service.jobs.values()))
        await eventually(slow_job.queue.full)
        before = service.engine.view(slow_job.request_id).forward_calls
        fast = Exchange(app, payload).start()
        await fast.finish()
        after = service.engine.view(slow_job.request_id).forward_calls
        snapshot = service.snapshot()
        slow.disconnect()
        await slow.finish()
        cases.append(result('bounded-backpressure', 'controlled_asgi_send',
                            {'slow_calls_before_peer': before, 'slow_calls_after_peer': after,
                             'snapshot_while_blocked': snapshot, 'records': list(service.history)},
                            {'paused_producer': before == after, 'queue_bounded': slow_job.peak_queue == 2,
                             'peer_finished': tokens(SSEDecoder().feed(fast.body)) == data['expected_ids'],
                             'released': service.engine.used_tokens == 0 and not service.jobs}))
    finally:
        if not slow.task.done():
            slow.disconnect()
            await slow.finish()
        await service.close()

    service = StreamService()
    await service.start()
    async def block_body(message):
        if message['type'] == 'http.response.body':
            await asyncio.Event().wait()
    try:
        ex = Exchange(StreamingApp(service, send_timeout=0.03), payload, send_hook=block_body).start()
        await ex.finish()
        cases.append(result('send-timeout', 'controlled_asgi_send', {'record': service.history[-1]},
                            {'timeout_recorded': service.history[-1]['transport_end'] == 'send_timeout',
                             'released': service.engine.used_tokens == 0 and not service.jobs}))
    finally:
        await service.close()

    class FailSecond(TinyCachedRunner):
        def advance(self, prompt, generated, context):
            token = super().advance(prompt, generated, context)
            if generated:
                raise RuntimeError('injected diagnostic, retained only by the server')
            return token
    service = StreamService(FailSecond())
    await service.start()
    try:
        ex = Exchange(StreamingApp(service), payload).start()
        await ex.finish()
        frames = SSEDecoder().feed(ex.body)
        cases.append(result('error-after-start', 'controlled_asgi', {'status': ex.status, 'events': frames},
                            {'status_stays_200': ex.status == 200, 'partial_output': tokens(frames) == [12],
                             'explicit_error': frames[-1]['event'] == 'error',
                             'no_private_error_on_wire': b'injected diagnostic' not in ex.body,
                             'released': service.engine.used_tokens == 0}))
    finally:
        await service.close()

    service = StreamService(max_requests=1)
    await service.start()
    try:
        app = StreamingApp(service, body_limit=128, body_timeout=0.02)
        large = await Exchange(app, raw=b'x'*129).start().finish()
        timeout = await Exchange(app, feed_body=False).start().finish()
        job = service.open(payload)
        busy = await Exchange(app, payload).start().finish()
        service.detach(job)
        cases.append(result('body-and-admission-limits', 'controlled_asgi',
                            {'statuses': [large.status, timeout.status, busy.status]},
                            {'expected_errors': [large.status, timeout.status, busy.status] == [413, 408, 503],
                             'released': service.engine.used_tokens == 0 and not service.jobs}))
    finally:
        await service.close()

    for name, config, reason in [('request-deadline', {'request_timeout': 0.02}, 'deadline_exceeded'),
                                 ('slow-consumer', {'slow_timeout': 0.02}, 'slow_consumer')]:
        service = StreamService(queue_size=1, **config)
        await service.start()
        try:
            job = service.open(payload)
            await eventually(lambda: job.terminal is not None)
            frames = await collect(service, job)
            service.detach(job)
            cases.append(result(name, 'controlled_service', {'events': frames, 'record': service.history[-1]},
                                {'bounded_terminal': [e['event'] for e in frames] == ['accepted', 'error'],
                                 'reason': frames[-1]['data']['reason'] == reason,
                                 'released': service.engine.used_tokens == 0}))
        finally:
            await service.close()
    service = StreamService()
    await service.start()
    service.open(payload)
    await service.close()
    cases.append(result('service-shutdown', 'controlled_service', {'records': list(service.history)},
                        {'no_jobs': not service.jobs, 'no_reservation': service.engine.used_tokens == 0,
                         'driver_finished': service._task.done()}))
    return cases


def protocol_case():
    original = [{'event': 'accepted', 'data': {'说明': '流式事件'}},
                {'event': 'token', 'data': {'token_id': 12}}, {'event': 'done', 'data': {}}]
    raw = b': ping\n\n'+b''.join(map(encode_event, original))
    chunks = []
    for step in (1, 2, 7, len(raw)):
        decoder, parsed = SSEDecoder(), []
        for start in range(0, len(raw), step):
            parsed.extend(decoder.feed(raw[start:start+step]))
        decoder.finish()
        chunks.append({'chunk_bytes': step, 'parsed': parsed})
    decoder = SSEDecoder()
    decoder.feed(b'event: done\ndata: {}\n')
    truncated = False
    try:
        decoder.finish()
    except ValueError:
        truncated = True
    return result('sse-byte-fragments', 'protocol_bytes', {'fragmentations': chunks},
                  {'all_equal': all(v['parsed'] == original for v in chunks), 'truncated_rejected': truncated})


async def prefix_case(data):
    import torch
    from zero2vllm.engine.prefix_runner import PrefixTinyRunner
    runner = PrefixTinyRunner(num_blocks=32)
    service = StreamService(runner, step_delay=0.02)
    async with local_server(StreamingApp(service)) as url:
        streams = [await asyncio.to_thread(read_http, url, data['normal']) for _ in range(2)]
        await eventually(lambda: not service.jobs)
        acquired = [e for e in runner.store.events if e['kind'] == 'acquired']
        pool = runner.store.pool_snapshot()
        return result('http-prefix-backend', 'real_loopback_http_cpu_torch',
                      {'torch': torch.__version__, 'streams': streams, 'acquisitions': acquired, 'pool': pool},
                      {'same_output': all(tokens(v) == data['expected_ids'] for v in streams),
                       'cold_then_warm': [len(e['hit_blocks']) for e in acquired] == [0, 2],
                       'no_active_refs': pool['active_block_references'] == 0,
                       'released': service.engine.used_tokens == 0})


async def run(args):
    data = json.loads((LESSON/'data/requests.json').read_text(encoding='utf-8'))
    model = TinyAttentionLM(max_context=32)
    reference = {
        'normal': generate(model, data['normal']['prompt_ids'], data['normal']['max_new_tokens'], use_cache=False).token_ids,
        'branch': generate(model, data['branch_prompt'], 3, use_cache=False).token_ids,
    }
    assert reference['normal'] == data['expected_ids'] and reference['branch'] == data['branch_expected_ids']
    cases = []
    if args.case in ('all', 'http'):
        cases += await http_cases(data)
    if args.case in ('all', 'controlled'):
        cases += await controlled_cases(data)
    if args.case in ('all', 'protocol'):
        cases.append(protocol_case())
    if args.include_prefix or args.case == 'prefix':
        cases.append(await prefix_case(data))
    return {'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'source': source_fingerprint(),
            'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                            'uvicorn': version('uvicorn'), 'h11': version('h11'), 'device': 'cpu'},
            'execution': {'http': 'real loopback sockets where labeled; controlled ASGI sends for deterministic stalls',
                          'engine': 'single-loop synchronous tiny forwards; no tensor batching',
                          'delay': 'artificial step delays expose cancellation; no performance claims'},
            'full_recompute_reference': reference,
            'gpu_status': {'cuda-a100': 'not_run', 'maca-c500': 'not_run'},
            'cases': cases, 'passed': all(c['passed'] for c in cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('all', 'http', 'controlled', 'protocol', 'prefix'), default='all')
    parser.add_argument('--include-prefix', action='store_true', help='Also run HTTP with lesson 2.6 CPU PyTorch.')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'.local/runs/part-02-engine/07-http-and-streaming-output')
    args = parser.parse_args()
    report = asyncio.run(run(args))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for case in report['cases']:
        print(('PASS' if case['passed'] else 'FAIL')+' '+case['name'])
    print('Summary:', args.output_dir/'summary.json')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
