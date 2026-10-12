"""POST JSON and consume our SSE frames incrementally, using only stdlib."""
import argparse
import json
import urllib.error
import urllib.request

from zero2vllm.serving.protocol import SSEDecoder, StreamValidator


def events(url, payload, *, timeout=35.0):
    request = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'),
                                     headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
    decoder, validator = SSEDecoder(), StreamValidator()
    # Bypass environment proxies for this explicit local tutorial client.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        if response.headers.get_content_type() != 'text/event-stream':
            raise ValueError('Expected a text/event-stream response.')
        while chunk := response.read1(4096):
            for event in decoder.feed(chunk):
                validator.accept(event)
                yield event
        decoder.finish()
        if not validator.terminal:
            raise ValueError('Connection ended without done/error; generation is incomplete.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000/generate')
    parser.add_argument('--prompt', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--max-new-tokens', type=int, default=3)
    parser.add_argument('--stop-after', type=int, default=0, help='Close after this many token events; 0 reads all.')
    args = parser.parse_args()
    stream = events(args.url, {'prompt_ids': args.prompt, 'max_new_tokens': args.max_new_tokens})
    count = 0
    try:
        for event in stream:
            print(json.dumps(event, ensure_ascii=False), flush=True)
            if event['event'] == 'error':
                raise SystemExit(1)
            if event['event'] == 'token':
                count += 1
                if args.stop_after > 0 and count >= args.stop_after:
                    print('Client closed early; no successful completion claimed.', flush=True)
                    break
    except urllib.error.HTTPError as error:
        print(f'HTTP {error.code}: {error.read().decode("utf-8")}', flush=True)
        raise SystemExit(1) from error
    finally:
        stream.close()


if __name__ == '__main__':
    main()
