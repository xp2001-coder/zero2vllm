"""Split SSE frames one byte at a time, including Chinese UTF-8 bytes."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from zero2vllm.serving.protocol import SSEDecoder, StreamValidator, encode_event


def main():
    frames = [
        {'event': 'accepted', 'data': {'request_id': 'demo', 'seq': 0, '说明': '开始'}},
        {'event': 'token', 'data': {'request_id': 'demo', 'seq': 1, 'index': 0, 'token_id': 12}},
        {'event': 'done', 'data': {'request_id': 'demo', 'seq': 2, 'output_tokens': 1}},
    ]
    raw = b': ping\n\n'+b''.join(map(encode_event, frames))
    decoder, validator, restored = SSEDecoder(), StreamValidator(), []
    for value in raw:
        for frame in decoder.feed(bytes([value])):
            validator.accept(frame)
            restored.append(frame)
            print('complete event:', frame['event'], 'seq=', frame['data']['seq'])
    decoder.finish()
    assert restored == frames and validator.terminal
    incomplete = SSEDecoder()
    assert incomplete.feed(b'event: done\ndata: {}\n') == []
    try:
        incomplete.finish()
    except ValueError:
        print('truncated frame: rejected')
    else:
        raise AssertionError('Missing blank line must not count as done.')
    print('PASS: byte fragments preserve event boundaries and UTF-8.')


if __name__ == '__main__':
    main()
