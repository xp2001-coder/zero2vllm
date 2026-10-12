"""Our JSON-over-SSE teaching protocol (LF/CRLF), not a browser EventSource clone."""
import json


def encode_event(event):
    data = json.dumps(event['data'], ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return f"event: {event['event']}\ndata: {data}\n\n".encode('utf-8')


class StreamValidator:
    """Check request routing, event order and terminal token accounting."""
    def __init__(self):
        self.request_id = None
        self.sequence = self.output_tokens = 0
        self.terminal = False

    def accept(self, event):
        kind, data = event['event'], event['data']
        if self.terminal or kind not in ('accepted', 'token', 'done', 'error') or not isinstance(data, dict):
            raise ValueError('Unexpected stream event.')
        if self.sequence == 0:
            if kind != 'accepted' or not isinstance(data.get('request_id'), str) or not data['request_id']:
                raise ValueError('The stream must start with an accepted request ID.')
            self.request_id = data['request_id']
        elif kind == 'accepted':
            raise ValueError('Duplicate accepted event.')
        if data.get('request_id') != self.request_id or type(data.get('seq')) is not int or data['seq'] != self.sequence:
            raise ValueError('Request ID or event sequence changed unexpectedly.')
        if kind == 'token':
            if type(data.get('index')) is not int or data['index'] != self.output_tokens:
                raise ValueError('Token index is missing or out of order.')
            if type(data.get('token_id')) is not int or data['token_id'] < 0:
                raise ValueError('Invalid token ID.')
            self.output_tokens += 1
        elif kind in ('done', 'error'):
            if type(data.get('output_tokens')) is not int or data['output_tokens'] != self.output_tokens:
                raise ValueError('Terminal token count does not match received tokens.')
            self.terminal = True
        self.sequence += 1


class SSEDecoder:
    """Accept arbitrary byte fragments, including split UTF-8 codepoints.

    Only complete blank-line-delimited events are yielded. JSON data is required;
    comments are ignored. id/retry/reconnect and lone-CR line endings are outside
    this course client's subset. EOF never acts as an implicit done event.
    """
    def __init__(self, max_event_bytes=16384):
        self.buffer = b''
        self.kind = 'message'
        self.data = []
        self.size = 0
        self.limit = max_event_bytes

    def feed(self, chunk):
        self.buffer += chunk
        events = []
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n', 1)
            self.size += len(line)+1
            if self.size > self.limit:
                raise ValueError('SSE event exceeds the client limit.')
            text = line.removesuffix(b'\r').decode('utf-8', errors='strict')
            if not text:
                if self.data:
                    events.append({'event': self.kind, 'data': json.loads('\n'.join(self.data))})
                self.kind, self.data, self.size = 'message', [], 0
            elif not text.startswith(':'):
                name, _, value = text.partition(':')
                if value.startswith(' '):
                    value = value[1:]
                if name == 'event':
                    self.kind = value
                elif name == 'data':
                    self.data.append(value)
        if self.size+len(self.buffer) > self.limit:
            raise ValueError('SSE event exceeds the client limit.')
        return events

    def finish(self):
        if self.buffer or self.data or self.size:
            raise ValueError('Stream ended in the middle of an SSE frame.')
