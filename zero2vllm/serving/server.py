"""Run the lesson's local HTTP server with one process and one event loop."""
import argparse

from zero2vllm.serving.app import StreamingApp
from zero2vllm.serving.service import StreamService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--backend', choices=('tiny', 'prefix'), default='tiny')
    parser.add_argument('--queue-size', type=int, default=2)
    parser.add_argument('--max-requests', type=int, default=8)
    parser.add_argument('--max-running', type=int, default=2)
    parser.add_argument('--step-delay', type=float, default=0.0)
    parser.add_argument('--request-timeout', type=float, default=30.0)
    parser.add_argument('--slow-timeout', type=float, default=5.0)
    parser.add_argument('--send-timeout', type=float, default=5.0)
    args = parser.parse_args()
    try:
        import uvicorn
    except ModuleNotFoundError as error:
        raise SystemExit('Install uvicorn==0.35.0; see lesson 2.7 lab.md.') from error
    runner = None
    if args.backend == 'prefix':
        try:
            from zero2vllm.engine.prefix_runner import PrefixTinyRunner
        except ModuleNotFoundError as error:
            raise SystemExit('The prefix backend also requires lesson 2.6 CPU PyTorch.') from error
        runner = PrefixTinyRunner(num_blocks=32, block_size=2)
    service = StreamService(runner, queue_size=args.queue_size, max_requests=args.max_requests,
                            max_running=args.max_running, step_delay=args.step_delay,
                            request_timeout=args.request_timeout, slow_timeout=args.slow_timeout)
    uvicorn.run(StreamingApp(service, send_timeout=args.send_timeout), host='127.0.0.1',
                port=args.port, loop='asyncio', http='h11', ws='none', lifespan='on',
                timeout_graceful_shutdown=5)


if __name__ == '__main__':
    main()
