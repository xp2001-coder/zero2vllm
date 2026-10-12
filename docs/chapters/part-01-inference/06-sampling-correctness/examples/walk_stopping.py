"""Compare output IDs, visible text, and consumed cache positions."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from zero2vllm.reference.next_token import teaching_model, VOCABULARY
from zero2vllm.reference.sampling import SamplingConfig, generate_sampled


def main():
    cases = [('zero', dict(max_new_tokens=0), [], '', 0, 'length'),
             ('eos', dict(max_new_tokens=3, eos_token_id=2), [2], '', 2, 'eos'),
             ('string', dict(max_new_tokens=3, stop_strings=('推理推理',)), [2, 2], '', 3, 'stop_string'),
             ('length', dict(max_new_tokens=3), [2, 2, 2], '推理推理推理', 4, 'length')]
    for name, options, ids, text, cache, reason in cases:
        result = generate_sampled(teaching_model(), [0, 1], VOCABULARY, SamplingConfig(temperature=0), **options)
        assert result['new_token_ids'] == ids and result['generated_text'] == text
        assert result['cache_length'] == cache and result['finish_reason'] == reason
        print(f"{name}: ids={ids}, text={text!r}, cache={cache}, reason={reason}")
    print('PASS: stopping, text removal, and final unconsumed output.')


if __name__ == '__main__':
    main()
