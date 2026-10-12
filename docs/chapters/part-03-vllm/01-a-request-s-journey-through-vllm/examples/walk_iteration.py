"""Hand-chosen data: pack two requests and route hypothetical model outputs."""
from itertools import permutations
import json

REQUESTS = {
    'A': {'ids': [10, 11, 12, 13, 14, 15], 'computed': 2, 'scheduled': 3,
          'blocks': [8, 3], 'hypothetical_output': []},
    'B': {'ids': [20, 21, 22, 23], 'computed': 3, 'scheduled': 1,
          'blocks': [5], 'hypothetical_output': [24]},
}


def pack(order, block_size=4):
    input_ids, positions, slots, boundaries = [], [], [], [0]
    for request_id in order:
        request = REQUESTS[request_id]
        start = request['computed']
        end = start + request['scheduled']
        assert end <= len(request['ids'])
        input_ids.extend(request['ids'][start:end])
        positions.extend(range(start, end))
        slots.extend(request['blocks'][p // block_size] * block_size + p % block_size
                     for p in range(start, end))
        boundaries.append(len(input_ids))
    return {'input_ids': input_ids, 'positions': positions, 'slot_mapping': slots,
            'illustrative_boundaries': boundaries,
            'req_id_to_index': {request_id: i for i, request_id in enumerate(order)},
            'sampled_token_ids': [REQUESTS[r]['hypothetical_output'] for r in order]}


def route(batch):
    return {request_id: batch['sampled_token_ids'][index]
            for request_id, index in batch['req_id_to_index'].items()}


def main():
    batch = pack(['B', 'A'])
    assert batch['input_ids'] == [23, 12, 13, 14]
    assert batch['positions'] == [3, 2, 3, 4]
    assert batch['slot_mapping'] == [23, 34, 35, 12]
    assert batch['illustrative_boundaries'] == [0, 1, 4]
    expected = {'A': [], 'B': [24]}
    for order in permutations(REQUESTS):
        assert route(pack(order)) == expected
    wrong = dict(zip(REQUESTS, batch['sampled_token_ids']))
    assert wrong != expected
    print('教学数据：未执行 vLLM 或任何模型；24 是预设输出。')
    print(json.dumps(batch, ensure_ascii=False, indent=2))
    print('按 ID 还原：', route(batch))
    print('错误地按登记顺序分配：', wrong)
    print('两种执行排列均通过；2 条请求、4 个输入位置、1 个有效新输出。')


if __name__ == '__main__':
    main()
