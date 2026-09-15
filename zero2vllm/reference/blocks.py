"""Logical KV block allocation simulation, independent of tensor storage."""

import heapq


class BlockPool:
    def __init__(self, num_blocks: int, block_size: int):
        if any(type(value) is not int or value <= 0 for value in (num_blocks, block_size)):
            raise ValueError("Block count and size must be positive integers.")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free = list(range(num_blocks))
        self._tables: dict[str, list[int]] = {}
        self._lengths: dict[str, int] = {}

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    def table(self, request_id: str) -> tuple[int, ...]:
        return tuple(self._tables[request_id])

    def reserve(self, request_id: str, num_tokens: int) -> tuple[int, ...]:
        """Grow a request atomically; insufficient capacity changes no state."""
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("A nonempty request ID is required.")
        if type(num_tokens) is not int or num_tokens < 0:
            raise ValueError("Token count must be a nonnegative integer.")
        if num_tokens < self._lengths.get(request_id, 0):
            raise ValueError("reserve only grows requests; use release to free them.")
        required = (num_tokens + self.block_size - 1) // self.block_size
        current = self._tables.get(request_id, [])
        extra = required - len(current)
        if extra > self.free_blocks:
            raise MemoryError("Insufficient free KV blocks.")
        table = self._tables.setdefault(request_id, [])
        table.extend(heapq.heappop(self._free) for _ in range(extra))
        self._lengths[request_id] = num_tokens
        return tuple(table)

    def locate(self, request_id: str, token_position: int) -> tuple[int, int]:
        if type(token_position) is not int or not 0 <= token_position < self._lengths[request_id]:
            raise IndexError("Token position is outside the reserved request length.")
        logical_block, offset = divmod(token_position, self.block_size)
        return self._tables[request_id][logical_block], offset

    def release(self, request_id: str) -> None:
        for block in self._tables.pop(request_id):
            heapq.heappush(self._free, block)
        del self._lengths[request_id]
