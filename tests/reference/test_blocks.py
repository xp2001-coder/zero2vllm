import random
import unittest

from zero2vllm.reference.blocks import BlockPool


class BlockPoolTests(unittest.TestCase):
    def test_blocks_can_be_noncontiguous(self):
        pool = BlockPool(3, 4)
        pool.reserve("a", 4)
        pool.reserve("b", 4)
        self.assertEqual(pool.reserve("a", 5), (0, 2))
        self.assertEqual(pool.locate("a", 4), (2, 0))
        with self.assertRaises(IndexError):
            pool.locate("a", 5)

    def test_out_of_memory_is_atomic(self):
        pool = BlockPool(2, 4)
        pool.reserve("a", 4)
        with self.assertRaises(MemoryError):
            pool.reserve("a", 12)
        self.assertEqual(pool.table("a"), (0,))
        self.assertEqual(pool.free_blocks, 1)
        with self.assertRaises(IndexError):
            pool.locate("a", 4)
        with self.assertRaises(MemoryError):
            pool.reserve("b", 12)
        with self.assertRaises(KeyError):
            pool.table("b")

    def test_release_reuses_blocks_and_rejects_double_release(self):
        pool = BlockPool(2, 4)
        pool.reserve("a", 8)
        pool.release("a")
        self.assertEqual(pool.reserve("b", 8), (0, 1))
        with self.assertRaises(KeyError):
            pool.release("a")

    def test_zero_and_repeated_reservations(self):
        pool = BlockPool(2, 4)
        self.assertEqual(pool.reserve("a", 0), ())
        pool.reserve("a", 3)
        pool.reserve("a", 3)
        self.assertEqual(pool.free_blocks, 1)
        with self.assertRaises(ValueError):
            pool.reserve("a", 2)
        pool.release("a")
        self.assertEqual(pool.free_blocks, 2)

    def test_randomized_allocation_conserves_capacity_and_ownership(self):
        rng = random.Random(12)
        pool = BlockPool(12, 4)
        lengths = {}
        for _ in range(300):
            request = f"r{rng.randrange(6)}"
            if request in lengths and rng.random() < 0.3:
                pool.release(request)
                del lengths[request]
            else:
                wanted = lengths.get(request, 0) + rng.randrange(1, 9)
                try:
                    pool.reserve(request, wanted)
                    lengths[request] = wanted
                except MemoryError:
                    pass
            allocated = [block for key in lengths for block in pool.table(key)]
            self.assertEqual(len(allocated), len(set(allocated)))
            self.assertEqual(len(allocated) + pool.free_blocks, 12)

    def test_invalid_arguments(self):
        for size in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                BlockPool(2, size)
        pool = BlockPool(2, 4)
        for count in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                pool.reserve("a", count)
        with self.assertRaises(ValueError):
            pool.reserve("", 1)


if __name__ == "__main__":
    unittest.main()
