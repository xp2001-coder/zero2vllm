"""CPU teaching format: group-wise symmetric INT4 weights, no GPU dependency.

W is [K,N], flattened in row-major order; the first value occupies the low
nibble. Quantization uses [-7,7], ties-to-even rounding and float64 scales.
This format is deliberately independent of AWQ/GPTQ/Marlin checkpoints.
"""

from dataclasses import dataclass
import math
import random


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _matrix(values):
    if not values or not values[0]:
        raise ValueError("matrix must be nonempty")
    width = len(values[0])
    result = []
    for row in values:
        if len(row) != width:
            raise ValueError("matrix must be rectangular")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in row):
            raise ValueError("matrix values must be finite numbers")
        result.append([float(v) for v in row])
    return result


def pack_int4(values):
    """Pack signed integers [-8,7]; zero-pad the final high nibble if needed."""
    data = bytearray((len(values) + 1) // 2)
    for i, value in enumerate(values):
        if type(value) is not int or not -8 <= value <= 7:
            raise ValueError("INT4 value must be an integer in [-8,7]")
        data[i // 2] |= (value & 15) << (4 * (i % 2))
    return bytes(data)


def unpack_int4(data, count):
    if type(count) is not int or count < 0 or len(data) != (count + 1) // 2:
        raise ValueError("packed length does not match count")
    if count % 2 and data[-1] >> 4:
        raise ValueError("unused high nibble must be zero")
    result = []
    for i in range(count):
        value = (data[i // 2] >> (4 * (i % 2))) & 15
        result.append(value - 16 if value >= 8 else value)
    return result


def quantize_value(value, scale):
    """Round to nearest even, then saturate to the narrow symmetric range."""
    if not math.isfinite(value) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("value must be finite and scale must be positive finite")
    normalized = max(-7.0, min(7.0, value / scale))
    return max(-7, min(7, round(normalized)))


@dataclass(frozen=True)
class Int4Weights:
    k: int
    n: int
    group_size: int
    data: bytes
    scales: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        for name in ("k", "n", "group_size"):
            _positive_int(getattr(self, name), name)
        groups = (self.k + self.group_size - 1) // self.group_size
        if len(self.scales) != groups or any(len(row) != self.n for row in self.scales):
            raise ValueError("scales must have shape [ceil(K/G),N]")
        if any(not math.isfinite(s) or s <= 0 for row in self.scales for s in row):
            raise ValueError("scales must be positive finite")
        if -8 in unpack_int4(self.data, self.k * self.n):
            raise ValueError("teaching quantization uses [-7,7], with -8 reserved")

    def value(self, k, n):
        index = k * self.n + n
        nibble = (self.data[index // 2] >> (4 * (index % 2))) & 15
        signed = nibble - 16 if nibble >= 8 else nibble
        return signed * self.scales[k // self.group_size][n]


def quantize_weights(weights, group_size=8):
    _positive_int(group_size, "group_size")
    weights = _matrix(weights)
    k, n = len(weights), len(weights[0])
    scales = []
    quantized = [[0] * n for _ in range(k)]
    for start in range(0, k, group_size):
        end = min(start + group_size, k)
        group_scales = []
        for column in range(n):
            amax = max(abs(weights[row][column]) for row in range(start, end))
            scale = amax / 7 if amax else 1.0
            if scale == 0:
                raise ValueError("scale underflow in float64 reference")
            group_scales.append(scale)
            for row in range(start, end):
                quantized[row][column] = quantize_value(weights[row][column], scale)
        scales.append(tuple(group_scales))
    return Int4Weights(k, n, group_size,
                       pack_int4([v for row in quantized for v in row]), tuple(scales))


def dequantize_weights(weights):
    return [[weights.value(k, n) for n in range(weights.n)] for k in range(weights.k)]


def matmul(inputs, weights):
    inputs, weights = _matrix(inputs), _matrix(weights)
    if len(inputs[0]) != len(weights):
        raise ValueError("incompatible matrix shapes")
    return [[sum(x * weights[k][n] for k, x in enumerate(row))
             for n in range(len(weights[0]))] for row in inputs]


def packed_linear(inputs, weights):
    """Read/dequantize on demand; a math reference, not an optimized kernel."""
    inputs = _matrix(inputs)
    if len(inputs[0]) != weights.k:
        raise ValueError("incompatible matrix shapes")
    return [[sum(x * weights.value(k, n) for k, x in enumerate(row))
             for n in range(weights.n)] for row in inputs]


def quantization_experiment(group_size=8, seed=7):
    rng = random.Random(seed)
    m, k, n = 3, 17, 5  # Both an incomplete group and an odd packed length.
    inputs = [[rng.uniform(-1, 1) for _ in range(k)] for _ in range(m)]
    weights = [[rng.uniform(-2, 2) for _ in range(n)] for _ in range(k)]
    packed = quantize_weights(weights, group_size)
    restored = dequantize_weights(packed)
    fp_output = matmul(inputs, weights)
    explicit_output = matmul(inputs, restored)
    packed_output = packed_linear(inputs, packed)
    def error(a, b):
        return max(abs(x-y) for ar, br in zip(a, b) for x, y in zip(ar, br))
    impl_error = error(explicit_output, packed_output)
    weight_error = error(weights, restored)
    tolerance = 1e-12
    bounded = all(abs(weights[i][j]-restored[i][j]) <= packed.scales[i//group_size][j]/2+tolerance
                  for i in range(k) for j in range(n))
    scale_bytes = 8 * len(packed.scales) * n
    return {
        "schema_version": 1, "experiment": "cpu_int4_linear_reference",
        "record_type": "measured_reference",
        "input": {"m": m, "k": k, "n": n, "group_size": group_size, "seed": seed,
                  "x": inputs, "w": weights},
        "format": {"name": "zero2vllm-int4-reference-v1", "qrange": [-7, 7],
                   "rounding": "ties_to_even", "zero_point": 0, "axis": "K",
                   "layout": "row-major W[K,N], low nibble first", "scale_dtype": "float64",
                   "arithmetic": "Python float64; no FP16 or INT4 hardware arithmetic"},
        "packed_hex": packed.data.hex(), "scales": packed.scales,
        "outputs": {"unquantized": fp_output, "explicit_dequant": explicit_output,
                    "packed_reference": packed_output},
        "storage": {"packed_bytes": len(packed.data), "scale_bytes": scale_bytes,
                    "logical_payload_bytes": len(packed.data)+scale_bytes,
                    "hypothetical_fp16_weight_bytes": 2*k*n,
                    "excludes": "Python object overhead, headers, workspace and alignment"},
        "verification": {"passed": impl_error <= tolerance and bounded,
                         "implementation_max_abs_error": impl_error,
                         "weight_max_abs_error": weight_error,
                         "output_quantization_max_abs_error": error(fp_output, packed_output),
                         "weight_half_scale_bound_passed": bounded, "absolute_tolerance": tolerance},
        "limitations": ["Synthetic tensors; output error is not model task quality.",
                        "On-demand unpacking is not a GPU fused GEMM or vLLM integration.",
                        "Payload bytes are not process memory, GPU memory, throughput or speedup."]}
