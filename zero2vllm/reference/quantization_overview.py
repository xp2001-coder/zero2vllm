"""Lesson 9.1: FP32 input values, unpacked logical INT4, float64 Q/DQ.

This intentionally allocates Python integers, not four-bit storage. Binary32
input conversion is explicit; scales and arithmetic use Python float64.
"""
import math
import struct

from .quantization import matmul, quantize_value


def as_fp32(value):
    if type(value) not in (int, float):
        raise ValueError('Expected a finite number, not a boolean or string')
    try:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('Expected a finite number')
        result = struct.unpack('<f', struct.pack('<f', number))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError('Input is outside finite FP32 range') from error
    if not math.isfinite(result):
        raise ValueError('Input is outside finite FP32 range')
    if number != 0 and result == 0:
        raise ValueError('Nonzero input underflows to zero in FP32')
    return result


def fp32_matrix(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError('Expected a nonempty matrix')
    if not isinstance(values[0], (list, tuple)) or not values[0]:
        raise ValueError('Expected nonempty matrix rows')
    width = len(values[0])
    if any(not isinstance(row, (list, tuple)) or len(row) != width for row in values):
        raise ValueError('Matrix must be rectangular')
    return [[as_fp32(v) for v in row] for row in values]


def _positive_scale(scale):
    if type(scale) not in (int, float):
        raise ValueError('scale must be positive finite')
    try:
        scale = float(scale)
    except OverflowError as error:
        raise ValueError('scale is outside the reference type') from error
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(7 * scale):
        raise ValueError('scale must be positive with a finite [-7s,7s] range')
    return scale


def dequantize_logical(qvalues, scale):
    scale = _positive_scale(scale)
    if not isinstance(qvalues, (list, tuple)) or not qvalues:
        raise ValueError('Expected a nonempty integer matrix')
    if not isinstance(qvalues[0], (list, tuple)) or not qvalues[0]:
        raise ValueError('Expected nonempty integer rows')
    width = len(qvalues[0])
    if any(not isinstance(row, (list, tuple)) or len(row) != width for row in qvalues):
        raise ValueError('Integer matrix must be rectangular')
    if any(type(v) is not int or not -7 <= v <= 7 for row in qvalues for v in row):
        raise ValueError('Teaching logical INT4 uses integers in [-7,7]')
    result = [[v * scale for v in row] for row in qvalues]
    if any(not math.isfinite(v) for row in result for v in row):
        raise ValueError('Dequantized output overflows the reference type')
    return result


def logical_int4_qdq(weights, *, scale=None):
    """One symmetric scale for the entire matrix; no grouping and no packing."""
    fp32 = fp32_matrix(weights)
    amax = max(abs(v) for row in fp32 for v in row)
    if scale is None:
        scale = amax / 7 if amax else 1.0
        scale_source = 'amax/7; all-zero tensor uses 1'
    else:
        scale_source = 'explicit teaching parameter'
    scale = _positive_scale(scale)
    q = [[quantize_value(v, scale) for v in row] for row in fp32]
    restored = dequantize_logical(q, scale)
    errors = [a-b for ar,br in zip(fp32,restored) for a,b in zip(ar,br)]
    clipped = sum(abs(v) > 7 * scale for row in fp32 for v in row)
    # Accommodate FP64 arithmetic, without permitting an arbitrary absolute
    # tolerance to hide errors on small-magnitude tensors.
    tolerance = 8 * math.ulp(max(amax, 7*scale))
    bound = max(abs(e) for e in errors) <= scale/2 + tolerance
    return dict(
        fp32_input=fp32, q=q, scale=scale, dequantized=restored,
        contract=dict(logical_range=[-7,7], zero_point=0, rounding='ties_to_even',
                      granularity='per_tensor', group_size=None, group_axis=None,
                      scale_dtype='float64', arithmetic='Python float64',
                      storage='Python integer lists; unpacked; not a checkpoint',
                      scale_source=scale_source),
        error=dict(max_abs=max(abs(e) for e in errors), mse=sum(e*e for e in errors)/len(errors),
                   out_of_range_count=clipped, half_scale_bound_applicable=clipped==0,
                   half_scale_bound_satisfied=bound),
    )


def linear_qdq(inputs, weights, *, scale=None):
    qdq = logical_int4_qdq(weights, scale=scale)
    x = fp32_matrix(inputs)
    original = matmul(x, qdq['fp32_input'])
    restored = matmul(x, qdq['dequantized'])
    return dict(x=x, qdq=qdq, original_output=original, quantized_output=restored,
                output_max_abs_error=max(abs(a-b) for ar,br in zip(original,restored) for a,b in zip(ar,br)),
                compute='FP32 input values promoted to Python float64 multiply/accumulate; no GPU kernel')


def _affine_contract(scale, zero_point, qmin, qmax):
    scale = _positive_scale(scale)
    if any(type(v) is not int for v in (zero_point, qmin, qmax)):
        raise ValueError('zero_point and integer bounds must be integers')
    if not qmin < qmax or not qmin <= zero_point <= qmax:
        raise ValueError('Require qmin < qmax and qmin <= zero_point <= qmax')
    try:
        endpoints = [(qmin-zero_point)*scale, (qmax-zero_point)*scale]
    except OverflowError as error:
        raise ValueError('Affine range overflows the reference type') from error
    if not all(math.isfinite(v) for v in endpoints):
        raise ValueError('Affine range must be finite')
    return scale


def _encode_affine(values, scale, zero_point, qmin, qmax):
    # Clamp before round only to avoid rounding inf for very small scales.
    # Bounds here are integers: this is equivalent to clamping the final q.
    # IMPORTANT: round(v / s) + z, not round(v / s + z). For odd z these
    # differ at halfway values under ties-to-even rounding.
    return [[max(qmin, min(qmax, round(max(qmin-zero_point,
                min(qmax-zero_point, v/scale))) + zero_point)) for v in row]
            for row in values]


def quantize_affine(values, *, scale, zero_point, qmin=0, qmax=15):
    """FP32 inputs -> logical integers; round to even BEFORE adding z."""
    scale = _affine_contract(scale, zero_point, qmin, qmax)
    return _encode_affine(fp32_matrix(values), scale, zero_point, qmin, qmax)


def dequantize_affine(values, *, scale, zero_point, qmin=0, qmax=15):
    """Logical q -> float64 s*(q-z); unsigned q may encode negative reals."""
    scale = _affine_contract(scale, zero_point, qmin, qmax)
    if not isinstance(values, (list, tuple)) or not values or not isinstance(values[0], (list, tuple)) or not values[0]:
        raise ValueError('Expected a nonempty integer matrix')
    width = len(values[0])
    if any(not isinstance(row, (list, tuple)) or len(row) != width for row in values):
        raise ValueError('Integer matrix must be rectangular')
    if any(type(v) is not int or not qmin <= v <= qmax for row in values for v in row):
        raise ValueError('Logical integer outside the declared affine range')
    return [[scale*(v-zero_point) for v in row] for row in values]


def requantize_affine(values, *, source_scale, source_zero_point,
                      target_scale, target_zero_point,
                      source_qmin=0, source_qmax=15, target_qmin=0, target_qmax=15):
    """Reference Q2(DQ1(q1)), with explicit FLOAT64 intermediate.

    This is not an integer multiplier/shift kernel. It never re-converts DQ
    values to FP32 and never has access to the original pre-Q1 tensor.
    """
    restored = dequantize_affine(values, scale=source_scale, zero_point=source_zero_point,
                                 qmin=source_qmin, qmax=source_qmax)
    target_scale = _affine_contract(target_scale, target_zero_point, target_qmin, target_qmax)
    return _encode_affine(restored, target_scale, target_zero_point, target_qmin, target_qmax)
