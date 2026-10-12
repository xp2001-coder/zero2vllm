"""Float64, small-matrix algorithm references; no checkpoint or GPU execution.

Convention: X[M,K] @ W[K,N]. Fixed grids have one scale/zero-point per
output column and span all K inputs. This is deliberately not a G128 exporter.
"""
from dataclasses import asdict
import itertools
import math

from .calibration import AffineParameters, error_metrics, parameters_from_range
from zero2vllm.reference.quantization import matmul


def matrix(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError('Expected nonempty matrix')
    if not isinstance(values[0], (list, tuple)) or not values[0]:
        raise ValueError('Expected nonempty matrix rows')
    n = len(values[0])
    if any(not isinstance(row, (list, tuple)) or len(row) != n for row in values):
        raise ValueError('Expected rectangular matrix')
    if any(type(v) not in (int, float) or not math.isfinite(v) for row in values for v in row):
        raise ValueError('Expected finite numeric values')
    return [[float(v) for v in row] for row in values]


def transpose(a):
    return [list(row) for row in zip(*matrix(a))]


def pair(x, w):
    x, w = matrix(x), matrix(w)
    if len(x[0]) != len(w):
        raise ValueError('X[M,K] and W[K,N] must agree on K')
    return x, w


def validate_grids(w, grids):
    if len(grids) != len(w[0]) or any(not isinstance(p, AffineParameters) for p in grids):
        raise ValueError('One AffineParameters grid per output column is required')


def column_grids(w, *, bits=4, scheme='symmetric', ratio=1.0):
    w = matrix(w)
    if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError('Clipping ratio must be in (0,1]')
    return [parameters_from_range(min(c)*ratio, max(c)*ratio, bits=bits, scheme=scheme)
            for c in transpose(w)]


def code(value, p):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Quantized value must be finite')
    u = max(p.qmin-p.zero_point, min(p.qmax-p.zero_point, value/p.scale))
    return round(u) + p.zero_point


def decode(q, p):
    return p.scale * (q-p.zero_point)


def rtn(w, grids=None):
    w = matrix(w)
    grids = column_grids(w) if grids is None else list(grids)
    validate_grids(w, grids)
    q = [[code(v, grids[n]) for n, v in enumerate(row)] for row in w]
    return dict(q=q, dq=[[decode(v, grids[n]) for n, v in enumerate(row)] for row in q],
                grids=[asdict(p) for p in grids])


def layer_metrics(x, w, candidate):
    x, w = pair(x, w)
    candidate = matrix(candidate)
    if len(candidate) != len(w) or len(candidate[0]) != len(w[0]):
        raise ValueError('Candidate must have W shape')
    return error_metrics(matmul(x, w), matmul(x, candidate))


def clipping_search(x, w, ratios, *, objective='layer', bits=4):
    x, w = pair(x, w)
    if objective not in ('layer', 'weight') or not ratios:
        raise ValueError('Need ratios and weight/layer objective')
    candidates = []
    for ratio in ratios:
        result = rtn(w, column_grids(w, bits=bits, ratio=ratio))
        metrics = layer_metrics(x, w, result['dq']) if objective == 'layer' else error_metrics(w, result['dq'])
        candidates.append(dict(ratio=ratio, objective_mse=metrics['mse'], **result))
    best = min(candidates, key=lambda r: (r['objective_mse'], -r['ratio']))
    return dict(objective=objective, candidates=candidates, best=best)


def optimize_rounding(x, w, grids, *, max_variables=16):
    """Exact finite search over each value's clipped floor/ceil choices.

    This is a tiny oracle, not AdaRound/AutoRound or global search over all
    integer codes. Equal losses retain RTN, then the first improving candidate.
    """
    x, w = pair(x, w)
    validate_grids(w, grids)
    if type(max_variables) is not int or not 0 <= max_variables <= 20:
        raise ValueError('max_variables must be an integer in [0,20]')
    options = []
    for row in w:
        for n, v in enumerate(row):
            p = grids[n]
            u = max(p.qmin-p.zero_point, min(p.qmax-p.zero_point, v/p.scale))
            options.append(sorted({math.floor(u)+p.zero_point, math.ceil(u)+p.zero_point}))
    variables = sum(len(o) == 2 for o in options)
    if variables > max_variables:
        raise ValueError('Too many rounding variables for exhaustive teaching search')
    baseline = rtn(w, grids)
    best = dict(**baseline, loss=layer_metrics(x, w, baseline['dq'])['sse'])
    n = len(w[0])
    evaluated = 0
    for values in itertools.product(*options):
        q = [list(values[i:i+n]) for i in range(0, len(values), n)]
        dq = [[decode(v, grids[j]) for j, v in enumerate(row)] for row in q]
        loss = layer_metrics(x, w, dq)['sse']
        if loss < best['loss']:
            best = dict(q=q, dq=dq, grids=baseline['grids'], loss=loss)
        evaluated += 1
    return dict(best=best, rtn=baseline, candidates=evaluated, free_variables=variables,
                scope='Clipped floor/ceil choices on fixed grids; calibration SSE only')


def gram_hessian(x, damping=0.01):
    x = matrix(x)
    if type(damping) not in (int, float) or not math.isfinite(damping) or damping < 0:
        raise ValueError('Relative damping must be finite and nonnegative')
    h = [[2*v for v in row] for row in matmul(transpose(x), x)]
    lam = damping * sum(h[i][i] for i in range(len(h))) / len(h)
    for i in range(len(h)):
        h[i][i] += lam
    return h, lam


def cholesky(a):
    a = matrix(a)
    n = len(a)
    if len(a[0]) != n or any(not math.isclose(a[i][j], a[j][i], rel_tol=1e-12, abs_tol=1e-14)
                              for i in range(n) for j in range(n)):
        raise ValueError('Cholesky requires a symmetric square matrix')
    lower = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            pivot = a[i][j] - sum(lower[i][k]*lower[j][k] for k in range(j))
            if i == j:
                if pivot <= 0:
                    raise ValueError('Hessian is not positive definite; check inputs/damping')
                lower[i][j] = math.sqrt(pivot)
            else:
                lower[i][j] = pivot/lower[j][j]
    return lower


def inverse_spd(a):
    lower = cholesky(a)
    n = len(lower)
    columns = []
    for c in range(n):
        y = [0.0]*n
        for i in range(n):
            y[i] = ((1.0 if i == c else 0.0)-sum(lower[i][j]*y[j] for j in range(i)))/lower[i][i]
        x = [0.0]*n
        for i in reversed(range(n)):
            x[i] = (y[i]-sum(lower[j][i]*x[j] for j in range(i+1, n)))/lower[i][i]
        columns.append(x)
    return transpose(columns)


def gptq_reference(x, w, grids, *, damping=0.01, act_order=False, block_size=1):
    """Cholesky error compensation with fixed per-output grids and lazy blocks.

    No dynamic groups, dead-channel repair, model traversal or exporter.
    H=2 X^T X + lambda I; U^T U = H^{-1}. Quantize rows of W[K,N].
    """
    x, w = pair(x, w)
    validate_grids(w, grids)
    if type(block_size) is not int or block_size <= 0 or type(act_order) is not bool:
        raise ValueError('Positive block_size and boolean act_order required')
    h, lam = gram_hessian(x, damping)
    k, n = len(w), len(w[0])
    order = sorted(range(k), key=lambda i: (-h[i][i], i)) if act_order else list(range(k))
    h = [[h[i][j] for j in order] for i in order]
    upper = transpose(cholesky(inverse_spd(h)))
    work = [w[i][:] for i in order]
    q, trace = [[0]*n for _ in range(k)], []
    for start in range(0, k, block_size):
        stop = min(k, start+block_size)
        errors = []
        for i in range(start, stop):
            current = work[i][:]
            q[i] = [code(v, grids[j]) for j, v in enumerate(current)]
            restored = [decode(v, grids[j]) for j, v in enumerate(q[i])]
            err = [(a-b)/upper[i][i] for a, b in zip(current, restored)]
            errors.append(err)
            for j in range(i, stop):
                for c in range(n):
                    work[j][c] -= upper[i][j]*err[c]
            trace.append(dict(step=i, original_feature=order[i], before=current, codes=q[i][:],
                              dq=restored, normalized_error=err, incremental_objective=sum(e*e for e in err)/2))
        for j in range(stop, k):
            for c in range(n):
                work[j][c] -= sum(upper[i][j]*errors[i-start][c] for i in range(start, stop))
    restored_order = [[0]*n for _ in range(k)]
    for i, original in enumerate(order):
        restored_order[original] = q[i]
    dq = [[decode(v, grids[j]) for j, v in enumerate(row)] for row in restored_order]
    metrics = layer_metrics(x, w, dq)
    penalty = lam*error_metrics(w, dq)['sse']/2
    return dict(q=restored_order, dq=dq, grids=[asdict(p) for p in grids], order=order,
                hessian=h, upper=upper, lambda_absolute=lam, trace=trace,
                layer_sse=metrics['sse'], regularization_penalty=penalty,
                regularized_objective=metrics['sse']+penalty,
                trace_objective=sum(t['incremental_objective'] for t in trace))


def diagonal_transform(x, w, scales):
    x, w = pair(x, w)
    if len(scales) != len(w) or any(type(s) not in (float, int) or not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError('One positive finite transform scale per input channel required')
    return [[v/scales[j] for j, v in enumerate(row)] for row in x], \
        [[v*scales[i] for v in row] for i, row in enumerate(w)]


def scaling_search(x, w, alphas=(0, .25, .5, .75, 1), *, bits=4):
    """AWQ-inspired activation-mean scaling, evaluated on one Linear only.

    Quantizer scales are recalculated for each transformed weight. No AWQ
    clipping, architecture-specific folding, or end-to-end model reproduction.
    """
    x, w = pair(x, w)
    if not alphas or any(type(a) not in (int, float) or not 0 <= a <= 1 for a in alphas):
        raise ValueError('Need alpha candidates in [0,1]')
    means = [sum(abs(v) for v in c)/len(c) for c in transpose(x)]
    candidates = []
    for alpha in alphas:
        scales = [max(m**alpha, 1e-4) for m in means]
        normalizer = math.sqrt(min(scales)*max(scales))
        scales = [s/normalizer for s in scales]
        transformed_x, transformed_w = diagonal_transform(x, w, scales)
        quantized = rtn(transformed_w, column_grids(transformed_w, bits=bits))
        effective_w = [[v/scales[i] for v in row] for i, row in enumerate(quantized['dq'])]
        candidates.append(dict(alpha=alpha, transform_scales=scales, quantized=quantized,
            effective_w=effective_w, calibration_mse=layer_metrics(x, w, effective_w)['mse'],
            identity_error=error_metrics(matmul(x,w),matmul(transformed_x,transformed_w))['max_abs']))
    return dict(activation_mean_abs=means, candidates=candidates,
                best=min(candidates, key=lambda r: (r['calibration_mse'], r['alpha'])))


def hadamard(size):
    if type(size) is not int or size <= 0 or size > 64 or size & (size-1):
        raise ValueError('Teaching Hadamard size must be power of two in [1,64]')
    h = [[1.0]]
    while len(h) < size:
        h = [r+r for r in h] + [r+[-v for v in r] for r in h]
    return [[v/math.sqrt(size) for v in r] for r in h]


def rotate(x, w, r):
    x, w = pair(x, w)
    r = matrix(r)
    if len(r) != len(w) or len(r[0]) != len(w):
        raise ValueError('Rotation must be K by K')
    identity = matmul(transpose(r), r)
    if any(abs(v-(i == j)) > 1e-10 for i, row in enumerate(identity) for j, v in enumerate(row)):
        raise ValueError('Rotation must be orthogonal')
    return matmul(x, r), matmul(transpose(r), w)


def stochastic_code(value, p, rng):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError('Expected finite value')
    u = max(p.qmin-p.zero_point, min(p.qmax-p.zero_point, value/p.scale))
    low = math.floor(u)
    return low + (rng.random() < u-low) + p.zero_point


def ste_derivatives(value, p):
    """Chosen clipped STE for z=0; boundaries use the interior convention.

    These are surrogate derivatives, not derivatives of the true staircase.
    ds is the unnormalized LSQ-style local expression, without gradient scaling.
    """
    if p.zero_point != 0 or type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError('STE demonstration requires finite value and zero-point 0')
    u = value/p.scale
    if u < p.qmin:
        return dict(dx=0.0, ds=float(p.qmin))
    if u > p.qmax:
        return dict(dx=0.0, ds=float(p.qmax))
    return dict(dx=1.0, ds=float(round(u)-u))
