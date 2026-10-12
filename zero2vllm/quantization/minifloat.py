"""Readable CPU encodings, not native FP8/FP4 arithmetic or a GPU emulator."""
from functools import lru_cache
import math
from zero2vllm.reference.quantization_overview import as_fp32

FORMATS={'e4m3fn':(4,3,7), 'e5m2':(5,2,15), 'e2m1':(2,1,1)}


def _format(name):
    if name not in FORMATS: raise ValueError('Choose e4m3fn, e5m2 or e2m1; suffixes matter')
    return FORMATS[name]


def decode_minifloat(code, name, *, ftz=False):
    e,m,bias=_format(name);bits=1+e+m
    if type(code) is not int or not 0<=code<2**bits or type(ftz) is not bool:
        raise ValueError('Invalid bit pattern or FTZ policy')
    sign=-1.0 if code>>(e+m) else 1.0
    exponent=(code>>m)&(2**e-1);fraction=code&(2**m-1)
    if name=='e4m3fn' and exponent==15 and fraction==7: return math.nan
    if name=='e5m2' and exponent==31: return sign*math.inf if fraction==0 else math.nan
    if exponent==0:
        value=0.0 if ftz else 2.0**(1-bias)*(fraction/2**m)
    else:
        value=2.0**(exponent-bias)*(1+fraction/2**m)
    return math.copysign(value,sign)


@lru_cache(maxsize=3)
def _positive_grid(name):
    e,m,_=_format(name)
    return tuple((code,decode_minifloat(code,name)) for code in range(2**(e+m))
                 if math.isfinite(decode_minifloat(code,name)))


def encode_minifloat(value,name,*,overflow='saturate'):
    """Finite-input, nearest-even enumeration; saturate or reject over-range input.

    This deliberately does not implement a vendor's NaN/Inf cast policy, nor
    the OCP OVF conversion mode. 'raise' rejects any input above max finite.
    """
    e,m,_=_format(name)
    if type(value) not in (int,float): raise ValueError('Expected a finite real value')
    try: value=float(value)
    except OverflowError as error: raise ValueError('Input overflows float64') from error
    if not math.isfinite(value) or overflow not in ('saturate','raise'):
        raise ValueError('Expected finite input and an explicit supported overflow policy')
    magnitude=abs(value);grid=_positive_grid(name)
    if magnitude>grid[-1][1] and overflow=='raise': raise ValueError('Value exceeds max finite')
    magnitude=min(magnitude,grid[-1][1])
    # Adjacent positive encodings order the finite values; low code bit is
    # the significand parity, including boundaries between exponent bins.
    code,_=min(grid,key=lambda pair:(abs(pair[1]-magnitude),pair[0]&1))
    return code | ((1 if math.copysign(1.0,value)<0 else 0)<<(e+m))


def decode_e8m0(code):
    if type(code) is not int or not 0<=code<=255: raise ValueError('E8M0 uses one unsigned byte')
    return math.nan if code==255 else 2.0**(code-127)


def decode_scaled_block(codes, *, kind, scale_code, global_scale=1.0):
    """Numerical decomposition of MXFP4 (32) / NVFP4 (16); no wire layout.

    The NVFP4 global_scale argument is the direct DQ multiplier, not an
    inverse-scale variable used by a particular kernel implementation.
    """
    if kind not in ('mxfp4','nvfp4'): raise ValueError('Choose mxfp4 or nvfp4')
    size=32 if kind=='mxfp4' else 16
    if not isinstance(codes,(list,tuple)) or len(codes)!=size: raise ValueError(f'Expected exactly {size} logical element codes')
    elements=[decode_minifloat(c,'e2m1') for c in codes]
    if kind=='mxfp4':
        if type(global_scale) not in (int,float) or global_scale!=1.0: raise ValueError('This MXFP4 decomposition has no extra global scale')
        scale=decode_e8m0(scale_code)
    else:
        scale=decode_minifloat(scale_code,'e4m3fn')
        if not math.isfinite(scale) or scale<0: raise ValueError('NVFP4 block scale must be finite and nonnegative')
        if type(global_scale) not in (int,float) or not math.isfinite(global_scale) or global_scale<=0:
            raise ValueError('Global DQ scale must be positive finite')
        global_scale=as_fp32(global_scale)
    result=[v*scale*global_scale for v in elements]
    if any(math.isinf(v) for v in result): raise ValueError('Decoded values overflow the float64 reference')
    return result
