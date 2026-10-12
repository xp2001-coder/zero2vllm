"""2-D integer quantization reference: explicit axes, tails and scale storage.

Inputs become FP32 values; computation uses float64. Logical codes are Python
integers, not packed payloads. All parameter statistics exclude padded cells.
"""
from dataclasses import asdict, dataclass
import math
import struct

from zero2vllm.reference.quantization_overview import fp32_matrix

SCALE_BITS = {'float16':16, 'float32':32, 'float64':64}


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def cast_scale(value, dtype):
    if dtype not in SCALE_BITS:
        raise ValueError('scale_dtype must be float16, float32 or float64')
    try:
        result = struct.unpack('<'+{'float16':'e','float32':'f','float64':'d'}[dtype],
                             struct.pack('<'+{'float16':'e','float32':'f','float64':'d'}[dtype],value))[0]
    except (OverflowError, struct.error, TypeError) as error:
        raise ValueError('Scale cannot be stored in the declared dtype') from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError('Stored scale must stay positive and finite; underflow is not silently fixed')
    return result


def _geometry(shape, axis, group_size, pad_to_group):
    if len(shape) != 2:
        raise ValueError('Expected a 2-D shape')
    for v in shape: _positive_int(v,'shape dimension')
    if type(pad_to_group) is not bool:
        raise ValueError('pad_to_group must be bool')
    if axis is None:
        if group_size is not None or pad_to_group:
            raise ValueError('Per-tensor mode uses axis=None, group_size=None, no group padding')
        return tuple(shape),(1,1),None
    if type(axis) is not int or axis not in (0,1):
        raise ValueError('axis must be 0, 1 or None')
    group_size = shape[axis] if group_size is None else group_size
    _positive_int(group_size,'group_size')
    groups = (shape[axis]+group_size-1)//group_size
    physical = list(shape)
    if pad_to_group: physical[axis] = groups*group_size
    params = (groups,shape[1]) if axis==0 else (shape[0],groups)
    return tuple(physical),params,group_size


@dataclass(frozen=True)
class QuantizedTensor:
    shape: tuple[int,int]
    axis: int | None
    group_size: int | None
    bits: int
    scheme: str
    scale_dtype: str
    pad_to_group: bool
    q: tuple[tuple[int,...],...]
    scales: tuple[tuple[float,...],...]
    zero_points: tuple[tuple[int,...],...]

    def __post_init__(self):
        object.__setattr__(self,'shape',tuple(self.shape))
        for field in ('q','scales','zero_points'):
            object.__setattr__(self,field,tuple(tuple(row) for row in getattr(self,field)))
        if type(self.bits) is not int or self.bits not in (2,3,4,8):
            raise ValueError('Teaching integer widths are 2,3,4,8')
        if self.scheme not in ('symmetric','asymmetric'):
            raise ValueError('Unknown integer quantization scheme')
        physical,params,g = _geometry(self.shape,self.axis,self.group_size,self.pad_to_group)
        object.__setattr__(self,'group_size',g)
        for name,expected in (('q',physical),('scales',params),('zero_points',params)):
            rows=getattr(self,name)
            if len(rows)!=expected[0] or any(len(r)!=expected[1] for r in rows):
                raise ValueError(f'{name} shape does not match axis/group contract: expected {expected}')
        low,high=self.integer_range
        for row in self.scales:
            for s in row:
                if type(s) not in (int,float) or cast_scale(s,self.scale_dtype)!=s:
                    raise ValueError('Scale values must match their declared stored dtype')
        for row in self.zero_points:
            for z in row:
                if type(z) is not int or not low<=z<=high or (self.scheme=='symmetric' and z!=0):
                    raise ValueError('Invalid zero-point for the scheme')
        for i,row in enumerate(self.q):
            for j,q in enumerate(row):
                if type(q) is not int or not low<=q<=high:
                    raise ValueError('Code is outside the declared integer range')
                if i>=self.shape[0] or j>=self.shape[1]:
                    a,b=self.parameter_index(i,j)
                    if q!=self.zero_points[a][b]:
                        raise ValueError('Padded cells must encode real zero: q_pad = zero_point')

    @property
    def integer_range(self):
        return (-(2**(self.bits-1)-1),2**(self.bits-1)-1) if self.scheme=='symmetric' else (0,2**self.bits-1)

    @property
    def physical_shape(self):
        return len(self.q),len(self.q[0])

    def parameter_index(self,i,j):
        if self.axis is None: return 0,0
        return (i//self.group_size,j) if self.axis==0 else (i,j//self.group_size)

    def to_dict(self):
        return dict(**asdict(self),physical_shape=self.physical_shape,
            parameter_shape=(len(self.scales),len(self.scales[0])),integer_range=self.integer_range,
            rounding='round_ties_to_even_before_adding_zero_point',input_dtype='FP32 values',
            reference_arithmetic='Python float64',storage='Unpacked Python integers; not a checkpoint')


def _quantize(values, *, scheme, bits, axis, group_size, scale_dtype, pad_to_group):
    values=fp32_matrix(values)
    shape=(len(values),len(values[0]))
    physical,params,g=_geometry(shape,axis,group_size,pad_to_group)
    if type(bits) is not int or bits not in (2,3,4,8):
        raise ValueError('Teaching integer widths are 2,3,4,8')
    qmin,qmax=(-(2**(bits-1)-1),2**(bits-1)-1) if scheme=='symmetric' else (0,2**bits-1)
    scales=[[0.0]*params[1] for _ in range(params[0])]
    zps=[[0]*params[1] for _ in range(params[0])]
    for a in range(params[0]):
        for b in range(params[1]):
            if axis is None: cells=[v for row in values for v in row]
            elif axis==0: cells=[values[i][b] for i in range(a*g,min((a+1)*g,shape[0]))]
            else: cells=values[a][b*g:min((b+1)*g,shape[1])]
            if scheme=='symmetric':
                amax=max(abs(v) for v in cells)
                scale=cast_scale(amax/qmax if amax else 1.0,scale_dtype);z=0
            else:
                low=min(0.0,min(cells));high=max(0.0,max(cells))
                scale=cast_scale((high-low)/(qmax-qmin) if high>low else 1.0,scale_dtype)
                z=max(qmin,min(qmax,round(qmin-low/scale)))
            scales[a][b]=scale;zps[a][b]=z
    codes=[]
    for i in range(physical[0]):
        row=[]
        for j in range(physical[1]):
            a,b=(0,0) if axis is None else ((i//g,j) if axis==0 else (i,j//g))
            s,z=scales[a][b],zps[a][b]
            v=values[i][j] if i<shape[0] and j<shape[1] else 0.0
            # Integer clipping endpoints make this equivalent to final clipping,
            # while guarding normalization overflow for unusually small scales.
            normalized=max(qmin-z,min(qmax-z,v/s))
            row.append(max(qmin,min(qmax,round(normalized)+z)))
        codes.append(row)
    return QuantizedTensor(shape,axis,g,bits,scheme,scale_dtype,pad_to_group,codes,scales,zps)


def quantize_symmetric(values, *, bits=4, axis=0, group_size=2, scale_dtype='float64', pad_to_group=False):
    return _quantize(values,scheme='symmetric',bits=bits,axis=axis,group_size=group_size,scale_dtype=scale_dtype,pad_to_group=pad_to_group)


def quantize_asymmetric(values, *, bits=4, axis=0, group_size=2, scale_dtype='float64', pad_to_group=False):
    return _quantize(values,scheme='asymmetric',bits=bits,axis=axis,group_size=group_size,scale_dtype=scale_dtype,pad_to_group=pad_to_group)


def dequantize(tensor, *, include_padding=False):
    if not isinstance(tensor,QuantizedTensor) or type(include_padding) is not bool:
        raise ValueError('Expected a QuantizedTensor and boolean include_padding')
    shape=tensor.physical_shape if include_padding else tensor.shape
    restored=[]
    for i in range(shape[0]):
        row=[]
        for j in range(shape[1]):
            a,b=tensor.parameter_index(i,j)
            row.append(tensor.scales[a][b]*(tensor.q[i][j]-tensor.zero_points[a][b]))
        restored.append(row)
    return restored


def calculate_effective_bpw(tensor, *, zero_point_bits=None, metadata_bytes=0, alignment_bytes=1):
    """Estimate three separately byte-rounded buffers plus a header and final alignment.

    No serialized file is created. Padding includes physical tensor cells,
    byte rounding per buffer, and alignment of the combined allocation.
    """
    if not isinstance(tensor,QuantizedTensor): raise ValueError('Expected a QuantizedTensor')
    if type(metadata_bytes) is not int or metadata_bytes<0: raise ValueError('metadata_bytes must be nonnegative integer')
    _positive_int(alignment_bytes,'alignment_bytes')
    if zero_point_bits is None: zero_point_bits=tensor.bits if tensor.scheme=='asymmetric' else 0
    if type(zero_point_bits) is not int or zero_point_bits not in (0,tensor.bits,8,16,32,64):
        raise ValueError('Unsupported zero-point storage width')
    if tensor.scheme=='asymmetric' and zero_point_bits<tensor.bits:
        raise ValueError('Asymmetric zero-points need sufficient declared storage bits')
    count=math.prod(tensor.shape);physical=math.prod(tensor.physical_shape)
    params=len(tensor.scales)*len(tensor.scales[0])
    weight=count*tensor.bits;tail=(physical-count)*tensor.bits
    scale=params*SCALE_BITS[tensor.scale_dtype];zp=params*zero_point_bits;meta=metadata_bytes*8
    buffers=(weight+tail,scale,zp)
    byte_padding=sum((-b)%8 for b in buffers)
    subtotal=weight+tail+scale+zp+meta+byte_padding
    alignment_padding=(-subtotal)%(alignment_bytes*8)
    total=subtotal+alignment_padding
    return dict(n_weights=count,nominal_bits=tensor.bits,weight_bits=weight,scale_bits=scale,
        zero_point_bits=zp,metadata_bits=meta,tensor_padding_bits=tail,byte_padding_bits=byte_padding,
        alignment_padding_bits=alignment_padding,total_bits=total,estimated_bytes=total//8,
        effective_bpw=total/count,scope='Declared buffer plan, not measured checkpoint size or GPU memory',
        scale_dtype=tensor.scale_dtype,zero_point_storage_bits=zero_point_bits,alignment_bytes=alignment_bytes)
