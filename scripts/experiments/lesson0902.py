"""Lesson 9.2: integer tensor contracts, minifloats and explicit storage accounting."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform

from scripts.common import ROOT, source_fingerprint
from zero2vllm.quantization import QuantizedModelPassport
from zero2vllm.quantization.tensor_contract import (quantize_symmetric, quantize_asymmetric,
    dequantize, calculate_effective_bpw, cast_scale)
from zero2vllm.quantization.minifloat import (decode_minifloat, encode_minifloat,
    decode_e8m0, decode_scaled_block)
from zero2vllm.reference.quantization import matmul

LESSON = ROOT/'docs/chapters/part-09-quantization/02-numerical-representations-and-tensor-contracts'
DEFAULT_OUTPUT = ROOT/'.local/runs/part-09-quantization/02-tensor-contracts'


def load_fixture():
    return json.loads((LESSON/'data/grouped-linear.json').read_text(encoding='utf8'))


def error_metrics(a,b):
    errors=[abs(x-y) for ar,br in zip(a,b) for x,y in zip(ar,br)]
    return dict(max_abs=max(errors),mse=sum(e*e for e in errors)/len(errors))


def rejection(call):
    try: call()
    except ValueError as error: return str(error)
    return ''


def run_experiments():
    f=load_fixture();w=f['w'];expected=f['symmetric']
    sym=quantize_symmetric(w)
    asym=quantize_asymmetric(w,scale_dtype='float16',pad_to_group=True)
    restored=dequantize(sym);y=matmul(f['x'],w);yq=matmul(f['x'],restored)
    linear=dict(x=f['x'],w=w,symmetric=sym.to_dict(),asymmetric_fp16_padded=asym.to_dict(),
        dq=restored,y_original=y,y_qdq=yq,weight_error=error_metrics(w,restored),output_error=error_metrics(y,yq))
    bpw=calculate_effective_bpw(asym,metadata_bytes=4,alignment_bytes=16)
    cases=[]
    def add(name,condition,**details): cases.append(dict(name=name,passed=bool(condition),**details))
    add('non_square_linear_golden', [list(r) for r in sym.q]==expected['q'] and restored==expected['dq']
        and y==expected['y_original'] and yq==expected['y_qdq']
        and linear['weight_error']==dict(max_abs=expected['weight_max_abs'],mse=expected['weight_mse'])
        and linear['output_error']['max_abs']==expected['output_max_abs'])
    small=quantize_asymmetric([[-1,0,1,2]],axis=None,group_size=None)
    add('asymmetric_zero_has_code_five',small.q==((0,5,10,15),) and small.zero_points==((5,),),result=small.to_dict())
    other=quantize_symmetric(w,axis=1)
    add('axis_changes_parameter_shape',sym.to_dict()['parameter_shape']==(3,3) and other.to_dict()['parameter_shape']==(5,2),axis1=other.to_dict())
    modes=[quantize_symmetric(w,axis=None,group_size=None),quantize_symmetric(w,group_size=None),
           quantize_symmetric(f['x'],axis=1,group_size=None)]
    add('tensor_channel_token_granularity',[t.to_dict()['parameter_shape'] for t in modes]==[(1,1),(1,3),(2,1)],
        parameter_shapes=[t.to_dict()['parameter_shape'] for t in modes])
    add('tail_keeps_one_valid_row',sym.scales[-1]==(2.5/7,2.0,0.5/7),tail_scales=sym.scales[-1])
    add('padding_uses_zero_point',asym.q[-1]==tuple(f['asymmetric_tail_zero_points']) and dequantize(asym,include_padding=True)[-1]==[0,0,0],
        padding_codes=asym.q[-1],wrong_dq_if_code_zero=-asym.scales[-1][-1]*15)
    nopad=quantize_asymmetric(w,scale_dtype='float16')
    add('padding_excluded_from_statistics',asym.scales==nopad.scales and asym.zero_points==nopad.zero_points)
    zeros=quantize_asymmetric([[0,0],[0,0]])
    add('all_zero_groups',zeros.scales==((1.0,1.0),) and zeros.q==((0,0),(0,0)))
    positive=quantize_asymmetric([[2],[2]])
    negative=quantize_asymmetric([[-2],[-2]])
    add('one_sided_constants_include_zero',positive.zero_points==((0,),) and negative.zero_points==((15,),)
        and dequantize(positive)==[[2],[2]] and dequantize(negative)==[[-2],[-2]])
    add('stored_scale_is_used',asym.scales[0][0]==cast_scale(3.5/15,'float16'),stored=asym.scales[0][0],ideal=3.5/15)
    failures=[rejection(lambda: quantize_symmetric([[1e-8]],scale_dtype='float16')),
              rejection(lambda: cast_scale(1e8,'float16'))]
    add('reject_unrepresentable_scales',all(failures),expected_errors=failures)
    add('storage_accounting',all(bpw[k]==v for k,v in f['asymmetric_fp16_padded_storage'].items() if k not in ('metadata_bytes',)),report=bpw)
    grid=[decode_minifloat(c,'e2m1') for c in range(8)]
    add('fp4_nonuniform_grid',grid==[0,0.5,1,1.5,2,3,4,6],positive_grid=grid)
    add('same_bits_different_interpretation',decode_minifloat(0x38,'e4m3fn')==1 and decode_minifloat(0x38,'e5m2')==0.5,
        byte_hex='0x38',uint8=56,e4m3fn=1,e5m2=0.5)
    counts={}
    for name,size in [('e4m3fn',256),('e5m2',256),('e2m1',16)]:
        vals=[decode_minifloat(c,name) for c in range(size)]
        counts[name]=dict(finite=sum(math.isfinite(v) for v in vals),inf=sum(math.isinf(v) for v in vals),nan=sum(math.isnan(v) for v in vals))
    add('special_bit_patterns',counts=={'e4m3fn':{'finite':254,'inf':0,'nan':2},'e5m2':{'finite':248,'inf':2,'nan':6},'e2m1':{'finite':16,'inf':0,'nan':0}},counts=counts)
    add('subnormal_and_ftz',decode_minifloat(1,'e4m3fn')==2**-9 and decode_minifloat(1,'e4m3fn',ftz=True)==0,
        preserved=2**-9,ftz=0)
    add('fp4_ties_and_saturation',encode_minifloat(2.5,'e2m1')==4 and encode_minifloat(5,'e2m1')==6
        and encode_minifloat(10,'e2m1')==7 and bool(rejection(lambda:encode_minifloat(10,'e2m1',overflow='raise'))))
    add('e8m0_is_not_an_integer_scale',decode_e8m0(127)==1 and decode_e8m0(128)==2 and math.isnan(decode_e8m0(255)),
        code127=1,code128=2,code255='NaN',zero_encoding=False)
    mx=decode_scaled_block([7]*32,kind='mxfp4',scale_code=128)
    nv=decode_scaled_block([7]*16,kind='nvfp4',scale_code=0x38,global_scale=0.5)
    add('scaled_fp4_has_multiple_parameters',mx==[12]*32 and nv==[3]*16,mxfp4_first=mx[0],nvfp4_first=nv[0])
    # Existing integer codes and group metadata may be sharded without a new Q
    # step only if the global group mapping is preserved.
    split=3
    local_naive=[i//2 for i in range(5-split)]
    global_ids=[(i+split)//2 for i in range(5-split)]
    add('shard_boundary_cuts_a_group',local_naive==[0,0] and global_ids==[1,2],
        global_group_ids=global_ids,incorrect_local_group_ids=local_naive,
        scope='Index arithmetic only; not a distributed kernel')
    return dict(linear=linear,storage=bpw,cases=cases,passed=all(c['passed'] for c in cases)),sym,asym


def check_torch_float8():
    """Independent bit reinterpretation oracle; no float8 GEMM or GPU use."""
    import torch
    records=[]
    for name in ('e4m3fn','e5m2'):
        actual=torch.tensor(list(range(256)),dtype=torch.uint8).view(getattr(torch,'float8_'+name)).float().tolist()
        mismatches=[]
        for code,got in enumerate(actual):
            want=decode_minifloat(code,name)
            same=(math.isnan(got) and math.isnan(want)) or (got==want and (got!=0 or math.copysign(1,got)==math.copysign(1,want)))
            if not same: mismatches.append(code)
        records.append(dict(format=name,patterns=256,passed=not mismatches,mismatch_codes=mismatches))
    return dict(torch_version=torch.__version__,device='cpu',operation='uint8 bit view to float8 then cast to float32',
        native_low_precision_arithmetic=False,formats=records,passed=all(r['passed'] for r in records))


def make_passport(report,tensor,name):
    p=QuantizedModelPassport('lesson-9.2-'+name,'Synthetic 2-D integer tensor; unpacked codes; float64 reference arithmetic')
    facts={
        'provenance':dict(model='Synthetic W[5,3], X[2,5]; no bias',quantization_framework='zero2vllm CPU reference',
            framework_revision=report['course_source'],algorithm='Group MinMax including zero' if tensor.scheme=='asymmetric' else 'Group amax / 7 symmetric RTN'),
        'numerical':dict(weight_representation='logical 4-bit integer; range '+str(tensor.integer_range),
            activation_representation='FP32 values',symmetry=tensor.scheme,group_size=tensor.group_size,
            group_axis={'axis':tensor.axis,'meaning':'partition K of W[K,N]; separate parameters for each N'},
            scale_dtype=tensor.scale_dtype,scale_granularity='per_group_per_output_channel',zero_point=tensor.zero_points,
            rounding='ties-to-even before adding zero-point',clipping=tensor.integer_range,saturation='integer clamp'),
        'execution':dict(qweight_shape=tensor.physical_shape,qweight_dtype='unpacked Python integers',
            scale_shape=(len(tensor.scales),len(tensor.scales[0])),zero_point_shape=(len(tensor.zero_points),len(tensor.zero_points[0])),
            in_memory_layout=dict(logical_shape=tensor.shape,physical_shape=tensor.physical_shape,padding='q_pad = z',container='Python tuple rows; no packed buffer'),
            activation_dtype='FP32 values promoted to float64',multiply_dtype='Python float64',accumulator_dtype='Python float64',
            output_dtype='Python float64',runtime='CPU teaching reference'),
        'validation':dict(hardware='Host CPU; GPU not used',framework={'python':platform.python_version()},
            correctness={'cases_passed':all(c['passed'] for c in report['cases']),'scope':'synthetic tensor and format checks'})}
    for section,items in facts.items():
        for key,value in items.items():
            p.record(section,key,status='verified',observed=value,evidence=('summary.json#linear','zero2vllm/quantization/tensor_contract.py'))
    for section,fields in {
        'provenance':('model_revision','tokenizer_revision','calibration_dataset','calibration_samples','token_budget','sequence_length','seed'),
        'numerical':('kv_representation',),
        'execution':('checkpoint_format','format_version','g_idx','packing','on_disk_layout','repack','quant_method','linear_backend','actual_kernel'),
        'validation':('driver','device_runtime')}.items():
        for key in fields: p.record(section,key,status='not_applicable',note='Deterministic CPU tensor reference; no model checkpoint or device runtime')
    return p


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--torch-check',action='store_true',help='Explicitly require installed PyTorch for 512 FP8 bit-pattern comparisons')
    args=parser.parse_args(argv)
    report=dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),course_source=source_fingerprint(),python=platform.python_version(),
        verification_level='cpu_tensor_contract_reference',gpu_validated=False,packed=False,passed=False,cases=[])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    try:
        result,sym,asym=run_experiments();report.update(result)
        if args.torch_check:
            report['torch_oracle']=check_torch_float8();report['passed'] &= report['torch_oracle']['passed']
        for name,tensor in [('symmetric',sym),('asymmetric',asym)]:
            p=make_passport(report,tensor,name)
            (args.output_dir/(name+'-passport.json')).write_text(json.dumps(p.to_dict(),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    except (ValueError,ImportError,RuntimeError,OSError) as error:
        report.update(passed=False,error=str(error))
    (args.output_dir/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    for c in report['cases']: print(c['name'],'PASS' if c['passed'] else 'FAIL')
    if 'torch_oracle' in report: print('PyTorch FP8 bit patterns:',report['torch_oracle'])
    if 'error' in report: print(report['error'])
    print('Unpacked CPU reference; bpw is a declared buffer plan, not file or GPU memory measurement.')
    print('Results:',args.output_dir)
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
