"""Lesson 9.1: logical INT4, affine Q/DQ/RQ, a Linear, and model passports."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

from scripts.common import ROOT, source_fingerprint
from scripts.experiments.source_records import inspect_sources
from zero2vllm.quantization import QuantizedModelPassport
from zero2vllm.reference.quantization import matmul
from zero2vllm.reference.quantization_overview import (dequantize_logical, linear_qdq, logical_int4_qdq,
    quantize_affine, dequantize_affine, requantize_affine)

LESSON = ROOT/'docs/chapters/part-09-quantization/01-quantization-system-overview'
DEFAULT_OUTPUT = ROOT/'.local/runs/part-09-quantization/01-system-overview'


def load_fixture():
    return json.loads((LESSON/'data/linear.json').read_text(encoding='utf-8'))


def target_passport():
    p = QuantizedModelPassport('qwen3-8b-w4a16-g128-plan', 'Planned model; no weights loaded or GPU execution')
    for section, name, value in [
        ('provenance', 'model', 'Qwen3-8B'),
        ('numerical', 'weight_representation', '4-bit; exact encoding unspecified'),
        ('numerical', 'activation_representation', '16-bit floating point; FP16/BF16 unspecified'),
        ('numerical', 'group_size', 128),
        ('execution', 'runtime', 'vLLM'),
    ]:
        p.record(section, name, status='declared', expected=value, note='Course target, not inspected model evidence')
    return p


def cpu_passport(report):
    p = QuantizedModelPassport('lesson-9.1-cpu-linear', 'Synthetic FP32 input values; unpacked integers and Python float64 arithmetic')
    source = 'zero2vllm/reference/quantization_overview.py'
    evidence = ('summary.json#linear', source)
    facts = {
        'provenance': {'model':'synthetic Linear [1,3] x [3,2], no bias',
                       'quantization_framework':'zero2vllm CPU reference',
                       'algorithm':'amax/7 symmetric RTN, ties-to-even'},
        'numerical': {'weight_representation':'logical signed INT4; quantizer range [-7,7]',
                      'activation_representation':'FP32 input values', 'symmetry':'symmetric',
                      'scale_dtype':'float64', 'scale_granularity':'per_tensor',
                      'zero_point':0, 'rounding':'ties_to_even', 'clipping':'[-7,7] integer range',
                      'saturation':'clamp; current amax-calibrated fixture has no out-of-range input'},
        'execution': {'qweight_shape':[3,2], 'qweight_dtype':'Python int lists, unpacked',
                      'scale_shape':[], 'zero_point_shape':[],
                      'in_memory_layout':'nested rows W[K,N]; no contiguous low-bit buffer',
                      'activation_dtype':'FP32 values promoted to Python float64',
                      'multiply_dtype':'Python float64', 'accumulator_dtype':'Python float64',
                      'output_dtype':'Python float64', 'runtime':'zero2vllm CPU teaching program'},
        'validation': {'hardware':'host CPU execution; physical GPU not queried',
                       'framework':{'python':platform.python_version(), 'implementation':platform.python_implementation()},
                       'correctness':report['verification']},
    }
    for section, items in facts.items():
        for name, value in items.items():
            p.record(section, name, observed=value, status='verified', evidence=evidence)
    for section, names, reason in [
        ('provenance', ('model_revision','tokenizer_revision','calibration_dataset','calibration_samples','token_budget','sequence_length','seed'),
         'Deterministic synthetic tensor fixture; no tokenizer, calibration corpus or random sampling'),
        ('numerical', ('group_size','group_axis','kv_representation'),
         'Per-tensor toy Linear; no groups and no KV state'),
        ('execution', ('checkpoint_format','format_version','packing','on_disk_layout','repack','g_idx','quant_method','linear_backend','actual_kernel'),
         'No model checkpoint, packing, GPU kernel or vLLM integration in this experiment'),
        ('validation', ('driver','device_runtime'), 'CPU-only Python reference; no device runtime used'),
    ]:
        for name in names:
            p.record(section, name, status='not_applicable', note=reason)
    p.record('provenance','framework_revision',status='verified',observed=report['course_source'],evidence=('summary.json#course_source',))
    return p


def rejection(call):
    try:
        call()
    except ValueError as error:
        return str(error)
    return ''


def affine_experiment():
    f = json.loads((LESSON/'data/affine.json').read_text(encoding='utf-8'))
    q1 = quantize_affine(f['x'], **f['source'])
    dq1 = dequantize_affine(q1, **f['source'])
    args = {**{'source_'+k:v for k,v in f['source'].items()},
            **{'target_'+k:v for k,v in f['target'].items()}}
    q2 = requantize_affine(q1, **args)
    dq2 = dequantize_affine(q2, **f['target'])
    double = [[f['double_rounding']['x']]]
    return dict(input=f['x'], source=f['source'], target=f['target'], q1=q1, dq1=dq1, q2=q2, dq2=dq2,
        expected=f['expected'], wrong_dq_without_z=dequantize_affine(q1,scale=0.5,zero_point=0),
        saturated_q2=requantize_affine(q1,**{**args,'target_qmax':3}),
        double_rounding=dict(x=double, q1=quantize_affine(double,**f['source']),
            via_q1=requantize_affine(quantize_affine(double,**f['source']),**args),
            direct=quantize_affine(double,**f['target'])),
        rounding_order=dict(correct=quantize_affine([[0.25]],**f['source']),
            incorrect_round_after_adding_z=[[round(0.25/0.5+3)]]),
        compute='Float64 reference with explicit DQ intermediate; no integer requantization kernel')


def run_experiments():
    fixture = load_fixture()
    linear = linear_qdq(fixture['x'], fixture['w'])
    expected = fixture['expected']
    checks = {
        'scale':linear['qdq']['scale']==expected['scale'],
        'logical_values':linear['qdq']['q']==expected['q'],
        'dequantized_values':linear['qdq']['dequantized']==expected['dequantized'],
        'original_output':linear['original_output']==expected['original_output'],
        'quantized_output':linear['quantized_output']==expected['quantized_output'],
        'weight_error':linear['qdq']['error']['max_abs']==expected['weight_max_abs_error'],
        'weight_mse':linear['qdq']['error']['mse']==expected['weight_mse'],
        'output_error':linear['output_max_abs_error']==expected['output_max_abs_error'],
    }
    cases = []
    def add(name, condition, **details):
        cases.append(dict(name=name, passed=bool(condition), **details))
    add('hand_computed_linear',all(checks.values()),checks=checks)
    ties = logical_int4_qdq([[-0.75,-0.25,0.25,0.75,1.25]],scale=0.5)
    add('rounding_ties',ties['q']==[[-2,0,0,2,2]],result=ties)
    zero = logical_int4_qdq([[0,0],[0,0]])
    add('zero_tensor',zero['scale']==1 and zero['dequantized']==[[0,0],[0,0]],result=zero)
    clipped = logical_int4_qdq([[-4,4]],scale=0.5)
    add('clipping_breaks_half_scale_bound',clipped['q']==[[-7,7]] and not clipped['error']['half_scale_bound_satisfied'] and clipped['error']['out_of_range_count']==2,result=clipped)
    wrong = matmul(linear['x'],dequantize_logical(linear['qdq']['q'],1.0))
    add('wrong_scale_changes_model',wrong==[[-12,-5]] and wrong!=linear['quantized_output'],wrong_scale=1.0,wrong_output=wrong,
        note='Deliberately wrong metadata; this is not the quality result of the intended quantizer.')
    fp = logical_int4_qdq([[0.1,3.5]])
    add('fp32_input_is_explicit',fp['fp32_input'][0][0]==0.10000000149011612,result=fp)
    errors = [rejection(lambda x=x: logical_int4_qdq(x)) for x in ([],[[1],[2,3]],[[True]],[[float('nan')]],[[float('inf')]],[[1e50]],[[1e-50]])]
    add('reject_invalid_inputs',all(errors),expected_errors=errors)
    plan = target_passport()
    add('name_is_not_execution_evidence',all(plan.sections['execution'][k].status=='unknown' for k in ('checkpoint_format','actual_kernel','accumulator_dtype')),
        statuses=plan.status_counts())
    missing = rejection(lambda: plan.record('execution','actual_kernel',status='verified',observed='Marlin'))
    add('verified_requires_evidence',bool(missing),expected_error=missing)
    restored = QuantizedModelPassport.from_dict(json.loads(json.dumps(plan.to_dict())))
    add('passport_roundtrip',restored.to_dict()==plan.to_dict())
    affine = affine_experiment()
    add('affine_zero_point_encodes_zero',affine['q1']==affine['expected']['q1'] and affine['dq1']==affine['input'])
    add('omitting_zero_point_shifts_every_value',all(b-a==1.5 for a,b in zip(affine['dq1'][0],affine['wrong_dq_without_z'][0])))
    add('requantization_changes_scale_and_zero_point',affine['q2']==affine['expected']['q2'] and affine['dq2']==affine['expected']['dq2'])
    add('requantization_saturates_target_range',affine['saturated_q2']==[[0,2,2,3,3]])
    add('requantization_is_not_direct_quantization',affine['double_rounding']['via_q1']==[[2]] and affine['double_rounding']['direct']==[[3]])
    add('round_before_adding_zero_point',affine['rounding_order']['correct']==[[3]] and affine['rounding_order']['incorrect_round_after_adding_z']==[[4]])
    add('reject_invalid_zero_point',bool(rejection(lambda: quantize_affine([[0]],scale=0.5,zero_point=16))))
    return linear, cases, checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--inspect-source',action='store_true',help='Inspect already cached pinned vLLM text; no import')
    parser.add_argument('--fetch-source',action='store_true',help='Explicitly download missing pinned vLLM source for inspection')
    parser.add_argument('--source-root',type=Path,default=ROOT/'.local/sources')
    args = parser.parse_args(argv)
    report = dict(schema_version=1, recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                  course_source=source_fingerprint(), verification_level='cpu_numerical_reference',
                  python=platform.python_version(), gpu_validated=False, upstream_executed=False,
                  packed=False, cases=[], passed=False)
    try:
        linear,cases,checks = run_experiments()
        report.update(linear=linear,affine=affine_experiment(),cases=cases,verification=checks,passed=all(c['passed'] for c in cases))
        passports = {'cpu-passport.json':cpu_passport(report), 'qwen-plan-passport.json':target_passport()}
        if args.inspect_source or args.fetch_source:
            records = inspect_sources([LESSON/'data/source-manifest.json'],args.source_root,args.fetch_source)
            report['source_inspection'] = records
            report['passed'] &= all(r['passed'] for r in records)
        args.output_dir.mkdir(parents=True,exist_ok=True)
        for name,passport in passports.items():
            (args.output_dir/name).write_text(json.dumps(passport.to_dict(),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        report['passports'] = {name:p.status_counts() for name,p in passports.items()}
    except (ValueError,OSError) as error:
        report.update(passed=False,error=str(error))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for item in report['cases']:
        print(item['name'], 'PASS' if item['passed'] else 'FAIL')
    for item in report.get('source_inspection',[]):
        print('Pinned source inspection:',item['passed'],item['errors'])
    if report.get('error'): print(report['error'])
    print('CPU Q/DQ/RQ reference: unpacked Python integers; no quantized checkpoint or GPU execution.')
    print('Results:',args.output_dir)
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
