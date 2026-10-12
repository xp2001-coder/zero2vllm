"""Run small-matrix quantization algorithm experiments; CPU, no downloads."""
import argparse
import json
import math
from pathlib import Path
import platform
import random
from datetime import datetime, timezone

from scripts.common import ROOT, source_fingerprint
from zero2vllm.quantization import QuantizedModelPassport
from zero2vllm.quantization.calibration import AffineParameters, digest, error_metrics
from zero2vllm.quantization.algorithms import (column_grids, rtn, layer_metrics, clipping_search,
    optimize_rounding, gptq_reference, gram_hessian, scaling_search, diagonal_transform,
    hadamard, rotate, ste_derivatives, stochastic_code, transpose, matmul)

LESSON = ROOT/'docs/chapters/part-09-quantization/04-quantization-algorithms-and-error-control'
DEFAULT_OUTPUT = ROOT/'.local/runs/part-09-quantization/04-algorithms'


def load_fixture():
    return json.loads((LESSON/'data/linear.json').read_text(encoding='utf8'))


def comparison(fixture):
    f = fixture['main']
    x, w, test = f['calibration'], f['weight'], f['held_out']
    grids = column_grids(w, bits=f['bits'])
    base = rtn(w, grids)
    clip = clipping_search(x, w, f['clipping_ratios'])
    rounding = optimize_rounding(x, w, grids)
    gptq = gptq_reference(x, w, grids, damping=f['damping'], block_size=2)
    scale = scaling_search(x, w, f['scaling_alphas'])
    rows = []
    for name, candidate, budget in [
        ('RTN', base['dq'], 'one nearest rounding pass'),
        ('clipping search', clip['best']['dq'], f'{len(clip["candidates"])} range candidates'),
        ('floor/ceil exhaustive', rounding['best']['dq'], f'{rounding["candidates"]} discrete candidates'),
        ('GPTQ core reference', gptq['dq'], 'one 3x3 inverse/Cholesky and 3 row decisions'),
        ('AWQ-inspired scaling', scale['best']['effective_w'], f'{len(scale["candidates"])} scaling candidates'),
    ]:
        rows.append(dict(method=name, weight_mse=error_metrics(w,candidate)['mse'],
            calibration_mse=layer_metrics(x,w,candidate)['mse'], held_out_mse=layer_metrics(test,w,candidate)['mse'],
            search_budget=budget, effective_weight=candidate))
    return dict(rows=rows, rtn=base, clipping=clip, rounding=rounding, gptq=gptq, scaling=scale,
        scope='Same W/X, INT4 per-output grids and held-out rows; range/scaling methods intentionally change grids. No model algorithm ranking.')


def torch_check(fixture):
    """Independent remaining-submatrix inverse and autograd surrogate oracles."""
    import torch
    f = fixture['main']; x, w = f['calibration'], f['weight']
    grids = column_grids(w)
    checks = []
    for act_order in (False, True):
        h, lam = gram_hessian(x, f['damping'])
        order = sorted(range(len(w)), key=lambda i:(-h[i][i],i)) if act_order else list(range(len(w)))
        ht = torch.tensor(h,dtype=torch.float64)[order][:,order]
        work = torch.tensor(w,dtype=torch.float64)[order].clone()
        out = torch.empty_like(work)
        scales = torch.tensor([p.scale for p in grids],dtype=torch.float64)
        zeros = torch.tensor([p.zero_point for p in grids],dtype=torch.float64)
        lo = torch.tensor([p.qmin for p in grids],dtype=torch.float64)
        hi = torch.tensor([p.qmax for p in grids],dtype=torch.float64)
        for i in range(len(w)):
            inv = torch.linalg.inv(ht[i:,i:])
            q = (torch.round(work[i]/scales)+zeros).clamp(min=lo,max=hi)
            out[i] = scales*(q-zeros)
            delta = work[i]-out[i]
            work[i:] -= inv[:,0,None]/inv[0,0]*delta[None,:]
        restored = torch.empty_like(out)
        for i, original in enumerate(order):restored[original]=out[i]
        ref = gptq_reference(x,w,grids,damping=f['damping'],act_order=act_order,block_size=2)
        diff = error_metrics(restored.tolist(),ref['dq'])['max_abs']
        checks.append(dict(name=f'GPTQ vs constrained inverse, act_order={act_order}',max_abs=diff,passed=diff<1e-12))
    grads = []
    p = AffineParameters(.5,0,-7,7,'symmetric')
    for value in (-4.0,-.6,.6,4.0):
        v = torch.tensor(value,dtype=torch.float64,requires_grad=True)
        s = torch.tensor(p.scale,dtype=torch.float64,requires_grad=True)
        normalized = (v/s).clamp(p.qmin,p.qmax)
        rounded = normalized + (normalized.round()-normalized).detach()
        y = s*rounded
        y.backward()
        expected = ste_derivatives(value,p)
        grads.append(dict(x=value,torch_dx=v.grad.item(),torch_ds=s.grad.item(),expected=expected,
            passed=abs(v.grad.item()-expected['dx'])<1e-12 and abs(s.grad.item()-expected['ds'])<1e-12))
    return dict(torch_version=torch.__version__,device='cpu',dtype='float64',gptq_oracles=checks,
        ste_oracles=grads,passed=all(c['passed'] for c in checks+grads),
        scope='Independent small-matrix math and explicitly chosen STE; not execution of upstream GPTQ/AWQ packages')


def run_experiments():
    f = load_fixture(); table = comparison(f)
    t = f['rounding']; x,w = t['calibration'],t['weight']
    p = [AffineParameters(t['scale'],t['zero_point'],t['qmin'],t['qmax'],'symmetric')]
    toy = dict(rtn=rtn(w,p), rounding=optimize_rounding(x,w,p), gptq=gptq_reference(x,w,p,damping=0))
    toy['held_out_rtn'] = layer_metrics(t['held_out'],w,toy['rtn']['dq'])
    toy['held_out_optimized'] = layer_metrics(t['held_out'],w,toy['rounding']['best']['dq'])
    scale = scaling_search(f['scaling']['calibration'],f['scaling']['weight'])
    r = hadamard(2); rx,rw = rotate([[8,0]],[[.25],[1]],r)
    rotation = dict(matrix=r, input_before=[[8,0]],input_after=rx,weight_after=rw,
        output_before=matmul([[8,0]],[[.25],[1]]),output_after=matmul(rx,rw))
    cases=[]
    def add(name,passed):cases.append(dict(name=name,passed=bool(passed)))
    def rejects(fn):
        try:fn()
        except ValueError:return True
        return False
    add('nearest_rounding_is_zero',toy['rtn']['q']==[[0],[0]])
    add('four_rounding_candidates',toy['rounding']['candidates']==4)
    add('reconstruction_selects_zero_one',toy['rounding']['best']['q']==[[0],[1]])
    add('hand_reconstruction_sse',math.isclose(toy['rounding']['best']['loss'],.2405))
    add('gptq_compensates_second_weight',math.isclose(toy['gptq']['trace'][1]['before'][0],.98))
    add('gptq_matches_toy_optimum',toy['gptq']['q']==toy['rounding']['best']['q'])
    add('calibration_gain_can_lose_held_out',toy['held_out_optimized']['mse']>toy['held_out_rtn']['mse'])
    main=f['main']; grids=column_grids(main['weight'])
    for size in (1,2,8):
        q=gptq_reference(main['calibration'],main['weight'],grids,block_size=size)
        add(f'lazy_block_{size}_same_codes',q['q']==table['gptq']['q'])
    for act in (False,True):
        q=gptq_reference(main['calibration'],main['weight'],grids,act_order=act)
        add(f'loss_accounting_act_order_{act}',math.isclose(q['trace_objective'],q['regularized_objective'],rel_tol=1e-10,abs_tol=1e-12))
    add('rank_deficiency_without_damping_rejected',rejects(lambda:gptq_reference([[1,1],[1,1]],w,p,damping=0)))
    repaired=gptq_reference([[1,1],[1,1]],w,p,damping=.01)
    add('damping_allows_rank_deficient_case',repaired['lambda_absolute']>0)
    add('exhaustive_size_guard',rejects(lambda:optimize_rounding(x,w,p,max_variables=1)))
    add('scaling_float_identity',max(c['identity_error'] for c in scale['candidates'])<1e-12)
    add('scaling_can_improve',scale['best']['calibration_mse']<scale['candidates'][0]['calibration_mse'])
    add('scaling_not_monotonic',scale['candidates'][1]['calibration_mse']>scale['candidates'][0]['calibration_mse'])
    add('rotation_float_identity',error_metrics(rotation['output_before'],rotation['output_after'])['max_abs']<1e-12)
    add('rotation_redistributes_peak',max(abs(v) for v in rx[0])<8 and math.isclose(sum(v*v for v in rx[0]),64))
    z=AffineParameters(.5,3,0,15,'asymmetric')
    add('nonzero_zero_point_preserved',rtn([[0],[.25]], [z])['q']==[[3],[3]])
    add('ste_has_zero_saturated_input_gradient',ste_derivatives(4,AffineParameters(.5,0,-7,7,'symmetric'))['dx']==0)
    rng=random.Random(7); sampled=[stochastic_code(.25,p[0],rng) for _ in range(10000)]
    stochastic=dict(seed=7,draws=len(sampled),mean_code=sum(sampled)/len(sampled),theoretical_mean=.25,
                    note='Within range only; expectation does not guarantee a smaller loss for one draw')
    add('stochastic_sampling_matches_declared_distribution',abs(stochastic['mean_code']-.25)<.02)
    return dict(comparison=table,rounding_toy=toy,scaling_toy=scale,rotation=rotation,
                stochastic=stochastic,cases=cases,passed=all(c['passed'] for c in cases))


def passport(report):
    p=QuantizedModelPassport('lesson-9.4-fixed-grid-gptq-core','3x2 synthetic Linear, CPU float64; algorithm reference only')
    grids=report['comparison']['gptq']['grids']
    facts={
        'provenance':dict(model='synthetic Linear W[3,2]',quantization_framework='zero2vllm teaching algorithms',
            framework_revision=report['course_source'],algorithm='GPTQ Cholesky core; fixed per-output grid, relative damping .01, block size 2',
            calibration_dataset={'fixture':'data/linear.json','sha256':report['fixture_sha256'],'split':'main.calibration'},
            calibration_samples=4,seed=0),
        'numerical':dict(weight_representation='logical signed INT4 [-7,7]',activation_representation='float64 unquantized',
            symmetry='symmetric',group_size=3,group_axis='K: axis 0 of W[K,N]',scale_dtype='float64',
            scale_granularity='one scale per output channel; one full-K group',zero_point=[g['zero_point'] for g in grids],
            rounding='nearest-even of compensated weight before adding zero-point',clipping=[-7,7],saturation='clamp integer codes'),
        'execution':dict(activation_dtype='float64',multiply_dtype='float64',accumulator_dtype='float64',output_dtype='float64',runtime='offline CPU reference'),
        'validation':dict(hardware='host CPU',framework={'python':platform.python_version()},
            correctness={'passed':report['passed'],'cases':len(report['cases'])})}
    for section,values in facts.items():
        for name,value in values.items():
            p.record(section,name,status='verified',observed=value,evidence=('summary.json','data/linear.json','zero2vllm/quantization/algorithms.py'))
    for section,names in {'provenance':('model_revision','tokenizer_revision','token_budget','sequence_length'),
        'numerical':('kv_representation',),'execution':('checkpoint_format','format_version','qweight_shape','qweight_dtype',
        'scale_shape','zero_point_shape','g_idx','packing','on_disk_layout','repack','in_memory_layout','quant_method','linear_backend','actual_kernel'),
        'validation':('driver','device_runtime')}.items():
        for name in names:p.record(section,name,status='not_applicable',note='Synthetic un-packed Linear reference; no model tokenizer/checkpoint/device kernel')
    return p.to_dict()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--torch-check',action='store_true')
    args=parser.parse_args(argv);args.output_dir.mkdir(parents=True,exist_ok=True)
    report=dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),
        course_source=source_fingerprint(),fixture_sha256=digest(load_fixture()),gpu_validated=False,
        real_model_evaluated=False,upstream_algorithm_packages_executed=False,passed=False,cases=[])
    try:
        report.update(run_experiments())
        if args.torch_check:
            report['torch_oracles']=torch_check(load_fixture())
            report['passed'] &= report['torch_oracles']['passed']
        (args.output_dir/'cpu-passport.json').write_text(json.dumps(passport(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except (ValueError,ImportError,RuntimeError,OSError) as error:
        report.update(passed=False,error=str(error))
    (args.output_dir/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    for c in report['cases']:print(c['name'],'PASS' if c['passed'] else 'FAIL')
    if 'error' in report:print(report['error'])
    print('CPU synthetic algorithm references; no Qwen quality or GPU performance measured.')
    print('Results:',args.output_dir)
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
