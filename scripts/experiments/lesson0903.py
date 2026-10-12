"""Lesson 9.3: controlled calibration, observer state, and error objectives."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import platform

from scripts.common import ROOT,source_fingerprint
from zero2vllm.quantization import QuantizedModelPassport
from zero2vllm.quantization.calibration import (prepare_samples,feature_rows,valid_rows,RangeObserver,
    FixedHistogram,parameters_from_range,percentile_threshold,mse_clip_search,error_metrics,classification_metrics,digest)
from zero2vllm.reference.quantization import matmul

LESSON=ROOT/'docs/chapters/part-09-quantization/03-calibration-observers-and-error-measurement'
DEFAULT_OUTPUT=ROOT/'.local/runs/part-09-quantization/03-calibration'


def load_fixture():
    return json.loads((LESSON/'data/corpus.json').read_text(encoding='utf8'))


def make_protocol(fixture, **kwargs):
    p=prepare_samples(fixture['records'],**kwargs)
    p['feature_source']='corpus.json#features; deterministic synthetic lookup, no transformer'
    p['feature_sha256']=digest(fixture['features'])
    p['calibration_target']='per-tensor static activation range, logical signed INT4 [-7,7]'
    p['upstream_input']='fixed unquantized feature lookup'
    p['batching']='one prepared document per observer update'
    return p


def calibrate(fixture, protocol, mode='minmax'):
    observer=RangeObserver(mode=mode,alpha=0.25)
    for tokens,mask in zip(protocol['input_ids'],protocol['attention_mask']):
        observer.update([fixture['features'][str(t)] for t in tokens],mask)
    parameters=observer.freeze()
    return parameters,observer.state()


def evaluate(fixture, protocol, parameters):
    rows,mask=feature_rows(protocol,fixture['features'])
    rows=valid_rows(rows,mask)
    qdq=parameters.qdq(rows)
    return dict(error=error_metrics(rows,qdq['dq']),saturation_fraction=qdq['saturation_fraction'],
        evaluated_tokens=len(rows),scope='Held-out synthetic activation Q/DQ, not model quality')


def controlled_studies(fixture):
    evaluation=make_protocol(fixture,split='test',sample_count=4,max_length=8,seed=0)
    count_study=[]
    for count in (2,4,8):
        p=make_protocol(fixture,split='calibration',domain='general',sample_count=count,max_length=8,seed=0)
        params,state=calibrate(fixture,p)
        count_study.append(dict(protocol=p,parameters=asdict(params),observer=state,evaluation=evaluate(fixture,evaluation,params)))
    distribution_study=[]
    for domain in ('general','code'):
        p=make_protocol(fixture,split='calibration',domain=domain,sample_count=4,max_length=8,seed=0,token_budget=16)
        params,state=calibrate(fixture,p)
        distribution_study.append(dict(protocol=p,parameters=asdict(params),observer=state,evaluation=evaluate(fixture,evaluation,params)))
    return dict(evaluation_protocol=evaluation,sample_count=count_study,fixed_token_budget=distribution_study)


def objective_examples():
    w=[[1],[1]];a=[[0.875],[1]];b=[[1],[0.75]];x=[[8,0]]
    weight_vs_layer={name:dict(weight=error_metrics(w,c),layer=error_metrics(matmul(x,w),matmul(x,c))) for name,c in [('A',a),('B',b)]}
    h_fp=[[4,1]];h_quant=[[3,1]];target=[[5]]
    reconstruction={name:dict(fp_input=error_metrics(target,matmul(h_fp,c)),
        quantized_upstream=error_metrics(target,matmul(h_quant,c))) for name,c in [('A',[[1],[1]]),('B',[[1.25],[1]])]}
    reference=[[1,0.875],[4,0]];qa=[[0.875,1],[4,0]];qb=[[1,0.5],[3,0]]
    logits={name:dict(error=error_metrics(reference,c),task=classification_metrics(c,[0,0])) for name,c in [('A',qa),('B',qb)]}
    return dict(weight_vs_layer=weight_vs_layer,reconstruction=reconstruction,logits_vs_accuracy=logits)


def clipping_experiment():
    values=[[-0.5,0.5]]*128+[[7,0]]
    search=mse_clip_search(values,[0.5,1,2,4,7])
    validation=[[0.5,-0.5],[7,-7]]
    held_out=[]
    for c in search['candidates']:
        params=parameters_from_range(-c['threshold'],c['threshold'])
        held_out.append(dict(threshold=c['threshold'],validation_mse=error_metrics(validation,params.qdq(validation)['dq'])['mse']))
    return dict(calibration_description='256 values at +/-0.5, one 7 and one 0',calibration_elements=258,
        search=search,validation=held_out,policy='Candidate chosen on calibration only; validation reveals a distribution gap')


def torch_check():
    import torch
    from torch.ao.quantization.observer import MinMaxObserver,MovingAverageMinMaxObserver
    records=[]
    batches=[[[-1,1]],[[-4,8]]]
    for mode,cls,kwargs in [('minmax',MinMaxObserver,{}),('ema',MovingAverageMinMaxObserver,{'averaging_constant':0.25})]:
        ours=RangeObserver(mode=mode,alpha=0.25)
        theirs=cls(dtype=torch.quint8,qscheme=torch.per_tensor_affine,quant_min=0,quant_max=15,**kwargs)
        for values in batches:
            ours.update(values);theirs(torch.tensor(values,dtype=torch.float32))
        p=ours.freeze(scheme='asymmetric');scale,z=theirs.calculate_qparams()
        passed=(ours.minimum==theirs.min_val.item() and ours.maximum==theirs.max_val.item()
            and math.isclose(p.scale,scale.item(),rel_tol=1e-6) and p.zero_point==z.item())
        records.append(dict(mode=mode,passed=passed,ours=asdict(p),torch_scale=scale.item(),torch_zero_point=z.item(),
            minimum=ours.minimum,maximum=ours.maximum))
    return dict(torch_version=torch.__version__,device='cpu',comparisons=records,passed=all(r['passed'] for r in records),
        scope='Two affine observer fixtures, matching range [0,15]; not full observer equivalence')


def run_experiments():
    f=load_fixture();studies=controlled_studies(f);objectives=objective_examples();clipping=clipping_experiment()
    cases=[]
    def add(name,passed,**details):cases.append(dict(name=name,passed=bool(passed),**details))
    def rejects(fn):
        try:fn()
        except ValueError:return True
        return False
    counts=studies['sample_count']
    add('nested_sample_count_study',counts[0]['protocol']['sample_ids']==counts[1]['protocol']['sample_ids'][:2]
        and counts[1]['protocol']['sample_ids']==counts[2]['protocol']['sample_ids'][:4])
    add('fixed_effective_token_budget',all(r['protocol']['valid_tokens']==16 for r in studies['fixed_token_budget']))
    all_cal={i for r in counts+studies['fixed_token_budget'] for i in r['protocol']['sample_ids']}
    add('held_out_sample_ids',not all_cal.intersection(studies['evaluation_protocol']['sample_ids']))
    p=counts[0]['protocol'];rows,mask=feature_rows(p,f['features'])
    obs=RangeObserver();obs.update(rows,mask)
    bad=RangeObserver();bad.update(rows)
    add('padding_excluded_before_observing',bad.maximum==999 and obs.maximum<999,padded_max=bad.maximum,valid_max=obs.maximum)
    alpha=[]
    for batches in ([[[-1,1]],[[-4,8]]],[[[-4,8]],[[-1,1]]]):
        ob=RangeObserver(mode='ema',alpha=0.25)
        for batch in batches:ob.update(batch)
        alpha.append(ob.state())
    add('ema_is_order_dependent',alpha[0]['minimum']==-1.75 and alpha[0]['maximum']==2.75
        and alpha[1]['minimum']==-3.25 and alpha[1]['maximum']==6.25,orders=alpha)
    single=RangeObserver(mode='ema',alpha=0.25);single.update([[-1,1],[-4,8]])
    add('ema_depends_on_batch_partition',single.minimum==-4 and single.maximum==8)
    obs.freeze();add('freeze_blocks_test_leakage',rejects(lambda:obs.update([[999]])))
    obs.reset();add('reset_removes_old_statistics',obs.minimum is None and obs.tokens==0 and not obs.frozen)
    empty=RangeObserver();empty.update([[999]],[False])
    add('empty_mask_does_not_initialize',empty.updates==0 and rejects(empty.freeze))
    h=FixedHistogram(-2,2,bins=4);h.update([[-3,-2,-1,0,1,2,3]])
    add('histogram_edges_and_out_of_range',h.counts==[1,1,1,2] and h.underflow==1 and h.overflow==1,histogram=h.state())
    add('nearest_rank_percentile',percentile_threshold([[1,2,3,100]],0.75)==3)
    c=clipping['search'];add('clipping_can_reduce_calibration_mse',c['best']['threshold']==4 and c['best']['mse']<c['candidates'][-1]['mse'])
    best_validation=next(r for r in clipping['validation'] if r['threshold']==c['best']['threshold'])
    add('calibration_best_can_lose_on_validation',best_validation['validation_mse']>clipping['validation'][-1]['validation_mse'])
    v=objectives['weight_vs_layer'];add('weight_mse_does_not_order_layer_error',v['A']['weight']['mse']<v['B']['weight']['mse'] and v['A']['layer']['mse']>v['B']['layer']['mse'])
    v=objectives['reconstruction'];add('upstream_choice_reverses_candidate_ranking',v['A']['fp_input']['mse']<v['B']['fp_input']['mse'] and v['A']['quantized_upstream']['mse']>v['B']['quantized_upstream']['mse'])
    v=objectives['logits_vs_accuracy'];add('small_logit_mse_can_flip_prediction',v['A']['error']['mse']<v['B']['error']['mse'] and v['A']['task']['accuracy']<v['B']['task']['accuracy'])
    zero=error_metrics([[0,0]],[[0,0]])
    add('zero_norm_metrics_are_explicit',zero['mse']==0 and zero['nmse'] is None and zero['cosine'] is None)
    scaling=error_metrics([[1,2]],[[2,4]])
    add('cosine_hides_scale_error',math.isclose(scaling['cosine'],1) and scaling['mse']==2.5)
    nll=classification_metrics([[0,0],[0,0],[999,0]],[0,1,-100],[True,True,False])
    add('nll_uses_valid_target_count',nll['valid_targets']==2 and math.isclose(nll['perplexity'],2),metrics=nll)
    add('shape_mismatch_cannot_silently_zip',rejects(lambda:error_metrics([[1],[2]],[[1]])))
    return dict(studies=studies,clipping=clipping,objectives=objectives,cases=cases,passed=all(c['passed'] for c in cases))


def passport(report):
    record=report['studies']['sample_count'][-1];protocol=record['protocol'];params=record['parameters']
    p=QuantizedModelPassport('lesson-9.3-static-activation-calibration','Synthetic lookup features; no LLM weights, tokenizer or inference kernel')
    facts={
        'provenance':dict(model='fixed 2-channel token-feature lookup',quantization_framework='zero2vllm CPU observer',
            framework_revision=report['course_source'],algorithm='per-tensor MinMax then symmetric RTN',
            calibration_dataset={'name':protocol['dataset'],'split':protocol['split'],'domain':protocol['domain'],
                'sha256':protocol['dataset_sha256'],'sample_ids':protocol['sample_ids']},
            calibration_samples=protocol['used_samples'],token_budget=protocol['valid_tokens'],sequence_length=protocol['max_length'],seed=protocol['seed']),
        'numerical':dict(activation_representation='logical signed INT4 [-7,7]; static activation Q/DQ',symmetry='symmetric',
            scale_dtype='float64',scale_granularity='per_tensor',zero_point=params['zero_point'],rounding='ties_to_even_before_adding_z',
            clipping=[params['qmin'],params['qmax']],saturation='clamp at frozen range'),
        'execution':dict(activation_dtype='FP32 lookup values',multiply_dtype='Python float64 reference',
            accumulator_dtype='Python float64 reference',output_dtype='Python float64 DQ',runtime='offline CPU calibration reference'),
        'validation':dict(hardware='host CPU only',framework={'python':platform.python_version()},
            correctness={'case_count':len(report['cases']),'passed':report['passed']})}
    for section,items in facts.items():
        for key,value in items.items():
            p.record(section,key,status='verified',observed=value,evidence=('summary.json','protocol.json','zero2vllm/quantization/calibration.py'))
    for section,keys in {'provenance':('model_revision','tokenizer_revision'),
        'numerical':('weight_representation','kv_representation','group_size','group_axis'),
        'execution':('checkpoint_format','format_version','qweight_shape','qweight_dtype','scale_shape','zero_point_shape','g_idx','packing','on_disk_layout','repack','in_memory_layout','quant_method','linear_backend','actual_kernel'),
        'validation':('driver','device_runtime')}.items():
        for key in keys:p.record(section,key,status='not_applicable',note='Synthetic activation experiment, no model weights/checkpoint/device kernel')
    return p


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--torch-check',action='store_true')
    args=parser.parse_args(argv)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report=dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),course_source=source_fingerprint(),
        gpu_validated=False,real_model_evaluated=False,mqbench_executed=False,passed=False,cases=[])
    try:
        report.update(run_experiments())
        if args.torch_check:
            report['torch_observers']=torch_check();report['passed'] &= report['torch_observers']['passed']
        outputs={'protocol.json':report['studies']['sample_count'][-1]['protocol'],'cpu-passport.json':passport(report).to_dict()}
        for name,data in outputs.items():
            (args.output_dir/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except (ValueError,ImportError,RuntimeError,OSError) as error:
        report.update(passed=False,error=str(error))
    (args.output_dir/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    for c in report['cases']:print(c['name'],'PASS' if c['passed'] else 'FAIL')
    if 'error' in report:print(report['error'])
    print('Calibration and error reference on synthetic features; no real model PPL or GPU results.')
    print('Results:',args.output_dir)
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
