"""Small, inspectable calibration protocols and observers, with no model download.

The corpus contains synthetic token IDs and a fixed feature table, not outputs
from a language model. Padding masks are applied before observing values.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import random

from zero2vllm.quantization.tensor_contract import cast_scale
from zero2vllm.reference.quantization_overview import fp32_matrix


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(encoded.encode('utf8')).hexdigest()


def validate_corpus(records):
    if not isinstance(records, list) or not records:
        raise ValueError('Expected a nonempty record list')
    ids, contents = set(), {}
    for record in records:
        if set(record) != {'id', 'split', 'domain', 'token_ids'}:
            raise ValueError('Record must contain id, split, domain and token_ids')
        if any(not isinstance(record[k], str) or not record[k] for k in ('id', 'split', 'domain')):
            raise ValueError('Record labels must be nonempty strings')
        if record['split'] not in {'calibration', 'validation', 'test'}:
            raise ValueError('Unknown split')
        if record['id'] in ids:
            raise ValueError('Duplicate sample ID')
        ids.add(record['id'])
        tokens = record['token_ids']
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 3 for t in tokens):
            raise ValueError('Body tokens are nonempty integer IDs >= 3; 0/1/2 are reserved')
        key = tuple(tokens)
        if key in contents and contents[key] != record['split']:
            raise ValueError('Identical token content leaks across dataset splits')
        contents[key] = record['split']


def prepare_samples(records, *, split, domain=None, sample_count=4, max_length=8,
                    seed=0, token_budget=None, add_bos=True, add_eos=True):
    """Shuffle document IDs, take a prefix, truncate, budget, then right-pad.

    Length truncation reserves BOS/EOS positions. A final budget cut keeps a
    prefix and may omit EOS. The report distinguishes requested and used counts.
    No cross-document concatenation or language-model tokenization is performed.
    """
    validate_corpus(records)
    for value, name in ((sample_count, 'sample_count'), (max_length, 'max_length')):
        _positive_int(value, name)
    if type(seed) is not int or type(add_bos) is not bool or type(add_eos) is not bool:
        raise ValueError('seed must be int; special-token options must be bool')
    if token_budget is not None:
        _positive_int(token_budget, 'token_budget')
    if max_length <= int(add_bos) + int(add_eos):
        raise ValueError('max_length must leave room for at least one body token')
    eligible = [r for r in records if r['split'] == split and (domain is None or r['domain'] == domain)]
    if len(eligible) < sample_count:
        raise ValueError('Not enough records for requested sample_count')
    random.Random(seed).shuffle(eligible)
    selected = eligible[:sample_count]
    rows, masks, sample_ids, body_dropped = [], [], [], 0
    remaining = token_budget
    for record in selected:
        if remaining == 0:
            break
        capacity = max_length - int(add_bos) - int(add_eos)
        body = record['token_ids'][:capacity]
        body_dropped += len(record['token_ids']) - len(body)
        tokens = ([1] if add_bos else []) + body + ([2] if add_eos else [])
        if remaining is not None:
            tokens = tokens[:remaining]
            remaining -= len(tokens)
        sample_ids.append(record['id'])
        masks.append([True] * len(tokens) + [False] * (max_length - len(tokens)))
        rows.append(tokens + [0] * (max_length - len(tokens)))
    if remaining is not None and remaining > 0:
        raise ValueError('Requested token budget exceeds selected valid tokens')
    valid = [t for row, mask in zip(rows, masks) for t, keep in zip(row, mask) if keep]
    return dict(schema_version=1, dataset='zero2vllm synthetic token fixture v1',
        dataset_sha256=digest(records), tokenizer={'name':'pretokenized-synthetic-ids', 'revision':'v1',
        'language_model_tokenizer':False}, split=split, domain=domain, seed=seed,
        requested_samples=sample_count, selected_ids=[r['id'] for r in selected], used_samples=len(rows),
        sample_ids=sample_ids, max_length=max_length, requested_token_budget=token_budget,
        valid_tokens=len(valid), body_tokens=sum(t >= 3 for t in valid),
        bos_tokens=valid.count(1), eos_tokens=valid.count(2), padding_tokens=len(rows)*max_length-len(valid),
        body_tokens_dropped_by_length=body_dropped,
        preprocessing={'bos':add_bos,'eos':add_eos,'length_policy':'reserve specials, truncate body on right',
        'budget_policy':'keep prefix of final sample; EOS can be omitted', 'padding':'right; token_id=0',
        'concatenation':False, 'statistics_mask':'all retained tokens including BOS/EOS; exclude padding'},
        input_ids=rows, attention_mask=masks, prepared_sha256=digest(dict(ids=sample_ids, rows=rows, masks=masks)))


def feature_rows(protocol, features):
    """Map synthetic IDs to features; masks remain explicit for the observer."""
    rows, mask = [], []
    for tokens, keep in zip(protocol['input_ids'], protocol['attention_mask']):
        for token, valid in zip(tokens, keep):
            if str(token) not in features:
                raise ValueError(f'Missing synthetic feature for token {token}')
            rows.append(list(features[str(token)]))
            mask.append(valid)
    return rows, mask


def valid_rows(values, mask=None):
    if not isinstance(values, (list, tuple)) or not values or not isinstance(values[0], (list, tuple)) or not values[0]:
        raise ValueError('Expected a nonempty rectangular [tokens,channels] matrix')
    width = len(values[0])
    if any(not isinstance(r, (list, tuple)) or len(r) != width for r in values):
        raise ValueError('Expected rectangular matrix')
    if mask is None:
        mask = [True] * len(values)
    if not isinstance(mask, (list, tuple)) or len(mask) != len(values) or any(type(v) is not bool for v in mask):
        raise ValueError('Mask must contain one boolean per token row')
    selected = [row for row, keep in zip(values, mask) if keep]
    return fp32_matrix(selected) if selected else []


@dataclass(frozen=True)
class AffineParameters:
    scale: float
    zero_point: int
    qmin: int
    qmax: int
    scheme: str

    def __post_init__(self):
        if type(self.scale) not in (float, int) or not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('scale must be positive finite')
        if any(type(x) is not int for x in (self.zero_point, self.qmin, self.qmax)):
            raise ValueError('Integer bounds and zero-point required')
        if not self.qmin < self.qmax or not self.qmin <= self.zero_point <= self.qmax:
            raise ValueError('Invalid affine integer bounds')

    def qdq(self, values):
        rows = fp32_matrix(values)
        # Clamp the normalized value at integer endpoints before round to avoid
        # overflow of v/scale; this has the same final saturated code.
        q = [[round(max(self.qmin-self.zero_point,
                       min(self.qmax-self.zero_point, v/self.scale))) + self.zero_point
              for v in row] for row in rows]
        dq = [[self.scale * (v-self.zero_point) for v in row] for row in q]
        lo, hi = self.scale*(self.qmin-self.zero_point), self.scale*(self.qmax-self.zero_point)
        clipped = sum(v < lo or v > hi for row in rows for v in row)
        return dict(q=q, dq=dq, saturation_fraction=clipped/(len(rows)*len(rows[0])))


def parameters_from_range(low, high, *, bits=4, scheme='symmetric'):
    if type(bits) is not int or bits not in (2,3,4,8):
        raise ValueError('Supported widths: 2,3,4,8')
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in (low,high)) or low > high:
        raise ValueError('Expected ordered finite range')
    if scheme == 'symmetric':
        qmax = 2**(bits-1)-1
        scale = max(abs(low), abs(high))/qmax or 1.0
        return AffineParameters(cast_scale(scale,'float64'),0,-qmax,qmax,scheme)
    if scheme == 'asymmetric':
        low, high = min(low,0), max(high,0)
        qmax = 2**bits-1
        scale = (high-low)/qmax or 1.0
        z = max(0,min(qmax,round(-low/scale)))
        return AffineParameters(cast_scale(scale,'float64'),z,0,qmax,scheme)
    raise ValueError('Choose symmetric or asymmetric')


class RangeObserver:
    """Per-tensor global MinMax or EMA of per-batch extrema.

    EMA alpha is the weight of the NEW batch, not the old-state coefficient.
    Empty masked batches do not initialize or advance the observer.
    """
    def __init__(self, *, mode='minmax', alpha=0.1):
        if mode not in ('minmax','ema') or type(alpha) not in (int,float) or not 0 < alpha <= 1:
            raise ValueError('Expected minmax/ema and alpha in (0,1]')
        self.mode, self.alpha = mode, float(alpha)
        self.reset()

    def reset(self):
        self.minimum = self.maximum = None
        self.updates = self.tokens = self.elements = 0
        self.frozen = False

    def update(self, values, mask=None):
        if self.frozen:
            raise ValueError('Observer is frozen; reset before new calibration')
        rows = valid_rows(values, mask)
        if not rows:
            return
        flat = [v for row in rows for v in row]
        low, high = min(flat), max(flat)
        if self.minimum is None:
            self.minimum, self.maximum = low, high
        elif self.mode == 'minmax':
            self.minimum, self.maximum = min(self.minimum,low), max(self.maximum,high)
        else:
            self.minimum += self.alpha * (low-self.minimum)
            self.maximum += self.alpha * (high-self.maximum)
        self.updates += 1
        self.tokens += len(rows)
        self.elements += len(flat)

    def freeze(self, *, bits=4, scheme='symmetric'):
        if self.minimum is None:
            raise ValueError('No valid observations')
        params = parameters_from_range(self.minimum,self.maximum,bits=bits,scheme=scheme)
        self.frozen = True
        return params

    def state(self):
        return dict(mode=self.mode,alpha_new_batch=self.alpha if self.mode=='ema' else None,
            minimum=self.minimum,maximum=self.maximum,updates=self.updates,tokens=self.tokens,
            elements=self.elements,frozen=self.frozen)


class FixedHistogram:
    """Exact counts in fixed bins; under/overflow are retained, not silently clipped.

    Range selection and a second counting pass are separate. This is not a
    replica of PyTorch's streaming rebinning or its threshold optimizer.
    """
    def __init__(self, low, high, bins=8):
        _positive_int(bins,'bins')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (low,high)) or low >= high:
            raise ValueError('Histogram range must be finite and nonempty')
        self.low, self.high, self.bins = low, high, bins
        self.counts = [0]*bins
        self.underflow = self.overflow = 0

    def update(self, values, mask=None):
        for row in valid_rows(values,mask):
            for v in row:
                if v < self.low:
                    self.underflow += 1
                elif v > self.high:
                    self.overflow += 1
                else:
                    index = min(self.bins-1,int((v-self.low)/(self.high-self.low)*self.bins))
                    self.counts[index] += 1

    def state(self):
        return dict(edges=[self.low+(self.high-self.low)*i/self.bins for i in range(self.bins+1)],
            counts=list(self.counts),underflow=self.underflow,overflow=self.overflow,
            interval='left-closed/right-open; final bin includes high')


def percentile_threshold(values, percentile=0.99, mask=None):
    if type(percentile) not in (int,float) or not 0 < percentile <= 1:
        raise ValueError('percentile must be in (0,1]')
    ordered = sorted(abs(v) for row in valid_rows(values,mask) for v in row)
    if not ordered:
        raise ValueError('No valid values')
    return ordered[math.ceil(percentile*len(ordered))-1]


def error_metrics(reference, candidate):
    # Keep float64 DQ outputs intact; shape validation must not zip away rows.
    if not reference or not candidate or len(reference) != len(candidate):
        raise ValueError('Metric matrices must be nonempty and have identical shape')
    width = len(reference[0])
    if not width or any(len(a)!=width or len(b)!=width for a,b in zip(reference,candidate)):
        raise ValueError('Metric matrices must have identical rectangular shape')
    a, b = [v for row in reference for v in row], [v for row in candidate for v in row]
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in a+b):
        raise ValueError('Metrics require finite numeric values')
    sse = math.fsum((x-y)**2 for x,y in zip(a,b))
    energy = math.fsum(x*x for x in a)
    other_energy = math.fsum(x*x for x in b)
    cosine = math.fsum(x*y for x,y in zip(a,b))/math.sqrt(energy*other_energy) if energy and other_energy else None
    return dict(elements=len(a),sse=sse,mse=sse/len(a),max_abs=max(abs(x-y) for x,y in zip(a,b)),
        reference_energy=energy,nmse=sse/energy if energy else None,
        cosine=max(-1.0,min(1.0,cosine)) if cosine is not None else None,
        undefined_policy='NMSE null if reference energy=0; cosine null if either norm=0; no epsilon added')


def mse_clip_search(values, thresholds, *, bits=4, mask=None):
    rows = valid_rows(values,mask)
    if not rows or not isinstance(thresholds,(list,tuple)) or not thresholds:
        raise ValueError('Need valid calibration values and candidate thresholds')
    candidates = []
    for threshold in thresholds:
        if type(threshold) not in (int,float) or not math.isfinite(threshold) or threshold <= 0:
            raise ValueError('Thresholds must be positive finite')
        params = parameters_from_range(-threshold,threshold,bits=bits)
        result = params.qdq(rows)
        candidates.append(dict(threshold=threshold,scale=params.scale,zero_point=params.zero_point,
            mse=error_metrics(rows,result['dq'])['mse'],saturation_fraction=result['saturation_fraction']))
    best = min(candidates,key=lambda c:(c['mse'],-c['threshold']))
    return dict(best=dict(best),candidates=candidates,objective='Element-weighted calibration MSE',
        tie_break='larger threshold',search='finite supplied candidate set, not a global optimum')


def classification_metrics(logits, labels, mask=None):
    """Stable NLL for already aligned targets; no implicit causal label shift."""
    if mask is None:
        mask = [True]*len(logits)
    if len(labels) != len(logits) or len(mask) != len(logits) or any(type(v) is not bool for v in mask):
        raise ValueError('Labels and boolean mask must match logit rows')
    rows = valid_rows(logits,mask)
    targets = [y for y,keep in zip(labels,mask) if keep]
    if not rows:
        raise ValueError('No valid targets')
    nlls, correct = [], 0
    for row, target in zip(rows,targets):
        if type(target) is not int or not 0 <= target < len(row):
            raise ValueError('Target outside class vocabulary')
        maximum = max(row)
        # Subtract the target before adding the small log-sum-exp correction;
        # adding it to a large shared logit first could round it away.
        nlls.append((maximum-row[target]) + math.log(math.fsum(math.exp(v-maximum) for v in row)))
        correct += max(range(len(row)),key=row.__getitem__) == target
    nll = math.fsum(nlls)
    mean = nll/len(rows)
    return dict(valid_targets=len(rows),nll_sum=nll,mean_nll=mean,
        perplexity=math.exp(mean) if mean < 709 else None,accuracy=correct/len(rows),
        scope='Synthetic aligned logits; not language-model PPL or benchmark accuracy')
