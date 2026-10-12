"""Lesson 3.5: explicit, bounded v0.8.5 configuration rules on CPU.

Not EngineArgs, not a YAML parser, not hardware detection. Unmodeled features
must be checked against pinned source and on the target platform.
"""
from copy import deepcopy


DEFAULTS = {'max_model_len': None, 'max_num_batched_tokens': None, 'max_num_seqs': None,
            'enable_prefix_caching': None, 'enable_chunked_prefill': None,
            'enforce_eager': None, 'block_size': None}
BOOL_FIELDS = {'enable_prefix_caching', 'enable_chunked_prefill', 'enforce_eager'}


def engine_choice(env_value, feature_status):
    """Feature status is an INPUT from source review, never detected here.

    'unsupported' models _raise_or_fallback; 'experimental' models
    _warn_or_fallback. Only explicit '0'/'1' accepted by this teaching API.
    """
    if env_value not in (None, '0', '1') or feature_status not in ('supported', 'experimental', 'unsupported'):
        raise ValueError('expected absent/0/1 and a classified feature status')
    if env_value == '0':
        return {'engine': 'V0', 'reason': 'explicit V0'}
    if feature_status == 'unsupported':
        if env_value == '1':
            raise ValueError('explicit V1 cannot satisfy unsupported feature')
        return {'engine': 'V0', 'reason': 'automatic fallback: unsupported feature'}
    if feature_status == 'experimental' and env_value is None:
        return {'engine': 'V0', 'reason': 'automatic fallback: experimental feature'}
    return {'engine': 'V1', 'reason': 'experimental opt-in' if feature_status == 'experimental' else 'supported feature'}


def normalized(values):
    output = {}
    for name, value in values.items():
        key = name.replace('-', '_')
        if key not in DEFAULTS or key in output:
            raise ValueError('unknown or duplicate normalized option: ' + name)
        if key in BOOL_FIELDS:
            if value is not None and type(value) is not bool:
                raise ValueError('boolean options require true/false/null, not strings')
        elif value is not None and (type(value) is not int or value < 1):
            raise ValueError('lengths and capacities require positive integers')
        output[key] = value
    return output


def verify_scheduler(tokens, seqs, model_len, chunked):
    if any(type(v) is not int or v < 1 for v in (tokens, seqs, model_len)) or type(chunked) is not bool:
        raise ValueError('positive capacities and a resolved boolean required')
    if not chunked and tokens < model_len:
        raise ValueError('unchunked prefill requires token budget >= model length')
    if tokens < seqs:
        raise ValueError('token budget must be >= max_num_seqs')


def resolve_v1(file_options=None, cli_options=None, *, usage='server', memory_gib=40, model_limit=4096, platform='cuda'):
    """Selected dense text, TP=1 V1 policies, with an origin trail per write.

    `file_options` is an already decoded flat mapping (JSON in our CLI); real
    vLLM --config uses its YAML/argparse path. No model config or device loaded.
    Positive/type checks and a simple model-limit check are teaching guards.
    """
    if usage not in ('server', 'offline') or platform != 'cuda':
        raise ValueError('this resolver only models CUDA server/offline defaults; other platforms need separate evidence')
    if type(memory_gib) not in (int, float) or not 0 <= memory_gib < float('inf') or type(model_limit) is not int or model_limit < 1:
        raise ValueError('invalid synthetic device capacity or model limit')
    values, origins, trace = {}, {}, []

    def assign(key, value, phase):
        trace.append({'field': key, 'before': values.get(key), 'after': value, 'phase': phase})
        values[key], origins[key] = value, phase

    for key, value in DEFAULTS.items(): assign(key, value, 'declared_default')
    for phase, options in [('file', file_options or {}), ('cli', cli_options or {})]:
        for key, value in normalized(options).items(): assign(key, value, phase)
    if values['max_model_len'] is None: assign('max_model_len', model_limit, 'model_config')
    if values['max_model_len'] > model_limit:
        raise ValueError('model length exceeds this fixture limit; RoPE overrides are outside scope')
    assign('enable_chunked_prefill', True, 'v1_policy: mandatory chunked prefill')
    if values['enable_prefix_caching'] is None: assign('enable_prefix_caching', True, 'v1_default')
    high = memory_gib >= 70
    budget = (8192 if usage == 'server' else 16384) if high else (2048 if usage == 'server' else 8192)
    if values['max_num_batched_tokens'] is None: assign('max_num_batched_tokens', budget, 'v1_device_usage_default')
    if values['max_num_seqs'] is None: assign('max_num_seqs', 1024 if high else 256, 'v1_device_default')
    if values['enforce_eager'] is None: assign('enforce_eager', False, 'model_default')
    verify_scheduler(values['max_num_batched_tokens'], values['max_num_seqs'], values['max_model_len'], values['enable_chunked_prefill'])
    if values['block_size'] is None: assign('block_size', 16, 'cuda_platform_default')
    return deepcopy({'values': values, 'origins': origins, 'trace': trace,
                     'graph_intent': 'disabled' if values['enforce_eager'] else 'eligible_for_configuration_not_proven_active',
                     'scope': 'v0.8.5 selected V1 CUDA defaults; synthetic context; no vLLM or GPU execution',
                     'runtime_checks_pending': ['model loading', 'backend dimensions/dtype', 'cache capacity', 'capture and replay', 'numerical validation']})
