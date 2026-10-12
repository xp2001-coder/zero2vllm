"""An evidence ledger, not a model loader or an automatic hardware verifier.

Expected settings and observed facts remain separate. A citation is required
for a verified entry, but the ledger cannot prove that a citation is truthful.
"""
from dataclasses import asdict, dataclass, field
import json


FIELDS = {
    'provenance': (
        'model', 'model_revision', 'tokenizer_revision', 'quantization_framework',
        'framework_revision', 'algorithm', 'calibration_dataset', 'calibration_samples',
        'token_budget', 'sequence_length', 'seed',
    ),
    'numerical': (
        'weight_representation', 'activation_representation', 'kv_representation',
        'symmetry', 'group_size', 'group_axis', 'scale_dtype', 'scale_granularity',
        'zero_point', 'rounding', 'clipping', 'saturation',
    ),
    'execution': (
        'checkpoint_format', 'format_version', 'qweight_shape', 'qweight_dtype',
        'scale_shape', 'zero_point_shape', 'g_idx', 'packing', 'on_disk_layout',
        'repack', 'in_memory_layout', 'activation_dtype', 'multiply_dtype',
        'accumulator_dtype', 'output_dtype', 'runtime', 'quant_method',
        'linear_backend', 'actual_kernel',
    ),
    'validation': (
        'hardware', 'driver', 'device_runtime', 'framework', 'correctness',
        'model_quality', 'model_memory', 'kv_memory', 'prefill', 'decode',
        'ttft', 'tpot', 'throughput', 'goodput',
    ),
}


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError('Passport values must be finite JSON data') from error


@dataclass(frozen=True)
class PassportEntry:
    expected: object = None
    observed: object = None
    status: str = 'unknown'
    evidence: tuple[str, ...] = ()
    note: str = ''

    def __post_init__(self):
        if self.status not in {'unknown', 'declared', 'verified', 'not_applicable'}:
            raise ValueError('Unknown passport status')
        if not isinstance(self.note, str):
            raise ValueError('note must be text')
        if not isinstance(self.evidence, (list, tuple)) or any(
            not isinstance(item, str) or not item.strip() for item in self.evidence
        ):
            raise ValueError('evidence must be a list of nonempty citations')
        object.__setattr__(self, 'evidence', tuple(self.evidence))
        object.__setattr__(self, 'expected', _json_copy(self.expected))
        object.__setattr__(self, 'observed', _json_copy(self.observed))
        if self.status == 'verified':
            if self.observed is None or not self.evidence:
                raise ValueError('Verified entries require observations and evidence')
        elif self.observed is not None:
            raise ValueError('Only verified entries may contain observations')
        if self.status == 'declared' and self.expected is None:
            raise ValueError('Declared entries require an expected value')
        if self.status in {'unknown', 'not_applicable'} and self.expected is not None:
            raise ValueError('Unknown/not-applicable entries cannot assert a value')
        if self.status == 'not_applicable' and not self.note.strip():
            raise ValueError('Not-applicable entries require a reason')


@dataclass
class QuantizedModelPassport:
    experiment_id: str
    scope: str
    sections: dict = field(init=False)

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in (self.experiment_id, self.scope)):
            raise ValueError('experiment_id and scope must be nonempty text')
        self.sections = {section: {name: PassportEntry() for name in names}
                         for section, names in FIELDS.items()}

    def record(self, section, name, **values):
        if section not in FIELDS or name not in FIELDS[section]:
            raise ValueError(f'Unknown passport field: {section}.{name}')
        self.sections[section][name] = PassportEntry(**values)

    def to_dict(self):
        # Revalidate on export: callers can inspect the ledger, but mutating it
        # directly must not silently bypass the evidence contract.
        if set(self.sections) != set(FIELDS):
            raise ValueError('Passport sections do not match schema 1.0')
        result = {}
        for section, names in FIELDS.items():
            if set(self.sections[section]) != set(names):
                raise ValueError(f'Passport fields do not match schema: {section}')
            result[section] = {}
            for name, entry in self.sections[section].items():
                if not isinstance(entry, PassportEntry):
                    raise ValueError('Passport fields must contain PassportEntry objects')
                result[section][name] = asdict(PassportEntry(**asdict(entry)))
        return _json_copy(dict(schema_version='1.0', experiment_id=self.experiment_id,
                               scope=self.scope, sections=result))

    @classmethod
    def from_dict(cls, value):
        try:
            if set(value) != {'schema_version', 'experiment_id', 'scope', 'sections'} or value['schema_version'] != '1.0':
                raise ValueError('Invalid passport schema')
            passport = cls(value['experiment_id'], value['scope'])
            if set(value['sections']) != set(FIELDS):
                raise ValueError('Invalid passport sections')
            for section, names in FIELDS.items():
                entries = value['sections'][section]
                if set(entries) != set(names):
                    raise ValueError(f'Invalid passport fields: {section}')
                for name, entry in entries.items():
                    if set(entry) != {'expected', 'observed', 'status', 'evidence', 'note'}:
                        raise ValueError('Invalid passport entry')
                    passport.record(section, name, **entry)
            passport.to_dict()
            return passport
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError('Invalid passport structure') from error

    def status_counts(self):
        counts = dict.fromkeys(('unknown', 'declared', 'verified', 'not_applicable'), 0)
        for fields in self.to_dict()['sections'].values():
            for entry in fields.values():
                counts[entry['status']] += 1
        return counts
