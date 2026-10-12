import json
import unittest

from zero2vllm.quantization import PassportEntry, QuantizedModelPassport
from scripts.experiments.lesson0901 import target_passport


class PassportTests(unittest.TestCase):
    def test_model_name_does_not_infer_algorithm_or_kernel(self):
        p=target_passport()
        for section,name in [('provenance','algorithm'),('execution','actual_kernel'),('execution','checkpoint_format'),('execution','accumulator_dtype'),('numerical','group_axis')]:
            self.assertEqual(p.sections[section][name].status,'unknown')
        self.assertEqual(p.sections['numerical']['group_size'].expected,128)
        self.assertIsNone(p.sections['numerical']['group_size'].observed)

    def test_verified_without_observation_or_citation_is_rejected(self):
        for values in ({'observed':'Marlin'}, {'evidence':['startup.log']}, {}):
            with self.subTest(values=values),self.assertRaises(ValueError):
                PassportEntry(status='verified',**values)

    def test_expected_is_not_observed_and_disagreement_is_preserved(self):
        p=QuantizedModelPassport('fallback','illustrative record; not a measured GPU run')
        p.record('execution','actual_kernel',expected='preferred',observed='fallback',status='verified',evidence=['example.log#L1'])
        entry=p.to_dict()['sections']['execution']['actual_kernel']
        self.assertEqual((entry['expected'],entry['observed']),('preferred','fallback'))

    def test_false_and_zero_are_valid_observations(self):
        for value in (False,0):
            entry=PassportEntry(observed=value,status='verified',evidence=['report.json'])
            self.assertEqual(entry.observed,value)

    def test_unknown_cannot_assert_values(self):
        for status in ('unknown','not_applicable'):
            with self.assertRaises(ValueError):
                PassportEntry(status=status,expected='A100',note='test')
        with self.assertRaises(ValueError):
            PassportEntry(status='declared',expected='A100',observed='A100')

    def test_not_applicable_requires_reason(self):
        with self.assertRaises(ValueError):
            PassportEntry(status='not_applicable')
        self.assertEqual(PassportEntry(status='not_applicable',note='No KV in a single Linear').status,'not_applicable')

    def test_json_roundtrip_preserves_all_fields(self):
        p=target_passport()
        data=json.loads(json.dumps(p.to_dict()))
        self.assertEqual(QuantizedModelPassport.from_dict(data).to_dict(),data)

    def test_import_rejects_unknown_missing_and_bad_schema(self):
        original=target_passport().to_dict()
        for action in ('version','missing','extra','bad_entry'):
            data=json.loads(json.dumps(original))
            if action=='version': data['schema_version']='99'
            if action=='missing': del data['sections']['execution']['actual_kernel']
            if action=='extra': data['sections']['execution']['kernal']={}
            if action=='bad_entry': data['sections']['execution']['actual_kernel']['surprise']=True
            with self.subTest(action=action),self.assertRaises(ValueError):
                QuantizedModelPassport.from_dict(data)

    def test_mutable_inputs_and_exports_are_detached(self):
        value={'shape':[3,2]}
        p=QuantizedModelPassport('detached','test')
        p.record('execution','qweight_shape',status='verified',observed=value,evidence=['report'])
        value['shape'][0]=100
        data=p.to_dict()
        data['sections']['execution']['qweight_shape']['observed']['shape'][0]=200
        self.assertEqual(p.to_dict()['sections']['execution']['qweight_shape']['observed'],{'shape':[3,2]})

    def test_invalid_json_and_citations_are_rejected(self):
        for value in (float('nan'),float('inf'),object()):
            with self.assertRaises(ValueError): PassportEntry(expected=value,status='declared')
        for evidence in ('report', [''],[3]):
            with self.assertRaises(ValueError): PassportEntry(observed='x',status='verified',evidence=evidence)

    def test_typo_and_direct_schema_mutation_are_rejected(self):
        p=QuantizedModelPassport('shape','test')
        with self.assertRaises(ValueError): p.record('execution','kernal',status='declared',expected='x')
        p.sections['execution'].pop('actual_kernel')
        with self.assertRaises(ValueError): p.to_dict()

    def test_invalid_identity_and_input_structure(self):
        for identity in ('',None,3):
            with self.assertRaises(ValueError): QuantizedModelPassport(identity,'test')
        for data in (None,[],{'schema_version':'1.0'}):
            with self.assertRaises(ValueError): QuantizedModelPassport.from_dict(data)


if __name__=='__main__': unittest.main()
