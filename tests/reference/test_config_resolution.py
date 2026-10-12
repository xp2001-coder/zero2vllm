import unittest
from scripts.experiments.lesson0305 import experiments
from zero2vllm.reference.config_resolution import engine_choice, normalized, resolve_v1, verify_scheduler


class ConfigurationTests(unittest.TestCase):
    def test_explicit_cli_over_file_and_defaults(self):
        r=resolve_v1({'max-num-seqs':64},{'max_num_seqs':32})
        self.assertEqual(r['values']['max_num_seqs'],32)
        self.assertEqual(r['origins']['max_num_seqs'],'cli')

    def test_false_is_not_missing(self):
        self.assertFalse(resolve_v1({'enable_prefix_caching':True},{'enable_prefix_caching':False})['values']['enable_prefix_caching'])
        self.assertTrue(resolve_v1({'enable_prefix_caching':None})['values']['enable_prefix_caching'])

    def test_engine_policy_overrides_false_and_records_why(self):
        r=resolve_v1(cli_options={'enable_chunked_prefill':False})
        events=[e for e in r['trace'] if e['field']=='enable_chunked_prefill']
        self.assertEqual([e['after'] for e in events],[None,False,True])
        self.assertIn('mandatory',events[-1]['phase'])

    def test_memory_threshold_is_inclusive_and_usage_specific(self):
        for mem,server,offline,seqs in [(69.9,2048,8192,256),(70,8192,16384,1024),(80,8192,16384,1024)]:
            for use,budget in [('server',server),('offline',offline)]:
                r=resolve_v1(memory_gib=mem,usage=use)['values']
                self.assertEqual((r['max_num_batched_tokens'],r['max_num_seqs']),(budget,seqs))

    def test_explicit_budget_not_replaced_by_hardware_default(self):
        r=resolve_v1(cli_options={'max_num_batched_tokens':4096},memory_gib=80)
        self.assertEqual(r['values']['max_num_batched_tokens'],4096)

    def test_trace_and_input_are_not_shared(self):
        source={'max_num_seqs':64};r=resolve_v1(source);r['values']['max_num_seqs']=9
        self.assertEqual(source,{'max_num_seqs':64});self.assertEqual(resolve_v1(source)['values']['max_num_seqs'],64)

    def test_invalid_types_unknowns_and_alias_collisions(self):
        for d in [{'unknown':1},{'max_num_seqs':True},{'max_num_seqs':0},{'enable_prefix_caching':'false'},{'max-num-seqs':2,'max_num_seqs':3}]:
            with self.assertRaises(ValueError):normalized(d)

    def test_model_limit_guard(self):
        self.assertEqual(resolve_v1(cli_options={'max_model_len':1024})['values']['max_model_len'],1024)
        with self.assertRaises(ValueError):resolve_v1(cli_options={'max_model_len':8192})

    def test_capacities_must_be_consistent_after_defaults(self):
        with self.assertRaises(ValueError):resolve_v1(cli_options={'max_num_batched_tokens':128})
        self.assertEqual(resolve_v1(cli_options={'max_num_batched_tokens':128,'max_num_seqs':64})['values']['max_num_seqs'],64)

    def test_chunking_changes_budget_constraint(self):
        verify_scheduler(512,16,4096,True)
        with self.assertRaises(ValueError):verify_scheduler(512,16,4096,False)

    def test_explicit_v1_unsupported_is_error_not_silent_fallback(self):
        with self.assertRaises(ValueError):engine_choice('1','unsupported')
        self.assertEqual(engine_choice(None,'unsupported')['engine'],'V0')

    def test_experimental_needs_explicit_opt_in(self):
        self.assertEqual(engine_choice(None,'experimental')['engine'],'V0')
        self.assertEqual(engine_choice('1','experimental')['engine'],'V1')

    def test_explicit_zero_disables_v1(self):
        for s in ['supported','experimental','unsupported']:self.assertEqual(engine_choice('0',s)['engine'],'V0')

    def test_environment_domain_is_explicit(self):
        for env in ['',True,'true','2']:
            with self.assertRaises(ValueError):engine_choice(env,'supported')

    def test_eager_default_does_not_prove_graph_running(self):
        self.assertEqual(resolve_v1(cli_options={'enforce_eager':True})['graph_intent'],'disabled')
        self.assertIn('not_proven_active',resolve_v1()['graph_intent'])

    def test_platform_and_backend_checks_are_not_faked(self):
        with self.assertRaises(ValueError):resolve_v1(platform='maca')
        r=resolve_v1(cli_options={'block_size':7})
        self.assertEqual(r['values']['block_size'],7);self.assertIn('backend dimensions/dtype',r['runtime_checks_pending'])

    def test_synthetic_hardware_input_is_checked(self):
        for v in [True,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):resolve_v1(memory_gib=v)
        self.assertEqual(resolve_v1(memory_gib=0)['values']['max_num_batched_tokens'],2048)

    def test_all_cases_pass(self):
        r=experiments();self.assertEqual(len(r),12);self.assertTrue(all(c['passed'] for c in r))


if __name__ == '__main__': unittest.main()
