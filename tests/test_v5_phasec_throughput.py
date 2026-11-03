"""C-T denominator, missing cells and actual GPU trace classification controls."""
import copy
import unittest

from pokezero import v5_phasec_throughput as ct, v5_training as training


def result(wall):
    return dict(declared_games=20000,status='COMPLETE_DECLARATION',
        jax_reload='VERIFIED_ORIGINAL_TRAINED_ARCHIVE_LOADED',steps=20100,
        original_run_wall_seconds=wall,collection_runtime={'kernel_blocked_seconds':33.0},
        original_phase_wall_seconds={'TRAIN_CACHES_INCLUSIVE':10.0},
        dispositions={'COMPLETE':19980,'CAPPED':20,'FAILED':0,'REFUSED':0,'RUNNING':0,'NOT_ATTEMPTED':0})


class PhaseCThroughputTests(unittest.TestCase):
    def test_all_collector_write_phases_survive_without_becoming_a_new_denominator(self):
        banks=[{'phases':{name:float(i+1)for name in ct.COLLECTOR_PHASES}}for i in range(10)]
        actual=ct.collector_phase_totals(banks)
        self.assertEqual(actual['banks'],10)
        for key in ('spool_write_wall_seconds','game_cache_wall_seconds','cache_merge_wall_seconds'):
            self.assertEqual(actual['wall_seconds'][key],55.0)
        with self.assertRaises(ValueError):ct.collector_phase_totals(banks[:-1])
        broken=copy.deepcopy(banks);del broken[4]['phases']['game_cache_wall_seconds']
        with self.assertRaises(ValueError):ct.collector_phase_totals(broken)
        broken=copy.deepcopy(banks);broken[0]['phases']['game_cache_wall_seconds']=float('nan')
        with self.assertRaises(ValueError):ct.collector_phase_totals(broken)

    def test_full_iteration_wall_is_the_denominator_and_unknowns_are_retained(self):
        for wall,status in ((1000,'TARGET20_MET'),(4000,'FLOOR5_MET_TARGET20_MISSED'),(4001,'BELOW_BINDING5')):
            original=result(wall);saved=copy.deepcopy(original)
            report=ct.gate_report(original,{'updates':20100})
            self.assertEqual(report['threshold'],status)
            self.assertEqual(report['declared_games_per_second'],20000/wall)
            self.assertEqual(report['natural_complete_per_second'],19980/wall)
            self.assertEqual(report['dispositions']['CAPPED'],20)
            self.assertEqual(original,saved)

    def test_partial_iterations_or_bad_wall_never_receive_an_admission(self):
        for change in ({'declared_games':19999},{'status':'STOPPED'}, {'steps':0},
                {'jax_reload':'UNAVAILABLE'},*({'original_run_wall_seconds':v}for v in (0,-1,True,float('nan'),float('inf')))):
            with self.subTest(change=change),self.assertRaises(ValueError):
                ct.gate_report(dict(result(1000),**change),{'updates':20100})
        with self.assertRaises(ValueError):ct.gate_report(result(1000),{'updates':20099})

    def test_only_actual_kernel_and_memory_categories_count_not_host_annotations(self):
        report=ct.gpu_trace_categories({'traceEvents':[
            {'ph':'X','cat':'kernel','dur':30},
            {'ph':'X','cat':'kernel','dur':50},
            {'ph':'X','cat':'gpu_memcpy','dur':7},
            {'ph':'X','cat':'gpu_memset','dur':3},
            {'ph':'X','cat':'user_annotation','dur':9000},
            {'ph':'X','cat':'cpu_op','dur':10000},
            {'ph':'i','cat':'kernel','dur':2000}]})
        self.assertEqual(report['kernel']['events'],2)
        self.assertAlmostEqual(report['kernel']['seconds'],80/1e6)
        self.assertAlmostEqual(report['gpu_memcpy']['seconds'],7/1e6)
        for duration in (True,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                ct.gpu_trace_categories({'traceEvents':[{'ph':'X','cat':'kernel','dur':duration}]})

    def test_disabled_execution_and_failed_instrumentation_restore_original_calls(self):
        with self.assertRaisesRegex(ValueError,'explicit finite'):
            ct.execute({},'/tmp/inert',resident_cache_bytes=1)
        original=training.train_caches;update=training._update
        with self.assertRaisesRegex(RuntimeError,'inert failure'):
            with ct.instrument_training('/tmp/inert',1024):
                self.assertIsNot(training.train_caches,original)
                raise RuntimeError('inert failure')
        self.assertIs(training.train_caches,original)
        self.assertIs(training._update,update)
        for budget in (True,0,-1):
            with self.assertRaises(ValueError):
                with ct.instrument_training('/tmp/inert',budget):pass


if __name__=='__main__':unittest.main()
