import unittest
from gpu_engine.phase_d_guard_diagnostic import Checks,payload_checks,summarize


class GuardDiagnosticTests(unittest.TestCase):
    def test_stacked_guards_are_not_first_failure(self):
        request=dict(side=dict(id='p1',pokemon=[dict(condition='1/2 slp',active=True)]+[{}]*5),active=[dict(trapped=True)])
        payload=dict(selfActiveRequestState=dict(trapped=True),pendingBatonPassSides=['p1'],sides=dict(p2=dict(
            volatiles=['solarbeam'],pokemon=[dict(condition='12/100 slp')],materializationBlockers=['toxic-stage-unknown'])))
        c=payload_checks(request,payload,['|switch|p1a: x|X|1/2','|switch|p2a: y|Y|100/100'],owner=0)
        kinds={r['kind']for r in c.rows if r['status']=='TRIGGERED'}
        self.assertTrue({'native-probed-hard-trapped-own-request','deferred-baton-phase','native-probed-or-hidden-volatile-clock'}<=kinds)
        self.assertFalse({'OWN_SLEEP_LATENT_CLOCK_UNAVAILABLE','native-probed-sleep-clock'}&kinds)
        # Item discharge is decided only with positive B2 facts in shared protocol checks.
        from gpu_engine.phase_d_protocol_world import preflight_checks
        rows=preflight_checks(payload,{})
        self.assertTrue(any(r['kind']=='public-materialization-blocker' and r['status']=='TRIGGERED' for r in rows))
        # An Encore alone no longer trips the hidden-clock check: its count is drawn from the public posterior.
        encored=dict(payload,sides=dict(p2=dict(volatiles=['encore'],pokemon=[dict(condition='12/100')])))
        clock=[r for r in preflight_checks(encored,{}) if r['kind']=='native-probed-or-hidden-volatile-clock']
        self.assertEqual([r['status'] for r in clock],['CLEAR'])
        c=payload_checks(request,encored,['|switch|p1a: x|X|1/2','|switch|p2a: y|Y|100/100'],owner=0)
        self.assertNotIn('native-probed-or-hidden-volatile-clock',{r['kind']for r in c.rows if r['status']=='TRIGGERED'})

    def test_missing_prerequisite_is_not_clear(self):
        c=Checks();c.call('x',lambda: {}['missing']);self.assertEqual(c.rows[0]['status'],'NOT_EVALUABLE')

    def test_typed_refusal_remains_triggered(self):
        class Refusal(ValueError):kind='x'
        def fail():raise Refusal('x')
        c=Checks();c.call('x',fail);self.assertEqual(c.rows[0]['status'],'TRIGGERED')

    def test_root_counts_distinct_from_occurrences(self):
        d=dict(checks=[dict(kind='x',status='TRIGGERED')]*2,all_refusals=['x'],not_evaluable=['y'])
        result=summarize([dict(band=0,diagnostic=d),dict(band=1,diagnostic=d)])
        self.assertEqual(result['trigger_occurrences'],{'x':4})
        self.assertEqual(result['roots_affected_by_band'],{'0':{'x':1},'1':{'x':1}})
        self.assertFalse(result['coverage_claim'])
