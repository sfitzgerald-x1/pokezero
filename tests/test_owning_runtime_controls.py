"""Synthetic protocol/deadline controls; no model, game or external process."""
import asyncio
from copy import deepcopy
from pathlib import Path
import time
import sys
import unittest
from unittest.mock import patch

import test_live_protocol_bridge as fixtures
import test_owning_encoder_study as studies
from pokezero.foulplay_bridge import ControlledFoulPlayConfig
from pokezero.live_protocol_actor import LiveProtocolActor
from pokezero import owning_encoder_study as study


class RuntimeControlsTests(unittest.TestCase):
    def fixture(self):
        f = fixtures.OwningBridgeTests(methodName='test_both_seats_actual_owner_only_exact_observation_choice_and_terminal')
        f.setUp(); self.addCleanup(f.doCleanups)
        return f

    def test_explicit_limit_and_single_line_dispatch_preserve_both_owners(self):
        f = self.fixture()
        original = LiveProtocolActor.accept_owner_stream
        for owner in ('p1','p2'):
            prefixes=[]; choices=[]; observed=[]
            for single in (False,True):
                groups=[]
                def accepted(actor, **kwargs):
                    groups.append(kwargs['lines']); return original(actor, **kwargs)
                config=f.config(owner=owner,owning_decision_timeout_seconds=.5,
                    owning_public_single_line=single)
                events=f.events(config,owner_lines=['|turn|1',fixtures.request(owner)])
                events[-1]['lines'].append('|')
                with patch.object(LiveProtocolActor,'accept_owner_stream',new=accepted):
                    result,memory,_=asyncio.run(f.game(config,events))
                self.assertEqual(all(len(group)==1 for group in groups[:-1]),single)
                self.assertEqual(groups[-1],('|win|'+config.pokezero_username,'|'))
                self.assertEqual(result.pokezero_submitted_choice_players,(owner,))
                prefixes.append(result.encoder_provenance['actor_receipt']['prefix_sha256'])
                choices.append(memory.sent[1]['choices'][owner])
                observed.append(list(f.backends[-1].lines))
                step=f.trajectories[-1].steps[0]
                self.assertEqual(step.metadata['owning_decision_timeout_seconds'],.5)
                self.assertEqual(step.metadata['policy_elapsed_scope'],
                    'owning-snapshot-encode-selection-and-choice')
            self.assertEqual(prefixes[0],prefixes[1])
            self.assertEqual(choices[0],choices[1])
            self.assertEqual(observed[0],observed[1])

    def test_canonical_tie_terminal_padding_preserves_literals_and_closure(self):
        f=self.fixture()
        for owner in ('p1','p2'):
            config=f.config(owner=owner,owning_decision_timeout_seconds=.5,
                owning_public_single_line=True)
            events=f.events(config)
            for event in events:
                if event.get('type')=='stream':
                    event['lines']=['|tie' if line.startswith('|win|') else line
                                    for line in event['lines']]
            events[-1]['lines']+=['|','|']
            result,_,_=asyncio.run(f.game(config,events))
            self.assertTrue(result.tied)
            self.assertEqual(f.backends[-1].lines[-3:],['|tie','|','|'])
            self.assertEqual(result.encoder_provenance['actor_receipt']['status'],'TERMINAL_UNQUALIFIED')

    def test_encode_timeout_submits_no_late_choice_or_opponent_work(self):
        f=self.fixture(); original=fixtures.FixturePythonBackend.encode
        def slow(backend,*args,**kwargs):
            time.sleep(.08); return original(backend,*args,**kwargs)
        for owner in ('p1','p2'):
            config=f.config(owner=owner,owning_decision_timeout_seconds=.002)
            memory=fixtures.MemoryBridge(f.events(config)); count=len(f.trajectories)
            with patch.object(fixtures.FixturePythonBackend,'encode',new=slow):
                with self.assertRaises(TimeoutError):
                    asyncio.run(f.game(config,memory.events,memory=memory))
            self.assertFalse(any(x.get('type')=='choices' for x in memory.sent))
            self.assertEqual(len(f.trajectories),count)
        f.fp.assert_not_awaited()

    def test_delayed_timeout_callback_cannot_publish_late_result(self):
        f=self.fixture(); original=fixtures.FixturePythonBackend.encode
        def holds_gil(backend,*args,**kwargs):
            until=time.perf_counter()+.04
            while time.perf_counter()<until: pass
            return original(backend,*args,**kwargs)
        previous=sys.getswitchinterval()
        try:
            sys.setswitchinterval(.1)
            for owner in ('p1','p2'):
                config=f.config(owner=owner,owning_decision_timeout_seconds=.002)
                memory=fixtures.MemoryBridge(f.events(config))
                count=len(f.trajectories)
                with patch.object(fixtures.FixturePythonBackend,'encode',new=holds_gil):
                    with self.assertRaises(TimeoutError):
                        asyncio.run(f.game(config,memory.events,memory=memory))
                self.assertFalse(any(x.get('type')=='choices' for x in memory.sent))
                self.assertEqual(len(f.trajectories),count)
            f.fp.assert_not_awaited()
        finally:sys.setswitchinterval(previous)

    def test_choice_decoding_is_inside_deadline(self):
        f=self.fixture();original=LiveProtocolActor.choice
        def slow(actor,snapshot):
            time.sleep(.08); return original(actor,snapshot)
        config=f.config(owning_decision_timeout_seconds=.002)
        memory=fixtures.MemoryBridge(f.events(config))
        with patch.object(LiveProtocolActor,'choice',new=slow):
            with self.assertRaises(TimeoutError):
                asyncio.run(f.game(config,memory.events,memory=memory))
        self.assertFalse(any(x.get('type')=='choices' for x in memory.sent))
        self.assertEqual(f.trajectories,[])
        f.fp.assert_not_awaited()

    def test_configuration_rejects_invalid_or_legacy_owning_controls(self):
        f=self.fixture()
        for limit in (True,0,-1,float('nan'),float('inf'),'60',3601):
            with self.assertRaises(ValueError): f.config(owning_decision_timeout_seconds=limit)
        for flag in (None,0,1,'line'):
            with self.assertRaises(ValueError): f.config(owning_public_single_line=flag)
        for args in ({'owning_public_single_line':True},{'owning_decision_timeout_seconds':1}):
            with self.assertRaises(ValueError):
                ControlledFoulPlayConfig(checkpoint=None,showdown_root=Path('/authored'),**args)
        self.assertIsNone(f.config().owning_decision_timeout_seconds)

    def test_study_requires_prospective_limit_and_binds_actual_dispatch(self):
        f=studies.StudyTests(methodName='test_fixed_full_roster_nonoverlap_and_counterbalanced_order')
        f.setUp();self.addCleanup(f.doCleanups)
        fields={k:f.spec[k] for k in ('study_id','runtime','admission')}
        for limit in (None,True,0,float('inf')):
            with self.assertRaises(ValueError):study.make_prespec(**fields,owning_decision_timeout_seconds=limit)
        config=study._config(f.spec,0)
        self.assertEqual(config.owning_decision_timeout_seconds,.25)
        self.assertTrue(config.owning_public_single_line)
        drift=deepcopy(f.spec);drift['controls']['owning_decision_timeout_seconds']=.5
        with self.assertRaisesRegex(ValueError,'drift'):study.validate_prespec(drift)
        self.assertTrue(study.SPEC_SCHEMA.endswith('.v2'))
        bad=studies.game(f.spec,0,0)
        bad['encoder_provenance']['owning_public_single_line']=False
        with self.assertRaises(ValueError):study.checked_game(bad,f.spec,0,0)
        bad=studies.game(f.spec,0,0)
        bad['encoder_provenance']['owning_decision_timeout_seconds']=None
        with self.assertRaises(ValueError):study.checked_game(bad,f.spec,0,0)
