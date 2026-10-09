"""Particle telemetry must retain genuine cancellation proof, not fabricate it."""
import copy
import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

from pokezero.mcts_eval.paper_reference import SamplingDeadlineExceeded, ReferenceRefusal
from pokezero.mcts_eval.paper_reference_particles import bootstrap_population, history_failure_diagnostic


class DeadlineEvidenceTests(unittest.TestCase):
    def certificate(self):
        return dict(schema='pokezero.world-sampling-deadline.v1',deadline_at=12.,checked_at=12.5,
            accepted_world=False,backed_up=False)

    def error(self):
        error=SamplingDeadlineExceeded('original cooperative deadline')
        error.sampling_diagnostic=self.certificate()
        return error

    def test_original_exception_clock_and_zero_backup_survive_partial_cleanup(self):
        error=self.error();created=[];released=[];receipt={};calls=0
        def check():
            nonlocal calls
            calls+=1
            if calls==3:
                raise error
        def initial(index):
            owner=object();created.append(owner);return owner
        with self.assertRaises(SamplingDeadlineExceeded) as caught:
            bootstrap_population(count=3,stages=1,initial=initial,advance=lambda p,s:None,
                release=released.append,rng=random.Random(1),check=check,receipt=receipt)
        self.assertIs(caught.exception,error)
        for key,value in self.certificate().items():
            self.assertEqual(error.sampling_diagnostic[key],value)
        self.assertEqual(released,created)
        self.assertFalse(error.sampling_diagnostic['bootstrap_population']['complete'])
        self.assertFalse(error.sampling_diagnostic['bootstrap_population']['partial_stage_used'])

    def test_nested_population_receipts_keep_original_clock_without_aliasing(self):
        error=self.error();original=copy.deepcopy(error.sampling_diagnostic)
        error.sampling_diagnostic=history_failure_diagnostic(error,{'scope':'prior'})
        outer=history_failure_diagnostic(error,{'scope':'current'})
        self.assertEqual({k:outer[k] for k in original},original)
        self.assertEqual(outer['bootstrap_population'],{'scope':'current'})
        self.assertEqual(outer['cause_sampling_diagnostic']['bootstrap_population'],{'scope':'prior'})
        outer['cause_sampling_diagnostic']['bootstrap_population']['scope']='changed-copy'
        self.assertEqual(error.sampling_diagnostic['bootstrap_population'],{'scope':'prior'})

    def test_missing_inconsistent_or_nonfinite_clock_is_not_synthesized(self):
        for bad in (None,dict(self.certificate(),checked_at=11.),
                    dict(self.certificate(),checked_at=float('nan')),
                    dict(self.certificate(),accepted_world=True),
                    dict(self.certificate(),backed_up=True)):
            error=self.error();error.sampling_diagnostic=bad
            diagnostic=history_failure_diagnostic(error,{'complete':False})
            self.assertEqual(diagnostic['schema'],'pokezero.bootstrap-history-conditioning-failure.v1')
            self.assertNotIn('checked_at',diagnostic)

    def test_non_deadline_failure_cannot_be_promoted_to_cancellation(self):
        error=ReferenceRefusal('native state refused');error.sampling_diagnostic=self.certificate()
        self.assertEqual(history_failure_diagnostic(error,{})['schema'],
            'pokezero.bootstrap-history-conditioning-failure.v1')

    def test_declared_chance_and_population_wrappers_preserve_the_exact_clock(self):
        error=self.error();original=copy.deepcopy(error.sampling_diagnostic)
        for schema in ('pokezero.fixed-chance-pool-failure.v1',
                       'pokezero.bootstrap-history-conditioning-failure.v1'):
            error.sampling_diagnostic=dict(schema=schema,
                cause_sampling_diagnostic=error.sampling_diagnostic,scope='retained')
        repaired=history_failure_diagnostic(error,{'complete':False})
        self.assertEqual({key:repaired[key] for key in original},original)
        self.assertEqual(repaired['cause_sampling_diagnostic'],error.sampling_diagnostic)

    def test_unknown_and_over_depth_wrappers_do_not_fabricate_cancel_certificates(self):
        error=self.error()
        error.sampling_diagnostic=dict(schema='unknown-private-wrapper',
            cause_sampling_diagnostic=error.sampling_diagnostic)
        self.assertEqual(history_failure_diagnostic(error,{})['schema'],
            'pokezero.bootstrap-history-conditioning-failure.v1')
        error=self.error()
        for _ in range(33):
            error.sampling_diagnostic=dict(schema='pokezero.fixed-chance-pool-failure.v1',
                cause_sampling_diagnostic=error.sampling_diagnostic)
        self.assertEqual(history_failure_diagnostic(error,{})['schema'],
            'pokezero.bootstrap-history-conditioning-failure.v1')

    def test_existing_whole_game_validator_accepts_proved_cancellation_only(self):
        path=Path(__file__).resolve().parents[1]/'scripts/wider_search_comparison.py'
        spec=importlib.util.spec_from_file_location('wider_deadline_validator',path)
        wider=importlib.util.module_from_spec(spec);spec.loader.exec_module(wider)
        diagnostic=history_failure_diagnostic(self.error(),{'complete':False})
        draw=dict(status='DEADLINE_CANCELLED',sampling_diagnostic=diagnostic)
        batch=SimpleNamespace(world_draws=2,trajectories=1,transitions=1,deadline_exhausted=True)
        measured=SimpleNamespace(worker_pids=tuple(range(20)),
            result=SimpleNamespace(trajectories=1,world_draws=2,transitions=1),
            worker_receipts=[dict(batch=batch,evidence=dict(draws=[
                dict(status='ROOT_VALIDATED',released=True),draw]))])
        wider.validate_reference_measurement(measured)
        draw['sampling_diagnostic']=history_failure_diagnostic(SamplingDeadlineExceeded('unproved'),{})
        with self.assertRaisesRegex(RuntimeError,'clock/zero-backup'):
            wider.validate_reference_measurement(measured)


if __name__=='__main__':
    unittest.main()
