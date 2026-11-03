"""Authored inert controls only: no model, simulator, dataset or games."""
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from pokezero import phase_d_owning_capture as capture
from pokezero import phase_d_saved_roots as saved
from pokezero.live_protocol_actor import ActorContract, LiveProtocolActor, OwnedProvider, RawPolicy, SHAPES, _request_targets
from pokezero.policy import PolicyDecision
from pokezero.showdown import _legal_action_mask
import numpy as np


def request():
    return '|request|' + json.dumps(dict(rqid=1, side=dict(id='p1', pokemon=[dict(
        ident=f'p1: Mon{i}', details='Snorlax, L80', condition='120/240', active=i==0,
        moves=['bodyslam'], stats=dict(atk=100,def_=100,spa=100,spd=100,spe=100)) for i in range(2)]),
        active=[dict(moves=[dict(id='bodyslam',pp=24,maxpp=24)])]))


class Backend(OwnedProvider):
    def ingest(self, lines):
        if '|bad|' in lines:raise ValueError('fixture refusal')
    def encode(self, snapshot):
        packet=json.loads(snapshot.own_request_line[9:])
        arrays={k:np.zeros(s,np.float32 if k=='numeric_features' else bool if k.endswith('mask') else np.int32) for k,s in SHAPES.items()}
        arrays['attention_mask'][:]=True
        arrays['legal_action_mask'][:]=_legal_action_mask(packet)
        return arrays,_request_targets(packet)


def actor():
    return LiveProtocolActor(ActorContract.champion(),backend_factory=lambda c,**k:Backend(c),
        policy_factory=lambda c:RawPolicy(c,lambda obs,*,rng:PolicyDecision(0,'fixture'),lambda:None))


def feed(a, lines):
    return a.accept_owner_stream(battle_id=a.battle_id, owner=a.owner, generation=a.generation,
        prefix_sha256=a.prefix_sha256, lines=tuple(lines))


class CaptureTests(unittest.TestCase):
    def test_opt_in_noop(self):
        a=actor();self.assertIs(capture.capture_actor(a),a);self.assertFalse(hasattr(a,'_phase_d_capture'))
    def test_literal_wire_and_request_action_rng_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            with capture.capture_to(root,cell={'ordinal':0},declaration_sha256='a'*64):a=capture.capture_actor(actor())
            a.begin_game(battle_id='authored',owner='p1',seed=1,stream_id=1)
            lines=('|turn|1',request());feed(a,lines)
            rng=random.Random(7);before=rng.getstate();snapshot=a.snapshot_boundary()
            result=a.decide(snapshot,rng=rng)
            self.assertEqual(result.action_index,0);self.assertEqual(a.choice(snapshot),'move 1');self.assertEqual(rng.getstate(),before)
            a.close_game();roots=list(capture.read_capture(Path(root)/'0.jsonl'))
            self.assertEqual(len(roots),1);self.assertEqual(roots[0]['lines'],lines);self.assertEqual(roots[0]['turn'],1)
    def test_exclusive_path_and_mutation_refusal(self):
        with tempfile.TemporaryDirectory() as root:
            with capture.capture_to(root,cell={'ordinal':0},declaration_sha256='a'*64):
                a=capture.capture_actor(actor())
                with self.assertRaises(FileExistsError):capture.capture_actor(actor())
            a.begin_game(battle_id='authored',owner='p1',seed=1,stream_id=1);feed(a,('|turn|1',request()))
            a.decide(a.snapshot_boundary(),rng=random.Random(1));a.close_game()
            p=Path(root)/'0.jsonl';blob=p.read_bytes();p.write_bytes(blob.replace(b'|turn|1',b'|turn|2'))
            with self.assertRaisesRegex(ValueError,'prefix'):list(capture.read_capture(p))
    def test_refused_wire_is_not_recorded(self):
        with tempfile.TemporaryDirectory() as root:
            with capture.capture_to(root,cell={'ordinal':0},declaration_sha256='a'*64):a=capture.capture_actor(actor())
            a.begin_game(battle_id='authored',owner='p1',seed=1,stream_id=1)
            with self.assertRaises(ValueError):feed(a,('|bad|',))
            a.close_game();rows=[json.loads(x) for x in (Path(root)/'0.jsonl').read_bytes().splitlines()]
            self.assertEqual([r['event'] for r in rows],['BEGIN','CLOSE'])
    def test_interrupted_battle_keeps_durable_roots(self):
        with tempfile.TemporaryDirectory() as root:
            with capture.capture_to(root,cell={'ordinal':0},declaration_sha256='a'*64):a=capture.capture_actor(actor())
            a.begin_game(battle_id='authored',owner='p1',seed=1,stream_id=1);feed(a,('|turn|1',request()))
            a.decide(a.snapshot_boundary(),rng=random.Random(1))
            self.assertEqual(len(list(capture.read_capture(Path(root)/'0.jsonl'))),1)
            a.close_game()


class SelectorTests(unittest.TestCase):
    def rows(self):
        return [dict(cell={'ordinal':i},owner='p1',decision_index=k,turn=turn,prefix_sha256=f'{i*10+k:064x}')
                for i in range(60) for k,turn in enumerate((1,4,10,20))]
    def test_selection_turn_balanced_order_independent_outcome_blind(self):
        rows=self.rows();d=saved.declare();a=saved.select(rows,d)
        for r in rows:r.update(outcome='LOST',constructor_status='REFUSED')
        self.assertEqual(a,saved.select(list(reversed(rows)),d));self.assertEqual(len(a),200)
        self.assertEqual([sum(saved.band(r['turn'])==k for r in a) for k in range(4)],[50]*4)
    def test_no_refill_and_missing_turn_supply_blocks(self):
        with self.assertRaisesRegex(ValueError,'insufficient'):saved.select(self.rows()[:196],saved.declare())
    def test_missing_results_stay_full_denominator(self):
        roster=saved.select(self.rows(),saved.declare());records=[dict(address=saved.address(r),requested_worlds=16,
            constructed_worlds=16,decision_seconds=1.,refusals=()) for r in roster[:180]]
        report=saved.summarize(roster,records,worlds=16)
        self.assertEqual(report['full_constructor_coverage'],.9);self.assertFalse(report['admitted'])
        self.assertIsNone(report['decision_p95_seconds']);self.assertEqual(report['refusals_by_root']['UNAVAILABLE_RESULT'],20)
    def test_full_requested_worlds_required(self):
        roster=saved.select(self.rows(),saved.declare());records=[dict(address=saved.address(r),requested_worlds=32,
            constructed_worlds=31,decision_seconds=None,refusals=('latent-clock',)) for r in roster]
        report=saved.summarize(roster,records,worlds=32)
        self.assertEqual(report['fully_constructed_roots'],0);self.assertEqual(report['refusals_by_root'],{'latent-clock':200})


if __name__=='__main__':unittest.main()
