"""Sealed operand selectors and no-draw guards, without a simulated battle."""
from collections import namedtuple
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from gpu_engine import fresh200_native as native

Chance = namedtuple('Chance', native.CHANCE_FIELDS)


@dataclass(frozen=True)
class Config:
    bridge_path: Path


class SealedOperandTests(unittest.TestCase):
    def setUp(self):
        self.chance = Chance(np.zeros((1, 2, 12), np.int32), np.full((1, 2), 3, np.int32),
            np.full((1, 2), 4, np.int32), np.full((1, 2), 5, np.int32),
            np.array([[1, 0]], np.int32), np.array([1], np.int32), np.zeros((1, 2, 5), np.int32))
        self.state = SimpleNamespace(pending_residual=np.array([False]),
            residual_draws=np.ones((1, 2, 12), np.int32), phazing_draws=np.full((1, 2), 6, np.int32),
            secondary_draws=np.full((1, 2), 7, np.int32), sleep_talk_draws=np.full((1, 2), 8, np.int32),
            effect_draws=np.ones((1, 2, 5), np.int32))

    def packet(self, chance=None, row=0):
        return native.step_packet(self.state, self.chance if chance is None else chance, row=row, stream=51, request=8)

    def test_fresh_packet_preserves_all_seven_saved_fields(self):
        packet = self.packet()
        self.assertEqual(packet['schema'], 'pokezero.gpu_chance_tape.v6')
        self.assertEqual(packet['nonce'], '51:8');self.assertFalse(packet['resuming'])
        for name in native.CHANCE_FIELDS:self.assertEqual(packet[name], np.asarray(getattr(self.chance, name))[0].tolist())

    def test_suspended_turn_reads_saved_operands_and_fresh_entry_ties(self):
        self.state.pending_residual[0] = True
        packet = self.packet();self.assertTrue(packet['resuming'])
        for name, saved in (('draws', 'residual_draws'), ('phazing', 'phazing_draws'),
                ('secondary', 'secondary_draws'), ('sleep_talk', 'sleep_talk_draws'), ('effect', 'effect_draws')):
            self.assertEqual(packet[name], getattr(self.state, saved)[0].tolist())
        self.assertEqual(packet['entry_ties'], [1, 0]);self.assertEqual(packet['initial_tie'], 1)

    def test_missing_out_of_bounds_float_and_batch_aliases_fail(self):
        for chance in (self.chance._replace(effect=None), self.chance._replace(phazing=np.full((1, 2), 60)),
                self.chance._replace(draws=self.chance.draws.astype(np.float32)),
                self.chance._replace(entry_ties=np.zeros((2, 2), np.int32))):
            with self.subTest(chance=repr(chance)[:70]), self.assertRaises(ValueError):self.packet(chance)
        for row in (-1, 1, True):
            with self.subTest(row=row), self.assertRaises(ValueError):self.packet(row=row)
        self.state.pending_residual[0] = True;self.state.effect_draws = None
        with self.assertRaises(ValueError):self.packet()

    def test_initial_packet_requires_literal_gender_and_unsplit_tie_rows(self):
        raw = SimpleNamespace(gender=np.array([[[0, 1, 2, 0, 1, 2]]*2], np.int32))
        packet = native.initial_packet(raw, np.array([1], np.int32), row=0)
        self.assertEqual(packet, dict(genders=raw.gender[0].tolist(), initial_tie=1, allow_inert_gender=False))
        for tie, row in ((np.array([2]), 0), (np.array([1]), -1), (np.array([1, 1]), 0)):
            with self.subTest(tie=tie, row=row), self.assertRaises(ValueError):native.initial_packet(raw, tie, row=row)
        raw.gender[0, 0, 0] = -1
        with self.assertRaises(ValueError):native.initial_packet(raw, np.array([1]), row=0)

    def test_unknown_initial_or_raw_rng_audit_is_fatal(self):
        for kind in ('UNRESOLVED', 'INITIAL_RANDOM', 'INERT_GENDER', 'NEW_UNREVIEWED_ROLE'):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'unresolved'):
                native.require_resolved([dict(kind=kind)])
        events = [dict(kind='COUPLED_INITIAL', raw=1), dict(kind='COUPLED', role='accuracy', raw=99)]
        self.assertIs(native.require_resolved(events), events)

    def test_control_files_require_initial_audit_and_never_overwrite_a_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'owned'
            packet = dict(genders=[[0]*6]*2, initial_tie=0, allow_inert_gender=False)
            control = native.SealedChanceControl(root, Config(Path('original.mjs')), packet)
            self.assertIn('fresh200_bridge.mjs', control.config.bridge_path.read_text())
            with self.assertRaisesRegex(ValueError, 'initialization audit'):
                control.begin(self.state, self.chance, row=0, stream=51, request=8)
            self.assertEqual(control.initialized(), [])
            value = control.begin(self.state, self.chance, row=0, stream=51, request=8)
            self.assertEqual(json.loads(control.control.read_text()), value)
            with self.assertRaises(FileExistsError):control.begin(self.state, self.chance, row=0, stream=51, request=8)
            control.audit.write_text(json.dumps(dict(kind='UNRESOLVED', role='RAW'))+'\n')
            events = control.end();self.assertFalse(control.control.exists())
            with self.assertRaisesRegex(ValueError, 'unresolved'):native.require_resolved(events)


SCRIPT = r'''
const assert=require('node:assert/strict'),hooks=require(process.argv[1]);
function fixture(){
 const counts={raw:0,next:0,seed:0,team:0,setSeed:0},events=[];
 class PRNG {
  constructor(seed){this.setSeed(seed);}
  setSeed(seed){counts.setSeed++;this.rng={next(){counts.next++;return 0;}};return this;}
  random(){counts.raw++;return 0;}
  static generateSeed(){counts.seed++;return [1,2,3,4];}
 }
 class Battle {
  setPlayer(slot,options){return this.getTeam(options);}
  getTeam(options){counts.team++;return options.team;}
 }
 return {PRNG,Battle,counts,events};
}
const f=fixture();
hooks.install(f.Battle,f.PRNG,()=>null,e=>f.events.push(e),()=>({allow_inert_gender:false}),'',{strictNoRng:true});
const p=new f.PRNG([1,2,3,4]);
assert.throws(()=>p.random(2),/UNMAPPED_RAW_RNG/);
assert.throws(()=>p.rng.next(),/UNMAPPED_RNG_NEXT/);
assert.throws(()=>f.PRNG.generateSeed(),/UNSEALED_SEED/);
assert.throws(()=>p.setSeed(undefined),/UNSEALED_SEED/);
const b=new f.Battle();
for(const input of [undefined,{}, {team:[]},{team:'random'}])
 assert.throws(()=>b.getTeam(input),/EXPLICIT_TEAM_REQUIRED/);
const literal=Array.from({length:6},()=>({species:'literal'}));assert.equal(b.getTeam({team:literal}),literal);
assert.deepEqual(f.counts,{raw:0,next:0,seed:0,team:1,setSeed:1});
const legacy=fixture();
hooks.install(legacy.Battle,legacy.PRNG,()=>null,e=>legacy.events.push(e),()=>({allow_inert_gender:false}));
const old=new legacy.PRNG([1,2,3,4]);assert.equal(old.random(2),0);old.rng.next();legacy.PRNG.generateSeed();
assert.deepEqual(legacy.counts,{raw:1,next:1,seed:1,team:0,setSeed:1});
assert.equal(legacy.events[0].kind,'INITIAL_RANDOM');
process.stdout.write(JSON.stringify({strict:f.counts,legacy:legacy.counts}));
'''


@unittest.skipUnless(shutil.which('node'), 'requires node')
class StrictNoDrawTests(unittest.TestCase):
    def test_strict_unknown_paths_never_call_raw_next_seed_or_team_generator(self):
        path = Path(__file__).resolve().parents[1]/'gpu_engine/chance_hooks.cjs'
        result = json.loads(subprocess.check_output([shutil.which('node'), '-e', SCRIPT, str(path)], text=True, timeout=30))
        self.assertEqual(result['strict']['raw'], 0);self.assertEqual(result['strict']['next'], 0)
        self.assertEqual(result['strict']['seed'], 0)


if __name__ == '__main__':unittest.main()
