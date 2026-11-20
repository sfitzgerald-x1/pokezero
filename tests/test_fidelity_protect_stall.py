"""The tied post-action expiry group from the Breloom/Umbreon witness."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from _showdown_root import requires_showdown, showdown_root
from pokezero.env import BattleStartOverride
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('fidelity_panel', ROOT / 'scripts/fidelity_panel.py')
panel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(panel)
NODE = shutil.which('node')
PACK_TEAMS = """
const {Teams}=require(process.argv[1]+'/dist/sim/teams.js');
const mon=(species,ability,level,spe,moves)=>({species,ability,level,moves,
  gender:'F', nature:'Serious',item:'Leftovers',
  ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},
  evs:{hp:85,atk:85,def:85,spa:85,spd:85,spe}});
console.log(JSON.stringify({
  p1:Teams.pack([mon('Breloom','Effect Spore',84,80,['Focus Punch','Spore'])]),
  p2:Teams.pack([mon('Umbreon','Synchronize',88,85,['Protect','Toxic'])]),
}));
"""


@requires_showdown()
@unittest.skipUnless(NODE, 'requires node')
class ProtectStallProofTests(unittest.TestCase):
    def test_both_native_orders_preserve_stall_and_public_carry(self):
        import tempfile

        team = json.loads(subprocess.check_output([NODE, '-e', PACK_TEAMS, str(showdown_root())], text=True))
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            bridge = folder / 'bridge.mjs'
            bridge.write_text(f'await import({json.dumps((ROOT / "gpu_engine/fidelity_bridge.mjs").as_uri())});\n')
            control, audit = folder / 'chance.json', folder / 'chance-audit.jsonl'
            config = LocalShowdownConfig(showdown_root=showdown_root(), node_binary=NODE, bridge_path=bridge)
            env = LocalShowdownEnv(config)
            try:
                env.reset_with_start_override(seed=51087, start_override=BattleStartOverride(
                    player_teams=team, format_id='gen3customgame', observation_format_id='gen3customgame',
                ))
                before = env.snapshot()
                speed = [side['pokemon'][0]['storedStats']['spe']
                         for side in before.bridge_snapshot['battle']['sides']]
                self.assertEqual(speed, [165, 165])
                tape = dict(schema='pokezero.gpu_chance_tape.v6', nonce='51087:witness',
                            draws=[[0] * 12 for _ in range(2)], secondary=[99, 99], sleep_talk=[0, 0],
                            effect=[[0] * 5 for _ in range(2)], phazing=[0, 0], entry_ties=[0, 0],
                            resuming=False, protect_stall_order=0)
                actions = {'p1': 0, 'p2': 0}
                events, failure = panel.reference_step(env, actions, tape, control, audit)
                self.assertIsNone(failure, events)
                order, = [e for e in events if e['kind'] == 'PROTECT_STALL_ORDER']
                second, proof = panel.verify_protect_stall_orders(env, before, actions, tape, control, audit, order)
                self.assertIsNotNone(second, proof)
                self.assertEqual(proof['orders'][1], proof['orders'][0][::-1])
                self.assertEqual(proof['stall'], [[[0, 0], [2, 1]], [[0, 0], [2, 1]]])
                self.assertTrue(proof['state_equal'], proof)
                self.assertTrue(proof['public_carry_equal'], proof)
                self.assertTrue(any(line.endswith('|Protect') and line.startswith('|-activate|')
                                    for line in second.protocol_lines))
            finally:
                env.close()
