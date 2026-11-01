"""Exact original public warning, no simulator/game/model/checkpoint execution."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import unittest

from gpu_engine.public_annotations import TURN_500_WARNING, classify


class Turn500LexicalTests(unittest.TestCase):
    def test_exact_pinned_source_literal_hash_and_closed_error_shapes(self):
        # Same source string digest as the failed cell, not a replay/retry of it.
        self.assertEqual(hashlib.sha256(TURN_500_WARNING.encode()).hexdigest(),
            '93b11525588ee0d97d4a64e077e55e4a252a88f66c0f3a56d84867dea0aa612f')
        import os
        root=Path(os.environ.get('POKEZERO_SHOWDOWN_ROOT',
            '/Users/scott/workspace/agents/pokemon-showdown-timing-smoke-f76228a'))
        self.assertEqual(subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
            'f76228a1354b5d0f307ca2d16101294ad3a2308b')
        source=(root/'sim/battle.ts').read_text()
        body=source.split('if (this.turn <= 100) return;',1)[1].split("if (!this.ruleTable.has('endlessbattleclause'))",1)[0]
        self.assertIn('(this.turn >= 500 && this.turn % 100 === 0)',body)
        self.assertIn('const turnsLeft = 1000 - this.turn;',body)
        self.assertIn('const turnsLeftText = (turnsLeft === 1 ? `1 turn` : `${turnsLeft} turns`);',body)
        self.assertIn("this.add('bigerror', `You will auto-tie if the battle doesn't end in ${turnsLeftText} (on turn 1000).`);",body)
        self.assertEqual(classify(TURN_500_WARNING),'text')
        for line in ('|bigerror|', '|bigerror|arbitrary fatal error',
            "|bigerror|Warning: One player isn't adhering to a 510 EV limit, and the other player is.",
            TURN_500_WARNING.replace('500 turns','100 turns'),
            TURN_500_WARNING.replace('1000','999'),TURN_500_WARNING+'|extra',
            TURN_500_WARNING+'|[from] ability: Pressure',TURN_500_WARNING+' ',
            TURN_500_WARNING+'\n',TURN_500_WARNING+'\r',' '+TURN_500_WARNING,
            TURN_500_WARNING.replace('bigerror','error')):
            with self.subTest(line=line):self.assertIsNone(classify(line))


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class Turn500ConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_gpu_engine_public_switch_context import SwitchContextTests
        SwitchContextTests.setUpClass()
        cls.f=SwitchContextTests.f

    def test_complete_original_raw_consumers_continuation_and_late_refusal_atomic(self):
        import numpy as np
        from gpu_engine import public_inputs, public_wire
        from pokezero.recorded_encoder_runner import JaxRecordedAdapter
        f=self.f
        # Original clock consumer requires every +1 turn; this is a fixed
        # synthetic literal protocol prefix, not simulator/game execution.
        lines=(*f.public.prefix,*(f'|turn|{turn}'for turn in range(1,500)))
        host,_=public_wire.project_lines(public_wire.WireState(),lines,**f.e.wire_kwargs(f.catalog))
        after,packet=public_wire.project_line(host,TURN_500_WARNING,**f.e.wire_kwargs(f.catalog))
        self.assertEqual(packet.raw_line,TURN_500_WARNING)
        self.assertEqual([event.kind for event in packet.events], [public_inputs.Kind.RAW_BEGIN,
            public_inputs.Kind.SWITCH_PRELUDE,public_inputs.Kind.MUDSHOT,public_inputs.Kind.TOXIC,
            public_inputs.Kind.NONPROC,public_inputs.Kind.TENDENCY])
        self.assertEqual(after.encounters,host.encounters)
        self.assertEqual(after.live,host.live)
        self.assertEqual(after.active,host.active)
        self.assertEqual((after.turn,after.phase),(host.turn,host.phase))
        self.assertEqual(after.next_sequence,host.next_sequence+6)
        self.assertFalse(packet.full_line_consumers_complete)
        # All173 actual synthetic source observation columns, both owning seats.
        # No neural model/forward or actual battle is invoked by this fixture.
        f.compare_source([*lines,TURN_500_WARNING,'|turn|500'])
        # Original B3 JAX feed path, preserving full controls and source line.
        adapter=JaxRecordedAdapter(f.catalog)
        adapter.reset(stream_id=1,battle_id='synthetic-turn500-control')
        adapter.feed((*lines,TURN_500_WARNING,'|turn|500'))
        self.assertEqual(adapter.host.turn,500)
        self.assertEqual(adapter.host.next_sequence,after.next_sequence+len(
            public_wire.project_line(after,'|turn|500',**f.e.wire_kwargs(f.catalog))[1].events))
        before,old_host=adapter.state,adapter.host
        for bad in ('|bigerror|arbitrary fatal error',TURN_500_WARNING.replace('500 turns','100 turns')):
            with self.subTest(line=bad),self.assertRaisesRegex(ValueError,'wire refused kind=bigerror'):
                adapter.feed(('|',bad))
            self.assertIs(adapter.state,before)
            self.assertEqual(adapter.host,old_host)
        self.assertTrue(all(np.isfinite(np.asarray(x)).all()for x in f.jax.tree.leaves(adapter.state)))


if __name__=='__main__':unittest.main()
