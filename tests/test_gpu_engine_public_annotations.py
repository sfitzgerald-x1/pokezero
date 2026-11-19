"""Canonical main annotations retain every shared V4 event consumer."""
import importlib.util
import unittest

SETUP=('|player|p1|One|1','|player|p2|Two|2|1000','|teamsize|p1|6','|teamsize|p2|6',
       '|gametype|singles','|gen|3','|tier|[Gen 3] Random Battle','|rule|Species Clause: Limit one of each species','|start')
ANNOTATIONS=('|-miss|p1a: Nick|p2a: Absol','|-crit|p2a: Absol','|-supereffective|p2a: Absol','|-resisted|p2a: Absol',
             '|-hitcount|p2a: Absol|1','|-hitcount|p2a: Absol|5','|-hint|Sleep Clause Mod activated.',
             '|-message|A public announcement.','|-fieldactivate|move: Perish Song',
             '|-activate|p2a: Absol|move: Heal Bell','|-singleturn|p1a: Nick|move: Focus Punch')


class AnnotationLexicalTests(unittest.TestCase):
    def test_literal_gen3_custom_game_setup_messages(self):
        from gpu_engine.public_annotations import classify

        tier = '|tier|[Gen 3] Custom Game'
        rules = ('|raw|<div class="infobox"><details class="readmore" open><summary><strong>'
                 '1 custom rule:</strong></summary> sleepclausemod</details></div>')
        self.assertEqual(classify(tier), 'setup')
        self.assertEqual(classify(rules), 'setup')
        for line in (tier + '|extra', '|tier|[Gen 3] Doubles Custom Game', tier.lower(),
                     rules + '|extra', rules.replace('sleepclausemod', 'other'), '|raw|<b>arbitrary</b>'):
            with self.subTest(line=line):
                self.assertIsNone(classify(line))

    def test_frozen_roots30_and32_owner_diagnostics(self):
        from gpu_engine.public_annotations import classify

        for line in ('|debug|Success chance: 50%', '|debug|Sunny Day water suppress'):
            with self.subTest(line=line):
                self.assertEqual(classify(line), 'text')
                self.assertIsNone(classify(line + '|extra'))
        for line in ('|debug|Success chance: 25%', '|debug|Success chance: 050%',
                     '|debug|Sunny Day fire boost', '|debug|Sunny Day water suppress '):
            with self.subTest(line=line):
                self.assertIsNone(classify(line))

    def test_frozen_roots29_and43_dispatch_diagnostics(self):
        from gpu_engine.public_annotations import classify

        for line in ('|debug|before turn callback: focuspunch', '|debug|BP: 40'):
            with self.subTest(line=line):
                self.assertEqual(classify(line), 'text')
                self.assertIsNone(classify(line + '|extra'))
        for line in ('|debug|before turn callback: Focus Punch', '|debug|before turn callback: shelltrap',
                     '|debug|BP: 040', '|debug|BP: 41'):
            with self.subTest(line=line):
                self.assertIsNone(classify(line))

    def test_literal_gen3_damage_debug_messages(self):
        from gpu_engine.public_annotations import classify

        for line in ('|debug|BP: 70', '|debug|Negating (sp)def boost/penalty.'):
            self.assertEqual(classify(line), 'text')
            self.assertIsNone(classify(line + '|extra'))
        for line in ('|debug|BP: 070', '|debug|BP: 71', '|debug|Negating (sp)atk boost/penalty.',
                     '|debug|unrecognized diagnostic'):
            with self.subTest(line=line):
                self.assertIsNone(classify(line))

    def test_frozen_root2_diagnostics_admit_only_exact_annotation_text(self):
        from gpu_engine.public_annotations import classify

        lines = ('|debug|move failed because it did nothing', '|debug|Shield Dust prevent secondary')
        for line in lines:
            with self.subTest(line=line):
                self.assertEqual(classify(line), 'text')
                self.assertIsNone(classify(line + '|p2a: Dustox'))
                self.assertIsNone(classify(line + '\n'))
        self.assertIsNone(classify('|debug|Shield Dust prevent primary'))
        self.assertIsNone(classify('|debug|move failed because it did something'))

    def test_exact_immunity_and_secondary_chance_diagnostics(self):
        from gpu_engine.public_annotations import classify

        diagnostics = ('|debug|natural status immunity', '|debug|weather immunity',
                       '|debug|immune to status', '|debug|doubling secondary chance')
        for line in diagnostics:
            with self.subTest(line=line):
                self.assertEqual(classify(line), 'text')
                self.assertIsNone(classify(line + '|p2a: Vileplume'))
                self.assertIsNone(classify(line + ' 2'))
        self.assertIsNone(classify('|debug|tripling secondary chance'))

    def test_frozen_root29_reversal_diagnostic(self):
        from gpu_engine.public_annotations import classify

        self.assertEqual(classify('|debug|BP: 20'), 'text')
        for line in ('|debug|BP: 020', '|debug|BP: 21', '|debug|BP: 20|extra', '|debug|BP: 42'):
            with self.subTest(line=line):
                self.assertIsNone(classify(line))

    def test_frozen_roots55_and57_damage_and_sleep_diagnostics(self):
        from gpu_engine.public_annotations import classify

        for line in ('|debug|damage event failed', '|debug|set status [slp] interrupted'):
            with self.subTest(line=line):
                self.assertEqual(classify(line), 'text')
                self.assertIsNone(classify(line + '|extra'))
        for line in ('|debug|damage event succeeded', '|debug|set status [tox] interrupted',
                     '|debug|set status [slp] succeeded', '|debug|set status [slp] interrupted '):
            with self.subTest(line=line):
                self.assertIsNone(classify(line))

    def test_witnessed_rain_and_interrupted_toxic_diagnostics(self):
        from gpu_engine.public_annotations import classify

        for line in ('|debug|rain water boost', '|debug|set status [tox] interrupted'):
            with self.subTest(line=line):
                self.assertEqual(classify(line), 'text')
                self.assertIsNone(classify(line + '|extra'))
        for line in ('|debug|rain water boosted', '|debug|set status [psn] interrupted',
                     '|debug|set status [tox] succeeded'):
            with self.subTest(line=line):
                self.assertIsNone(classify(line))

    def test_literal_mirror_coat_before_turn_diagnostic(self):
        from gpu_engine.public_annotations import classify

        line = '|debug|before turn callback: mirrorcoat'
        self.assertEqual(classify(line), 'text')
        for altered in (line + '|extra', line + ' ', line.replace('mirrorcoat', 'counter')):
            with self.subTest(line=altered):
                self.assertIsNone(classify(altered))

    def test_literal_rain_fire_suppression_diagnostic(self):
        from gpu_engine.public_annotations import classify

        line = '|debug|rain fire suppress'
        self.assertEqual(classify(line), 'text')
        for altered in (line + '|extra', line + ' ', line.replace('fire', 'electric')):
            with self.subTest(line=altered):
                self.assertIsNone(classify(altered))

    def test_literal_artificial_status_immunity_diagnostic(self):
        from gpu_engine.public_annotations import classify

        line = '|debug|artificial status immunity'
        self.assertEqual(classify(line), 'text')
        for altered in (line + '|extra', line + ' ', line.replace('status', 'move')):
            with self.subTest(line=altered):
                self.assertIsNone(classify(altered))

    def test_literal_compound_eyes_accuracy_diagnostic(self):
        from gpu_engine.public_annotations import classify

        line = '|debug|compoundeyes - enhancing accuracy'
        self.assertEqual(classify(line), 'text')
        for altered in (line + '|extra', line + ' ', line.replace('compoundeyes', 'hustle')):
            with self.subTest(line=altered):
                self.assertIsNone(classify(altered))

    def test_literal_wonder_guard_return_diagnostic(self):
        from gpu_engine.public_annotations import classify

        line = '|debug|Wonder Guard immunity: return'
        self.assertEqual(classify(line), 'text')
        for altered in (line + '|extra', line + ' ', line.replace('return', 'frustration')):
            with self.subTest(line=altered):
                self.assertIsNone(classify(altered))

    def test_literal_wonder_guard_dragon_claw_diagnostic(self):
        from gpu_engine.public_annotations import classify

        line = '|debug|Wonder Guard immunity: dragonclaw'
        self.assertEqual(classify(line), 'text')
        for altered in (line + '|extra', line + ' ', line.replace('dragonclaw', 'dragonbreath')):
            with self.subTest(line=altered):
                self.assertIsNone(classify(altered))

    def test_bounded_canonical_shapes_and_refusals(self):
        from gpu_engine.public_annotations import classify
        for line in (*SETUP,*ANNOTATIONS):self.assertIsNotNone(classify(line),line)
        self.assertEqual(classify('|-activate|p2a: Gengar|move: Destiny Bond'),'fainted_target')
        for line in ('|gen|4','|gametype|doubles','|tier|gen9randombattle','|start|extra',
                '|teamsize|p1|7','|teamsize|p3|6','|player|p3|Three','|player|p1|',
                '|-miss','|-miss|p1a: Nick','|-miss|p1: Nick|p2a: Absol',
                '|-miss|p1a: Nick|p2: Absol','|-miss|p1a: Nick|p2a: Absol|extra',
                '|-miss|p1a: Nick|p2a: Absol|[from] ability: Pressure',
                '|-crit','|-crit|p2: Absol','|-crit|p2a: Absol|extra','|-hitcount|p2a: Absol|0',
                '|-hitcount|p2a: Absol|6','|-hitcount|p2a: Absol|01','|-hint|','|-message|x|extra',
                '|-fieldactivate|move: Mud Sport','|-fieldactivate|move: Perish Song|extra',
                '|-activate|p2: Absol|move: Heal Bell','|-activate|p2a: Absol|move: Heal Bell|extra',
                '|-activate|p2a: Absol|move: Heal Bell|[from] ability: Pressure',
                '|-activate|p2: Gengar|move: Destiny Bond','|-activate|p2a: Gengar|Destiny Bond',
                '|-activate|p2a: Gengar|move: Destiny Bond|extra',
                '|-activate|p2a: Gengar|move: Destiny Bond|[of] p1a: Grumpig',
                '|-activate|p2a: Gengar|move: Destiny Bond|[from] ability: Pressure',
                '|-activate|p2a: Absol|move: Unknown','|-singleturn|p1a: Nick|move: Unknown',
                '|-singleturn|p1: Nick|move: Focus Punch','|-singleturn|p1a: Nick|move: Focus Punch|extra',
                '|-hint|[from] ability: Pressure|[of] p2a: Absol','|request|{}','|UNSUPPORTED|x'):
            self.assertIsNone(classify(line),line)
        for line in ('|-crit|p2a: Absol\n','|-hint|two\rlines','|gen|3 '):self.assertIsNone(classify(line),line)


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class AnnotationStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_gpu_engine_public_switch_context import SwitchContextTests
        SwitchContextTests.setUpClass();cls.f=SwitchContextTests.f;cls.j,cls.jax=cls.f.j,cls.f.jax

    def test_setup_and_annotations_match_all173_source_columns_and_retain_names(self):
        from gpu_engine import public_wire,public_inputs
        lines=[*SETUP,*self.f.public.prefix,'|turn|1','|move|p1a: Nick|Surf|p2a: Absol',
            *ANNOTATIONS,'|-damage|p2a: Absol|80/100']
        host,projections=public_wire.project_lines(public_wire.WireState(),lines,**self.f.e.wire_kwargs(self.f.catalog))
        self.assertEqual(host.player_names,('One','Two'))
        self.assertEqual(len(projections),len(lines))
        for projection in projections:
            self.assertFalse(projection.full_line_consumers_complete)
            self.assertEqual([event.kind for event in projection.events[:2]],
                [public_inputs.Kind.RAW_BEGIN,public_inputs.Kind.SWITCH_PRELUDE])
            self.assertEqual(sum(event.kind==public_inputs.Kind.NONPROC for event in projection.events),1)
            self.assertEqual(sum(event.kind==public_inputs.Kind.MUDSHOT for event in projection.events),1)
        self.f.compare_source(lines)

    def test_each_annotation_is_a_real_switch_boundary_and_closes_trick(self):
        from gpu_engine import public_wire,public_belief_snapshot
        from pokezero import showdown
        from pokezero.belief import PublicBattleBeliefEngine
        before=[self.f.public.prefix[1],self.f.public.prefix[0]]
        base=self.f.ingest(before)
        step=self.f.shared()[0]
        for line in (*ANNOTATIONS,*SETUP):
            engine=PublicBattleBeliefEngine(format_id='gen3randombattle',set_source=self.f.public.source)
            calls=[];original=engine._with_set_summary
            def record(pokemon):
                result=original(pokemon);calls.append(result.key);return result
            engine._with_set_summary=record
            for raw in before:engine.ingest_event(showdown._public_event_from_line(raw))
            self.assertTrue(engine._pending_switches,line)
            projections=self.f.projections([*before,line]);state=base
            for chunk in self.f.chunks(projections[-1:]):
                state,error=step(state,chunk);self.assertEqual(error.tolist(),[0],line)
            engine.ingest_event(showdown._public_event_from_line(line))
            self.assertFalse(engine._pending_switches,line);self.assertEqual(state.context.switches.side.tolist(),[-1])
            summary,ok=public_belief_snapshot.read(state.context.refresh.controller.visible,state.beliefs,self.f.catalog.source.device)
            self.assertTrue(bool(ok.all()))
            for side in (0,1):
                for slot,pokemon in enumerate(engine.snapshot().side(f'p{side+1}')):
                    self.assertEqual(int(summary.count[0,1-side,slot]),pokemon.candidate_set_count,line)
                    self.assertEqual(int(state.context.refresh.controller.visible.generation[0,1-side,slot]),calls.count(pokemon.key),line)
        trick=[*self.f.public.prefix,'|-activate|p1a: Nick|move: Trick|[of] p2a: Absol']
        for line in ANNOTATIONS:
            state=self.f.ingest([*trick,line]);self.assertEqual(state.context.items.pending.tolist(),[False],line)
            engine=PublicBattleBeliefEngine.from_events([showdown._public_event_from_line(raw) for raw in (*trick,line)],
                format_id='gen3randombattle',set_source=self.f.public.source)
            self.assertIsNone(engine._pending_trick,line)

    def test_unsupported_or_wrong_target_does_not_advance_host(self):
        from gpu_engine import public_wire
        host,_=public_wire.project_lines(public_wire.WireState(),self.f.public.prefix,**self.f.e.wire_kwargs(self.f.catalog))
        for line in ('|-miss|p1a: Unknown|p2a: Absol','|-miss|p1a: Nick|p2a: Unknown',
                '|-miss|p1a: Nick|p2a: Absol|extra','|-crit|p2a: Unknown','|-hitcount|p2: Absol|2','|-fieldactivate|move: Mud Sport',
                '|gen|4','|gametype|doubles','|request|{}'):
            with self.assertRaises(ValueError):public_wire.project_line(host,line,**self.f.e.wire_kwargs(self.f.catalog))
        self.assertEqual(host.player_names,(None,None))

    def test_destiny_bond_activation_requires_current_fainted_identity_and_retains_raw_consumers(self):
        from gpu_engine import public_wire,public_inputs
        prefix=[*self.f.public.prefix,'|turn|1','|-singlemove|p2a: Absol|Destiny Bond',
            '|move|p1a: Nick|Surf|p2a: Absol','|-damage|p2a: Absol|0 fnt','|faint|p2a: Absol']
        line='|-activate|p2a: Absol|move: Destiny Bond'
        host,_=public_wire.project_lines(public_wire.WireState(),prefix,**self.f.e.wire_kwargs(self.f.catalog))
        next_host,projection=public_wire.project_line(host,line,**self.f.e.wire_kwargs(self.f.catalog))
        self.assertEqual(next_host.encounters,host.encounters)
        self.assertEqual(next_host.live,host.live)
        self.assertEqual(next_host.active,host.active)
        self.assertGreater(next_host.next_sequence,host.next_sequence)
        self.assertEqual([event.kind for event in projection.events[:2]],
            [public_inputs.Kind.RAW_BEGIN,public_inputs.Kind.SWITCH_PRELUDE])
        self.assertEqual(sum(event.kind==public_inputs.Kind.NONPROC for event in projection.events),1)
        self.assertEqual(sum(event.kind==public_inputs.Kind.MUDSHOT for event in projection.events),1)
        self.assertFalse(any(event.kind in (public_inputs.Kind.FAINT,public_inputs.Kind.ACTIVE)
            for event in projection.events))
        self.assertFalse(projection.full_line_consumers_complete)
        # The activation cannot substitute for either recorded faint. Continue
        # through the independent attacker's faint and actual replacement lines.
        lines=[*prefix,line,'|faint|p1a: Nick',
            '|switch|p1a: Gardevoir|Gardevoir, L80|100/100',
            '|switch|p2a: Gardevoir|Gardevoir, L80|100/100']
        self.f.compare_source(lines)
        live,_=public_wire.project_lines(public_wire.WireState(),self.f.public.prefix,
            **self.f.e.wire_kwargs(self.f.catalog))
        for initial,bad in ((live,line),(host,'|-activate|p1a: Nick|move: Destiny Bond'),
            (host,'|-activate|p2a: Unknown|move: Destiny Bond'),
            (host,line+'|extra'),(host,line+'|[of] p1a: Nick'),
            (host,'|-activate|p2a: Absol|move: Grudge')):
            with self.subTest(line=bad),self.assertRaises(ValueError):
                public_wire.project_line(initial,bad,**self.f.e.wire_kwargs(self.f.catalog))
        # A known benched encounter is not this side's fainted active.
        bench=[self.f.public.prefix[0],
            '|switch|p2a: Gardevoir|Gardevoir, L80|100/100',self.f.public.prefix[1],'|faint|p2a: Absol']
        host,_=public_wire.project_lines(public_wire.WireState(),bench,**self.f.e.wire_kwargs(self.f.catalog))
        with self.assertRaisesRegex(ValueError,'exact registered fainted active'):
            public_wire.project_line(host,'|-activate|p2a: Gardevoir|move: Destiny Bond',
                **self.f.e.wire_kwargs(self.f.catalog))
