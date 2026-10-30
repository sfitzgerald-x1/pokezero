"""Existing live-type consumers also run inside the full source-refresh frame."""
from dataclasses import replace
import importlib.util
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class FormechangeStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_v4_encoder as fixtures
        from gpu_engine import public_belief
        from pokezero.category_vocab import CategoryVocabulary
        from pokezero.dex import ShowdownDex, SpeciesInfo
        from pokezero.randbat import Gen3RandbatSpeciesUniverse, Gen3RandbatVariant

        class Fixture(fixtures.V4EncoderTests):
            def request(self, owner=0):
                packet = super().request(owner)
                if owner == 1:
                    packet['side']['pokemon'][1].update(ident='p2: Castform',
                        details='Castform, L80', ability='Forecast')
                return packet

        Fixture.setUpClass()
        raw = Fixture.public.source
        variant = Gen3RandbatVariant('Castform', 'synthetic', 'Castform', 'synthetic',
            80, ('Body Slam', 'Surf'), 'Forecast', 'Leftovers')
        raw = replace(raw, universes=dict(raw.universes,
            castform=Gen3RandbatSpeciesUniverse('Castform', 80, (variant,))))
        stats = dict(hp=70, atk=70, **{'def':70}, spa=70, spd=70, spe=70)
        species = dict(Fixture.inputdex.species)
        species['castform'] = SpeciesInfo('castform', 'Castform', ('Normal',), stats, 0.8)
        # Sunny exercises the real dex path; Rainy/Snowy use source fallbacks.
        species['castformsunny'] = SpeciesInfo('castformsunny', 'Castform-Sunny', ('Fire',), stats, 0.8)
        dex = ShowdownDex(Fixture.inputdex.moves, species, Fixture.inputdex.type_chart)
        labels = set(Fixture.vocab.tokens) | {'species:castform', 'ability:forecast',
            'belief:possible_ability:forecast'}
        Fixture.vocab = CategoryVocabulary(tuple(sorted(labels)))
        Fixture.source = public_belief.compile_source(raw)
        Fixture.inputdex = dex
        Fixture.catalog = Fixture.e.compile_encoder(Fixture.source, dex, Fixture.vocab,
            checkpoint_sha256='a'*64, feature_masks=Fixture.masks)
        Fixture.public = SimpleNamespace(source=raw, prefix=(
            '|switch|p1a: Snorlax|Snorlax, L80|100/100',
            '|switch|p2a: Castform|Castform, L80|100/100'))
        cls.f = Fixture()

    def test_forecast_real_shape_and_reversion_match_all173_source_columns(self):
        from gpu_engine import public_wire, public_inputs
        lines = [*self.f.public.prefix, '|turn|1', '|-weather|SunnyDay',
            '|-formechange|p2a: Castform|Castform-Sunny|[msg]|[from] ability: Forecast']
        host, projections = public_wire.project_lines(public_wire.WireState(), lines,
            **self.f.e.wire_kwargs(self.f.catalog))
        frame = projections[-1]
        self.assertEqual(frame.events[0].kind, public_inputs.Kind.RAW_BEGIN)
        self.assertEqual(sum(e.kind == public_inputs.Kind.TYPES for e in frame.events), 1)
        self.assertEqual(sum(e.kind == public_inputs.Kind.ABILITY for e in frame.events), 1)
        self.assertEqual(sum(e.kind == public_inputs.Kind.SWITCH_PRELUDE for e in frame.events), 1)
        self.assertEqual(sum(e.kind == public_inputs.Kind.NONPROC for e in frame.events), 1)
        self.assertEqual(sum(e.kind == public_inputs.Kind.MUDSHOT for e in frame.events), 1)
        self.assertFalse(frame.full_line_consumers_complete)
        self.assertEqual(host.encounters[1][0].species, 'Castform')
        self.f.compare_source(lines)
        for extra in (
            ['|-weather|RainDance', '|-formechange|p2a: Castform|Castform-Rainy|[from] ability: Forecast'],
            ['|-weather|Hail', '|-formechange|p2a: Castform|Castform-Snowy|[from] ability: Forecast'],
            ['|-weather|none', '|-formechange|p2a: Castform|Castform|[msg]|[from] ability: Forecast'],
        ):
            self.f.compare_source(lines + extra)

    def test_side_attribution_and_namespace_refusals_remain_source_exact(self):
        import numpy as np
        from gpu_engine import public_wire
        base = [*self.f.public.prefix, '|turn|1']
        # Existing Python semantics are by public side, including stale names.
        self.f.compare_source(base + ['|-formechange|p2a: Earlier Name|Castform-Sunny'])
        self.f.compare_source(base + ['|-formechange|p2: Earlier Name|Castform-Rainy'])
        host, _ = public_wire.project_lines(public_wire.WireState(), base,
            **self.f.e.wire_kwargs(self.f.catalog))
        for line in ('|-formechange|p3a: Castform|Castform-Sunny', '|-formechange',
                     '|-formechange|p2a: Castform'):
            with self.subTest(line=line), self.assertRaises(ValueError):
                public_wire.project_line(host, line, **self.f.e.wire_kwargs(self.f.catalog))
        with self.assertRaises(ValueError):
            public_wire.project_line(host, '|-formechange|p2a: Castform|Castform-Sunny',
                **dict(self.f.e.wire_kwargs(self.f.catalog), type_catalog=None))
        before = self.f.ingest(base)
        _, projections = public_wire.project_lines(host,
            ['|-formechange|p2a: Castform|Castform-Sunny|[msg]|[from] ability: Forecast'],
            **self.f.e.wire_kwargs(self.f.catalog))
        chunk = self.f.chunks(projections)[0]
        bad = chunk._replace(type_binding=chunk.type_binding.at[0].add(1))
        after, error = self.f.jax.jit(lambda s,c:self.f.e.fold_chunk(s,c,self.f.catalog))(before,bad)
        self.assertNotEqual(error.tolist(), [0])
        for a,b in zip(self.f.jax.tree.leaves(before),self.f.jax.tree.leaves(after),strict=True):
            np.testing.assert_array_equal(a,b)
