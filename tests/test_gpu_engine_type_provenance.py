"""Type-event history follows the host parser, including restoration to native typing."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class TypeProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_gpu_engine_public_types_context import PublicTypeContextTests as fixture
        fixture.setUpClass.__func__(cls)
        from gpu_engine import public_team
        from pokezero.category_vocab import CategoryVocabulary
        from pokezero.dex import ShowdownDex, MoveInfo
        cls.vocab = CategoryVocabulary(cls.vocab.tokens + ('ability:colorchange', 'ability:forecast', 'move:return'))
        moves = dict(cls.dex.moves)
        moves['return'] = MoveInfo('return', 'Return', 'Normal', 'Physical', 'Physical', 102, 100, 0,
                                  False, False, False, None, {}, 'normal', False, pp=20)
        cls.dex = ShowdownDex(moves, cls.dex.species, cls.dex.type_chart)
        cls.catalog = cls.p.compile_moves(cls.host, cls.dex, cls.vocab)
        cls.abilities = cls.a.compile_abilities(cls.host, cls.vocab,
            ability_names=('Pressure', 'Trace', 'Immunity', 'Levitate', 'Color Change', 'Forecast'))
        cls.item_catalog = cls.items.compile_items(cls.host, cls.vocab, item_names=('Mystery Orb',))
        cls.species_catalog = public_team.compile_species(cls.host, cls.dex, cls.vocab, cls.abilities)
        cls.field_catalog = cls.f.compile_field(cls.vocab)
        cls.type_catalog = cls.t.compile_types(cls.species_catalog, cls.dex, cls.vocab)
        cls.contexts = {n: cls.c.compile_context(cls.host, cls.abilities, cls.catalog, items=cls.item_catalog,
            item_belief_narrowing=n, field_catalog=cls.field_catalog, type_catalog=cls.type_catalog) for n in (False, True)}
        cls.kwargs = dict(source=cls.host, species_catalog=cls.species_catalog, moves=cls.catalog,
            abilities=cls.abilities, items=cls.item_catalog, field_catalog=cls.field_catalog, type_catalog=cls.type_catalog)

    def test_protocol_corpus_matches_host_at_every_prefix(self):
        from pokezero import tier2
        cases = {
            'color_change_native_restore': [
                '|-start|p2a: Lizard|typechange|Fire|[from] ability: Color Change',
                '|-start|p2a: Lizard|typechange|Normal|[from] ability: Color Change',
                '|move|p2a: Lizard|Return|p1a: Cloud',
                '|-end|p2a: Lizard|typechange', '|faint|p2a: Lizard'],
            'conversion': ['|-start|p2a: Lizard|typechange|Fire|[from] move: Conversion'],
            'conversion2': ['|-start|p2a: Lizard|typechange|Water|[from] move: Conversion 2'],
            'camouflage': ['|-start|p2a: Lizard|typechange|Normal|[from] move: Camouflage'],
            'transform': ['|-transform|p2a: Lizard|p1a: Castform'],
            'forecast_native_restore': ['|-formechange|p1a: Cloud|Castform-Sunny',
                                       '|-formechange|p1a: Cloud|Castform'],
        }
        for name, events in cases.items():
            host = tier2._IncrementalContextFold()
            lines = list(self.prefix)
            # Arrival clears the previous occupant's history.
            events = [*events, '|switch|p2a: Spare|Absol, L80|100/100']
            for event in events:
                lines.append(event);host.process(lines)
                with self.subTest(case=name, event=event):
                    result, _ = self.run_lines(lines)
                    self.assertEqual(result.context.types.type_changed.tolist(),
                                     [[host.type_changed['p1'], host.type_changed['p2']]])
            if name == 'color_change_native_restore':
                before_faint, _ = self.run_lines([*self.prefix, *events[:4]])
                self.assertTrue(bool(before_faint.context.types.type_changed[0, 1]))
                self.assertFalse(bool(before_faint.context.types.overridden[0, 1]))

    def test_device_event_changes_only_appended_history(self):
        from typing import NamedTuple
        import jax
        import jax.numpy as j
        import numpy as np
        from gpu_engine import engine_device_public_hooks as hooks
        class Carry(NamedTuple):
            error: object
            old_leaf: object
            type_changed: object
        class Context(NamedTuple):
            trackers: object
        c = Context(Carry(j.zeros(2, j.int32), j.asarray([71, 83], j.int32), j.zeros((2, 2), bool)))
        before = jax.tree_util.tree_flatten_with_path(c.trackers)[0]
        c = hooks.type_event(c, 1, j.asarray([True, False]))
        c = hooks.type_event(c, 0, j.asarray([False, True]))
        after = jax.tree_util.tree_flatten_with_path(c.trackers)[0]
        self.assertEqual([jax.tree_util.keystr(k) for k, _ in before],
                         [jax.tree_util.keystr(k) for k, _ in after])
        for (key, old), (_, new) in zip(before[:-1], after[:-1]):
            self.assertEqual(old.dtype, new.dtype)
            self.assertEqual(old.shape, new.shape)
            self.assertEqual(np.asarray(old).tobytes(), np.asarray(new).tobytes())
        self.assertEqual(c.trackers.type_changed.tolist(), [[False, True], [True, False]])


# Reuse the canonical public-input fold fixture without rediscovering its test methods.
if importlib.util.find_spec('jax'):
    import test_gpu_engine_public_types_context as fixture
    for name in ('project', 'initial', 'fold', 'run_lines', 'chunks'):
        setattr(TypeProvenanceTests, name, getattr(fixture.PublicTypeContextTests, name))
    TypeProvenanceTests.carry = staticmethod(fixture.PublicTypeContextTests.carry)


if __name__ == '__main__':unittest.main()
