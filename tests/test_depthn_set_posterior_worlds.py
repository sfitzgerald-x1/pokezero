"""set_posterior in the protocol world route, on real screen r5a roots (tests/fixtures/set_posterior): capture_root's own
belief view gives the fixture's posterior; the start sampler is byte-identical under uniform weights and draws Choice Band
when a strike settles it; phase_d_protocol_world.sample hands the root's posteriors to the sampler (and the switch restores
today's call); materialize_protocol refuses a root whose hard support differs from B2's, in either direction, by name (or,
non-fatal, records it), and every world carries a non-fatal PUBLIC_SET_POSTERIOR row. Light (no engine compile): depthn-light."""
from contextlib import ExitStack
from dataclasses import replace
import random
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from _showdown_root import requires_showdown, showdown_root_str
from test_depthn_set_posterior import FIXTURES, beliefs, fixture
from gpu_engine import phase_d_protocol_world as pw
from pokezero import determinization as D, set_posterior as SP, v5_contract
from pokezero.category_vocab import build_category_vocabulary
from pokezero.dex import load_showdown_dex_cached, normalize_id
from pokezero.engine_world import unpack_team
from pokezero.live_protocol_actor import ActorContract, _vocab_sha
from pokezero.local_showdown import LocalShowdownConfig
from pokezero.observation import ObservationFeatureMasks
from pokezero.randbat import load_gen3_randbat_source_cached
from pokezero.v5_contract import DecisionAddress
from pokezero.v5_teacher_public_root import capture_root

SEED = 20261103


class Detached:
    """Expose the preceding trap agreement; other own fields mark completion of both posterior checks."""
    class Own(dict):
        def __missing__(self, key):
            raise ValueError('reached-after-posterior-checks')

    def __init__(self, address):
        self.address = address
        self._own = self.Own(active=0, volatile=(0,) * 6)

    @property
    def own(self):
        return self._own


@requires_showdown('the protocol route needs the pinned gen3 randbats source and dex')
class ProtocolRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dex = load_showdown_dex_cached(showdown_root_str())
        cls.src = load_gen3_randbat_source_cached(showdown_root_str())
        base = ActorContract.champion()
        # The belief view does not read the category vocabulary: a one-token vocabulary (every other token hashed) keeps the
        # roots process-free and independent of the champion's vocabulary file.
        with patch('pokezero.category_vocab.logger.warning'):
            vocab = build_category_vocabulary(['request_kind:move'])
        cls.contract = replace(base, category_vocab_sha256=_vocab_sha(vocab), categorical_vocab_size=vocab.size)
        masks = dict(base.feature_masks);masks['opponent_tendency_stats_block'] = masks.pop('stats_block')
        cls.config = LocalShowdownConfig(showdown_root=showdown_root_str(), set_belief_source=True, category_vocab=vocab,
                                         observation_spec=v5_contract.V4_SPEC, feature_masks=ObservationFeatureMasks(**masks))
        cls.roots = {}

    def root(self, name):
        if name not in self.roots:
            data = fixture(name);request = data['request']
            with patch('pokezero.category_vocab.logger.warning'):
                root = capture_root(config=self.config, contract=self.contract, lines=tuple(request['lines']), battle_id=request['battle_id'],
                    address=DecisionAddress(request['stream_id'], request['decision_index'], int(request['owner'][1]) - 1), seed=SEED & 0xffffffff)
            self.roots[name] = (data, root)
        return self.roots[name]

    def inputs(self, data, binding='bound'):
        return pw.ProtocolInputs(None, tuple(data['request']['lines']), NS(dex=self.dex, source=NS(source_hash=self.src.metadata.source_hash)),
                                 {}, NS(original=self.src, binding_sha256=binding))

    def start(self, root, seed, **kwargs):
        rng = random.Random(seed)
        result = D._gen3_randbat_belief_start_override_result(context=root.context, set_source=self.src, rng=rng, team_size=6,
                                                              canonicalize_revealed_variants=True, **kwargs)
        return result, rng.getstate()

    def test_the_root_belief_view_gives_the_fixture_posterior(self):
        for name in sorted(p.stem for p in FIXTURES.glob('*.json')):
            with self.subTest(fixture=name):
                data, root = self.root(name)
                posts = pw._set_posteriors(self.inputs(data), root)
                expected = SP.posterior(data['request']['lines'], None, source=self.src, owner=data['owner'], beliefs=beliefs(data),
                                        format_id=data['format_id'], dex=self.dex, binding='bound')
                self.assertEqual(posts, expected)
                self.assertEqual({k: [i for i, h in enumerate(p.hard) if h] for k, p in posts.items()},
                                 {k: v['hard'] for k, v in data['posterior'].items()})

    def test_the_start_sampler_is_byte_identical_under_uniform_weights(self):
        for name in ('refusal_trace', 'refusal_item_changed', 'refusal_hit_substitute', 'cb_collapse'):
            data, root = self.root(name)
            posts = SP.posterior(data['request']['lines'], None, source=self.src, owner=data['owner'], beliefs=beliefs(data),
                                 format_id=data['format_id'], dex=self.dex)
            uniform = {k: replace(p, weights=tuple(int(h) for h in p.hard), support=p.hard) for k, p in posts.items()}
            game = SP.game_lines_sha256(data['request']['lines'])
            for index in range(16):
                seed, _ = pw.world_seeds(root, NS(address=root.address), game, SEED, index)
                with self.subTest(fixture=name, world=index):
                    old = self.start(root, seed)
                    self.assertIsNotNone(old[0][0], old[0][1])
                    self.assertEqual(self.start(root, seed, set_posteriors=lambda: uniform), old)
                    if name != 'cb_collapse':   # their strikes are all refused: today's weights exactly
                        self.assertEqual(self.start(root, seed, set_posteriors=posts), old)

    def test_a_settled_choice_band_is_drawn_in_every_world(self):
        data, root = self.root('cb_collapse')
        posts = pw._set_posteriors(self.inputs(data), root)
        game = SP.game_lines_sha256(data['request']['lines'])
        foe = 'p1' if data['owner'] == 'p2' else 'p2'
        today = set()
        for index in range(16):
            seed, _ = pw.world_seeds(root, NS(address=root.address), game, SEED, index)
            (new, _), _ = self.start(root, seed, set_posteriors=posts)
            (old, _), _ = self.start(root, seed)
            item = lambda start: next(normalize_id(m.item or '') for m in unpack_team(start.player_teams[foe]) if normalize_id(m.species) == 'rhydon')
            self.assertEqual(item(new), 'choiceband')
            today.add(item(old))
        self.assertGreater(len(today), 1)

    def test_sample_hands_the_roots_posteriors_to_the_sampler_and_the_draw_switch_restores_today(self):
        data, root = self.root('cb_collapse')
        detached = NS(address=root.address, source_binding_sha256='bound')
        calls = []

        def sampler(**kwargs):
            calls.append(kwargs);return None, 'recorded'
        inputs = self.inputs(data)
        for switch in (True, False):
            calls.clear()
            with ExitStack() as stack:
                stack.enter_context(patch.object(pw, 'SET_POSTERIOR_DRAW', switch))
                stack.enter_context(patch.object(pw, '_gen3_randbat_belief_start_override_result', side_effect=sampler))
                stack.enter_context(patch.object(pw, '_carried_root', return_value=detached))
                made = pw.sample(root, detached, lines=inputs.lines, catalog=inputs.catalog, source=inputs.source, metadata={}, config=None,
                                 battle_id=root.context.battle_id, seed=9, count=2)
            self.assertEqual([w.refusals for w in made.worlds], [('recorded',)] * 2)
            if switch:
                posts = [call['set_posteriors']() for call in calls]
                self.assertIs(posts[0], posts[1])   # one posterior per root, shared by its worlds
                self.assertEqual(posts[0], SP.posterior(inputs.lines, None, source=self.src, owner=data['owner'], beliefs=beliefs(data),
                                                        format_id=data['format_id'], dex=self.dex, binding='bound'))
            else:
                self.assertTrue(all('set_posteriors' not in call for call in calls))

    def posts(self, data):
        return SP.posterior(data['request']['lines'], None, source=self.src, owner=data['owner'], beliefs=beliefs(data),
                            format_id=data['format_id'], dex=self.dex, binding='bound')

    def world_start(self, name, index, posts):
        data, root = self.root(name)
        seed = pw.world_seeds(root, NS(address=root.address), SP.game_lines_sha256(data['request']['lines']), SEED, index)[0]
        (start, reason), _ = self.start(root, seed, set_posteriors=posts)
        self.assertIsNotNone(start, reason)
        return start

    def materialize(self, name, rows_of, start=None):
        """materialize_protocol on a real root and a start (default: its posterior-drawn world 0), with B2's posterior rows
        rows_of(posts) and trap-free own agreement facts (Detached: other own reads raise 'reached-after-posterior-checks'); the
        tail facts, removals and preflight are patched clear."""
        data, root = self.root(name)
        inputs = self.inputs(data)
        inputs = pw.ProtocolInputs(Detached(root.address), inputs.lines, inputs.catalog, {}, inputs.source)
        posts = self.posts(data)
        start = self.world_start(name, 0, posts) if start is None else start
        rows = rows_of(posts)
        with ExitStack() as stack:
            stack.enter_context(patch.object(pw.tail, 'facts', return_value=dict(fresh_substitutes={}, replacement=None)))
            stack.enter_context(patch.object(pw.tail, 'defer_substitutes', side_effect=lambda payload, fresh: payload))
            stack.enter_context(patch.object(pw, '_removed_items', return_value={}))
            stack.enter_context(patch.object(pw, 'preflight_checks', return_value=()))
            stack.enter_context(patch.object(pw.public_worlds, '_revealed', return_value=rows))
            return pw.materialize_protocol(root, start, inputs=inputs, seed=1), posts

    def rows(self, posts, change=None):
        out = []
        for slot, (key, post) in enumerate(posts.items()):
            universe = self.src.universe_for(key)
            indices = [i for i, h in enumerate(post.hard) if h]
            if key == change:indices = indices[1:]
            out.append((slot, universe, tuple((i, universe.variants[i]) for i in indices), None, universe.species))
        return tuple(out)

    @staticmethod
    def row(made, kind):
        found = [x for x in made.mapper_diagnostics if x['kind'] == kind]
        return found[0] if len(found) == 1 else found

    def test_materialize_refuses_a_root_whose_hard_support_differs_from_b2s(self):
        name = 'refusal_item_changed'   # five revealed foes, one with eight hard variants
        made, _ = self.materialize(name, self.rows)
        self.assertEqual(made.refusals, ('reached-after-posterior-checks',))
        self.assertTrue(self.row(made, 'PUBLIC_SET_POSTERIOR')['b2_agrees']);self.assertEqual(self.row(made, SP.B2_DISAGREEMENT), [])
        made, posts = self.materialize(name, lambda posts: self.rows(posts, 'golduck'))
        self.assertEqual(made.refusals, (SP.B2_DISAGREEMENT,))
        row = self.row(made, SP.B2_DISAGREEMENT)
        first = [i for i, h in enumerate(posts['golduck'].hard) if h][0]
        self.assertEqual((row['status'], row['fatal'], row['species']),
                         ('TRIGGERED', True, [dict(species='golduck', set_posterior_only=[first], b2_only=[])]))
        self.assertIs(self.row(made, 'PUBLIC_SET_POSTERIOR')['b2_agrees'], False)
        # A species set_posterior holds and B2 does not: refused too.
        made, _ = self.materialize(name, lambda posts: self.rows(posts)[1:])
        self.assertEqual(made.refusals, (SP.B2_DISAGREEMENT,))
        # Non-fatal: the same TRIGGERED row, fatal False, and the world goes on (to today's one-way check, which a species B2
        # lacks passes).
        with patch.object(pw, 'SET_POSTERIOR_CHECK_FATAL', False):
            made, posts = self.materialize(name, lambda posts: self.rows(posts)[1:])
        self.assertEqual(made.refusals, ('reached-after-posterior-checks',))
        first = next(iter(posts))
        self.assertEqual(self.row(made, SP.B2_DISAGREEMENT)['species'],
                         [dict(species=first, set_posterior_only=[i for i, h in enumerate(posts[first].hard) if h], b2_only=[])])
        self.assertEqual((self.row(made, SP.B2_DISAGREEMENT)['fatal'], self.row(made, 'PUBLIC_SET_POSTERIOR')['b2_agrees']), (False, False))
        # The check off: today's one-way check only, which a species B2 lacks passes; the diagnostic row stays, unchecked.
        with patch.object(pw, 'SET_POSTERIOR_CHECK', False):
            made, _ = self.materialize(name, lambda posts: self.rows(posts)[1:])
        self.assertEqual(made.refusals, ('reached-after-posterior-checks',))
        self.assertEqual(self.row(made, SP.B2_DISAGREEMENT), []);self.assertIsNone(self.row(made, 'PUBLIC_SET_POSTERIOR')['b2_agrees'])
        # Both off: no set_posterior row at all.
        with patch.object(pw, 'SET_POSTERIOR_CHECK', False), patch.object(pw, 'SET_POSTERIOR_DRAW', False):
            made, _ = self.materialize(name, lambda posts: self.rows(posts)[1:])
        self.assertEqual(self.row(made, 'PUBLIC_SET_POSTERIOR'), [])

    def test_every_world_carries_a_non_fatal_set_posterior_row(self):
        data, root = self.root('cb_collapse')
        posts = self.posts(data)
        made, _ = self.materialize('cb_collapse', self.rows)
        row = self.row(made, 'PUBLIC_SET_POSTERIOR')
        cb = next(i for i, w in enumerate(posts['rhydon'].weights) if w)
        self.assertEqual({k: row[k] for k in ('status', 'fatal', 'drawn_by', 'zero_weight_draws', 'provenance', 'b2_agrees')},
                         dict(status='CLEAR', fatal=False, drawn_by='set_posterior', zero_weight_draws=[], provenance=posts['rhydon'].provenance,
                              b2_agrees=True))
        self.assertEqual(row['species'], [dict(species='rhydon', hard=4, weighted=1, admitted=1, refused={}, off_model_drops=0,
                                               drawn=[cb], drawn_weight=1)])
        # A world today's uniform draw gave a damage-excluded Rhydon: the row names the zero-weight draw, still non-fatal.
        uniform = {k: replace(p, weights=tuple(int(h) for h in p.hard), support=p.hard) for k, p in posts.items()}
        start = next(start for start in (self.world_start('cb_collapse', i, uniform) for i in range(16))
                     if next(normalize_id(m.item or '') for m in unpack_team(start.player_teams['p1' if data['owner'] == 'p2' else 'p2'])
                             if normalize_id(m.species) == 'rhydon') != 'choiceband')
        with patch.object(pw, 'SET_POSTERIOR_DRAW', False):
            made, _ = self.materialize('cb_collapse', self.rows, start=start)
        row = self.row(made, 'PUBLIC_SET_POSTERIOR')
        self.assertEqual(made.refusals, ('reached-after-posterior-checks',))
        self.assertEqual((row['status'], row['fatal'], row['drawn_by'], row['zero_weight_draws'], row['species'][0]['drawn_weight']),
                         ('TRIGGERED', False, 'uniform', ['rhydon'], 0))


if __name__ == '__main__':
    unittest.main()
