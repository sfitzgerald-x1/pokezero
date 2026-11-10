"""CPU tests of scripts/depthn_rr_reprice.py --true-worlds (the MC ladder's R1/R2 worlds): no GPU, no Showdown, no pricer. The
constructor samples through phase_d_protocol_world.sample as ProtocolConstructor.prepare does (the OV tests' sampler stand-in);
the true world's materialization is a fake, the overlay the OV harness's own; main runs the real DepthNLiveSearch over
tests/test_depthn_rr_reprice's fake pricing, with build's all-W rule."""
from contextlib import ExitStack, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from scripts import depthn_opponent_view_parity as P, depthn_rr_reprice as R
from tests import test_depthn_opponent_view_parity as PT, test_depthn_rr_reprice as RT
from gpu_engine import constructor_worlds as cw, mc_ladder_opponents as M, phase_d_protocol_world as pw
from gpu_engine.phase_d_live_worlds import Refusal, WorldSet

ORIGINAL_SAMPLE, ORIGINAL_FAKE_BUILD = pw.sample, RT.fake_build
CELL = dict(ordinal=1001, seed=7, target_kind='foulplay')
HP = [100, 0, 50, 0, 1, 2]   # RT.STATE's opponent HP (Snorlax, slot 2, at 50)


def write_sidecar(directory, *, kind='foulplay', ordinal=1001, seed=7, ours='p1', roots=None):
    """An opponent sidecar as mc_ladder_opponents writes it: at decision 0 Foul Play's request (Snorlax 60/100, PP 1/2/3), at
    decision 2 it waits. Returns its path."""
    folder = Path(directory) / 'opponent';folder.mkdir(parents=True, exist_ok=True)
    if roots is None:roots = {0: dict(decision_index=0, round=0, opponent_requested=True, root_request=PT.fp_request('60/100', (1, 2, 3))),
        2: dict(decision_index=2, round=2, opponent_requested=False, root_request=None)}
    path = folder / f'{kind}-{ordinal}.json'
    path.write_text(json.dumps(M.document(kind, ordinal, seed, ours, PT.TEAM, PT.fp_request(), roots)))
    return path


def fake_true_world(seen, *, reason=None, refused=False, error=None):
    """build_true_world's stand-in: RT's world under public marker `index` (the peer self-check's world) with sampled world i's
    kernel seed 1000 + i; or the team not reconstructed (reason), the materialization refused, or an error."""
    def build(call, team, request0, *, index=0):
        seen.append((index, team, request0))
        if error is not None:raise error
        if reason is not None:return None, dict(reason=reason)
        info = dict(reason=None, kernel_seed=1000 + index, constructed=not refused, refusals=['MATERIALIZE_REFUSED'] if refused else [],
            species=['snorlax'], happiness=[255], notes=[], packed_sha256='p' * 64)
        world = SimpleNamespace(state=None if refused else RT.STATE, public=None if refused else dict(w=jnp.asarray(index, jnp.int32)))
        return world, info
    return build


class SamplingConstructor(RT.FakeConstructor):
    """RT's constructor, its prepare sampling through phase_d_protocol_world.sample as ProtocolConstructor.prepare does, with its
    metadata (whose vocabulary the overlay reads)."""
    def prepare(self, request, index):
        self.calls.append(request['decision_index'])
        if request['decision_index'] in self.refuse:raise Refusal('TEST_REFUSAL', 'no worlds')
        detached = SimpleNamespace(address=SimpleNamespace(stream=1, frame=request['decision_index'], owner=0), public=PT.ROOT_CARRY)
        made = pw.sample(SimpleNamespace(kind='public'), detached, lines=tuple(request['lines']), catalog='catalog', source='source',
            metadata=self.metadata, config='config', battle_id=request['battle_id'], seed=20261103, count=16)
        return made.root, made, RT.LEGAL


def all_w_build(constructor, prepared, *arguments, **keywords):
    """RT's fake build with build's all-W rule (only_complete): a root with a world not built is refused."""
    held = [cw.RootRefusal(item.request_id, 'WORLD_CONSTRUCTOR_REFUSED', '', item.seconds, item.binding)
        if type(item) is cw.Prepared and not item.made.fully_constructed else item for item in prepared]
    return ORIGINAL_FAKE_BUILD(constructor, held, *arguments, **keywords)


class WrapperTests(unittest.TestCase):
    def prepare(self, directory, kind, decision=0, cell=CELL, build=None):
        """One prepare() under the wrapper: (its WorldSet, the record, build_true_world's calls). The sampler is restored after."""
        seen = []
        with ExitStack() as stack:
            stack.enter_context(patch.object(pw, 'sample', PT.fake_sample))
            stack.enter_context(patch.object(P, 'build_true_world', build or fake_true_world(seen)))
            with R.TrueWorlds(directory, kind).installed(cell, decision) as record:
                self.assertIsNot(pw.sample, PT.fake_sample)
                _, made, _ = SamplingConstructor().prepare(RT.body(decision), 0)
            self.assertIs(pw.sample, PT.fake_sample)
        self.assertIs(pw.sample, ORIGINAL_SAMPLE)
        return made, record, seen

    def test_true_private_prices_w_draws_of_the_true_state_with_the_roots_hp_and_pp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_sidecar(tmp);made, record, seen = self.prepare(tmp, 'true_private')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertIsInstance(made, WorldSet);self.assertTrue(made.fully_constructed)
        self.assertEqual((len(made.worlds), made.requested_worlds, made.seeds), (16, 16, tuple(500 + i for i in range(16))))   # the sampler's seeds
        self.assertEqual(made.root.address.frame, 0)   # the sampler's carried root
        self.assertEqual([(w.index, w.kernel_seed, int(np.asarray(w.public['w'])), w.refusals) for w in made.worlds], [(i, 1000 + i, i, ()) for i in range(16)])
        self.assertEqual(seen, [(i, PT.TEAM, PT.fp_request()) for i in range(16)])   # the sidecar's team and round-0 request, world i's seed
        for w in made.worlds:
            self.assertEqual(np.asarray(w.state.hp)[0, 1].tolist(), [100, 0, 60, 0, 1, 2])   # the root request's exact HP
            self.assertEqual(np.asarray(w.state.pp)[0, 1, 2].tolist(), [1, 2, 3, 0])   # and its active's PP
        self.assertEqual(np.asarray(RT.STATE.hp)[0, 1].tolist(), HP)   # the built world itself is not modified
        call = record['calls'][0]
        self.assertEqual((record['requested_kind'], record['problem'], record['sidecar_sha256'], call['kind'], call['fallback'], call['problem']),
            ('true_private', None, digest, 'true_private', None, None))
        self.assertEqual(call['worlds'][3], dict(world=3, true_kind='true_private', built=True, reason=None, notes=[], kernel_seed=1003,
            kernel_seed_is_sampled=True, refusals=[]))
        self.assertEqual(call['start'], dict(species=['snorlax'], happiness=[255], notes=[], packed_sha256='p' * 64))

    def test_true_keeps_the_materialized_draws_and_a_waiting_opponent_falls_back_to_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_sidecar(tmp)
            plain, record, _ = self.prepare(tmp, 'true')
            waiting, fallback, _ = self.prepare(tmp, 'true_private', decision=2)   # Foul Play waits at decision 2
            missing_root, _, _ = self.prepare(tmp, 'true', decision=3)   # true needs no root entry
        for made in (plain, waiting, missing_root):
            self.assertTrue(made.fully_constructed);self.assertTrue(all(np.asarray(w.state.hp)[0, 1].tolist() == HP for w in made.worlds))
        self.assertEqual((record['calls'][0]['kind'], record['calls'][0]['fallback']), ('true', None))
        call = fallback['calls'][0]
        self.assertEqual((fallback['requested_kind'], call['kind'], call['fallback'], call['problem']),
            ('true_private', 'true', 'the opponent waits at the root: no request to overlay', None))
        self.assertTrue(all(f['true_kind'] == 'true' and f['built'] for f in call['worlds']))

    def test_every_world_fails_closed_when_the_true_world_cannot_be_built(self):
        cases = (('no sidecar', dict(), 'true', 0, None, 'no opponent sidecar foulplay-1001.json'),
            ('another seed', dict(seed=8), 'true', 0, None, 'the sidecar names seed 8, the cell 7'),
            ('another seat', dict(ours='p2'), 'true', 0, None, 'the sidecar is seat p2, the root seat p1'),
            ('no root entry', dict(write=True), 'true_private', 3, None, 'the sidecar has no root 3'),
            ('a root the writer could not align', dict(roots={0: dict(decision_index=0, error='no owner capture row')}), 'true_private', 0, None,
                'the sidecar\'s root 0: no owner capture row'),
            ('the team', dict(write=True), 'true', 0, dict(reason='foul play team: stats'), 'foul play team: stats'),
            ('the materialization', dict(write=True), 'true', 0, dict(refused=True), 'the true world\'s materialization was refused'),
            ('an error', dict(write=True), 'true', 0, dict(error=KeyError('slot')), 'KeyError: \'slot\''))
        for name, sidecar, kind, decision, build, reason in cases:
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / 'opponent').mkdir()
                if sidecar:write_sidecar(tmp, **{k: v for k, v in sidecar.items() if k != 'write'})
                made, record, _ = self.prepare(tmp, kind, decision, build=None if build is None else fake_true_world([], **build))
                self.assertFalse(made.fully_constructed);self.assertEqual(len(made.worlds), 16)   # the live search refuses the root
                self.assertTrue(all(w.state is None and w.public is None for w in made.worlds))
                facts = record['calls'][0]['worlds']
                self.assertEqual({(f['built'], f['reason']) for f in facts}, {(False, reason)})
                want = ['MATERIALIZE_REFUSED'] if build and build.get('refused') else [reason]
                self.assertEqual([list(w.refusals) for w in made.worlds], [want] * 16);self.assertEqual(facts[0]['refusals'], want)

    def test_the_samplers_refusal_is_raised_on_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            write_sidecar(tmp);stack.enter_context(patch.object(pw, 'sample', PT.fake_sample))
            with R.TrueWorlds(tmp, 'true').installed(dict(CELL), 99) as record, self.assertRaises(Refusal):
                SamplingConstructor().prepare(RT.body(99), 0)   # frame 99: the stand-in refuses the source binding
            self.assertIs(pw.sample, PT.fake_sample)
            with self.assertRaises(KeyError), R.TrueWorlds(tmp, 'true').installed(dict(CELL), 0):raise KeyError('boom')
            self.assertIs(pw.sample, PT.fake_sample)   # restored when the block raises
        self.assertTrue(record['calls'][0]['error'].startswith('Refusal: SAME_PUBLIC_ROOT_SOURCE_BINDING_REQUIRED'))

    def test_sidecars_are_found_by_the_cells_ordinal_and_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp = write_sidecar(tmp);mine = write_sidecar(tmp, kind='self', ordinal=12, ours='p2');write_sidecar(tmp, ordinal=11001)
            self.assertEqual(R.sidecar_path(tmp, CELL), (fp, None))
            self.assertEqual(R.sidecar_path(tmp, dict(ordinal=12, target_kind='self')), (mine, None))
            self.assertEqual(R.sidecar_path(tmp, dict(ordinal=12)), (mine, None))   # no kind: the ordinal's only sidecar
            self.assertEqual(R.sidecar_path(tmp, dict(ordinal=12, target_kind='foulplay')), (None, 'no opponent sidecar foulplay-12.json'))
            self.assertEqual(R.sidecar_path(tmp, dict(ordinal=5)), (None, 'no opponent sidecar for cell 5'))
            write_sidecar(tmp, kind='self', ordinal=1001)
            self.assertEqual(R.sidecar_path(tmp, dict(ordinal=1001)), (None, 'cell 1001 has 2 sidecars and no target kind'))
            self.assertEqual(R.sidecar_path(tmp, CELL), (fp, None))
            with self.assertRaisesRegex(ValueError, 'true-kind'):R.TrueWorlds(tmp, 'sampled')
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, 'no opponent/'):R.TrueWorlds(tmp, 'true')


class MainTests(unittest.TestCase):
    setUp = RT.MainTests.setUp

    def replay(self, tmp, *extra, cells=('--cells', '1001')):
        """RT's main replay of cell 1001 (decisions 0-4; 1 single-legal; 4 refused by the constructor before it samples) with a
        sampling constructor, build's all-W rule and the true-world stand-in."""
        with ExitStack() as stack:
            stack.enter_context(patch.object(RT, 'FakeConstructor', SamplingConstructor))
            stack.enter_context(patch.object(RT, 'fake_build', all_w_build))
            stack.enter_context(patch.object(pw, 'sample', PT.fake_sample))
            stack.enter_context(patch.object(P, 'build_true_world', fake_true_world([])))
            return RT.MainTests.replay(self, tmp, *extra, cells=cells)

    def test_options_pair_the_flags(self):
        run = ['--run-directory', 'r', '--out', 'o', '--cells', '1001']
        with patch.object(R.server.evaluators, 'check_arguments'):
            args = R.options(RT.BASE + RT.CHAMPION + run + ['--true-worlds', 'd', '--true-kind', 'true_private'])
            self.assertEqual((args.true_worlds, args.true_kind), ('d', 'true_private'))
            self.assertEqual((R.options(RT.BASE + RT.CHAMPION + run).true_worlds, R.options(RT.BASE + RT.CHAMPION + run).true_kind), (None, None))
            for extra in (['--true-worlds', 'd'], ['--true-kind', 'true'], ['--true-worlds', 'd', '--true-kind', 'sampled']):
                with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):R.options(RT.BASE + RT.CHAMPION + run + extra)

    def test_main_checks_the_samplers_call_site_before_anything_loads(self):
        self.assertTrue(P.check_call_site())   # the real ProtocolConstructor.prepare
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'out'
            argv = RT.BASE + RT.CHAMPION + ['--run-directory', tmp, '--out', str(out), '--cells', '1001', '--true-worlds', tmp, '--true-kind', 'true']
            for problem, regex in ((dict(side_effect=RuntimeError('prepare keeps its own sample')), 'its own sample'), (dict(return_value=True), 'no opponent/')):
                with self.subTest(regex), ExitStack() as stack:
                    stack.enter_context(patch.object(R, 'options', return_value=R.parser().parse_args(argv)))
                    stack.enter_context(patch.object(R, 'tree_guard', return_value={}))
                    stack.enter_context(patch.object(P, 'check_call_site', **problem));build = stack.enter_context(patch.object(R.server, 'build'))
                    with self.assertRaisesRegex(SystemExit, regex):R.main([])
                    build.assert_not_called();self.assertFalse(out.exists())

    def test_main_records_true_worlds_falls_back_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            sidecars = Path(tmp) / 'sidecars';path = write_sidecar(sidecars)
            targets = Path(tmp) / 'targets.jsonl'
            targets.write_text(''.join(json.dumps(dict(ordinal=1001, decision_index=i, kind='foulplay', seed=7, ours='p1')) + '\n' for i in (0, 2, 3, 4)))
            code, rows, summary, header, constructor, *_ = self.replay(tmp, '--true-worlds', str(sidecars), '--true-kind', 'true_private',
                cells=('--targets', str(targets)))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual((code, constructor.calls, [r['decision_index'] for r in rows]), (0, [0, 2, 3, 4], [0, 2, 3, 4]))
        self.assertIs(pw.sample, ORIGINAL_SAMPLE)
        self.assertEqual(header['true_worlds'], dict(directory=str(sidecars), kind='true_private'))
        first, waiting, no_root, refused = rows
        self.assertEqual(first['true_worlds'], dict(sidecar=str(path), sidecar_sha256=digest, problem=None, requested_kind='true_private', sample_calls=1,
            error=None, kind='true_private', fallback=None, start=dict(species=['snorlax'], happiness=[255], notes=[], packed_sha256='p' * 64), worlds=16, built=16))
        self.assertEqual((first['replay']['status'], waiting['replay']['status']), ('COMPLETE', 'COMPLETE'))
        self.assertEqual([w['true_world']['kernel_seed'] for w in first['worlds']], [1000 + i for i in range(16)])
        self.assertEqual([w['kernel_seed'] for w in first['worlds']], [1000 + i for i in range(16)])   # the priced worlds are the true ones
        self.assertEqual(first['worlds'][0]['opponent']['party'][2]['hp'], 60)   # the overlay reached the priced world
        self.assertEqual(waiting['worlds'][0]['opponent']['party'][2]['hp'], 50)
        self.assertEqual((waiting['true_worlds']['kind'], waiting['true_worlds']['fallback']), ('true', 'the opponent waits at the root: no request to overlay'))
        self.assertEqual((no_root['replay']['status'], no_root['replay']['cause'], no_root['root']['refusal_cause']), ('REFUSED', 'constructor_refused', 'constructor_refused'))
        self.assertEqual((no_root['true_worlds']['problem'], no_root['true_worlds']['built']), ('the sidecar has no root 3', 0))
        self.assertTrue(all(not w['constructed'] and w['refusals'] == ['the sidecar has no root 3'] for w in no_root['worlds']))
        self.assertEqual((refused['prepared']['kind'], refused['true_worlds']['sample_calls'], refused['true_worlds']['kind']), ('RootRefusal', 0, None))
        self.assertEqual(summary['true_worlds'], dict(requested_kind='true_private', rows=4, played=dict(true_private=2, true=1),
            fallbacks=[dict(ordinal=1001, decision_index=2, fallback='the opponent waits at the root: no request to overlay')],
            failed_closed=[dict(ordinal=1001, decision_index=3, problem='the sidecar has no root 3', built=0, worlds=16)],
            never_sampled=[dict(ordinal=1001, decision_index=4)]))
        self.assertTrue(summary['peer_inputs_check']['passed'])   # the pricer's views read the true worlds in block order

    def test_main_without_a_sidecar_refuses_every_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'sidecars' / 'opponent').mkdir(parents=True)
            code, rows, summary, *_ = self.replay(tmp, '--true-worlds', str(Path(tmp) / 'sidecars'), '--true-kind', 'true')
        self.assertEqual((code, [r['replay']['status'] for r in rows], summary['complete']), (0, ['REFUSED'] * 4, 0))
        self.assertEqual({r['true_worlds']['problem'] for r in rows}, {'no opponent sidecar for cell 1001'})
        self.assertEqual([f['decision_index'] for f in summary['true_worlds']['failed_closed']], [0, 2, 3])
        self.assertEqual(summary['true_worlds']['never_sampled'], [dict(ordinal=1001, decision_index=4)])


if __name__ == '__main__':
    unittest.main()
