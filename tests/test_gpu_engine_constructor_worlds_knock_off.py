"""constructor_worlds' Knock Off fills (fix_knocked_items: the engine leaf item_knocked; fill_item_ledger: M1's per-mon item ledger
item_mutated / item_original) on two real Showdown Knock Off battles, held to the device path and to B2's ledger (CPU only).

The battles are real gen3randombattle games at the pinned Showdown (node, dist/sim/battle.js), between teams of real Gen 3
random-battle sets (pokezero.randbat's own source variants, each with the spread _fixture_from_variant_payload gives it, so from_b2
joins the owner's request to them exactly), driven by a fixed script:
- KNOCK: p1's Sableye knocks off p2's Snorlax's Leftovers (turn 1); Snorlax switches out (turn 2) and back in (turn 3).
- STICKY: Sableye's Knock Off on p2's Muk is blocked by Sticky Hold (-activate: nothing removed, turn 1); p2's Hariyama knocks
  off Sableye's Leftovers (turn 3).
At each battle's turn-4 decision, for both owners (so each Knock Off is once on the owner's side and once on the peer's, and the
Sticky Hold member once each):
- the constructor path: B2's fold (JaxRecordedAdapter) of the owner's own channel, phase_d_live_worlds.from_b2 with the owner's own
  request, 16 sampled worlds, each completed by constructor_worlds.complete_world with the lines' LineFacts and the root's carry;
- reference (a), the device path: the GPU engine with the device hooks over the same choices (transition.initialize_with_public and
  step_with_public on the generator's own static metadata, the full real table, run eagerly; a neutral chance tape), its state's
  item_knocked and the hooks' item_mutated / item_original; Showdown's own itemKnockedOff, item and active at every step anchor
  that trajectory;
- reference (b), B2's ledger: the same fold's public_items state (mutated, original), the original mapped to the engine item id.
Every completed world equals both references on every filled cell (the state's item_knocked of each party member; the carry's
ledger at each public slot), joined by species. The raw from_b2 worlds do not (the fills close that gap), and planted wrong fills
(another item, the next slot, a dropped state flag, a fill on the Sticky Hold member) fail the same comparison: the negative control.
One shared setup (setUpClass): one compile of the B2 catalog's fold and encode; the engine runs eagerly (no compile)."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from _showdown_root import showdown_root

ROOT = Path(__file__).resolve().parents[1]
# The generator's own static metadata prefix (gpu_engine/generate_teams.cjs: vocabulary, moves, type chart), evaluated without its team
# loop (tests/test_gpu_engine_depthn_constructor_gate's TablesBridgeTests rule).
METADATA_SCRIPT = (r"""const fs=require('fs'),vm=require('vm'),s=fs.readFileSync(process.argv[1],'utf8'),start=s.indexOf('async function generate('),"""
    r"""body=s.slice(s.indexOf('{',start)+1,s.indexOf('  async function emit(',start)).split('\n').filter(x=>!x.includes("const {Teams}")&&"""
    r"""!x.includes('const rngPackage')).join('\n');const code=s.slice(0,start)+'\nfunction metadata(root){'+body+"return {format:'gen3randombattle',"""
    r"""vocabulary,moves,type_chart:chart,stats_order:STATS};}\nJSON.stringify(metadata(ROOT))";"""
    r"""process.stdout.write(vm.runInNewContext(code,{require,ROOT:process.argv[2]}));""")
# One real battle per spec (stdin JSON): gen3randombattle with the given teams (Battle.getTeam uses a given team), the script's
# choices (`switch K` names the authored 1-based member, as gpu_engine/effect_probe.cjs's choice()); after every step: each player's
# own channel (sim/battle.ts extractChannelMessages, channels 1 and 2; |t:| timestamps dropped), each side's request, every member's
# item / itemKnockedOff / HP in authored order and the actives. Plus the engine pool row of every member (gpu_engine/team_rows.cjs).
BATTLE_SCRIPT = r"""
const path=require('path'),[root,repo]=process.argv.slice(1);
const {Battle,extractChannelMessages}=require(path.join(root,'dist/sim/battle.js'));
const rows=require(path.join(repo,'gpu_engine/team_rows.cjs'));const pool=rows.load(root);
const input=JSON.parse(require('fs').readFileSync(0,'utf8'));
const battles=input.battles.map(spec=>{
  const b=new Battle({formatid:'gen3randombattle',seed:spec.seed});
  spec.teams.forEach((team,s)=>b.setPlayer('p'+(s+1),{name:'p'+(s+1),team:team.map(x=>({...x}))}));
  b.sides.forEach(side=>side.pokemon.forEach((p,i)=>{p.pzIndex=i;}));
  const snap=()=>{
    const log=b.log.filter(l=>!l.startsWith('|t:|'));const ch=extractChannelMessages(log.join('\n'),[1,2]);
    return {turn:b.turn,ended:!!b.ended,channels:[ch[1],ch[2]],requests:b.sides.map(s=>s.activeRequest||null),active:b.sides.map(s=>s.active[0].pzIndex),
      mons:b.sides.map(s=>s.pokemon.slice().sort((x,y)=>x.pzIndex-y.pzIndex).map(p=>({species:p.species.id,item:p.item||'',knocked:!!p.itemKnockedOff,hp:p.hp,fainted:!!p.fainted})))};
  };
  const steps=[];
  for(const pair of spec.script){
    b.makeChoices(...pair.map((text,s)=>{
      const [kind,value]=text.split(/\s+/,2);if(kind!=='switch')return text;
      const index=b.sides[s].pokemon.findIndex(p=>p.pzIndex===Number(value)-1);if(index<0)throw new Error('no member '+value);return 'switch '+(index+1);}));
    steps.push(snap());
  }
  return {steps,rows:spec.teams.map(team=>team.map(set=>rows.memberRow(pool.dex,pool.id,set)))};
});
process.stdout.write(JSON.stringify({battles,vocabulary:pool.vocabulary}));
"""
SABLEYE, STARMIE, SWAMPERT, METAGROSS, HARIYAMA, SNORLAX, MUK = (('sableye', 'sableye-1-variant-1'), ('starmie', 'starmie-1-variant-5'),
    ('swampert', 'swampert-1-variant-2'), ('metagross', 'metagross-2-variant-7'), ('hariyama', 'hariyama-1-variant-4'),
    ('snorlax', 'snorlax-1-variant-2'), ('muk', 'muk-2-variant-4'))
P1 = (SABLEYE, STARMIE, SWAMPERT, METAGROSS, HARIYAMA, SNORLAX)
# name: (p1 team, p2 team, script, Showdown seed). Sableye: Knock Off, Recover, Seismic Toss, Toxic (Keen Eye, Leftovers); Snorlax:
# Body Slam, Curse, Earthquake, Shadow Ball; Muk: Curse, Explosion, Hidden Power Ground, Sludge Bomb (Sticky Hold); Hariyama: Bulk Up,
# Cross Chop, Earthquake, Knock Off. Every member holds Leftovers.
BATTLES = {
    'knock': (P1, (SNORLAX, STARMIE, MUK, HARIYAMA, SWAMPERT, METAGROSS), (('move 1', 'move 2'), ('move 3', 'switch 2'), ('move 3', 'switch 1')), (1, 2, 3, 4)),
    'sticky': (P1, (MUK, HARIYAMA, STARMIE, SNORLAX, SWAMPERT, METAGROSS), (('move 1', 'move 1'), ('move 3', 'switch 2'), ('move 3', 'move 4')), (5, 6, 7, 8)),
}
WORLDS = 16


def showdown_set(source, species, variant_id):
    """The Showdown set of one real source variant: its moves, ability, item and level, the spread _fixture_from_variant_payload gives
    it (the one from_b2 checks the request's stats against), a neutral nature and the species' own gender (M when mixed)."""
    from pokezero.determinization import _fixture_from_variant_payload
    universe = source.universes[species];variant = next(v for v in universe.variants if v.variant_id == variant_id)
    gender = source.species_metadata[species].get('gender') or 'M'
    fixture = _fixture_from_variant_payload(variant.to_summary(), fallback_species=universe.species, set_source=source, gender=gender)
    return dict(species=universe.species, level=fixture.level, moves=list(fixture.moves), ability=fixture.ability, item=fixture.item or '',
        evs=dict(fixture.evs), ivs=dict(fixture.ivs), nature='Serious', gender=gender)


def names(table, ids):
    """Normalized names of `ids` in `table` ('' for 0 or out of range)."""
    from pokezero.dex import normalize_id
    return [normalize_id(table[int(x)]) if 0 < int(x) < len(table) else '' for x in np.asarray(ids).reshape(-1)]


class KnockOffCarryParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        from pokezero import v5_contract
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.live_protocol_runtime import CompiledEncoderFunctions
        from pokezero.randbat import Gen3RandbatSource, RandbatSourceMetadata, load_gen3_randbat_source_cached
        from pokezero.randbat_vocab import gen3_category_vocabulary
        from pokezero.recorded_mirror_roots import CHAMPION_CONTRACT
        from gpu_engine import constructor_worlds as cw, device_public_world_state as states, device_public_worlds as worlds
        from gpu_engine import public_belief, transition, v4_encoder
        cls.jax, cls.cw = jax, cw
        root = showdown_root()
        cls.metadata = json.loads(subprocess.check_output(['node', '-e', METADATA_SCRIPT, str(ROOT / 'gpu_engine/generate_teams.cjs'), str(root)], text=True))
        # The real source restricted to the battles' species (a light catalog; every variant, spread and metadata row is the real one).
        real = load_gen3_randbat_source_cached(root)
        kept = sorted({s for p1, p2, _, _ in BATTLES.values() for s, _ in p1 + p2})
        cls.source = Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle', 3, None, None, None, 'knock-off-' + real.metadata.source_hash),
            {k: real.universes[k] for k in kept}, move_metadata=real.move_metadata, species_metadata=real.species_metadata)
        cls.catalog = v4_encoder.compile_encoder(public_belief.compile_source(cls.source), load_showdown_dex_cached(root),
            gen3_category_vocabulary(root, include_feature_pack_v4=True), checkpoint_sha256='a' * 64, feature_masks=CHAMPION_CONTRACT['feature_masks'])
        cls.compiled = worlds.compile_source(cls.source, cls.metadata, expected_source_hash=cls.source.metadata.source_hash,
            expected_source_rows_sha256=v5_contract.source_variant_census(cls.source)['source_rows_sha256'])
        cls.materializer = states.compile_tables(cls.compiled, cls.metadata)
        cls.tables = cw.carry_tables(cls.metadata, cls.catalog)
        # One B2 fold/encode compile for every root (the live fold's shared jits), which run under the live server's precision
        # (live_protocol_runtime.validate: jax_default_matmul_precision 'highest'): set for this class, restored after it.
        precision = jax.config.jax_default_matmul_precision;jax.config.update('jax_default_matmul_precision', 'highest')
        cls.addClassCleanup(jax.config.update, 'jax_default_matmul_precision', precision)
        cls.functions = CompiledEncoderFunctions(cls.catalog)
        # The real battles (Showdown) and every member's engine pool row.
        specs = [dict(teams=[[showdown_set(real, s, v) for s, v in team] for team in (p1, p2)], script=[list(x) for x in script], seed=list(seed))
            for p1, p2, script, seed in BATTLES.values()]
        played = json.loads(subprocess.check_output(['node', '-e', BATTLE_SCRIPT, str(root), str(ROOT)], input=json.dumps(dict(battles=specs)), text=True))
        cls.vocabulary = played['vocabulary'];cls.showdown = dict(zip(BATTLES, (b['steps'] for b in played['battles'])))
        # Reference (a): the engine with the device hooks over the same choices, one lane per battle, a neutral chance tape.
        cls.table = transition.tables_from_metadata(cls.metadata)
        rows = [team for b in played['battles'] for team in b['rows']]
        pool = {k: np.asarray([[m[k] for m in team] for team in rows]) for k in ('species', 'level', 'types', 'stats', 'moves', 'pp', 'item', 'ability',
            'hidden_power_type', 'hidden_power_power')}
        state = transition.state_from_pool(pool, np.arange(len(rows)).reshape(-1, 2), seed=0)
        # Eagerly (jax.disable_jit): the full real table's step_with_public compiles above 26 GiB on CPU; op by op it runs in about a
        # minute per step under 1 GiB (the same engine code, not compiled).
        cls.lane = {name: b for b, name in enumerate(BATTLES)};cls.engine = [];cls.invalid = []
        with jax.disable_jit():
            init = transition.initialize_with_public(state, cls.table);state, trackers = init.state, init.trackers
            for k in range(max(len(script) for _, _, script, _ in BATTLES.values())):   # the scripts are equally long
                actions = np.asarray([[cls.engine_action(script[k][s], int(np.asarray(state.active)[b, s])) for s in (0, 1)]
                    for b, (_, _, script, _) in enumerate(BATTLES.values())], np.int32)
                out = transition.step_with_public(state, actions, cls.table, trackers, chance_inputs=cls.chance(len(BATTLES)))
                state, trackers = out.step.state, out.trackers;cls.invalid.append(np.asarray(out.step.invalid_actions))
                cls.engine.append(jax.device_get((state, trackers)))   # host copies, both lanes, after every step
        # The constructor path at each battle's last decision, both owners.
        cls.roots = {(name, owner): cls.construct(name, owner) for name in BATTLES for owner in (0, 1)}

    @classmethod
    def engine_action(cls, text, active):
        from gpu_engine.effect_probe import engine_action
        return engine_action(text, active)

    @classmethod
    def chance(cls, n):
        """gpu_engine/effect_probe's neutral tape (hits, no crit, no secondary, no full paralysis) for `n` lanes."""
        import jax.numpy as j
        from gpu_engine import transition
        from gpu_engine.effect_probe import DEFAULT_AUX, DEFAULT_TAPE
        tile = lambda value, dtype=j.int32: j.asarray(np.broadcast_to(np.asarray(value), (n,) + np.shape(value)), dtype)
        return transition.ChanceInputs(draws=tile(DEFAULT_TAPE['draws']), secondary=tile(DEFAULT_TAPE['secondary']), sleep_talk=tile(DEFAULT_TAPE['sleep_talk']),
            phazing=tile(DEFAULT_AUX['phazing']), entry_ties=tile(DEFAULT_AUX['entry_ties']), initial_tie=tile(DEFAULT_AUX['initial_tie']),
            effect=tile(DEFAULT_TAPE['effect']))

    @classmethod
    def construct(cls, name, owner):
        """dict(adapter, root, raw (from_b2's sampled worlds), done (completed), facts, lines) at battle `name`'s last decision."""
        from pokezero.recorded_encoder_runner import JaxRecordedAdapter
        from pokezero.v5_contract import DecisionAddress
        from gpu_engine import phase_d_live_worlds as live
        last = cls.showdown[name][-1];lines = tuple(last['channels'][owner])
        adapter = JaxRecordedAdapter(cls.catalog, owner_side=owner, compiled_functions=cls.functions)
        adapter.reset(stream_id=1, battle_id=f'{name}-{owner}');adapter.feed(lines)
        root = live.from_b2(adapter, last['requests'][owner], cls.compiled, cls.metadata, address=DecisionAddress(1, len(cls.showdown[name]), owner))
        made = live.sample_worlds(root, cls.compiled, cls.materializer, count=WORLDS, seed=9)
        facts = cls.cw.line_facts(lines, owner)
        done = [cls.cw.complete_world(w, cls.tables, owner, facts=facts, root=root.public) for w in made.worlds]
        return dict(adapter=adapter, root=root, raw=made.worlds, done=done, facts=facts, lines=lines)

    # References -------------------------------------------------------------------------------------------------------------------
    def engine_refs(self, name):
        """(knocked, ledger) of the engine at battle `name`'s last decision: per side {species: item_knocked} over its party and
        {species: (item_mutated, item_original)} over the device carry's public slots."""
        (state, trackers), b = self.engine[-1], self.lane[name];species = self.metadata['vocabulary']['species']
        knocked = [dict(zip(names(species, np.asarray(state.species)[b, side]), map(bool, np.asarray(state.item_knocked)[b, side]))) for side in (0, 1)]
        f = trackers.mon_features;ledger = []
        for side in (0, 1):
            ledger.append({sp: (int(np.asarray(f['item_mutated'])[b, side, k]), int(np.asarray(f['item_original'])[b, side, k]))
                for k, sp in enumerate(names(species, np.asarray(trackers.species)[b, side])) if sp})
        return [{k: v for k, v in side.items() if k} for side in knocked], ledger

    def b2_ledger(self, built):
        """{species: (mutated, original as the engine item id)} per side from B2's own fold (public_items: mutated, original)."""
        state = built['adapter'].state;items = state.context.items;engine_items = [names(self.metadata['vocabulary']['items'], [i])[0]
            for i in range(len(self.metadata['vocabulary']['items']))]
        out = []
        for side in (0, 1):
            row = {}
            for k, sp in enumerate(names(self.catalog.species.names, np.asarray(state.mons.species)[0, side])):
                if not sp:continue
                original = names(self.catalog.items.names, [np.asarray(items.original)[0, side, k]])[0]
                row[sp] = (int(bool(np.asarray(items.mutated)[0, side, k])), engine_items.index(original) if original else 0)
            out.append(row)
        return out

    def mismatches(self, world, name, built):
        """Every filled cell of `world` (a state and carry of battle `name`'s constructor root `built`) that differs from reference (a)
        (the engine's item_knocked; the device hooks' ledger) or (b) (B2's ledger), joined by species; [] when both agree."""
        species = self.metadata['vocabulary']['species'];knocked, device = self.engine_refs(name);b2 = self.b2_ledger(built);out = []
        state, public = world.state, world.public
        for side in (0, 1):
            for p, sp in enumerate(names(species, np.asarray(state.species)[0, side])):
                if not sp:continue
                got = bool(np.asarray(state.item_knocked)[0, side, p]);want = knocked[side].get(sp, False)   # a sampled member: never knocked
                if got != want:out.append(f'state.item_knocked p{side + 1} {sp}: {got}, the engine {want}')
            f = public.mon_features
            for k, sp in enumerate(names(species, np.asarray(public.species)[0, side])):
                if not sp:continue
                got = (int(np.asarray(f['item_mutated'])[0, side, k]), int(np.asarray(f['item_original'])[0, side, k]))
                for ref, label in ((device, 'device'), (b2, 'B2')):
                    if got != ref[side].get(sp):out.append(f'ledger p{side + 1} {sp}: {got}, {label} {ref[side].get(sp)}')
        return out

    # Tests ------------------------------------------------------------------------------------------------------------------------
    def test_the_engine_follows_showdowns_battle(self):
        """Reference (a) is the same game: at every step the engine's actives, items and item_knocked are Showdown's (itemKnockedOff),
        nobody faints and no action is invalid; the Knock Offs and the Sticky Hold block are on Showdown's own lines."""
        items = self.metadata['vocabulary']['items'];self.assertEqual(self.vocabulary, self.metadata['vocabulary'])
        self.assertFalse(np.asarray(self.invalid).any())
        for name, b in self.lane.items():
            self.assertEqual(len(self.showdown[name]), len(self.engine))
            for k, (sd, (state, _)) in enumerate(zip(self.showdown[name], self.engine)):
                with self.subTest(battle=name, step=k):
                    self.assertEqual(np.asarray(state.active)[b].tolist(), sd['active']);self.assertFalse(bool(np.asarray(state.done)[b]))
                    for side in (0, 1):
                        self.assertEqual(np.asarray(state.item)[b, side].tolist(), [items.index(m['item']) if m['item'] else 0 for m in sd['mons'][side]])
                        self.assertEqual(np.asarray(state.item_knocked)[b, side].tolist(), [m['knocked'] for m in sd['mons'][side]])
                        self.assertTrue(all(m['hp'] > 0 and not m['fainted'] for m in sd['mons'][side]));self.assertTrue((np.asarray(state.hp)[b, side] > 0).all())
        lines = '\n'.join(self.showdown['knock'][-1]['channels'][0])
        self.assertIn('|-enditem|p2a: Snorlax|Leftovers|[from] move: Knock Off|[of] p1a: Sableye', lines)
        self.assertEqual(lines.count('|switch|p2a: Snorlax|'), 2)   # out at turn 2, back at turn 3
        lines = '\n'.join(self.showdown['sticky'][-1]['channels'][1])
        self.assertIn('|-activate|p2a: Muk|ability: Sticky Hold', lines)
        self.assertIn('|-enditem|p1a: Sableye|Leftovers|[from] move: Knock Off|[of] p2a: Hariyama', lines)
        self.assertNotIn('|-enditem|p2a: Muk|', lines)
        last = {name: steps[-1]['mons'] for name, steps in self.showdown.items()}
        self.assertEqual([(m['species'], m['knocked']) for m in last['knock'][1] if m['knocked']], [('snorlax', True)])
        self.assertEqual([(m['species'], m['knocked']) for m in last['sticky'][0] if m['knocked']], [('sableye', True)])
        self.assertEqual((last['sticky'][1][0]['item'], last['sticky'][1][0]['knocked']), ('leftovers', False))   # Muk keeps it

    def test_the_lines_facts_name_the_knocked_members(self):
        for owner in (0, 1):
            with self.subTest(owner=owner):
                self.assertEqual(self.roots[('knock', owner)]['facts'].knocked, ((), (('snorlax', 'leftovers'),)))
                self.assertEqual(self.roots[('sticky', owner)]['facts'].knocked, ((('sableye', 'leftovers'),), ()))   # not Muk

    def test_completed_worlds_equal_the_device_path_and_b2s_ledger(self):
        """Every completed world of every root, on both fills, equals (a) and (b): the knocked member (the owner's own or the peer's,
        Snorlax back in after its switch, Sableye) item_knocked and mutated with its Leftovers as the original; the Sticky Hold Muk and
        every other member untouched. The two references agree with each other, and a second completion pass is a no-op."""
        leftovers = self.metadata['vocabulary']['items'].index('leftovers')
        for (name, owner), built in self.roots.items():
            with self.subTest(battle=name, owner=owner):
                knocked, device = self.engine_refs(name);b2 = self.b2_ledger(built)
                victim, side = ('snorlax', 1) if name == 'knock' else ('sableye', 0)
                self.assertEqual((knocked[side][victim], device[side][victim], b2[side][victim]), (True, (1, leftovers), (1, leftovers)))
                for s in (0, 1):self.assertEqual({sp: v for sp, v in b2[s].items()}, {sp: device[s][sp] for sp in b2[s]})   # (a) = (b)
                if name == 'sticky':
                    self.assertEqual((knocked[1]['muk'], device[1]['muk'], b2[1]['muk']), (False, (0, 0), (0, 0)))
                self.assertEqual(len(built['done']), WORLDS);self.assertTrue(all(w.state is not None for w in built['done']))
                for w in built['done']:
                    self.assertEqual(self.mismatches(w, name, built), [], f'world {w.index}')
                    self.assertIs(self.cw.complete_world(w, self.tables, owner, facts=built['facts'], root=built['root'].public), w)

    def test_negative_control_a_planted_wrong_fill_fails(self):
        """Without the fills (from_b2's raw worlds) and with each planted wrong fill, the same comparison fails on every root's
        worlds: a wrong original item, the ledger written one slot over, the state flag dropped; and a fill planted on the Sticky Hold
        Muk fails it too (its state fill refuses outright: Muk still holds its Leftovers)."""
        import jax.numpy as j
        cw = self.cw;real_fill = cw.fill_item_ledger
        def other_item(public, facts, tables):
            out = real_fill(public, facts, tables);f = dict(out.mon_features);o = np.asarray(f['item_original']).copy();o[o > 0] += 1
            return out._replace(mon_features={**f, 'item_original': j.asarray(o)})
        def next_slot(public, facts, tables):
            out = real_fill(public, facts, tables);f = dict(out.mon_features)
            moved = {k: j.asarray(np.roll(np.asarray(f[k]) - np.asarray(public.mon_features[k]), 1, axis=-1) + np.asarray(public.mon_features[k]))
                for k in ('item_mutated', 'item_original')}
            return out._replace(mon_features={**f, **moved})
        plants = dict(other_item=('fill_item_ledger', other_item), next_slot=('fill_item_ledger', next_slot),
            no_state_flag=('fix_knocked_items', lambda state, owner, facts, tables: state))
        for (name, owner), built in self.roots.items():
            with self.subTest(battle=name, owner=owner, plant='raw'):
                self.assertTrue(all(self.mismatches(w, name, built) for w in built['raw']))   # the gap the fills close
            for plant, (target, wrong) in plants.items():
                with self.subTest(battle=name, owner=owner, plant=plant), patch.object(cw, target, wrong):
                    planted = [cw.complete_world(w, self.tables, owner, facts=built['facts'], root=built['root'].public) for w in built['raw']]
                    self.assertTrue(all(self.mismatches(w, name, built) for w in planted))
        for owner in (0, 1):
            built = self.roots[('sticky', owner)];facts = replace(built['facts'], knocked=(built['facts'].knocked[0], (('muk', 'leftovers'),)))
            with self.subTest(owner=owner, plant='sticky_hold_member'):
                world = built['done'][0];planted = replace(world, public=cw.fill_item_ledger(world.public, facts, self.tables))
                self.assertTrue(any('muk' in m for m in self.mismatches(planted, 'sticky', built)))
                with self.assertRaisesRegex(cw.KnockedItemUnavailable, 'muk: knocked off on the lines, but this world holds item'):
                    cw.fix_knocked_items(world.state, owner, facts, self.tables)


if __name__ == '__main__':unittest.main()
