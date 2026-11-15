"""Fidelity panel runner, merger and first-divergence attribution (diagnostic tooling, outside the engine tree).

Mirrors gpu_engine.fidelity.run game for game (same roster rule, keys, action law, chance tape, comparator and stop
statuses), but over any contiguous game range and in shards, so a panel can run in parallel and a fresh sample can be
drawn beyond the fixed 200-pair panel (games 50200..50399). `run --first-game 50200 --games 200` reproduces
fidelity.run's per-game results; `check-canonical` compares a merged panel with a fidelity.run result file.

  run         --tree T --first-game G --games N [--shard i/k] --out OUT.json [--diag]   (imports the engine: CPU pod only)
  merge       OUT.json SHARD.json...                                                  (json only)
  attribute   PANEL.json --pool POOL.npz --out TABLE.json [--md TABLE.md]             (numpy + json only)
  check-canonical PANEL.json FIDELITY.json                                            (json only)

--diag launches Showdown through a launcher that installs the tree's own chance hooks unchanged, then records, for an
UNRESOLVED role only, the JS stack, the turn, the current event/effect/move and the battle log tail. It never changes a
draw or a status; it only adds fields to that audit row.
"""
import argparse
import collections
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

DIAG_LAUNCHER = r"""
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const require=createRequire(import.meta.url);
const TREE=%(tree)s;
const at=process.argv.indexOf('--showdown-root');
if(at<0)throw Error('explicit pinned Showdown root required');
const root=process.argv[at+1],dir=path.dirname(process.argv[1]);
const control=path.join(dir,'chance.json'),auditPath=path.join(dir,'chance-audit.jsonl'),initial=path.join(dir,'initializing.json');
const {Battle}=require(path.join(root,'dist/sim/battle.js'));
const {PRNG}=require(path.join(root,'dist/sim/prng.js'));
let last=null;
const audit=row=>{
  if(row.kind==='UNRESOLVED'||row.kind==='INITIAL_RANDOM'){
    row.diag_stack=new Error().stack.split('\n').slice(2,18).map(s=>s.trim());
    if(last){try{row.diag=dict(last);}catch(e){row.diag={error:String(e)};}}
  }
  fs.appendFileSync(auditPath,JSON.stringify(row)+'\n');
};
function dict(b){
  const name=x=>x?(x.id||x.name||String(x)):null;
  return {turn:b.turn,event:name(b.event),effect:name(b.effect),activeMove:name(b.activeMove),
    activePokemon:b.activePokemon?String(b.activePokemon):null,log:b.log.slice(-30),
    actives:b.sides.map(s=>s.active.map(p=>p?{species:p.species.id,ability:p.ability,item:p.item,hp:p.hp,speed:p.getStat('spe',false,false),status:p.status}:null))};
}
require(path.join(TREE,'gpu_engine/chance_hooks.cjs')).install(Battle,PRNG,
  ()=>fs.existsSync(control)?JSON.parse(fs.readFileSync(control,'utf8')):null,
  audit,
  ()=>fs.existsSync(initial)?JSON.parse(fs.readFileSync(initial,'utf8')):null,
  path.join(root,'dist/sim/pokemon.js'));
// Only entry points are wrapped: the hooks read their immediate caller's stack frame (caller() = stack[3]), so no
// wrapper may sit between a hooked random/sample/randomChance and its caller.
for(const m of ['makeChoices','setPlayer','start']){
  const f=Battle.prototype[m];if(typeof f!=='function')continue;
  Battle.prototype[m]=function(...a){last=this;return f.apply(this,a);};
}
await import(pathToFileURL(path.join(TREE,'scripts/battle_bridge.mjs')).href);
"""


def _engine(tree):
    sys.path[:0] = [str(tree), str(tree / 'src')]
    import jax, jax.numpy as j, numpy as np  # noqa: E401
    from gpu_engine import fidelity as F, transition
    return jax, j, np, F, transition


def roster(F, pool, metadata, showdown, node, first_game, games):
    """fidelity.reconstruct for any contiguous game range: regenerate every team from its seed and check every stored field."""
    start, count = 2 * first_game, 2 * games
    if start < 0 or start + count > len(pool['seeds']):raise ValueError('game range outside the pool')
    seeds = pool['seeds'][start:start + count].tolist()
    if seeds != list(range(seeds[0], seeds[0] + count)):raise ValueError('contiguous original seeds required')
    raw = subprocess.check_output([node, str(F.ROOT / 'gpu_engine/generate_teams.cjs'), str(showdown), str(count), str(seeds[0]),
                                   metadata['source_commit_label']], text=True)
    records = [json.loads(x) for x in raw.splitlines()];header = records[0]
    for key in ('compiled_tree', 'rng_dependency', 'vocabulary', 'node_version'):
        if header[key] != metadata[key]:raise ValueError(f'generator binding differs: {key}')
    if len(records) != count + 2 or records[-1] != {'kind': 'terminal', 'team_count': count}:raise ValueError('incomplete regeneration')
    fields = ('species', 'level', 'types', 'stats', 'moves', 'pp', 'item', 'ability', 'nature', 'ivs', 'evs', 'hidden_power_type',
              'hidden_power_power')
    for index, row in enumerate(records[1:-1]):
        if row['kind'] != 'team' or row['index'] != index or row['seed'] != seeds[index]:raise ValueError('regenerated roster differs')
        for field in fields:
            if [mon[field] for mon in row['members']] != pool[field][start + index].tolist():
                raise ValueError(f'accepted tensor differs: team {start + index}/{field}')
    packed = json.loads(subprocess.check_output([node, str(F.ROOT / 'gpu_engine/regenerate_packed.cjs'), str(showdown)],
                                                input=json.dumps(seeds), text=True))
    if [r['seed'] for r in packed] != seeds or any(not r['packed'] for r in packed):raise ValueError('packed roster differs')
    return packed, dict(first_game=first_game, games=games, team_indices=[start, start + count], original_seed_range=[seeds[0], seeds[-1]],
                        regeneration_sha256=hashlib.sha256(raw.encode()).hexdigest())


def run(a):
    tree = Path(a.tree).resolve();jax, j, np, F, transition = _engine(tree)
    blob = Path(a.team_pool).read_bytes();pool_sha = hashlib.sha256(blob).hexdigest()
    if pool_sha != a.expected_pool_sha:raise ValueError('accepted pool digest differs')
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        metadata = json.loads(archive['metadata_utf8'].tobytes());pool = {k: archive[k] for k in archive.files if k != 'metadata_utf8'}
    git = lambda *args: subprocess.check_output(['git', '-C', str(tree), *args], text=True).strip()
    bound = ('gpu_engine', 'tests', 'src', 'scripts')
    if git('status', '--porcelain', '--', *bound):raise RuntimeError('dirty diagnostic source')
    source = git('rev-parse', 'HEAD')
    i, k = (int(x) for x in a.shard.split('/'))
    per = -(-a.games // k);lo = a.first_game + i * per;hi = min(a.first_game + a.games, lo + per)
    if lo >= hi:raise ValueError('empty shard')
    packed, ros = roster(F, pool, metadata, Path(a.showdown_root), a.node, lo, hi - lo)
    table = transition.tables_from_metadata(metadata);F.reference_actions.table = table
    n = hi - lo
    base = transition.state_from_pool(pool, np.arange(2 * lo, 2 * hi).reshape(n, 2), seed=719)
    base = base._replace(keys=jax.vmap(lambda g: jax.random.fold_in(jax.random.PRNGKey(719), g))(j.arange(lo, hi)))
    base = jax.block_until_ready(transition.initialize(base, table));advance = jax.jit(lambda s, act: transition.step(s, act, table))
    header = dict(schema='pokezero.fidelity_panel_shard.v1', source_commit=source, pool_sha256=pool_sha, roster=ros, panel=dict(
        first_game=a.first_game, games=a.games, shard=a.shard, game_range=[lo, hi]), format_id=a.format, chance_tape_schema=F.TAPE_SCHEMA,
        mirrors='gpu_engine.fidelity.run (per-game body, statuses and seeds)', diag=bool(a.diag), jax=jax.__version__,
        backend=jax.default_backend())
    out = Path(a.out);rows = []
    with out.with_suffix('.jsonl').open('x') as stream:
        stream.write(json.dumps(dict(kind='ENTRY', **header)) + '\n');stream.flush()
        for offset in range(n):
            game = lo + offset;state = jax.tree.map(lambda x: x[offset:offset + 1], base)
            original = [[metadata['vocabulary']['species'][int(x)] for x in state.species[0, s]] for s in range(2)]
            row = dict(game_id=game, team_ids=[2 * game, 2 * game + 1], steps=[], status='REQUEST_CAP', first_divergence=None)
            rng = random.Random(f'gpu-fidelity-actions-v1:{game}')
            with tempfile.TemporaryDirectory(prefix='pokezero-fidelity-') as folder:
                folder = Path(folder);bridge = folder / 'bridge.mjs'
                if a.diag:bridge.write_text(DIAG_LAUNCHER % dict(tree=json.dumps(str(tree))))
                else:bridge.write_text('await import(' + json.dumps((tree / 'gpu_engine/fidelity_bridge.mjs').as_uri()) + ');\n')
                control = folder / 'chance.json';audit = folder / 'chance-audit.jsonl';initializing = folder / 'initializing.json'
                gender_sensitive = any(metadata['vocabulary']['moves'][int(x)] == 'attract' for x in np.asarray(state.moves).flat)
                gender_sensitive |= any(metadata['vocabulary']['abilities'][int(x)] == 'cutecharm' for x in np.asarray(state.ability).flat)
                initial=F.initial_chance(state,allow_inert_gender=not gender_sensitive) if hasattr(F,'initial_chance') else dict(allow_inert_gender=not gender_sensitive)
                initializing.write_text(json.dumps(initial))
                env = F.LocalShowdownEnv(F.LocalShowdownConfig(showdown_root=Path(a.showdown_root), node_binary=a.node, bridge_path=bridge,
                    feature_masks=F.ObservationFeatureMasks(transition_token_budget=0)))
                try:
                    env.reset_with_start_override(seed=670000000 + (game - F.FIRST_GAME), start_override=F.BattleStartOverride(
                        player_teams={'p1': packed[2 * offset]['packed'], 'p2': packed[2 * offset + 1]['packed']},
                        format_id=a.format, observation_format_id='gen3customgame'))
                    initializing.unlink()
                    initial_events = [json.loads(line) for line in audit.read_text().splitlines()] if audit.exists() else []
                    row['initial_chance_audit'] = initial_events
                    if audit.exists():audit.unlink()
                    snapshot = env.snapshot();diff = F.compare(state, snapshot, original, metadata['vocabulary'], initial=True)
                    if any(e['kind'] == 'INITIAL_RANDOM' for e in initial_events):
                        row.update(status='UNRESOLVED_INITIAL_CHANCE', first_divergence=diff or None)
                    elif diff:row.update(status='INITIAL_CONTRACT_MISMATCH', first_divergence=diff)
                    else:
                        saved_effect = None
                        for index in range(200):
                            native, actions, diff, maps = F.reference_actions(env, state, snapshot, original, metadata['vocabulary'], rng)
                            if diff:row.update(status='LEGAL_ACTION_MISMATCH', first_divergence=diff);break
                            if not native:row['status'] = 'NO_REFERENCE_REQUEST';break
                            row.pop('post_boundary', None);row.pop('native_log_tail', None)
                            row['pre_boundary'] = F.boundary_projection(state, snapshot, original, metadata['vocabulary'])
                            tape = F.chance_tape(state, game, index, saved_effect);control.write_text(json.dumps(tape))
                            if not bool(np.asarray(state.pending_residual)[0]):saved_effect = tape['effect']
                            step = dict(index=index, showdown_actions=native, gpu_actions=actions.tolist(), identity_maps=maps, chance=tape)
                            failure = None
                            try:env.step(native)
                            except Exception as error:failure = str(error)
                            finally:control.unlink(missing_ok=True)
                            events = [json.loads(line) for line in audit.read_text().splitlines()] if audit.exists() else []
                            step['chance_audit'] = events;step['reference_error'] = failure
                            if audit.exists():audit.unlink()
                            row['steps'].append(step)
                            if any(e['kind'] == 'UNRESOLVED' for e in events):row['status'] = 'UNRESOLVED_CHANCE_ROLE';break
                            if failure:row['status'] = 'HARNESS_ERROR';break
                            result = jax.block_until_ready(advance(state, actions));state = result.state
                            if bool(np.asarray(result.invalid_actions)[0]):row['status'] = 'GPU_INVALID_ACTION';break
                            snapshot = env.snapshot();diff = F.compare(state, snapshot, original, metadata['vocabulary'])
                            row['post_boundary'] = F.boundary_projection(state, snapshot, original, metadata['vocabulary'])
                            row['native_log_tail'] = snapshot.bridge_snapshot['battle'].get('log', [])[-80:]
                            if diff:row.update(status='STATE_MISMATCH', first_divergence=diff);break
                            status, diff = F.completion(state, env.terminal())
                            if status == 'MATCHED_THROUGH_TURN20':
                                _, _, boundary_diff, _ = F.reference_actions(env, state, snapshot, original, metadata['vocabulary'], rng, choose=False)
                                if boundary_diff:status, diff = 'LEGAL_ACTION_MISMATCH', boundary_diff
                            if status:row.update(status=status, first_divergence=diff);break
                except Exception as error:row.update(status='HARNESS_ERROR', error=str(error))
                finally:
                    try:env.close()
                    except Exception as error:row.update(status='HARNESS_ERROR', close_error=str(error))
            row['gpu_ordinary_turns'] = int(state.turn[0]);rows.append(row)
            stream.write(json.dumps(dict(kind='GAME', **row)) + '\n');stream.flush()
            print(json.dumps(dict(game_id=game, status=row['status'], ordinary_turns=row['gpu_ordinary_turns'])), flush=True)
    if git('rev-parse', 'HEAD') != source or git('status', '--porcelain', '--', *bound):raise RuntimeError('bound source changed during trace')
    counts = collections.Counter(r['status'] for r in rows)
    with out.open('x') as target:
        json.dump(dict(**header, status_counts=dict(counts), rows=rows), target, sort_keys=True);target.write('\n')


MATCHED = ('MATCHED_TERMINAL', 'MATCHED_THROUGH_TURN20')


def merge(a):
    shards = [json.loads(Path(p).read_text()) for p in a.shards]
    panels = {json.dumps(s['panel'] | dict(shard=None, game_range=None), sort_keys=True) for s in shards}
    commits = {s['source_commit'] for s in shards}
    if len(panels) != 1 or len(commits) != 1:raise ValueError(f'shards disagree: {panels} {commits}')
    for key in ('pool_sha256','format_id','chance_tape_schema'):
        if len({s[key] for s in shards})!=1:raise ValueError(f'shards disagree: {key}')
    panel = shards[0]['panel'];rows = sorted((r for s in shards for r in s['rows']), key=lambda r: r['game_id'])
    ids = [r['game_id'] for r in rows];want = list(range(panel['first_game'], panel['first_game'] + panel['games']))
    if ids != want:raise ValueError(f'missing or repeated games: {sorted(set(want) - set(ids))[:10]} {len(ids)} vs {len(want)}')
    counts = collections.Counter(r['status'] for r in rows);matched = sum(counts[s] for s in MATCHED)
    out = dict(schema='pokezero.fidelity_panel.v1', source_commit=commits.pop(), pool_sha256=shards[0]['pool_sha256'],
               panel=dict(first_game=panel['first_game'], games=panel['games']), format_id=shards[0]['format_id'],
               chance_tape_schema=shards[0]['chance_tape_schema'], diag=any(s.get('diag') for s in shards),
               status_counts=dict(counts), matched=matched, games=len(rows), rows=rows)
    with open(a.out, 'x') as f:json.dump(out, f, sort_keys=True);f.write('\n')
    print(f"matched {matched}/{len(rows)}", dict(counts))


def check_canonical(a):
    p = json.load(open(a.panel));f = json.load(open(a.fidelity))
    fr = {r['game_id']: r for r in f['rows']};bad = []
    for r in p['rows']:
        x = fr.get(r['game_id'])
        if x is None:bad.append((r['game_id'], 'absent'));continue
        if (r['status'], len(r['steps']), r['gpu_ordinary_turns'], r.get('first_divergence')) != \
                (x['status'], len(x['steps']), x['gpu_ordinary_turns'], x.get('first_divergence')):
            bad.append((r['game_id'], r['status'], len(r['steps']), x['status'], len(x['steps'])))
    print(f"{len(p['rows']) - len(bad)}/{len(p['rows'])} games identical to the fidelity.run result (status, steps, turns, first divergence)")
    for b in bad[:20]:print('DIFF', b)
    return 1 if bad else 0


# ---- attribution -------------------------------------------------------------------------------------------------------
# The engine's tie columns (transition.step_with_chance: residual tie, switch_coin, entry ties, move tie) that a tape row
# would have to name; chance_hooks.cjs maps only the move tie (draws[0][0]) and the Wrap / Perish pairs (effect[s][3]).
RESIDUAL_KIND = {'leftovers': 'heal', 'petayaberry': 'heal', 'salacberry': 'heal', 'liechiberry': 'heal', 'ganlonberry': 'heal',
                 'apicotberry': 'heal', 'speedboost': 'ability', 'leechseed': 'seed', 'brn': 'status', 'psn': 'status', 'tox': 'status',
                 'encore': 'encore'}
TIE_COLUMN = {'heal': 'draws[0][9]', 'ability': 'draws[1][0]', 'seed': 'draws[0][10]', 'status': 'draws[1][9]', 'encore': 'draws[1][11]'}


def classify(row, teams, vocab_has):
    """(cause, class, detail) for an unmatched row. class: a unmodelled effect, b known open bug, c convention row (an engine
    chance input exists or the order is inert, but the comparator's tape has no row for it), d harness/comparator noise."""
    status = row['status']
    if status == 'UNRESOLVED_INITIAL_CHANCE':
        ev = [e for e in row.get('initial_chance_audit', []) if e['kind'] == 'INITIAL_RANDOM']
        frames = ' '.join(s for e in ev for s in e.get('stack', []))
        if 'new Pokemon' in frames:
            cc = sorted({m['species'] for t in teams for m in t if m['ability'] == 'cutecharm'})
            at = sorted({m['species'] for t in teams for m in t if 'attract' in m['moves']})
            if cc and not at:return ('Cute Charm (gender read at team construction)', 'a',
                                     f"cutecharm on {', '.join(cc)}; the comparator stops at the gender draw (gender.py: no Attract/Cute Charm)")
            return ('Attract (gender read at team construction)', 'a', f"attract on {', '.join(at)}; cutecharm {cc}")
        if 'randomFoe' in frames:
            return ('Trace at lead: randomFoe over one foe', 'c', 'sample over a 1-element list in singles (inert; the probes force it)')
        if 'insertChoice' in frames or 'speedSort' in frames:
            return ('lead switch-in Speed tie', 'c', 'initial order tie; the engine reads ChanceInputs.initial_tie, the tape has no initial row')
        return ('initial randomness: ' + frames[:80], 'd', '')
    if status == 'UNRESOLVED_CHANCE_ROLE':
        st = row['steps'][-1];u = [e for e in st['chance_audit'] if e['kind'] == 'UNRESOLVED'][0]
        role, detail = u.get('role'), u.get('detail');stack = ' '.join(u.get('diag_stack') or [])
        diag = u.get('diag') or {}
        if role == 'TIE_ORDER' and isinstance(detail, list):
            kinds = [RESIDUAL_KIND.get(x) for x in detail]
            if all(kinds) and len(set(kinds)) == 1:
                kind = kinds[0]
                return (f'residual {kind} tie ({"/".join(detail)}) at equal Speed', 'c',
                        f'engine tie bit {TIE_COLUMN[kind]}; no tape row')
            if all(kinds):return (f'residual tie across kinds ({"/".join(detail)})', 'd', 'unclassified: mixed subOrder kinds need a proof')
            if detail == ['switch', 'switch']:
                return ('both players switch at equal Speed', 'c', 'engine switch_coin draws[0][11]; no tape row')
            if set(detail) <= {'choicelock', 'encore'}:
                return ('Choice lock / Encore handler tie', 'c', 'same-mon DisableMove handlers; order-inert, no inert row')
            if detail == ['Pokemon', 'Pokemon']:
                acts = [a[0] for a in diag.get('actives', []) if a and a[0]]
                who = ', '.join(f"{a.get('species')} ({a.get('ability')}/{a.get('item') or '-'})" for a in acts)
                if 'eachEvent' in stack and 'runAction' in stack:
                    return ("eachEvent('Update') Pokemon-order tie (equal Speed, a handler present)", 'd',
                            f'unclassified: actual handlers need a commutation proof; actives {who}')
                if 'insertChoice' in stack:
                    return ('switch-in insertion Speed tie (insertChoice)', 'c', f'engine entry tie bit; actives {who}')
                return ('Pokemon-order tie', 'd', f'unclassified: actives {who}; {stack[:120]}')
            return (f'tie {detail}', 'd', 'unclassified: no reviewed mapping or commutation proof')
        if role == 'MISSING_ACTOR':
            if 'randomFoe' in stack:return ('Trace on a switch-in: randomFoe over one foe', 'c',
                                            'sample over the single foe in singles: inert, but the tape has no row (the probes force it)')
            if 'insertChoice' in stack:return ('switch-in insertion Speed tie (insertChoice)', 'c', 'engine entry tie bit; no tape row')
            if 'onResidual' in str(detail) or 'onResidual' in stack:
                acts = [a[0] for a in diag.get('actives', []) if a and a[0]]
                if any(a.get('ability') == 'shedskin' and a.get('status') for a in acts):
                    return ('Shed Skin (residual cure roll)', 'a', 'unmodelled in the engine; data/abilities.ts shedskin randomChance(33,100)')
                return ('residual random with no active move', 'd', 'unclassified: '+stack[:160])
            if 'insertChoice' in str(detail) or 'insertChoice' in stack:
                return ('switch insertion Speed tie (insertChoice)', 'c', 'engine entry tie; no tape row')
            return ('sample with no active move: ' + (diag.get('effect') or '?'), 'd', 'unclassified: '+stack[:160])
        if role == 'SAMPLE' and (detail or {}).get('move') in ('roar','whirlwind'):
            return (f"{(detail or {}).get('move')} target pick over {(detail or {}).get('length')}", 'c',
                    'engine phazing[s] drag roll; no tape row')
        return (f'{role} {str(detail)[:60]}', 'd', 'unclassified: '+stack[:160])
    if status == 'LEGAL_ACTION_MISMATCH' and 'transform' in json.dumps(row.get('first_divergence')):
        return ('Transform (move identities after -transform)', 'a', 'open PR #1970')
    if status in ('STATE_MISMATCH', 'TERMINAL_MISMATCH'):
        return state_cause(row, teams)
    return (status, 'd', json.dumps(row.get('first_divergence'))[:120])


def state_cause(row, teams):
    """Heuristic mechanic for a state mismatch from the failing step's Showdown log (the last turn) and the teams."""
    log = row.get('native_log_tail') or []
    turn = max([i for i, l in enumerate(log) if l.startswith('|turn|')][-2:-1] or [0])
    last = [l for l in log[turn:] if l]
    fd = json.dumps(row.get('first_divergence'))[:100]
    if any(l.startswith('|-transform|') for l in log):return ('Transform', 'a', 'open PR #1970; ' + fd)
    if any('Forecast' in l for l in log):return ('Forecast (Castform type change)', 'a', 'Forecast agent; ' + fd)
    toid = lambda x: ''.join(c for c in x.lower() if c.isalnum())
    for l in last:
        parts = l.split('|')
        if len(parts) > 3 and parts[1] == 'move':
            side = 0 if parts[2].startswith('p1') else 1
            name = toid(parts[2].split(': ', 1)[-1])
            mon = next((m for m in teams[side] if m['species'].startswith(name) or name.startswith(m['species'])), None)
            item = (mon or {}).get('item')
            if item in ('silkscarf', 'twistedspoon'):
                return ({'silkscarf': 'Silk Scarf', 'twistedspoon': 'Twisted Spoon'}[item] + ' (type-boost item)', 'a',
                        f"{parts[3]} by {mon['species']}; damage-mods agent; " + fd)
            if item == 'stick' and any(x.startswith('|-crit|') for x in last):
                return ('Stick (critical-hit stage)', 'a', f"{parts[3]} by {mon['species']}; " + fd)
    return ('state mismatch: ' + fd, 'a', 'unattributed')


def attribute(a):
    import numpy as np
    p = json.load(open(a.panel));pool = np.load(a.pool);meta = json.loads(pool['metadata_utf8'].tobytes());V = meta['vocabulary']
    def team(t):
        return [dict(species=V['species'][int(pool['species'][t, m])], ability=V['abilities'][int(pool['ability'][t, m])],
                     item=V['items'][int(pool['item'][t, m])] if int(pool['item'][t, m]) else '',
                     moves=[V['moves'][int(x)] for x in pool['moves'][t, m] if int(x)]) for m in range(6)]
    out = []
    for r in p['rows']:
        if r['status'] in MATCHED:continue
        teams = [team(t) for t in r['team_ids']]
        cause, cls, detail = classify(r, teams, None)
        log = (r.get('pre_boundary') or {}).get('showdown', {}).get('log_tail') or []
        turns = [int(l.split('|')[2]) for l in log if l.startswith('|turn|')]
        out.append(dict(game_id=r['game_id'], status=r['status'], decision=(r['steps'][-1]['index'] if r['steps'] else None),
                        showdown_turn=(turns[-1] if turns else 0), gpu_turns=r['gpu_ordinary_turns'], cause=cause, cls=cls, detail=detail))
    counts = collections.Counter((x['cause'], x['cls']) for x in out)
    res = dict(source_commit=p['source_commit'], panel=p['panel'], matched=p['matched'], games=p['games'], unmatched=len(out),
               rows=out, causes=[dict(cause=c, cls=k, games=n) for (c, k), n in counts.most_common()])
    with open(a.out, 'x') as f:json.dump(res, f, indent=1)
    print(f"matched {p['matched']}/{p['games']}; unmatched {len(out)}")
    for (c, k), n in counts.most_common():print(f'{n:4d}  ({k})  {c}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0]);sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run');r.add_argument('--tree', required=True);r.add_argument('--first-game', type=int, default=50200)
    r.add_argument('--games', type=int, default=200);r.add_argument('--shard', default='0/1');r.add_argument('--out', required=True)
    r.add_argument('--team-pool', required=True);r.add_argument('--expected-pool-sha', required=True)
    r.add_argument('--showdown-root', required=True);r.add_argument('--node', required=True)
    r.add_argument('--format', default='gen3customgame@@@sleepclausemod');r.add_argument('--diag', action='store_true')
    m = sub.add_parser('merge');m.add_argument('out');m.add_argument('shards', nargs='+')
    t = sub.add_parser('attribute');t.add_argument('panel');t.add_argument('--pool', required=True);t.add_argument('--out', required=True)
    c = sub.add_parser('check-canonical');c.add_argument('panel');c.add_argument('fidelity')
    a = ap.parse_args()
    if a.cmd == 'run':run(a)
    elif a.cmd == 'merge':merge(a)
    elif a.cmd == 'attribute':attribute(a)
    else:raise SystemExit(check_canonical(a))


if __name__ == '__main__':
    main()
