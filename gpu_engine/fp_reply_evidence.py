"""Reply-model evidence on Foul Play's sets, for the definitive RR's reweighted worlds.

usage: python -m gpu_engine.fp_reply_evidence export RUN_DIRECTORY OUT --pi-fp PT [--shard I --shards N] [--limit N]
           [--extra-sets RR/shard-*/rows-*.jsonl] [--ordinals ORDINALS.json]
       python -m gpu_engine.fp_reply_evidence weights --evidence OUT --rows ROWS/rows-*.jsonl --out WEIGHTS.jsonl
           [--drop-battles DROP.json] [--allow-partial]

export: replay each game of a run directory (fp_reply_data.load_game's layout, Foul Play's journal included) as
fp_reply_reweight.replay_game does, scoring Foul Play's public choices with the reply model under every candidate set of its
active Pokémon, and write evidence-<shard>.jsonl: one row per decision of our seat, {ordinal, decision_index, round, mons:
{species: {n, stats_aligned, sets: [[sorted move keys, ability, item, loglik], ...]}}}, the loglik being the summed log q of
the choices made public in earlier rounds (fp_reply_reweight's evidence; -inf is written as null). The reply model is the
π_FP checkpoint (its fp_reply marker's redaction is applied, as in the reweighting test). Games whose battle seed is in π_FP's
training data are refused: the evidence must come from games the model never saw. --extra-sets: the RR re-pricer's rows; each
set a decided root's world drew for a Foul Play Pokémon (with its item, not one removed in public) is scored beside the belief
engine's candidates when that Pokémon is active (the live constructor draws from a pool of generated teams, which can hold
sets the belief engine does not offer). A set's loglik does not depend on the other candidates, and the extra sets are scored
in their own batch, so every candidate's loglik is bit-identical to an export without them. --ordinals: only those games (a JSON
list of ordinals, e.g. the MC gate bank's Foul Play games).

weights: join the evidence with the RR re-pricer's rows. A world's loglik is the sum over Foul Play's Pokémon in that world
with evidence (n > 0) of the loglik of the world's set for it (set_key: move keys, ability, item, as fp_reply_reweight's
variant_key). A world whose set for such a Pokémon is not among its scored sets, or has loglik -inf, has none; every world of
that decision is then left out (the scorer counts the decision as world_weights_missing). Writes {ordinal, decision_index,
world, loglik} rows for depthn_rr_score --world-weights, and WEIGHTS.summary.json with the counts and the join coverage.
Coverage is per world, over every world of every decided root (root.decided false: never scored, counted) outside
--drop-battles: each failure is counted by reason. A world's item is its item at the root; one removed in public ('') joins
the one scored set with the same moves and ability (counted). Unown's letter forms are one species. The
ability in the key is base_ability, so a Trace holder keys on 'trace' whatever it has copied: tracer worlds (some member's
base_ability is trace) are counted, with those whose tracer has evidence at that decision (the join exercised the 'trace'
key) and the tracer members whose current ability differs (a copy at the root). Below 100% coverage nothing but the summary
is written (exit 1) unless --allow-partial.
"""
import argparse
import collections
import glob
import json
import math
import re
import time
from pathlib import Path

from gpu_engine import fp_reply_data as data_mod
from gpu_engine import fp_reply_reweight as rw

SCHEMA = 'v2s-fp-reply-evidence.v1'
# The RR re-pricer's per-world opponent sets (scripts/depthn_rr_reprice.py at 5a035eb7; one place: a rename is a one-line
# change): world[WORLD_SETS] is a list over party slots of {species, moves (Hidden Power typed), base_ability, ability, item}.
# The set key uses base_ability (a Trace holder's current ability is the copy); ability only when base_ability is absent.
WORLD_SETS = ('opponent', 'party')
SET_FIELDS = dict(species='species', moves='moves', ability=('base_ability', 'ability'), item='item')


def set_key(moves, ability, item):
    """fp_reply_reweight.variant_key's key from a set's parts (an empty move slot, None in the re-pricer's party, is no move)."""
    return frozenset(rw.move_key(m) for m in moves or () if m), rw.toid(ability), rw.toid(item)


def species_key(species):
    """A species id for the join: Unown's letter forms ('unownu', the protocol's details) are one species ('unown', the
    re-pricer's); every other species is its toid."""
    mon = rw.toid(species)
    return 'unown' if mon.startswith('unown') else mon


def snapshot(evidence, evidence_n, misaligned):
    """The JSON form of fp_reply_reweight's evidence at one decision: only Pokémon with evidence."""
    out = {}
    for mon, table in evidence.items():
        if not evidence_n[mon]:continue
        out[mon] = dict(n=int(evidence_n[mon]), stats_aligned=mon not in misaligned,
                        sets=[[sorted(key[0]), key[1], key[2], INCOMPLETE if math.isnan(value) else None if not math.isfinite(value) else float(value)]
                              for key, value in sorted(table.items(), key=lambda kv: (sorted(kv[0][0]), kv[0][1], kv[0][2]))])
    return out


def table_of(mons):
    """{species: {set_key: loglik or None}} from a snapshot."""
    return {species_key(mon): {(frozenset(moves), ability, item): value for moves, ability, item, value in entry['sets']} for mon, entry in mons.items()}


def resolve(scored, key, notes=None):
    """The scored set key a world's set key joins to, or (None, why). A world's item is its item at the root: '' once the
    battle has publicly removed it (a consumed berry, Knock Off). Such a key joins the one scored set with the same moves and
    ability (notes['item_removed_resolved']); several such sets are 'item_removed_ambiguous'."""
    if key in scored:return key, None
    if not key[2]:
        same = [s for s in scored if s[0] == key[0] and s[1] == key[1]]
        if len(same) == 1:
            if notes is not None:notes['item_removed_resolved'] += 1
            return same[0], None
        if same:return None, 'item_removed_ambiguous'
    return None, 'set_not_scored'


def world_loglik(table, sets, notes=None):
    """(loglik, None) for a world whose Foul Play sets are `sets` ({species: set_key}), or (None, why)."""
    total = 0.
    for mon, scored in table.items():
        if mon not in sets:return None, 'species_not_in_world'
        key, why = resolve(scored, sets[mon], notes)
        if key is None:return None, why
        value = scored[key]
        if value is None:return None, 'set_ruled_out'
        if value == INCOMPLETE:return None, 'set_incomplete'
        total += value
    return total, None


TRACE = 'trace'
INCOMPLETE = 'incomplete'   # an extra set skipped at a round with evidence (a new move into a disabled slot): its loglik misses a term


def world_party(world):
    """The re-pricer world's opponent party (a list of member dicts with a species), or None without the field."""
    value = world
    for key in WORLD_SETS:value = value.get(key) if type(value) is dict else None
    if type(value) is not list or any(type(m) is not dict or not m.get(SET_FIELDS['species']) for m in value):return None
    return value


def member_ability(member):
    """The ability a member's set is keyed on: base_ability, else ability (counted by weights as a fallback)."""
    return next((member[k] for k in SET_FIELDS['ability'] if member.get(k)), None)


def world_sets(world):
    """{species: set_key} from one re-pricer world, or None without the field."""
    party = world_party(world)
    if party is None:return None
    return {species_key(m[SET_FIELDS['species']]): set_key(m.get(SET_FIELDS['moves']), member_ability(m), m.get(SET_FIELDS['item'])) for m in party}


def weights(evidence_rows, rr_rows, drop=()):
    """(world-weight rows, counts): every world of each RR decision with evidence, or none of them (module docstring).
    Decisions of a battle in `drop` are counted and skipped; every other decision's worlds enter the coverage counts."""
    by_decision = {(r['ordinal'], r['decision_index']): table_of(r['mons']) for r in evidence_rows}
    out, counts = [], collections.Counter()
    for row in rr_rows:
        counts['decisions'] += 1;key = (row.get('ordinal'), row.get('decision_index'));worlds = row.get('worlds')
        if key[0] in drop:counts['dropped_battle_decisions'] += 1;continue
        if (row.get('root') or {}).get('decided') is False:counts['undecided_decisions'] += 1;continue   # never scored
        if type(worlds) is not list or not worlds:counts['no_worlds'] += 1;continue
        table = by_decision.get(key);logliks = []
        for world in worlds:
            counts['worlds'] += 1;party = world_party(world) if type(world) is dict else None
            tracers = [] if party is None else [m for m in party if rw.toid(m.get('base_ability')) == TRACE]
            if party is not None:
                counts['members_keyed_by_current_ability'] += sum(not m.get('base_ability') for m in party)
            if tracers:
                counts['tracer_worlds'] += 1;counts['tracer_members_copied_ability'] += sum(rw.toid(m.get('ability')) not in ('', TRACE) for m in tracers)
                exercised = table is not None and any(species_key(m[SET_FIELDS['species']]) in table for m in tracers)
                counts['tracer_worlds_with_evidence'] += exercised
            if party is None:value, why = None, 'world_sets_missing'
            elif table is None:value, why = None, 'no_evidence_row'
            else:value, why = world_loglik(table, world_sets(world), counts)
            if value is None:counts['worlds_failed_' + why] += 1;logliks.append(None);continue
            counts['worlds_joined'] += 1
            if tracers:counts['tracer_worlds_joined'] += 1;counts['tracer_worlds_joined_with_evidence'] += exercised
            logliks.append(value)
        if any(x is None for x in logliks):counts['decisions_not_weighted'] += 1;continue
        counts['weighted'] += 1;counts['with_evidence'] += bool(table);counts['worlds_weighted'] += len(logliks)
        out += [dict(ordinal=row['ordinal'], decision_index=row['decision_index'], world=w, loglik=x) for w, x in enumerate(logliks)]
    return out, counts


def coverage(counts):
    """The share of in-scope worlds joined (None without worlds)."""
    return counts['worlds_joined'] / counts['worlds'] if counts['worlds'] else None


def load_drop(path):
    """{ordinal: reason} from depthn_rr_score --drop-battles' JSON list of {ordinal (int), reason (str)}."""
    out = {}
    for number, item in enumerate(json.loads(Path(path).read_text())):
        if type(item) is not dict or type(item.get('ordinal')) is not int or type(item.get('reason')) is not str or not item['reason']:
            raise SystemExit(f'{path}[{number}]: {{ordinal: int, reason: str}} required')
        if item['ordinal'] in out:raise SystemExit(f'{path}[{number}]: ordinal {item["ordinal"]} listed twice')
        out[item['ordinal']] = item['reason']
    return out


def extra_sets(rr_rows):
    """{ordinal: {species_key: {set_key: variant dict}}} from the re-pricer rows' decided worlds (members with an item)."""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for row in rr_rows:
        if (row.get('root') or {}).get('decided') is False or type(row.get('worlds')) is not list:continue
        for world in row['worlds']:
            party = world_party(world) if type(world) is dict else None
            for m in party or ():
                if not m.get(SET_FIELDS['item']):continue
                variant = dict(moves=[x for x in m.get(SET_FIELDS['moves']) or () if x], ability=member_ability(m), item=m[SET_FIELDS['item']])
                out[row['ordinal']][species_key(m[SET_FIELDS['species']])].setdefault(set_key(variant['moves'], variant['ability'], variant['item']), variant)
    return out


def training_seeds(data_dir):
    """The battle seeds of π_FP's training data: only each shard's seed array is read (fp_reply_train.load reads every array)."""
    import numpy as np
    seeds = set()
    for path in sorted(glob.glob(str(Path(data_dir) / 'shard-*.npz'))):
        with np.load(path) as shard:
            if 'seed' in shard.files:seeds.update(int(x) for x in shard['seed'])
    return seeds


def export(args):
    import torch
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv, LocalShowdownError
    from pokezero.neural_policy import (category_vocab_from_model_config, checkpoint_file_sha256, feature_masks_from_model_config,
                                        load_transformer_checkpoint, observation_spec_from_model_config)
    from gpu_engine import fp_reply_train as train_mod
    torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False);torch.set_float32_matmul_precision('highest')
    run, out = Path(args.run_directory), Path(args.out);out.mkdir(parents=True, exist_ok=True);started = time.time()
    marker = torch.load(args.pi_fp, map_location='cpu', weights_only=True).get('fp_reply') or {}
    if not marker.get('data'):raise SystemExit(f'{args.pi_fp} has no fp_reply.data marker: its training games are unknown')
    trained_seeds = training_seeds(marker['data'])
    model, result = load_transformer_checkpoint(args.pi_fp, map_location='cpu');model.eval()
    if int(result.model_config.window_size) != 1:raise SystemExit(f'π_FP has window {result.model_config.window_size}; this export feeds window 1')
    paths = sorted(glob.glob(str(run / 'results' / '*.json')), key=lambda p: int(Path(p).stem))
    paths = [p for p in paths if int(Path(p).stem) % args.shards == args.shard]
    if args.ordinals:   # only these games (e.g. the MC gate bank's Foul Play games)
        keep = {int(x) for x in json.loads(Path(args.ordinals).read_text())};paths = [p for p in paths if int(Path(p).stem) in keep]
    data_mod.SCREEN_BAND = (0, 0)   # RR's games are the screen's
    extras = extra_sets(_jsonl(sorted(args.extra_sets))) if args.extra_sets else {}
    checks, mismatches, examples = collections.Counter(), collections.Counter(), {}
    env = dex = tables = config_key = None;games = 0;rows = []
    for path in paths:
        if args.limit and games >= args.limit:break
        game, skip = data_mod.load_game(run, path)
        if game is None:checks[f'skipped_{skip}'] += 1;continue
        if game['seed'] in trained_seeds:raise SystemExit(f"battle seed {game['seed']} (ordinal {game['ordinal']}) is in π_FP's training data")
        base = game['base_config'];key = (base['showdown_root'], base['node_binary'], bool(base['belief_set_source']))
        if not base['belief_set_source']:checks['skipped_no_belief_source'] += 1;continue
        if config_key is None:config_key = key
        elif key != config_key:checks['skipped_other_runtime'] += 1;continue
        if dex is None:dex = load_showdown_dex_cached(base['showdown_root']);tables = rw.generator_tables(base['showdown_root'], base['node_binary'])
        if env is None:
            env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=base['showdown_root'], node_binary=base['node_binary'],
                observation_spec=observation_spec_from_model_config(result.model_config),
                category_vocab=category_vocab_from_model_config(result.model_config, base['showdown_root']),
                feature_masks=feature_masks_from_model_config(result.model_config), set_belief_source=True))
        game_rows = []

        def on_ours(index, round_, evidence, evidence_n, misaligned, game=game, game_rows=game_rows):
            game_rows.append(dict(ordinal=game['ordinal'], decision_index=index, round=round_, mons=snapshot(evidence, evidence_n, misaligned)))
        sets_of = extras.get(game['ordinal'], {})
        extra = (lambda mon, sets_of=sets_of: list(sets_of.get(species_key(mon), {}).values())) if args.extra_sets else None
        try:_, _, _, game_checks = rw.replay_game(env, model, game, dex, torch, tables, marker.get('redaction'), on_ours, extra)
        except data_mod.Mismatch as error:
            reason = re.sub(r"'[^']*'|\"[^\"]*\"|\d+", '#', str(error));mismatches[reason] += 1
            examples.setdefault(reason, dict(ordinal=game['ordinal'], detail=str(error)));continue
        except (LocalShowdownError, ValueError, KeyError, StopIteration) as error:
            reason = f'env error {type(error).__name__}';mismatches[reason] += 1
            examples.setdefault(reason, dict(ordinal=game['ordinal'], detail=str(error)[:500]));env.close();env = None;continue
        games += 1;checks.update(game_checks);rows += game_rows
        if games % 10 == 0:print(json.dumps(dict(event='PROGRESS', shard=args.shard, games=games, rows=len(rows), seconds=round(time.time() - started, 1))), flush=True)
    if env is not None:env.close()
    (out / f'evidence-{args.shard}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    summary = dict(schema=SCHEMA, run_directory=str(run), pi_fp=str(args.pi_fp), pi_fp_sha256=checkpoint_file_sha256(args.pi_fp),
                   pi_fp_redaction=marker.get('redaction'), shard=args.shard, shards=args.shards, games=games, rows=len(rows), checks=dict(checks),
                   extra_sets=sorted(str(Path(p).resolve()) for p in args.extra_sets or ()),
                   mismatched=sum(mismatches.values()), mismatch_reasons=dict(mismatches), mismatch_examples=examples,
                   seconds=round(time.time() - started, 1))
    (out / f'shard-{args.shard}.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(dict(event='SHARD_DONE', **{k: summary[k] for k in ('shard', 'games', 'rows', 'mismatched', 'seconds')})), flush=True)


def _jsonl(paths):
    for path in paths:
        for raw in Path(path).read_bytes().splitlines():
            if raw.strip():yield json.loads(raw)


def build_weights(args):
    evidence = list(_jsonl(sorted(glob.glob(str(Path(args.evidence) / 'evidence-*.jsonl')))))
    drop = load_drop(args.drop_battles) if args.drop_battles else {}
    rows, counts = weights(evidence, _jsonl(args.rows), drop)
    out = Path(args.out)
    if out.exists():raise SystemExit(f'{out} exists')
    share = coverage(counts);full = share == 1.
    summary = dict(schema=SCHEMA + ':weights', evidence=str(Path(args.evidence).resolve()), rows=[str(Path(p).resolve()) for p in args.rows],
                   evidence_rows=len(evidence), dropped_battles=drop, counts=dict(counts),
                   coverage=dict(worlds=counts['worlds'], joined=counts['worlds_joined'], share=share, full=full,
                                 failed={k[len('worlds_failed_'):]: v for k, v in counts.items() if k.startswith('worlds_failed_')}),
                   tracers={k: counts[k] for k in ('tracer_worlds', 'tracer_worlds_with_evidence', 'tracer_worlds_joined',
                                                   'tracer_worlds_joined_with_evidence', 'tracer_members_copied_ability')},
                   written=full or args.allow_partial)
    Path(str(out) + '.summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(dict(event='COVERAGE', **summary['coverage'], tracers=summary['tracers'])))
    if not summary['written']:print(json.dumps(dict(event='COVERAGE_BELOW_FULL', out=None)));return 1
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print(json.dumps(summary['counts']))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0]);sub = parser.add_subparsers(dest='mode', required=True)
    e = sub.add_parser('export');e.add_argument('run_directory');e.add_argument('out');e.add_argument('--pi-fp', required=True)
    e.add_argument('--shard', type=int, default=0);e.add_argument('--shards', type=int, default=1);e.add_argument('--limit', type=int, default=0)
    e.add_argument('--extra-sets', nargs='+', help="the RR re-pricer's rows: score their worlds' sets too")
    e.add_argument('--ordinals', help="a JSON list of the run directory's game ordinals: export only those games")
    w = sub.add_parser('weights');w.add_argument('--evidence', required=True);w.add_argument('--rows', nargs='+', required=True)
    w.add_argument('--out', required=True);w.add_argument('--drop-battles', help="depthn_rr_score's --drop-battles JSON")
    w.add_argument('--allow-partial', action='store_true', help='write the weights below 100%% world coverage')
    args = parser.parse_args(argv)
    if args.mode == 'export':export(args);return 0
    return build_weights(args)


if __name__ == '__main__':raise SystemExit(main())
