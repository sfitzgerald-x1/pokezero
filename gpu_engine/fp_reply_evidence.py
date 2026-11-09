"""Reply-model evidence on Foul Play's sets, for the definitive RR's reweighted worlds.

usage: python -m gpu_engine.fp_reply_evidence export RUN_DIRECTORY OUT --pi-fp PT [--shard I --shards N] [--limit N]
       python -m gpu_engine.fp_reply_evidence weights --evidence OUT --rows ROWS/rows-*.jsonl --out WEIGHTS.jsonl

export: replay each game of a run directory (fp_reply_data.load_game's layout, Foul Play's journal included) as
fp_reply_reweight.replay_game does, scoring Foul Play's public choices with the reply model under every candidate set of its
active Pokémon, and write evidence-<shard>.jsonl: one row per decision of our seat, {ordinal, decision_index, round, mons:
{species: {n, stats_aligned, sets: [[sorted move keys, ability, item, loglik], ...]}}}, the loglik being the summed log q of
the choices made public in earlier rounds (fp_reply_reweight's evidence; -inf is written as null). The reply model is the
π_FP checkpoint (its fp_reply marker's redaction is applied, as in the reweighting test). Games whose battle seed is in π_FP's
training data are refused: the evidence must come from games the model never saw.

weights: join the evidence with the RR re-pricer's rows. A world's loglik is the sum over Foul Play's Pokémon in that world
with evidence (n > 0) of the loglik of the world's set for it (set_key: move keys, ability, item, as fp_reply_reweight's
variant_key). A world whose set for such a Pokémon is not among its scored sets, or has loglik -inf, has none; every world of
that decision is then left out (the scorer counts the decision as world_weights_missing). Writes {ordinal, decision_index,
world, loglik} rows for depthn_rr_score --world-weights, and WEIGHTS.summary.json with the counts.
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
# The RR re-pricer's per-world opponent sets (one place: a rename is a one-line change): world[WORLD_SETS] is a list over
# party slots of {species, moves, ability, item}.
WORLD_SETS = ('opponent', 'sets')
SET_FIELDS = dict(species='species', moves='moves', ability='ability', item='item')


def set_key(moves, ability, item):
    """fp_reply_reweight.variant_key's key from a set's parts."""
    return frozenset(rw.move_key(m) for m in moves or ()), rw.toid(ability), rw.toid(item)


def snapshot(evidence, evidence_n, misaligned):
    """The JSON form of fp_reply_reweight's evidence at one decision: only Pokémon with evidence."""
    out = {}
    for mon, table in evidence.items():
        if not evidence_n[mon]:continue
        out[mon] = dict(n=int(evidence_n[mon]), stats_aligned=mon not in misaligned,
                        sets=[[sorted(key[0]), key[1], key[2], None if not math.isfinite(value) else float(value)]
                              for key, value in sorted(table.items(), key=lambda kv: (sorted(kv[0][0]), kv[0][1], kv[0][2]))])
    return out


def table_of(mons):
    """{species: {set_key: loglik or None}} from a snapshot."""
    return {mon: {(frozenset(moves), ability, item): value for moves, ability, item, value in entry['sets']} for mon, entry in mons.items()}


def world_loglik(table, sets):
    """(loglik, None) for a world whose Foul Play sets are `sets` ({species: set_key}), or (None, why)."""
    total = 0.
    for mon, scored in table.items():
        if mon not in sets:return None, 'species_not_in_world'
        value = scored.get(sets[mon], 'missing')
        if value == 'missing':return None, 'set_not_scored'
        if value is None:return None, 'set_ruled_out'
        total += value
    return total, None


def world_sets(world):
    """{species: set_key} from one re-pricer world, or None without the field."""
    value = world
    for key in WORLD_SETS:value = value.get(key) if type(value) is dict else None
    if type(value) is not list:return None
    out = {}
    for member in value:
        if type(member) is not dict or not member.get(SET_FIELDS['species']):return None
        out[rw.toid(member[SET_FIELDS['species']])] = set_key(member.get(SET_FIELDS['moves']), member.get(SET_FIELDS['ability']),
                                                                 member.get(SET_FIELDS['item']))
    return out


def weights(evidence_rows, rr_rows):
    """(world-weight rows, counts): every world of each RR decision with evidence, or none of them (module docstring)."""
    by_decision = {(r['ordinal'], r['decision_index']): table_of(r['mons']) for r in evidence_rows}
    out, counts = [], collections.Counter()
    for row in rr_rows:
        counts['decisions'] += 1;worlds = row.get('worlds')
        if type(worlds) is not list or not worlds:counts['no_worlds'] += 1;continue
        table = by_decision.get((row.get('ordinal'), row.get('decision_index')))
        if table is None:counts['no_evidence_row'] += 1;continue
        logliks = []
        for world in worlds:
            sets = world_sets(world) if type(world) is dict else None
            if sets is None:logliks = None;counts['world_sets_missing'] += 1;break
            value, why = world_loglik(table, sets)
            if value is None:logliks = None;counts[why] += 1;break
            logliks.append(value)
        if logliks is None:continue
        counts['weighted'] += 1;counts['with_evidence'] += bool(table)
        out += [dict(ordinal=row['ordinal'], decision_index=row['decision_index'], world=w, loglik=x) for w, x in enumerate(logliks)]
    return out, counts


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
    trained_seeds = {int(x) for x in train_mod.load(marker['data'])['seed']}
    model, result = load_transformer_checkpoint(args.pi_fp, map_location='cpu');model.eval()
    if int(result.model_config.window_size) != 1:raise SystemExit(f'π_FP has window {result.model_config.window_size}; this export feeds window 1')
    paths = sorted(glob.glob(str(run / 'results' / '*.json')), key=lambda p: int(Path(p).stem))
    paths = [p for p in paths if int(Path(p).stem) % args.shards == args.shard]
    data_mod.SCREEN_BAND = (0, 0)   # RR's games are the screen's
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
        try:_, _, _, game_checks = rw.replay_game(env, model, game, dex, torch, tables, marker.get('redaction'), on_ours)
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
    rows, counts = weights(evidence, _jsonl(args.rows))
    out = Path(args.out)
    if out.exists():raise SystemExit(f'{out} exists')
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    summary = dict(schema=SCHEMA + ':weights', evidence=str(Path(args.evidence).resolve()), rows=[str(Path(p).resolve()) for p in args.rows],
                   evidence_rows=len(evidence), counts=dict(counts))
    Path(str(out) + '.summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary['counts']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0]);sub = parser.add_subparsers(dest='mode', required=True)
    e = sub.add_parser('export');e.add_argument('run_directory');e.add_argument('out');e.add_argument('--pi-fp', required=True)
    e.add_argument('--shard', type=int, default=0);e.add_argument('--shards', type=int, default=1);e.add_argument('--limit', type=int, default=0)
    w = sub.add_parser('weights');w.add_argument('--evidence', required=True);w.add_argument('--rows', nargs='+', required=True)
    w.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    (export if args.mode == 'export' else build_weights)(args)
    return 0


if __name__ == '__main__':raise SystemExit(main())
