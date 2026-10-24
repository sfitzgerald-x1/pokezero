#!/usr/bin/env python3
"""Read-only P2 layout adapter for the already-frozen descriptive arithmetic.

Prepared after collection began; this changes neither the statistical method nor
the collection. It never attests normal completion or enables interim inference.
Frozen arithmetic and terminal checks are loaded from the entry's source commit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType


SHARDS = (
    ('candidate-p1', 'p1', 'foul-play', 'informed-one-ply', 'candidate'),
    ('raw-p1', 'p1', 'foul-play', 'raw', 'raw'),
    ('candidate-p2', 'p2', 'foul-play', 'informed-one-ply', 'candidate'),
    ('raw-p2', 'p2', 'foul-play', 'raw', 'raw'),
    ('h2h-p1', 'p1', 'raw', 'informed-one-ply', 'h2h'),
    ('h2h-p2', 'p2', 'raw', 'informed-one-ply', 'h2h'),
)
READER = 'src/pokezero/mcts_eval/fullgame_strength.py'
VALIDATOR = 'scripts/phase0_paired_dry_run.py'


def argument(args, name, default=None):
    positions = [i for i, value in enumerate(args) if value == name]
    if len(positions) > 1 or (positions and positions[0] + 1 >= len(args)):
        raise ValueError('ambiguous registered argument: ' + name)
    return args[positions[0] + 1] if positions else default


def read_outcomes(entry, reader, checked_outcome, summaries_dir):
    """Return full-roster input rows, not execution or provenance attestation."""
    start, count = entry['seed_start'], entry['seed_count']
    if (entry.get('schema') != 'pokezero.p2-informed-one-ply-gameplay-entry.v1'
            or entry.get('information_model') != 'informed_adversary_root_minimax'
            or type(start) is not int or type(count) is not int or count != 128
            or not 0 <= start <= 2**32 - count
            or [(s['name'], s['seat']) for s in entry['shards']]
            != [(s[0], s[1]) for s in SHARDS]):
        raise ValueError('exact registered P2 roster and six-shard layout required')
    seeds = list(range(start, start + count))
    common = entry['common_arguments']
    round_cap = int(argument(common, '--max-decision-rounds'))
    arms, snapshots = {'candidate': [], 'raw': [], 'h2h': []}, {}
    for shard, (name, seat, opponent, policy, arm) in zip(entry['shards'], SHARDS):
        args = common + shard['arguments']
        if (argument(args, '--policy-mode') != policy
                or argument(args, '--opponent-policy-mode') != opponent):
            raise ValueError('registered shard policy identity disagrees')
        # CLI modes are not summary wire labels. Match the frozen bridge's
        # _opponent_policy_id_label and ControlledFoulPlayConfig name rewrite.
        opponent_id = 'foul-play' if opponent == 'foul-play' else 'pokezero-' + opponent
        opponent_name = argument(args, '--foulplay-username', 'FoulPlayBot')
        if opponent != 'foul-play' and opponent_name == 'FoulPlayBot':
            opponent_name = 'PokeZero' + opponent.title().replace('-', '') + 'Bot'
        path = Path(summaries_dir) / (name + '.json')
        if not path.exists():
            continue  # The frozen reader fills every missing seed-seat cell.
        payload = path.read_bytes()  # Atomic summary snapshot, read exactly once.
        doc = json.loads(payload)
        rows = doc['game_results']
        if (not isinstance(rows, list) or type(doc.get('completed_games')) is not int
                or doc['completed_games'] != len(rows) or len(rows) > count
                or any(type(r.get('seed')) is not int for r in rows)
                or [r['seed'] for r in rows] != seeds[:len(rows)]
                or doc.get('pokezero_player') != seat
                or doc.get('policy_mode') != policy
                or doc.get('opponent_policy_id') != opponent_id):
            raise ValueError('summary is not the registered policy/seat chronological prefix')
        outcomes = [checked_outcome(reader, row, seat, round_cap=round_cap,
            candidate_name=argument(args, '--pokezero-username', 'PokeZeroBot'),
            opponent_name=opponent_name,
            format_id=argument(args, '--format', 'gen3randombattle')) for row in rows]
        arms[arm].extend(outcomes)
        snapshots[name] = dict(sha256=hashlib.sha256(payload).hexdigest(),
            recorded=len(rows), statuses=dict(Counter(o.status for o in outcomes)))
    return seeds, arms, snapshots


def describe_panel(entry, reader, checked_outcome, summaries_dir):
    seeds, arms, snapshots = read_outcomes(entry, reader, checked_outcome, summaries_dir)
    result = reader.summarize_capped_pairs(seeds, arms['candidate'], arms['raw'],
                                          normal_complete_attempt=False)
    assert result['primary_test'] is None and not result['screening_rule_passed']
    return dict(schema='pokezero.p2-paired-live-dry-run.v1', snapshots=snapshots,
        planned_game_instances=6 * len(seeds), planned_primary_cells=4 * len(seeds),
        recorded_primary_cells=len(arms['candidate']) + len(arms['raw']),
        planned_h2h_cells=2 * len(seeds), recorded_h2h_cells=len(arms['h2h']),
        h2h_statuses=dict(Counter(o.status for o in arms['h2h'])), result=result,
        independent_final_validation=False,
        interpretation='Descriptive only: all missing/capped/conflicting primary cells '
            'remain [0,1]; H2H never enters the FoulPlay contrast. No interim test, '
            'normal-completion attestation, final qualification or advancement.')


def frozen_module(repository, commit, path, expected_sha256=None):
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('full registered source commit required')
    payload = subprocess.check_output(['git', '-C', str(repository), 'show', commit + ':' + path])
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('frozen reader hash mismatch')
    module = ModuleType('p2_frozen_' + digest)
    module.__file__ = str(Path(repository) / path)
    sys.modules[module.__name__] = module  # Required by dataclasses.
    exec(compile(payload, module.__file__, 'exec'), module.__dict__)
    return module, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', required=True, type=Path)
    parser.add_argument('--expected-entry-sha256', required=True)
    parser.add_argument('--repository', type=Path,
                        help='Local repository containing the exact registered commit.')
    parser.add_argument('--summaries-dir', type=Path,
                        help='Optional copied observation directory, never a new roster.')
    args = parser.parse_args()
    payload = args.entry.read_bytes()
    entry_sha = hashlib.sha256(payload).hexdigest()
    if entry_sha != args.expected_entry_sha256:
        raise ValueError('exact entry hash mismatch')
    entry = json.loads(payload)
    frozen = entry['analysis']['reader']
    if (frozen['path'] != READER or frozen['function'] != 'summarize_capped_pairs'
            or frozen['terminal_validator'] != VALIDATOR + ':checked_outcome'):
        raise ValueError('unexpected registered arithmetic or terminal validator')
    commit = entry['repositories'][entry['runtime']]['commit']
    repository = args.repository or Path(entry['runtime'])
    reader, reader_sha = frozen_module(repository, commit, READER, frozen['sha256'])
    validator, validator_sha = frozen_module(repository, commit, VALIDATOR)
    result = describe_panel(entry, reader, validator.checked_outcome,
                            args.summaries_dir or Path(entry['run_directory']))
    result.update(entry_sha256=entry_sha, source_commit=commit,
                  reader_sha256=reader_sha, terminal_validator_sha256=validator_sha)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
