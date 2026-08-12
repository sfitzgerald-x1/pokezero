"""Controlled foul-play benchmark harness for context-aware PokeZero policies.

The existing live-server foul-play benchmark is useful for raw online play, but it cannot exercise
context-aware replay-from-root search: the online client only has protocol lines, while
``RootPUCTSearchPolicy`` needs a deterministic seed, action trajectory, and both players' current
legal requests.

This module keeps foul-play across the GPL boundary by running it as a separate websocket client,
but owns the Showdown ``BattleStream`` process so PokeZero can build the exact ``PolicyContext``
required by root-PUCT.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import math
import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from .actions import ACTION_COUNT
from .belief import PublicBattleBeliefEngine
from .category_vocab import CategoryVocabulary
from .collection import RolloutRecord, write_rollout_record
from .determinization import gen3_randbat_belief_start_override_planner
from .deep_line_audit import PROTOCOL_SIGNATURE_SCHEMA_VERSION, protocol_signature_counts
from .dex import ShowdownDex, load_showdown_dex_cached
from .env import PlayerId, TerminalState
from .fallback_replay import RefusalRecord, attach_refusal_recorder
from .local_showdown import (
    BRIDGE_PATH,
    LocalShowdownConfig,
    LocalShowdownEnv,
    PublicBattleMaterializationState,
    actor_move_states_from_request_history,
    belief_set_source_env_enabled,
    showdown_seed_from_int,
)
from .mcts_diagnostics import (
    root_puct_fallback_category,
    sanitize_root_puct_missing_sampled_world_reason_categories,
)
from .neural_policy import (
    TransformerInferenceTimingAccumulator,
    TransformerSoftmaxPolicy,
    category_vocab_from_model_config,
    feature_masks_from_model_config,
    evaluate_transformer_action_priors,
    evaluate_transformer_observation_value,
    evaluate_transformer_opponent_action_priors,
    load_transformer_checkpoint,
    observation_spec_from_model_config,
    require_compatible_transformer_value_checkpoint,
)
from .observation import (
    DEFAULT_OBSERVATION_FEATURE_MASKS,
    OBSERVATION_SCHEMA_VERSION_V3,
    TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS,
    ObservationFeatureMasks,
    PokeZeroObservationV0,
)
from .policy import Policy, PolicyContext, PolicyDecision, RandomLegalPolicy
from .public_action_capture import append_public_action_round, public_action_round_from_protocol_lines
from .public_decision_corpus import (
    PublicDecisionCorpusWriter,
    PublicResolvedActionRound,
    public_corpus_manifest,
    public_decision_records_from_trajectory,
)
from .encoding_collision_audit import CollisionSketchWriter, collision_sketch_manifest
from .randbat import load_gen3_randbat_source_cached
from .randbat_vocab import gen3_category_vocabulary
from .rollout import RolloutConfig
from .search_policy import (
    EntropyMarginVisitBudgetSelector,
    FixedExtraVisitBudgetSelector,
    RootPUCTSearchPolicy,
    greedy_opponent_action_planner,
    prior_top_k_opponent_action_scenario_planner,
)
from .search import RootPUCTSearchTiming
from .showdown import (
    PlayerRelativeBattleState,
    normalize_for_player,
    observation_from_player_state,
    observation_schema_version_from_choice,
    observation_spec_for_schema,
    parse_showdown_replay,
    showdown_choice_for_action,
)
from .teacher_capture import action_index_from_choice_string
from .trajectory import BattleTrajectory, TrajectoryStep


SCHEMA_VERSION = "pokezero.controlled-foulplay-benchmark.v1"
COMPARISON_SCHEMA_VERSION = "pokezero.controlled-foulplay-comparison.v1"
DEFAULT_FOULPLAY_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "foul-play"
DEFAULT_BATTLE_ID_PREFIX = "battle-gen3randombattle-controlled"
DEFAULT_START_OVERRIDE_ATTEMPTS = 10
_WILSON_95_Z = 1.959963984540054
_MIN_STRENGTH_SAMPLE_GAMES = 300
ControlledFoulPlayProgressCallback = Callable[["ControlledFoulPlayBenchmarkResult"], None]
ControlledFoulPlayComparisonProgressCallback = Callable[["ControlledFoulPlayComparisonResult"], None]
ControlledFoulPlayTrajectoryCallback = Callable[[BattleTrajectory], None]
ControlledFoulPlayCaptureProgressCallback = Callable[[Mapping[str, Any]], None]
_COMPARISON_MODES = {"per-seed", "per-arm"}
_ROOT_PUCT_TIMING_FIELD_NAMES = tuple(entry.name for entry in fields(RootPUCTSearchTiming))

# The OPPONENT-MOVE JOURNAL.
#
# `EngineMctsStats.fallback_samples` records `{battle_id, round, seat, reason}` for
# every fallback decision, and the battle id carries the BattleStream seed. None of
# the 1,140 addresses recorded across eras 61-64 is replayable. The blocker is the
# OPPONENT, not our seed discipline:
#
#   * foul-play searches on a WALL-CLOCK budget (`fp/search/main.py:56`), so its
#     iteration count differs run to run even on identical input;
#   * one layer below that, poke-engine's `sample_node` (`src/mcts.rs:106-113`) does
#     `let mut rng = rng();` -- `rand::rng()`, OS-entropy, constructed fresh on every
#     chance-node sample. No seed reaches it from Python, and pokezero's seeding of
#     foul-play's `random` module (`_foulplay_env`) cannot reach it.
#
# Measured, on one live state at a FIXED iteration count: five runs, five different
# visit distributions. Pinning the budget does not fix it.
#
# So the opponent's move is not re-derivable and must be RECORDED. This journal is
# the producer half: it files the `/choose` body foul-play actually submitted, per
# decision round, in the same document as `fallback_samples`. A replay driver feeds
# these back instead of re-running foul-play.
#
# It records; it does not replay. Nothing here rescues the existing 1,140 addresses
# -- those battles are over and their opponent moves were never written down. Only
# addresses recorded by a run WITH journaling on become replayable.
#
# A JOURNALLED ADDRESS IS NOT A ONE-TURN REPLAY. The comment on
# `EngineMctsStats.fallback_samples` says "any entry here replays as a single turn"; that
# is wrong for the mode this bridge runs, and a replay driver written to it would
# start in the wrong place. `EngineMctsPolicy._live_folds` is keyed
# `(battle_id, seat)` and advanced over
# exactly the new public lines at EVERY decision (`EngineMctsPolicy._advance_live_fold`),
# never refolded from a whole log, and `leaf_eval="model"` refuses the
# decision outright when that fold is missing (`live_fold_broken`). So reaching round
# R means replaying rounds 0..R-1 first. That is why the `addressed` mode below keeps
# a PREFIX rather than the single addressed round.
#
# NECESSARY, NOT SUFFICIENT. Removing the opponent's nondeterminism leaves OUR side.
# For a bridge shard the mode is not a variable: `_build_policy` HARDCODES
# `leaf_eval="model"` (see `EngineMctsConfig(leaf_eval="model", ...)` below), and
# `_engine_policy_stats` returns None outside `policy_mode="engine-mcts"` -- so a
# shard that carries `fallback_samples` at all was produced by exactly one path:
#
#   * `leaf_eval="model"` reaches the crate through
#     `native.search_batched_multi_encoded(*search_args)` (in `_search_model`'s
#     `run_world`), with the seed passed POSITIONALLY as `record["seed"]` (assembled by
#     `native_search_args`), drawn at `world_seed = rng.getrandbits(63)`. It does NOT
#     call `puct_search_multi` -- that is the `_search_hp_fraction_crate` branch, a
#     different mode this bridge cannot select. An earlier revision of this comment
#     cited the wrong function for the only path that ships.
#   * `rng` there is `random.Random(f"{seed}:{player_id}:{decision_round_index}")`
#     from `_select_policy_decision`, so the world seeds are a pure function of
#     (battle seed, seat, round) and the draws taken before them.
#   * The crate seeds every generator it owns with `StdRng::seed_from_u64`; there is
#     no `rand::rng()` anywhere in `rust/pokezero-search/src`.
#   * Iteration count is `config.search_sims`, not a clock.
#
# So this path plus the journal is a CANDIDATE for exact replay. The residual is
# float and threading nondeterminism in the model forward, which is NOT verified
# here and is not verifiable in a checkout without libtorch.
#
# `leaf_eval="hp_fraction"` (the `EngineMctsConfig` dataclass default) would be
# unreplayable for the same two reasons as foul-play -- poke-engine's own wall-clock
# `monte_carlo_tree_search` over the unseeded `rand::rng()`. It is recorded here only
# so the contrast is not rediscovered: NO bridge shard can be in that mode, because
# of the hardcode above. Do not read this as "the common case is unreplayable".
OPPONENT_JOURNAL_SCHEMA_VERSION = "pokezero.opponent-journal.v1"

# SIZE, measured with THIS serializer against the real era 61-64 corpus (377 bridge
# summaries, 2,981 games, 145,163 decision rounds; journals replayed in at the real
# per-game round counts):
#
#   mode        corpus bytes    delta   median summary
#   base          14,837,932              39,149
#   addressed     17,160,541   +15.7%     43,921
#   full          39,647,286  +167.2%    102,748
#
# "addressed" (default): journal only the battles that recorded at least one fallback
# ADDRESS, truncated to the last such round -- the prefix a replay of that address
# needs and nothing more. 14.4% of games carry an address and 8.9% of decision rounds
# are journalled, which is where the ~10x saving over "full" comes from.
#
# "full": every opponent move of every battle. Reserved for work that needs to replay
# decisions which did NOT fall back.
#
# "off": record nothing.
#
# RECORDING TIME is negligible: the whole block costs 1.18 us per opponent decision
# (measured, dominated by the sha256 of a ~2.3 KB request line) against a decision
# boundary whose measured median wall is 5.02 s on our side plus foul-play's own
# fixed 1,000 ms search -- about 2e-7 of the boundary.
#
# SERIALIZATION time is the axis that actually scales, and it is superlinear:
# `_write_json` rewrites the ENTIRE summary after every game when `--summary-out` is
# set, so journal bytes are re-serialized O(games^2). Measured against the real mean
# per-game row (3,383 B) and mean journal bytes per game (812 addressed, 6,184 full),
# cumulative bytes written over a run:
#
#   --games 250      off 106 MB    addressed 132 MB    full 300 MB
#
# Against a 250-game run that takes hours of search wall that is not a bottleneck at
# the default, but "there is nothing to switch off for speed" would be wrong for
# `full` on a long run, so it is not claimed. `off` exists for that case.
#
# `addressed` is the default because it serves the stated purpose at a seventh of the
# bytes. Its known boundary is REAL AND LARGE, and is NOT the one the obvious counter
# measures:
#
#   * `fallback_sample_addresses_dropped` counts ONLY the 256-KEY ceiling
#     (`_fallback`'s `_FALLBACK_SAMPLE_KEY_CEILING` arm). It is 0 across all 552 shards
#     of eras 61-64, and that fact says nothing about the drop below.
#   * `_FALLBACK_SAMPLES_PER_CLASS = 3` (applied in `_fallback`'s `len(bucket) >=` guard)
#     and the one-address-per-battle rule beside it both `continue` with NO COUNTER.
#     That is where the addresses actually go.
#
# HOW BIG, measured per CUMULATIVE STATS BLOCK on ONE shard family (the 377 bridge
# summaries; the merged paired-eval shards mirror them and adding the two families
# double-counts):
#
#   denominator  sum(fallback_reasons.values()) = fallback DECISIONS, uncapped.
#                Cross-checked against the block's own `fallback_decisions`: 2,093
#                both ways, 0 per-block mismatches.
#   numerator    DISTINCT (battle_id, round, seat) across all keys of that block's
#                `fallback_samples` = decisions that have a replayable address.
#                DISTINCT is load-bearing: `_fallback`'s
#                `for key in [f"fallback:{reason}", *delta]` loop files one address
#                PER KEY, so a single decision appears under `fallback:<reason>` AND
#                under every world-failure class in its delta.
#
#   2,093 fallback decisions -> 609 addressed -> 1,484 (70.9%) with NO address.
#
# Invariant checked, since a wrong denominator is exactly the mistake being corrected
# here: numerator <= denominator in every one of the 313 blocks, 0 violations. All
# 1,140 address entries have an integer `round`, so reading them by raw dict access
# gives the same set as `fallback_addresses.iter_shard_addresses`, whose int check
# would otherwise drop the `round=None` that `_fallback` can file from a context
# without `decision_round_index`.
#
# WHICH RULE DOES THE DROPPING -- the cap of 3 is NOT the main one. Per-key bucket
# sizes within a block are 719 x 1, 140 x 2, 47 x 3 (906 keys, none over 3). Only
# those 47 (5.2%) ever reached `_FALLBACK_SAMPLES_PER_CLASS`; the other 859 (94.8%)
# were never limited by it, so every address they did not retain was refused by
# `_fallback`'s "One address per BATTLE" guard
# (`any(entry["battle_id"] == str(battle_id) for entry in bucket)`). The per-battle
# rule is the dominant mechanism and the cap is the ~5% tail.
#
# That split is in KEYS, which is what these shards can support. The DECISION-level
# split is NOT measurable from them: attributing an unaddressed decision needs to know
# which battles fell back, and `engine_mcts_fallbacks` is a
# `ControlledFoulPlayGameResult` field that `to_dict` never emits -- the per-game rows
# carry no fallback count at all. So "N fallback decisions per fallen-back battle"
# cannot be computed here; dividing 2,093 by the 430 ADDRESSED battles instead would
# assume every fallback lives in a battle that kept an address, which is the
# unstated-population move that produced the withdrawn numbers below.
#
# None of this contradicts "14.4% of games carry an address" above -- different
# questions in different units. 14.4% is games-with-an-address over GAMES (which
# battles get a journal); 70.9% is unaddressed DECISIONS over fallback decisions
# (which refusals are reachable at all).
#
# WITHDRAWN, recorded so it is not re-derived: an earlier revision said "45.2% of
# 4,022 occurrences" and "30 of 35 classes at the cap". Both were unsound. 4,022 and
# 2,203 were sums over BOTH shard families (2,203 is literally 1,140 + 1,063), the
# ratio divided ADDRESSES by DECISIONS -- different units, and multi-key filing means
# it can exceed 1 -- and the class count was taken corpus-wide when the cap is
# per-block. The true gap is worse than the number that was withdrawn, not better.
#
# Three published figures were wrong across three review rounds and all three were the
# same species: a counter measuring a DIFFERENT DROP, a ratio summed over TWO
# POPULATIONS, and a count taken at the WRONG SCOPE. None was an arithmetic slip. The
# question that catches all three, and the one to ask of any number before it is
# written down here: WHAT UNIT, OVER WHAT POPULATION.
#
# That is a coverage limit, not corruption: a battle with no address needs no
# journal, because nothing points at it to replay. It IS the reason to reach for
# `--opponent-journal full` whenever the question is "which decisions of this class
# can I replay" rather than "can I replay the addresses I have".
OPPONENT_JOURNAL_MODES = ("off", "addressed", "full")

# The REFUSAL RECORDER, wired on by default.
#
# `fallback_samples` files an ADDRESS -- `{battle_id, round, seat, reason}` -- and
# nothing else. Reading the state behind that address then requires replaying the
# battle, and for a bridge shard the battle does not replay: foul-play's search is
# wall-clock-budgeted over an unseeded `rand::rng()` (see the journal block above).
# So for the corpus this bridge produces, "replay the address and look" is not a
# thing anyone can do, and four eras of refusals were theorised about instead.
#
# `fallback_replay.attach_refusal_recorder` (#1180) captures the same state LIVE, at
# the moment of the refusal, where no replay is needed: the world-failure classes
# that fired ON THAT DECISION with counts, the worlds attempted/constructed/searched,
# the engine's proposed choices, the request's legal set in the ENGINE'S vocabulary,
# their disagreement, and the decision RNG seed. That is why this is on by default
# and not behind an opt-in flag -- an instrument nobody remembers to switch on
# records nothing, and every shard produced without it is another era of addresses
# that cannot be read.
#
# COST. Two axes, both measured, neither asserted.
#
# 1. DECISION TIME. The recorder's work is three bound-method wrappers around
#    `select_action_with_context`, `_map_choices` and `_fallback`; it sits OUTSIDE
#    the timed native call and touches no RNG. TWO per-decision costs, not one: the
#    first two wrappers run on every decision and the third only on a refusal, so
#    quoting the cheap one alone would understate exactly the run this exists for.
#    Measured (`scripts/refusal_recorder_cost.py --decisions 20000 --repeats 5`, four
#    invocations on this checkout, median of five timing runs each):
#
#      non-refusing decision (snapshot only)     1.03 - 1.25 us
#      refusing decision (snapshot + record)    11.16 - 13.77 us
#
#    Against the decision boundary this bridge actually runs -- a prior measurement
#    put the median at 5.02 s on our side, plus foul-play's fixed 1,000 ms search --
#    that is 2.1e-7 to 2.5e-7 of the boundary for a non-refusing decision and
#    2.2e-6 to 2.7e-6 for a refusing one. Blended at the measured era-64 cell-D
#    fallback rate of 1.00% (55 fallbacks in 5,513 decisions): 1.13 - 1.36 us, i.e.
#    about 2.5e-7 of the boundary.
#
# 2. SUMMARY BYTES, measured the way the journal block measures them, because that
#    is the axis that caught out the last author. `_write_json` rewrites the ENTIRE
#    summary after every game when `--summary-out` is set, so bytes re-serialize
#    O(games^2) and the document size, not the write count, is what scales.
#
#    Parameters taken from the REAL era-64 cell-D shards (`fp-d-probe-*-p{1,2}.json`,
#    16 seat shards, 128 games, 5,513 decisions, 55 fallback decisions): 0.430
#    refusals per game, 2,684 B mean per-game row, 37,270 B median seat shard.
#    Record sizes taken from 93 REAL records captured by this recorder on a local
#    d4/s1024/w4 batch, not from a fabricated one:
#
#      per refusal record          550 B median serialized (469 min, 777 max)
#      per game                   +236 B  (+8.8% of a 2,684 B row)
#      8-game seat shard        +1,892 B on 37,270 B: +5.1%
#
#    The O(games^2) axis, same per-game rate, cumulative bytes written:
#
#      --games 250   off 88.2 MB   refusals-on 95.6 MB   (+7.4 MB, +8.4%)
#
#    THAT +5.1% IS A RATE-CONDITIONAL NUMBER AND IT DOES NOT BOUND ANYTHING. A
#    refusal record is EXPENSIVE (550 B, ~4x a journal entry) and cell D's refusals
#    are RARE (0.430 per game); the product is small, and the product is all the
#    +5.1% is. Scale the rate and it scales linearly with no ceiling: at 4.3
#    refusals per game the shard is +51%, and a 100% fallback rate -- which is a
#    REAL state this very file documents (`--no-search-fallback` exists because a
#    searcher can play uniform-legal on every decision) -- puts 43 records on every
#    game row and the cumulative write into the hundreds of MB.
#
#    So the payload is CAPPED rather than argued about, the way the producer it
#    mirrors caps (`_FALLBACK_SAMPLES_PER_CLASS = 3` plus the one-address-per-battle
#    rule, both in `EngineMctsPolicy._fallback`). See `_REFUSAL_RECORDS_PER_BATTLE` and
#    `_REFUSAL_RECORDS_PER_RUN` below for the ceilings and the arithmetic that
#    picked them. What the cap drops is COUNTED and published, because
#    `fallback_sample_addresses_dropped` is right there as the example of a
#    truncation nobody can see.
#
#    What the ceilings actually buy, same tool, `--refusals-per-game 43` (the 100%
#    fallback rate), 250 games:
#
#                        off        on (capped)      if UNCAPPED
#      cumulative     88.2 MB       105.4 MB          830.2 MB
#      8-game shard        -          +94.4%          +963%
#
#    The reserve is also CHEAPER than the prefix it replaced (105.4 vs 120.5 MB),
#    which is not a coincidence: a prefix front-loads the whole payload into the
#    earliest writes and `_write_json` re-serializes it once per remaining game.
#    The unbiased sample is the cheap one.
#
#    At cell D's real 0.430 they agree exactly (+7.4 MB either way): the cap cannot
#    bite on the case the default was justified against, and removes the tail on the
#    case it was not.
#
#    The run budget is spent as an even PER-GAME RESERVE, not as a prefix -- a
#    prefix bounds the bytes and biases the sample, which for an instrument whose
#    job is to characterise a cell is the wrong trade. See `_refusal_allowance`.
#
#    And the loss is reported as TWO numbers, never one. A truncation to a published
#    ceiling and a record that reached no game row are both "records the document
#    does not carry", and a single count made them indistinguishable -- with the
#    ceilings published right beside it, so a reader would attribute either to them.
#    `records_dropped_to_ceiling` is bounded and does not condemn the run;
#    `records_unrowed` is unexplained data loss and blocks `trustworthy`.
#
#    `--no-refusal-records` still exists, but it is NOT the answer to the size
#    question and must not be sold as one: switching it off requires knowing in
#    advance that you are in a high-refusal cell, which is the thing the recorder
#    exists to discover. The cap is what makes the default safe; the flag is for a
#    caller who wants the bytes back on a cell they have already read.
#
# HEALTH IS NEVER SILENT. `RefusalRecorder` deliberately does not raise into the
# search -- an instrument that turns a handled refusal into a crash has changed the
# outcome it exists to explain -- so a capture failure lands on `recorder.errors`
# and an empty record list would otherwise read as "this run had no refusals".
# `RefusalRecorderHealth` carries `attached`, `attach_error`, `health_reported`,
# `instrument_errors` and `degraded_records` into the summary, and `trustworthy` is
# a CONJUNCTION with `health_reported` so that silence reads as unknown rather than
# as clean -- the same rule `ReplayResult.trustworthy` states.
#
# INVISIBLE TO THE ADDRESS READER. `fallback_addresses` accepts a mapping as a
# cumulative stats scope iff it CONTAINS `fallback_samples` (`_walk_stats_blocks`)
# and harvests addresses from any mapping so NAMED (`_walk_sample_blocks`). The
# header is `refusal_recorder`, the rows are `refusals`, and no key inside a
# serialized `RefusalRecord` is `fallback_samples`, `world_failure_reasons` or
# `fallback_reasons` -- the record spells its own delta `world_failures`. So the
# block can add neither a scope, nor an address, nor an occurrence count.
# `RefusalRecordReaderInvariantTest` pins that against a real `scan_corpus`.
REFUSAL_RECORDER_SCHEMA_VERSION = "pokezero.refusal-records.v1"

# Ceilings on what reaches the DOCUMENT. Neither bounds the recorder, which is
# #1180's object and keeps everything it captures in memory (250 records is ~140 KB
# of Python objects; the problem was never RAM). They bound the summary, because
# `_write_json` re-serializes the whole document once per game.
#
# PER BATTLE, and this one is read off real records rather than guessed. A refusal
# cause typically closes worlds for the rest of the battle it appears in: measured on
# a local d4/s1024/w4 batch, seed 600016 filed TEN consecutive `no_worlds_constructed`
# refusals at rounds 7-16, every one of them `attempted=16 constructed=0` with the
# same two world-failure classes and the same request legal set. Rounds 11-16 are six
# more views of one incident. Eight is deliberately not one: unlike the address store,
# whose job is a replay coordinate, a record's job is to be READ, and the first few
# repeats are what shows the reader that it repeats -- the run of identical rounds is
# itself the evidence that the ability in the message is a bystander. Eight keeps that
# and drops the tail.
_REFUSAL_RECORDS_PER_BATTLE = 8

# PER RUN. 250 records x 550 B median = ~138 KB added to the document, so the
# O(games^2) contribution over a 250-game run is bounded by 250 x 138 KB = ~34 MB
# whatever the fallback rate does -- against ~978 MB uncapped at a 100% rate. It is
# also comfortably above what an ordinary cell produces: cell D's measured 0.430
# refusals per game reaches 108 over a full 250-game run, so the cap cannot bite on
# the case the default was justified against.
#
# SPENT AS A PER-GAME RESERVE, NOT FIRST-COME-FIRST-SERVED. A run ceiling taken as a
# prefix exhausts after `_REFUSAL_RECORDS_PER_RUN / _REFUSAL_RECORDS_PER_BATTLE`
# games -- 31.25 here. Measured over 34 high-refusal games it emitted
# `[8]*31 + [2, 0, 0]`, so on a 250-game run 219 games would contribute nothing and
# the shard's entire refusal evidence would come from the first 12% of the seed band.
# For an instrument whose job is to CHARACTERISE A CELL that is a biased sample
# presented as the run's records, and counting the loss does not fix a bias --
# it documents one. `_refusal_allowance` spends the budget evenly instead.
_REFUSAL_RECORDS_PER_RUN = 250

# Instrument errors are uncapped on the recorder and one string per failed capture:
# 500 degraded decisions produced a 47 KB header, re-serialized into every progress
# write. The header keeps a sample and the TOTAL, so the number is never lost even
# though the strings are. SAMPLED BY STRIDE, not by prefix, for the same reason the
# record budget is: the first 20 failures of a run that failed 500 times are 20 views
# of whatever went wrong first.
_INSTRUMENT_ERRORS_IN_HEADER = 20


def _strided_sample(values: Sequence[str], limit: int) -> tuple[str, ...]:
    """At most ``limit`` entries, evenly spaced, always including first and last."""
    if len(values) <= limit:
        return tuple(values)
    if limit <= 1:
        return (values[0],)
    last = len(values) - 1
    return tuple(values[round(index * last / (limit - 1))] for index in range(limit))


def _refusal_allowance(*, room: int, games_remaining: int) -> int:
    """How many records THIS game may contribute.

    ``room`` is what is left of the run budget and ``games_remaining`` counts this
    game. The even share is ``room // games_remaining``, clamped to the per-battle
    ceiling above and floored at one so no game is silently excluded from the
    sample. Slack returns to the pool automatically: a game that files fewer than
    its share leaves ``room`` higher, so the share for the games after it rises.

    The floor is what makes this unbiased rather than merely bounded. On a short run
    -- the 8-game invocation `foulplay_paired_eval` actually issues -- the even share
    is 31, so the clamp to `_REFUSAL_RECORDS_PER_BATTLE` binds first and the reserve
    changes nothing at all. It only engages once games outnumber the budget, which is
    exactly the case where a prefix would have thrown away the tail of the band.
    """
    if games_remaining <= 0:
        return min(_REFUSAL_RECORDS_PER_BATTLE, max(0, room))
    return max(0, min(_REFUSAL_RECORDS_PER_BATTLE, max(1, room // games_remaining), room))


@dataclass(frozen=True)
class ControlledFoulPlayConfig:
    checkpoint: Path | None
    showdown_root: Path
    value_checkpoint: Path | None = None
    foulplay_root: Path = DEFAULT_FOULPLAY_ROOT
    foulplay_python: Path | None = None
    games: int = 1
    seed_start: int = 1
    foulplay_random_seed: int | None = None
    search_time_ms: int = 1000
    max_decision_rounds: int = 250
    format_id: str = "gen3randombattle"
    policy_mode: str = "root-puct"
    # Frozen Rust search configuration (policy_mode='engine-mcts'). These are the
    # axes of the study's config_id; every one changes search semantics or wall
    # time, so they are part of the frozen contract, never defaulted silently.
    engine_model_path: Path | None = None
    engine_tables_path: Path | None = None
    engine_depth: int = 4
    engine_sims: int = 1024
    engine_batch: int = 16
    engine_worlds: int = 4
    engine_c_puct: float = 1.4
    engine_model_priors: bool = True
    # Opponent-side model priors in the native search (campaign cells B/E).
    # Default OFF -- flag-off must reproduce the uniform-opponent search.
    engine_opponent_priors: bool = False
    device: str | None = None
    temperature: float = 1.0
    cpuct: float = 1.25
    selection_mode: str = "visits"
    root_prior_temperature: float | None = None
    minimum_value_improvement: float | None = None
    minimum_override_prior_ratio: float | None = None
    minimum_score_improvement: float | None = None
    root_visit_budget: int | None = 16
    root_extra_visits: int | None = None
    adaptive_root_contested_extra_visits: int | None = None
    adaptive_root_uncontested_extra_visits: int = 0
    adaptive_root_policy_entropy_threshold: float | None = None
    adaptive_root_value_margin_threshold: float | None = None
    root_time_budget_ms: int | None = None
    root_opponent_action_scenarios: int = 1
    root_opponent_action_candidate_scenarios: int = ACTION_COUNT
    leaf_rollout_rounds: int = 0
    leaf_rollout_sampling: bool = False
    belief_start_overrides: bool = False
    start_override_attempts: int = DEFAULT_START_OVERRIDE_ATTEMPTS
    belief_start_override_samples: int = 1
    start_override_hp_fraction_tolerance: float = 0.02
    opponent_legal_mask_mode: str = "hidden"
    opponent_crash_retries: int = 1
    # Candidate-set source for player-relative belief views. None defers to the
    # POKEZERO_BELIEF_SET_SOURCE env gate (the single flip point shared with local_showdown), so
    # benchmark observations match training observations without per-harness wiring.
    belief_set_source: bool | None = None
    allow_search_fallback: bool = True
    node_binary: str = "node"
    pokezero_username: str = "PokeZeroBot"
    foulplay_username: str = "FoulPlayBot"
    pokezero_player: PlayerId = "p1"
    websocket_host: str = "127.0.0.1"
    # Collision auditing can intentionally explore with a non-trained policy so
    # it never has to substitute a legacy-schema checkpoint just to emit v3
    # observations. This is capture-only, not a strength-evaluation mode.
    capture_driver: str = "checkpoint"
    audit_observation_schema: str | None = None
    # One of OPPONENT_JOURNAL_MODES; see the module-level block for the measurement
    # behind the default.
    opponent_journal: str = "addressed"
    # Attach the #1180 refusal recorder to the engine policy and write what it
    # captures into the summary. ON, because the state behind a bridge refusal is
    # not recoverable any other way -- see the REFUSAL RECORDER block above for the
    # cost and size measurements behind that default, and `--no-refusal-records`
    # for the way out on a long run.
    record_refusals: bool = True

    def __post_init__(self) -> None:
        if self.games <= 0:
            raise ValueError("games must be positive.")
        if self.seed_start < 0:
            raise ValueError("seed_start must be non-negative.")
        if self.foulplay_random_seed is not None and self.foulplay_random_seed < 0:
            raise ValueError("foulplay_random_seed must be non-negative when set.")
        if self.search_time_ms <= 0:
            raise ValueError("search_time_ms must be positive.")
        if self.max_decision_rounds <= 0:
            raise ValueError("max_decision_rounds must be positive.")
        if self.policy_mode not in {"raw", "root-puct", "engine-mcts"}:
            raise ValueError("policy_mode must be 'raw', 'root-puct', or 'engine-mcts'.")
        if self.policy_mode == "engine-mcts":
            missing = [
                name for name in ("engine_model_path", "engine_tables_path")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"policy_mode='engine-mcts' requires {', '.join(missing)} "
                    "(export them with mcts_eval.materialize_search_artifacts)."
                )
        if self.opponent_journal not in OPPONENT_JOURNAL_MODES:
            raise ValueError(
                "opponent_journal must be one of "
                f"{', '.join(OPPONENT_JOURNAL_MODES)}, got {self.opponent_journal!r}."
            )
        if self.capture_driver not in {"checkpoint", "random-legal"}:
            raise ValueError("capture_driver must be 'checkpoint' or 'random-legal'.")
        if self.capture_driver == "checkpoint":
            if self.checkpoint is None:
                raise ValueError("checkpoint capture requires a checkpoint.")
            if self.audit_observation_schema is not None:
                raise ValueError("audit_observation_schema is only valid for random-legal capture.")
        else:
            if self.checkpoint is not None:
                raise ValueError("random-legal capture must not provide a checkpoint.")
            if self.value_checkpoint is not None:
                raise ValueError("random-legal capture cannot use a value checkpoint.")
            if self.policy_mode != "raw":
                raise ValueError("random-legal capture requires policy_mode='raw'.")
            if self.audit_observation_schema != "v3":
                raise ValueError("random-legal capture requires audit_observation_schema='v3'.")
        if self.selection_mode not in {"puct", "value", "visits"}:
            raise ValueError("selection_mode must be 'puct', 'value', or 'visits'.")
        if self.root_prior_temperature is not None and (
            self.root_prior_temperature <= 0.0 or not math.isfinite(self.root_prior_temperature)
        ):
            raise ValueError("root_prior_temperature must be a finite positive value when set.")
        if self.minimum_value_improvement is not None and (
            self.minimum_value_improvement < 0.0 or not math.isfinite(self.minimum_value_improvement)
        ):
            raise ValueError("minimum_value_improvement must be a finite non-negative value when set.")
        if self.minimum_override_prior_ratio is not None and (
            self.minimum_override_prior_ratio < 0.0 or not math.isfinite(self.minimum_override_prior_ratio)
        ):
            raise ValueError("minimum_override_prior_ratio must be a finite non-negative value when set.")
        if self.minimum_score_improvement is not None and (
            self.minimum_score_improvement < 0.0 or not math.isfinite(self.minimum_score_improvement)
        ):
            raise ValueError("minimum_score_improvement must be a finite non-negative value when set.")
        if self.root_visit_budget is not None and self.root_visit_budget <= 0:
            raise ValueError("root_visit_budget must be positive when set.")
        if self.root_extra_visits is not None and self.root_extra_visits < 0:
            raise ValueError("root_extra_visits must be non-negative when set.")
        adaptive_configured = self.adaptive_root_contested_extra_visits is not None
        adaptive_threshold_configured = (
            self.adaptive_root_policy_entropy_threshold is not None
            or self.adaptive_root_value_margin_threshold is not None
        )
        if self.root_extra_visits is not None and (
            adaptive_configured
            or adaptive_threshold_configured
            or self.adaptive_root_uncontested_extra_visits != 0
        ):
            raise ValueError("root_extra_visits cannot be combined with adaptive root budgeting.")
        if not adaptive_configured and (
            adaptive_threshold_configured or self.adaptive_root_uncontested_extra_visits != 0
        ):
            raise ValueError(
                "adaptive root thresholds and uncontested extra visits require "
                "adaptive_root_contested_extra_visits."
            )
        if adaptive_configured:
            EntropyMarginVisitBudgetSelector(
                contested_extra_visits=self.adaptive_root_contested_extra_visits,
                uncontested_extra_visits=self.adaptive_root_uncontested_extra_visits,
                minimum_policy_entropy=self.adaptive_root_policy_entropy_threshold,
                maximum_value_margin=self.adaptive_root_value_margin_threshold,
            )
        if self.root_time_budget_ms is not None and self.root_time_budget_ms <= 0:
            raise ValueError("root_time_budget_ms must be positive when set.")
        if self.root_time_budget_ms is not None and (self.root_extra_visits is not None or adaptive_configured):
            raise ValueError(
                "root_time_budget_ms cannot be combined with fixed or adaptive post-sweep visit budgets."
            )
        if self.root_time_budget_ms is not None:
            # Time-bounded search must not inherit the default 16-visit cap.
            object.__setattr__(self, "root_visit_budget", None)
        if self.root_opponent_action_scenarios <= 0:
            raise ValueError("root_opponent_action_scenarios must be positive.")
        if self.root_opponent_action_candidate_scenarios <= 0:
            raise ValueError("root_opponent_action_candidate_scenarios must be positive.")
        if self.root_opponent_action_candidate_scenarios < self.root_opponent_action_scenarios:
            raise ValueError(
                "root_opponent_action_candidate_scenarios must be greater than or equal to "
                "root_opponent_action_scenarios."
            )
        if self.leaf_rollout_rounds < 0:
            raise ValueError("leaf_rollout_rounds must be non-negative.")
        if self.leaf_rollout_sampling and self.leaf_rollout_rounds <= 0:
            raise ValueError("leaf_rollout_sampling requires positive leaf_rollout_rounds.")
        if self.start_override_attempts <= 0:
            raise ValueError("start_override_attempts must be positive.")
        if self.belief_start_override_samples <= 0:
            raise ValueError("belief_start_override_samples must be positive.")
        if self.belief_start_override_samples > 1 and not self.belief_start_overrides:
            raise ValueError("belief_start_override_samples requires belief_start_overrides.")
        if self.start_override_hp_fraction_tolerance < 0.0 or not math.isfinite(
            self.start_override_hp_fraction_tolerance
        ):
            raise ValueError("start_override_hp_fraction_tolerance must be a finite non-negative value.")
        if self.opponent_legal_mask_mode not in {"hidden", "privileged"}:
            raise ValueError("opponent_legal_mask_mode must be 'hidden' or 'privileged'.")
        if self.opponent_crash_retries < 0:
            raise ValueError("opponent_crash_retries must be non-negative.")
        if self.pokezero_player not in {"p1", "p2"}:
            raise ValueError("pokezero_player must be 'p1' or 'p2'.")

    @property
    def resolved_foulplay_python(self) -> Path:
        if self.foulplay_python is not None:
            return self.foulplay_python
        return self.foulplay_root / ".venv" / "bin" / "python"

    @property
    def resolved_foulplay_random_seed(self) -> int:
        if self.foulplay_random_seed is not None:
            return self.foulplay_random_seed
        return self.seed_start

    def belief_set_source_enabled(self) -> bool:
        if self.belief_set_source is not None:
            return self.belief_set_source
        return belief_set_source_env_enabled()

    @property
    def effective_root_prior_temperature(self) -> float:
        if self.root_prior_temperature is not None:
            return self.root_prior_temperature
        return self.temperature

    def root_visit_budget_selector(
        self,
    ) -> FixedExtraVisitBudgetSelector | EntropyMarginVisitBudgetSelector | None:
        if self.root_extra_visits is not None:
            return FixedExtraVisitBudgetSelector(extra_visits=self.root_extra_visits)
        if self.adaptive_root_contested_extra_visits is None:
            return None
        return EntropyMarginVisitBudgetSelector(
            contested_extra_visits=self.adaptive_root_contested_extra_visits,
            uncontested_extra_visits=self.adaptive_root_uncontested_extra_visits,
            minimum_policy_entropy=self.adaptive_root_policy_entropy_threshold,
            maximum_value_margin=self.adaptive_root_value_margin_threshold,
        )

    @property
    def foulplay_player(self) -> PlayerId:
        return "p2" if self.pokezero_player == "p1" else "p1"


@dataclass(frozen=True)
class OpponentJournalEntry:
    """One opponent move, as SUBMITTED -- never inferred from the transcript.

    ``choice`` is the ``/choose`` body lifted off foul-play's outgoing websocket
    message (``_choice_body_from_outgoing_message``) and handed straight to the
    BattleStream in the ``choices`` event. There is no second derivation of it, so
    it cannot disagree with what was played.

    Field by field, each earning its bytes:

    ``round``
        ``decision_round``, the SAME counter that keys ``FallbackAddress.round``:
        the bridge passes it as ``PolicyContext.decision_round_index`` and
        ``engine_search._fallback`` files it verbatim. It increments once per
        request boundary, so a force-switch is its own round. Recorded EXPLICITLY
        rather than implied by list position, because the opponent is journalled
        only on rounds where it is in ``requested_players`` -- our own force-switch
        rounds leave a hole, and positional indexing would silently shift every
        later move by one.
    ``seat``
        The opponent's seat. ``battle_id`` is ``f"{prefix}-{seed}"`` and carries the
        seed and nothing else, while ``foulplay_paired_eval`` runs the SAME seed
        band from BOTH seats -- so ``battle-...-7800000`` exists in two summaries
        with the opponent on opposite sides. Without the seat, pooling two seat
        summaries silently mixes two different battles. This is the same reason
        ``FallbackAddress`` carries a seat.
    ``action``
        The decoded 0-8 action index. Already computed at the recording site to
        build the trajectory step, so it is free, and it lets a consumer key on our
        action space without re-implementing ``action_index_from_choice_string``.
    ``request_sha256``
        Digest (first 12 hex) of the raw ``|request|`` line this choice answers.
        A replay must FAIL CLOSED, not play on: ``move 1`` is 1-based into *that*
        request's active-move list, so applying it to a state that has drifted
        picks a different move and produces a confident, wrong replay. The digest
        is over the raw BattleStream line, not the copy forwarded to foul-play,
        because the forwarded copy carries a bridge-assigned rqid that is an
        artifact of the bridge rather than of the battle.
    """

    round: int
    seat: PlayerId
    choice: str
    action: int
    request_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "seat": self.seat,
            "choice": self.choice,
            "action": self.action,
            "request_sha256": self.request_sha256,
        }


def _request_digest(request_line: str | None) -> str:
    """Short digest of the request a recorded choice answers, or "" if absent.

    Empty rather than absent-key or a fake digest: a consumer that cannot verify the
    state must be able to SEE that it cannot, and "" is not a valid sha256 prefix, so
    it can never be mistaken for a match.
    """
    if not request_line:
        return ""
    return hashlib.sha256(request_line.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class RefusalRecorderHealth:
    """Whether the refusal capture can be believed -- and whether it ran at all.

    Every field here exists because its absence has a wrong reading:

    ``enabled``
        The run was configured to record. ``False`` means ``--no-refusal-records``,
        not "no refusals".
    ``attached``
        The recorder actually installed on the policy. It CANNOT install on a raw
        or root-PUCT policy -- ``RefusalRecorder._validate`` requires ``stats`` plus
        callable ``select_action_with_context``/``_map_choices``/``_fallback``, and
        only ``EngineMctsPolicy`` has them -- and those arms file no
        ``fallback_samples`` either, so there is nothing to record. That is a
        correct outcome, but it must not look like a clean engine-mcts run.
    ``attach_error``
        Why not, verbatim. A boolean alone sends the reader to the source.
    ``health_reported``
        Whether this result had an error channel at all. Defaults ``False`` and is
        a conjunct of :attr:`trustworthy` for the reason
        :class:`fallback_replay.BattleRun` records at length: an empty error list
        from a runner with no error channel says nothing, and reading it as healthy
        is the same defect as reading an empty record list as "no refusals".
    ``instrument_errors``
        `RefusalRecorder.errors`. Non-empty means SOME RECORD IS INCOMPLETE. The
        recorder never raises into the search -- an instrument that turns a handled
        refusal into a crash has changed the outcome it exists to explain -- so this
        list is the only place a swallowed capture failure surfaces.
    ``degraded_records``
        Records whose pre-decision baseline was lost, so their deltas may span more
        than one decision. Counted separately from errors because the record is
        still worth reading, just not as a per-decision measurement.
    ``recorded_refusals`` / ``emitted_refusals``
        What the instrument SAW and what the document CARRIES. The journal's
        ``recorded`` vs ``emitted`` pair is the same idea one feature earlier.
    ``records_dropped_to_ceiling`` vs ``records_unrowed``
        TWO NUMBERS, AND THEY MUST NOT RENDER ALIKE. Both are records that did not
        reach the document and a single total conflated them:

        * ``records_dropped_to_ceiling`` is bounded, expected, and its ceilings are
          published right here -- a reader can reconstruct exactly what happened. It
          does NOT block :attr:`trustworthy`.
        * ``records_unrowed`` is DATA LOSS: a refusal filed under a battle that
          produced no game row, because the game raised after refusing or a
          battle_id drifted. Nothing bounds it and nothing explains it. It blocks
          :attr:`trustworthy`.

        Under one combined field the two read identically (`dropped=6` vs
        `dropped=8`, both `trustworthy: True`), and since the header publishes the
        ceilings a reader would attribute either to them. Worse, the end-of-run
        accounting that keeps the identity true was UPGRADING a correct False to
        True: the partial summaries said `reconciled: False, trustworthy: False`
        after each game and the fixup made the final one clean. On a feature whose
        organising principle is that silence must never read as clean, converting a
        detected loss into a clean verdict is the wrong direction even with a number
        left behind.
    ``reconciled``
        ``recorded == emitted + dropped_to_ceiling + unrowed``. Book-keeping that
        every captured record is accounted for somewhere, so a NEW way of losing one
        shows up as a broken identity rather than as an absence.
    ``instrument_errors_total``
        The full count. ``instrument_errors`` is a SAMPLE (`_INSTRUMENT_ERRORS_IN_HEADER`),
        because the list is one string per failed capture, uncapped, and it is
        re-serialized into every progress write.
    """

    enabled: bool = False
    attached: bool = False
    attach_error: str | None = None
    health_reported: bool = False
    instrument_errors: tuple[str, ...] = ()
    instrument_errors_total: int = 0
    degraded_records: int = 0
    recorded_refusals: int = 0
    emitted_refusals: int = 0
    records_dropped_to_ceiling: int = 0
    records_unrowed: int = 0

    @property
    def reconciled(self) -> bool:
        """Every captured record reached a row, a ceiling, or the unrowed count."""
        return self.recorded_refusals == (
            self.emitted_refusals + self.records_dropped_to_ceiling + self.records_unrowed
        )

    @property
    def trustworthy(self) -> bool:
        """True only when health was REPORTED, the recorder ran, and it was clean.

        ``records_dropped_to_ceiling`` is deliberately NOT a conjunct: a truncation
        to a published ceiling is a documented, reconstructible loss, and making it
        block the verdict would mean every high-refusal cell reported an untrustworthy
        instrument for behaving exactly as designed. ``records_unrowed`` IS a
        conjunct, because nothing bounds or explains it.

        ``enabled`` is a conjunct as well as ``attached``. They are not the same
        claim and only one of them is currently implied by the other: a future
        caller that constructs this object by hand can set ``attached`` without
        ``enabled``, and "the recorder was switched off" must never be able to
        render as a trustworthy measurement of anything.
        """
        return (
            self.enabled
            and self.health_reported
            and self.attached
            and self.reconciled
            and self.records_unrowed == 0
            and not self.instrument_errors_total
            and self.degraded_records == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REFUSAL_RECORDER_SCHEMA_VERSION,
            # Where the rows live. Same contract as the journal header: every
            # per-game key this feature writes is `records_key` or
            # `records_key + "_" + <suffix>`, so a consumer scans the row for the
            # prefix rather than being told each name.
            "records_key": "refusals",
            "enabled": self.enabled,
            "attached": self.attached,
            "attach_error": self.attach_error,
            "health_reported": self.health_reported,
            "instrument_errors": list(self.instrument_errors),
            "instrument_errors_total": self.instrument_errors_total,
            "degraded_records": self.degraded_records,
            "recorded_refusals": self.recorded_refusals,
            "emitted_refusals": self.emitted_refusals,
            # Deliberately NOT summed into one `records_dropped`: see the class
            # docstring. A bounded truncation and a data loss must not render alike.
            "records_dropped_to_ceiling": self.records_dropped_to_ceiling,
            "records_unrowed": self.records_unrowed,
            # The ceilings, published so a reader of a truncated shard can tell
            # WHICH bound bit without reading this file.
            "records_per_battle_ceiling": _REFUSAL_RECORDS_PER_BATTLE,
            "records_per_run_ceiling": _REFUSAL_RECORDS_PER_RUN,
            "reconciled": self.reconciled,
            "trustworthy": self.trustworthy,
        }


class _RefusalCapture:
    """Owns one :func:`attach_refusal_recorder` for the lifetime of a bridge run.

    Attached once around the whole game loop rather than per game, because the
    bridge builds ONE policy and reuses it for every seed -- attaching per game
    would install and tear down a wrapper 250 times and, worse, would silently
    depend on the teardown order that ``_Hook`` exists to make irrelevant. Records
    are partitioned back out per game by ``battle_id``, which is
    ``f"{DEFAULT_BATTLE_ID_PREFIX}-{seed}"`` and unique within a run.

    Attach failure is CAPTURED, NEVER RAISED. The recorder validates the policy at
    attach time, and on a raw or root-PUCT arm that validation fails by design.
    Letting that AttributeError escape would make a diagnostic switched on by
    default able to abort a strength benchmark -- so it lands on
    :attr:`attach_error` and the run continues, with the summary saying so.
    """

    def __init__(self, policy: Any, *, enabled: bool, games: int = 1) -> None:
        self.enabled = bool(enabled)
        self._recorder: Any = None
        self._attach_error: str | None = None
        #: How many games the run intends to play, so the per-run budget can be
        #: spent as an even reserve instead of a prefix. See `_refusal_allowance`.
        self._games = max(1, int(games))
        self._rows_built = 0
        #: Emitted and dropped are decided HERE, at the moment a game row is built,
        #: and remembered -- not recomputed from the recorder at `health()` time.
        #: Recomputing would have to re-derive which records a previous row already
        #: took, and getting that wrong is exactly the silent-truncation shape the
        #: reconciliation exists to catch.
        self._emitted = 0
        self._dropped_to_ceiling = 0
        self._unrowed = 0
        if not self.enabled:
            return
        try:
            self._recorder = attach_refusal_recorder(policy)
        except Exception as error:  # noqa: BLE001 -- an instrument must not abort the run
            self._attach_error = f"{type(error).__name__}: {error}"

    def records_for(self, battle_id: str) -> tuple[RefusalRecord, ...]:
        """This battle's records, truncated to its ALLOWANCE, with the loss counted.

        Called ONCE per game as its row is built, and it advances the run's budget
        bookkeeping -- so it is not idempotent and must not be called twice for the
        same game.

        The allowance is an even share of what is left, not a prefix of the run
        budget: see :func:`_refusal_allowance`. Everything it refuses is counted as
        `records_dropped_to_ceiling`, which is bounded and reconstructible and
        therefore does NOT block `trustworthy`.
        """
        if self._recorder is None:
            return ()
        self._rows_built += 1
        mine = [
            record for record in self._recorder.records if record.battle_id == battle_id
        ]
        allowance = _refusal_allowance(
            room=max(0, _REFUSAL_RECORDS_PER_RUN - self._emitted),
            games_remaining=self._games - self._rows_built + 1,
        )
        kept = tuple(mine[:allowance])
        self._emitted += len(kept)
        self._dropped_to_ceiling += len(mine) - len(kept)
        return kept

    def account_for_unrowed_records(self) -> None:
        """Count records whose battle never produced a row, at end of run.

        A game that raised before its result was built takes its records with it.
        This keeps the reconciliation identity true so that a FUTURE way of losing a
        record still shows up as a broken identity -- but it files them under
        ``records_unrowed``, NOT under the ceiling count.

        That distinction is the whole point. An earlier revision folded these into
        the ceiling total, which made the end-of-run summary report a clean,
        trustworthy instrument for a run that had demonstrably lost data -- and it
        did so by UPGRADING the honest `trustworthy: False` the partial summaries had
        already published. Reconciling a loss is not the same as excusing it.
        """
        if self._recorder is None:
            return
        unaccounted = (
            len(self._recorder.records)
            - self._emitted
            - self._dropped_to_ceiling
            - self._unrowed
        )
        if unaccounted > 0:
            self._unrowed += unaccounted

    def health(self) -> RefusalRecorderHealth:
        if self._recorder is None:
            return RefusalRecorderHealth(
                enabled=self.enabled, attached=False, attach_error=self._attach_error
            )
        records = self._recorder.records
        errors = tuple(self._recorder.errors)
        return RefusalRecorderHealth(
            enabled=self.enabled,
            attached=True,
            attach_error=None,
            # The recorder HAS an error channel and we are reading it, which is
            # exactly what this flag asserts.
            health_reported=True,
            instrument_errors=_strided_sample(errors, _INSTRUMENT_ERRORS_IN_HEADER),
            instrument_errors_total=len(errors),
            degraded_records=sum(1 for record in records if record.degraded),
            recorded_refusals=len(records),
            emitted_refusals=self._emitted,
            records_dropped_to_ceiling=self._dropped_to_ceiling,
            records_unrowed=self._unrowed,
        )

    def detach(self) -> None:
        if self._recorder is not None:
            self._recorder.detach()


@dataclass(frozen=True)
class ControlledFoulPlayGameResult:
    battle_id: str
    seed: int
    winner: str | None
    pokezero_won: bool
    decision_rounds: int
    pokezero_decisions: int
    root_puct_searches: int
    root_puct_fallbacks: int
    # engine-mcts (policy_mode='engine-mcts') counterparts. The native searcher
    # records its own fallbacks in decision metadata exactly as Root-PUCT does;
    # without these the FoulPlay summary reported no fallback at all for that
    # mode, so a run could not tell a clean measurement from one where the
    # searcher was playing uniform-legal on FoulPlay-side states.
    engine_mcts_decisions: int = 0
    engine_mcts_fallbacks: int = 0
    engine_mcts_fallback_reasons: Mapping[str, int] = field(default_factory=dict)
    # Actual planner identities emitted by executed Root-PUCT decisions. This
    # is evidence for the primary hidden-information capstone contract.
    root_puct_opponent_action_policies: Mapping[str, int] = field(default_factory=dict)
    root_puct_total_visits: int = 0
    root_puct_effective_total_visits: int = 0
    root_puct_opponent_action_scenarios_generated: int = 0
    root_puct_opponent_action_scenarios_skipped: int = 0
    root_puct_opponent_action_scenarios_unsearched: int = 0
    root_puct_opponent_action_skip_categories: Mapping[str, int] = field(default_factory=dict)
    root_puct_opponent_action_missing_sampled_world_reason_categories: Mapping[str, int] = field(
        default_factory=dict
    )
    root_puct_opponent_action_replay_rejection_decision_rounds: Mapping[str, int] = field(
        default_factory=dict
    )
    root_puct_opponent_action_replay_request_mismatch_decision_rounds: Mapping[str, int] = field(
        default_factory=dict
    )
    root_puct_opponent_action_replay_request_mismatch_players: Mapping[str, int] = field(
        default_factory=dict
    )
    root_puct_opponent_action_replay_request_mismatch_shapes: Mapping[str, int] = field(
        default_factory=dict
    )
    root_puct_opponent_action_start_override_mismatch_decision_rounds: Mapping[str, int] = field(
        default_factory=dict
    )
    root_puct_opponent_action_first_observation_mismatch_paths: Mapping[str, int] = field(default_factory=dict)
    root_puct_opponent_action_groups_generated: int = 0
    root_puct_opponent_action_groups_used: int = 0
    root_puct_opponent_action_groups_skipped: int = 0
    root_puct_opponent_action_groups_unsearched: int = 0
    root_puct_selected_prior_action_changes: int = 0
    root_puct_pre_gate_prior_action_changes: int = 0
    root_puct_time_budget_exhaustions: int = 0
    root_puct_start_override_sources_used: int = 0
    root_puct_start_override_attempts_used: int = 0
    root_puct_start_override_duplicate_attempts: int = 0
    root_puct_start_override_shared_samples: int = 0
    root_puct_start_override_shared_samples_accepted: int = 0
    root_puct_start_override_shared_samples_rejected: int = 0
    root_puct_start_override_direct_materializations: int = 0
    root_puct_start_override_replay_materializations: int = 0
    root_puct_prior_action_change_details: tuple[Mapping[str, Any], ...] = ()
    root_puct_fallback_reasons: Mapping[str, int] = field(default_factory=dict)
    root_puct_fallback_categories: Mapping[str, int] = field(default_factory=dict)
    root_puct_average_elapsed_seconds: float | None = None
    # Wall-clock policy selection time for every PokeZero decision, including raw-policy arms.
    # This deliberately includes the dispatch boundary, so capstone reports measure the cost a
    # caller observes rather than only root-PUCT's internal timer.
    policy_elapsed_seconds: tuple[float, ...] = ()
    # Per-decision Root-PUCT timing is retained so W2 can recover stage splits
    # and include graceful fallbacks in the same accounting.
    root_puct_timings: tuple[Mapping[str, float | int], ...] = ()
    tied: bool = False
    capped: bool = False
    # Seats observed while the bridge ran, rather than the requested config values. These make
    # mirrored-seat smoke artifacts capable of detecting a dispatch/submission-side regression.
    pokezero_decision_players: tuple[PlayerId, ...] = ()
    pokezero_submitted_choice_players: tuple[PlayerId, ...] = ()
    # The opponent moves KEPT for this battle after the journal mode was applied.
    # `opponent_journal_recorded` is how many were observed and
    # `opponent_journal_failures` how many could not be recorded at all, so a
    # truncated, suppressed or lossy journal is visible as a number rather than as
    # an absence. That distinction is the whole lesson of
    # `fallback_sample_addresses_dropped`, which counts only ONE of the three ways an
    # address is discarded and reads 0 while most of them go missing -- see the
    # module block for the measurement.
    opponent_journal: tuple[OpponentJournalEntry, ...] = ()
    opponent_journal_recorded: int = 0
    opponent_journal_failures: int = 0
    # Every refusal the #1180 recorder captured in THIS battle, with the
    # per-decision state that produced it. Filed on the game row rather than only in
    # a run-level list so a record travels with the battle it belongs to; the run
    # level carries only the health header, which is a property of the instrument
    # rather than of any one battle.
    refusal_records: tuple[RefusalRecord, ...] = ()

    @property
    def outcome_score(self) -> float:
        """Capstone score: wins are one point; ties and decision caps split the point."""

        if self.pokezero_won:
            return 1.0
        if self.tied or self.capped:
            return 0.5
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "battle_id": self.battle_id,
            "seed": self.seed,
            "winner": self.winner,
            "pokezero_won": self.pokezero_won,
            "pokezero_score": self.outcome_score,
            "pokezero_decision_players": list(self.pokezero_decision_players),
            "pokezero_submitted_choice_players": list(self.pokezero_submitted_choice_players),
            "tied": self.tied,
            "capped": self.capped,
            "decision_rounds": self.decision_rounds,
            "pokezero_decisions": self.pokezero_decisions,
            "root_puct_searches": self.root_puct_searches,
            "root_puct_fallbacks": self.root_puct_fallbacks,
            "root_puct_opponent_action_policies": dict(
                sorted(self.root_puct_opponent_action_policies.items())
            ),
            "root_puct_total_visits": self.root_puct_total_visits,
            "root_puct_opponent_action_scenarios_generated": self.root_puct_opponent_action_scenarios_generated,
            "root_puct_opponent_action_scenarios_skipped": self.root_puct_opponent_action_scenarios_skipped,
            "root_puct_opponent_action_scenarios_unsearched": self.root_puct_opponent_action_scenarios_unsearched,
            "root_puct_opponent_action_groups_generated": self.root_puct_opponent_action_groups_generated,
            "root_puct_opponent_action_groups_used": self.root_puct_opponent_action_groups_used,
            "root_puct_opponent_action_groups_skipped": self.root_puct_opponent_action_groups_skipped,
            "root_puct_opponent_action_groups_unsearched": self.root_puct_opponent_action_groups_unsearched,
            "root_puct_selected_prior_action_changes": self.root_puct_selected_prior_action_changes,
            "root_puct_pre_gate_prior_action_changes": self.root_puct_pre_gate_prior_action_changes,
            "root_puct_time_budget_exhaustions": self.root_puct_time_budget_exhaustions,
            "root_puct_start_override_sources_used": self.root_puct_start_override_sources_used,
            "root_puct_start_override_attempts_used": self.root_puct_start_override_attempts_used,
            "root_puct_start_override_duplicate_attempts": self.root_puct_start_override_duplicate_attempts,
            "root_puct_start_override_shared_samples": self.root_puct_start_override_shared_samples,
            "root_puct_start_override_shared_samples_accepted": (
                self.root_puct_start_override_shared_samples_accepted
            ),
            "root_puct_start_override_shared_samples_rejected": (
                self.root_puct_start_override_shared_samples_rejected
            ),
            "root_puct_start_override_direct_materializations": (
                self.root_puct_start_override_direct_materializations
            ),
            "root_puct_start_override_replay_materializations": (
                self.root_puct_start_override_replay_materializations
            ),
        }
        if self.opponent_journal:
            # `opponent_MOVES`, deliberately NOT `opponent_journal`.
            #
            # The summary root carries an `opponent_journal` MAPPING (the header:
            # mode, schema version, counts). If the per-game rows used the same name
            # for a LIST, one key would have two JSON shapes in one document -- and
            # the idiom every consumer in this repo uses to find things
            # (`fallback_addresses._walk_sample_blocks`, recursive, by name) would
            # hand a driver the header's keys as if they were journal entries. One
            # name, one shape.
            #
            # Neither name is one the address reader dispatches on. It accepts a
            # mapping as a cumulative stats scope iff it contains `fallback_samples`
            # (`_walk_stats_blocks`) and harvests addresses from any mapping so NAMED
            # (`_walk_sample_blocks`), so both blocks are invisible to it and neither
            # can add a scope, an address, or an occurrence count.
            payload["opponent_moves"] = [entry.to_dict() for entry in self.opponent_journal]
        if self.opponent_journal_recorded:
            payload["opponent_moves_recorded"] = self.opponent_journal_recorded
        if self.opponent_journal_failures:
            # ON THE ROW, not only in the summary header. A header total says a run
            # lost rounds; it cannot say WHICH battle, and an unrecorded round makes
            # every later round of THAT battle unreplayable while leaving the others
            # sound. Without this a driver has to treat the whole shard as suspect.
            payload["opponent_moves_record_failures"] = self.opponent_journal_failures
        if self.refusal_records:
            # `refusals`, and NOT a name the address reader dispatches on. Every key
            # inside a serialized `RefusalRecord` is likewise distinct from the
            # reader's markers -- the per-decision world-failure delta is spelled
            # `world_failures`, never `world_failure_reasons` -- so this list can add
            # neither a cumulative scope nor an occurrence count. See the REFUSAL
            # RECORDER block.
            payload["refusals"] = [record.to_dict() for record in self.refusal_records]
        if self.root_puct_effective_total_visits:
            payload["root_puct_effective_total_visits"] = self.root_puct_effective_total_visits
        if self.root_puct_opponent_action_skip_categories:
            payload["root_puct_opponent_action_skip_categories"] = dict(
                sorted(self.root_puct_opponent_action_skip_categories.items())
            )
        if self.root_puct_opponent_action_missing_sampled_world_reason_categories:
            payload["root_puct_opponent_action_missing_sampled_world_reason_categories"] = dict(
                sorted(self.root_puct_opponent_action_missing_sampled_world_reason_categories.items())
            )
        if self.root_puct_opponent_action_replay_rejection_decision_rounds:
            payload["root_puct_opponent_action_replay_rejection_decision_rounds"] = dict(
                sorted(
                    self.root_puct_opponent_action_replay_rejection_decision_rounds.items(),
                    key=lambda item: int(item[0]),
                )
            )
        if self.root_puct_opponent_action_replay_request_mismatch_decision_rounds:
            payload["root_puct_opponent_action_replay_request_mismatch_decision_rounds"] = dict(
                sorted(
                    self.root_puct_opponent_action_replay_request_mismatch_decision_rounds.items(),
                    key=lambda item: int(item[0]),
                )
            )
        if self.root_puct_opponent_action_replay_request_mismatch_players:
            payload["root_puct_opponent_action_replay_request_mismatch_players"] = dict(
                sorted(self.root_puct_opponent_action_replay_request_mismatch_players.items())
            )
        if self.root_puct_opponent_action_replay_request_mismatch_shapes:
            payload["root_puct_opponent_action_replay_request_mismatch_shapes"] = dict(
                sorted(self.root_puct_opponent_action_replay_request_mismatch_shapes.items())
            )
        if self.root_puct_opponent_action_start_override_mismatch_decision_rounds:
            payload["root_puct_opponent_action_start_override_mismatch_decision_rounds"] = dict(
                sorted(
                    self.root_puct_opponent_action_start_override_mismatch_decision_rounds.items(),
                    key=lambda item: int(item[0]),
                )
            )
        if self.root_puct_opponent_action_first_observation_mismatch_paths:
            payload["root_puct_opponent_action_first_observation_mismatch_paths"] = dict(
                sorted(self.root_puct_opponent_action_first_observation_mismatch_paths.items())
            )
        if self.root_puct_average_elapsed_seconds is not None:
            payload["root_puct_average_elapsed_seconds"] = self.root_puct_average_elapsed_seconds
        if self.policy_elapsed_seconds:
            payload["policy_elapsed_seconds"] = list(self.policy_elapsed_seconds)
        if self.root_puct_timings:
            payload["root_puct_timing"] = [dict(timing) for timing in self.root_puct_timings]
        if self.root_puct_prior_action_change_details:
            payload["root_puct_prior_action_change_details"] = [
                dict(detail)
                for detail in self.root_puct_prior_action_change_details
            ]
        if self.root_puct_fallback_reasons:
            payload["root_puct_fallback_reasons"] = dict(sorted(self.root_puct_fallback_reasons.items()))
        fallback_categories = _fallback_categories_from_reasons(
            self.root_puct_fallback_reasons,
            self.root_puct_fallback_categories,
        )
        if fallback_categories:
            payload["root_puct_fallback_categories"] = dict(sorted(fallback_categories.items()))
        return payload


@dataclass(frozen=True)
class ControlledFoulPlayOpponentCrash:
    """A seed abandoned because the external foul-play process exited before completing the game."""

    seed: int
    policy_mode: str
    returncode: int | None
    attempts: int
    stage: str
    stderr_tail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "policy_mode": self.policy_mode,
            "returncode": self.returncode,
            "attempts": self.attempts,
            "stage": self.stage,
            "stderr_tail": self.stderr_tail,
        }


def _engine_policy_stats(policy: Any, policy_mode: str) -> Mapping[str, Any] | None:
    """``EngineMctsStats.to_dict()`` for an engine-mcts run, else None.

    Deliberately NOT a ``hasattr`` probe over an arbitrary policy: outside
    engine-mcts there is no engine searcher and None is the honest answer,
    while inside it a missing serializer is a contract break that should raise
    rather than quietly produce an empty telemetry block (the failure mode that
    left every acceptance shard with ``"policy_stats": {}``).
    """
    if policy_mode != "engine-mcts":
        return None
    return policy.stats.to_dict()


@dataclass(frozen=True)
class ControlledFoulPlayBenchmarkResult:
    config: ControlledFoulPlayConfig
    policy_id: str
    games: tuple[ControlledFoulPlayGameResult, ...]
    checkpoint_sha256: str | None = None
    foulplay_random_seed_schedule: tuple[int, ...] | None = None
    value_leaf_provenance: Mapping[str, object] | None = None
    # EngineMctsStats.to_dict() for the run's engine policy, or None outside
    # engine-mcts mode. This is the ONLY path by which
    # search_wall_per_searched_decision -- the field the 20 s/turn rejection
    # rule is defined on -- reaches a shard summary; policy_timing measures the
    # bridge-level per-decision wall, which is a different quantity (it counts
    # non-searched decisions too).
    policy_stats: Mapping[str, Any] | None = None
    # Instrument health for the #1180 refusal recorder. None means the caller did
    # not report any -- which `to_dict` renders as the default header, i.e.
    # `health_reported: false`, i.e. UNKNOWN. It is never rendered as clean.
    refusal_recorder: "RefusalRecorderHealth | None" = None

    @property
    def completed_games(self) -> int:
        return len(self.games)

    @property
    def wins(self) -> int:
        return sum(1 for game in self.games if game.pokezero_won)

    @property
    def ties(self) -> int:
        return sum(1 for game in self.games if game.tied)

    @property
    def capped_games(self) -> int:
        return sum(1 for game in self.games if game.capped)

    @property
    def score(self) -> float:
        return sum(game.outcome_score for game in self.games)

    @property
    def score_rate(self) -> float:
        return self.score / self.completed_games if self.completed_games else 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.completed_games if self.completed_games else 0.0

    def to_dict(self) -> dict[str, Any]:
        root_searches = sum(game.root_puct_searches for game in self.games)
        root_fallbacks = sum(game.root_puct_fallbacks for game in self.games)
        engine_decisions = sum(game.engine_mcts_decisions for game in self.games)
        engine_fallbacks = sum(game.engine_mcts_fallbacks for game in self.games)
        engine_fallback_reasons: dict[str, int] = {}
        for game in self.games:
            for reason, count in (game.engine_mcts_fallback_reasons or {}).items():
                engine_fallback_reasons[reason] = engine_fallback_reasons.get(reason, 0) + count
        root_total_visits = sum(game.root_puct_total_visits for game in self.games)
        root_effective_total_visits = sum(game.root_puct_effective_total_visits for game in self.games)
        root_scenarios_generated = sum(game.root_puct_opponent_action_scenarios_generated for game in self.games)
        root_scenarios_skipped = sum(game.root_puct_opponent_action_scenarios_skipped for game in self.games)
        root_scenarios_unsearched = sum(game.root_puct_opponent_action_scenarios_unsearched for game in self.games)
        root_scenario_skip_categories: dict[str, int] = {}
        root_missing_sampled_world_reason_categories: dict[str, int] = {}
        root_replay_rejection_decision_rounds: dict[str, int] = {}
        root_replay_request_mismatch_decision_rounds: dict[str, int] = {}
        root_replay_request_mismatch_players: dict[str, int] = {}
        root_replay_request_mismatch_shapes: dict[str, int] = {}
        root_start_override_mismatch_decision_rounds: dict[str, int] = {}
        root_first_observation_mismatch_paths: dict[str, int] = {}
        for game in self.games:
            _merge_count_mapping(
                root_scenario_skip_categories,
                game.root_puct_opponent_action_skip_categories,
            )
            _merge_count_mapping(
                root_missing_sampled_world_reason_categories,
                game.root_puct_opponent_action_missing_sampled_world_reason_categories,
            )
            _merge_count_mapping(
                root_replay_rejection_decision_rounds,
                game.root_puct_opponent_action_replay_rejection_decision_rounds,
            )
            _merge_count_mapping(
                root_replay_request_mismatch_decision_rounds,
                game.root_puct_opponent_action_replay_request_mismatch_decision_rounds,
            )
            _merge_count_mapping(
                root_replay_request_mismatch_players,
                game.root_puct_opponent_action_replay_request_mismatch_players,
            )
            _merge_count_mapping(
                root_replay_request_mismatch_shapes,
                game.root_puct_opponent_action_replay_request_mismatch_shapes,
            )
            _merge_count_mapping(
                root_start_override_mismatch_decision_rounds,
                game.root_puct_opponent_action_start_override_mismatch_decision_rounds,
            )
            _merge_count_mapping(
                root_first_observation_mismatch_paths,
                game.root_puct_opponent_action_first_observation_mismatch_paths,
            )
        root_action_groups_generated = sum(game.root_puct_opponent_action_groups_generated for game in self.games)
        root_action_groups_used = sum(game.root_puct_opponent_action_groups_used for game in self.games)
        root_action_groups_skipped = sum(game.root_puct_opponent_action_groups_skipped for game in self.games)
        root_action_groups_unsearched = sum(game.root_puct_opponent_action_groups_unsearched for game in self.games)
        root_selected_prior_action_changes = sum(game.root_puct_selected_prior_action_changes for game in self.games)
        root_pre_gate_prior_action_changes = sum(game.root_puct_pre_gate_prior_action_changes for game in self.games)
        root_time_budget_exhaustions = sum(game.root_puct_time_budget_exhaustions for game in self.games)
        root_start_override_sources_used = sum(game.root_puct_start_override_sources_used for game in self.games)
        root_start_override_attempts_used = sum(game.root_puct_start_override_attempts_used for game in self.games)
        root_start_override_duplicate_attempts = sum(
            game.root_puct_start_override_duplicate_attempts for game in self.games
        )
        root_start_override_shared_samples = sum(game.root_puct_start_override_shared_samples for game in self.games)
        root_start_override_shared_samples_accepted = sum(
            game.root_puct_start_override_shared_samples_accepted for game in self.games
        )
        root_start_override_shared_samples_rejected = sum(
            game.root_puct_start_override_shared_samples_rejected for game in self.games
        )
        root_fallback_reasons: dict[str, int] = {}
        root_fallback_categories: dict[str, int] = {}
        root_opponent_action_policies: dict[str, int] = {}
        for game in self.games:
            for reason, count in game.root_puct_fallback_reasons.items():
                root_fallback_reasons[reason] = root_fallback_reasons.get(reason, 0) + count
            for category, count in _fallback_categories_from_reasons(
                game.root_puct_fallback_reasons,
                game.root_puct_fallback_categories,
            ).items():
                root_fallback_categories[category] = root_fallback_categories.get(category, 0) + count
            for planner_id, count in game.root_puct_opponent_action_policies.items():
                root_opponent_action_policies[planner_id] = (
                    root_opponent_action_policies.get(planner_id, 0) + count
                )
        elapsed_values = [
            game.root_puct_average_elapsed_seconds
            for game in self.games
            if game.root_puct_average_elapsed_seconds is not None
        ]
        policy_elapsed_values = [
            elapsed
            for game in self.games
            for elapsed in game.policy_elapsed_seconds
        ]
        root_puct_timings = [
            timing
            for game in self.games
            for timing in game.root_puct_timings
        ]
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint": _checkpoint_path_label(self.config),
            "checkpoint_sha256": self.checkpoint_sha256,
            "capture_driver": self.config.capture_driver,
            "audit_observation_schema": _audit_observation_schema_version(self.config),
            "format_id": self.config.format_id,
            "policy_id": self.policy_id,
            "policy_mode": self.config.policy_mode,
            "opponent_policy_id": "foul-play",
            "pokezero_player": self.config.pokezero_player,
            "foulplay_player": self.config.foulplay_player,
            "games": self.config.games,
            "completed_games": self.completed_games,
            "complete": self.completed_games >= self.config.games,
            "status": "complete" if self.completed_games >= self.config.games else "partial",
            "wins": self.wins,
            "win_rate": self.win_rate,
            "ties": self.ties,
            "capped_games": self.capped_games,
            "score": self.score,
            "score_rate": self.score_rate,
            "outcome_scoring": _OUTCOME_SCORING,
            "seed_start": self.config.seed_start,
            "foulplay_random_seed": self.config.resolved_foulplay_random_seed,
            "max_decision_rounds": self.config.max_decision_rounds,
            "belief_set_source": self.config.belief_set_source_enabled(),
            # Sibling of root_puct, populated only when policy_mode is
            # engine-mcts. Fallback here means the native searcher could not
            # construct a single belief world and played uniform-legal instead,
            # so a run without this block cannot distinguish a clean strength
            # measurement from a contaminated one.
            "engine_mcts": {
                "decisions": engine_decisions,
                "fallback_decisions": engine_fallbacks,
                "fallback_rate": (engine_fallbacks / engine_decisions) if engine_decisions else None,
                "fallback_reasons": dict(sorted(engine_fallback_reasons.items())),
                "depth": self.config.engine_depth,
                "sims": self.config.engine_sims,
                "batch": self.config.engine_batch,
                "worlds": self.config.engine_worlds,
                # Part of the cell's identity, not a footnote: cells B and E
                # are read entirely against whether this was on.
                "opponent_priors": self.config.engine_opponent_priors,
                # The searcher's own telemetry. `search_wall_per_searched_decision`
                # is lifted to the top of this block because it, not
                # policy_timing.average_elapsed_seconds, is what the 20 s/turn
                # rejection rule is defined on: policy_timing averages over ALL
                # decisions including un-searched ones, so it reads low exactly
                # when the fallback rate is high.
                "search_wall_per_searched_decision": (
                    (self.policy_stats or {}).get("search_wall_per_searched_decision")
                ),
                "policy_stats": self.policy_stats,
            } if self.config.policy_mode == "engine-mcts" else None,
            "root_puct": {
                "cpuct": self.config.cpuct,
                "selection_mode": self.config.selection_mode,
                "root_prior_temperature": self.config.effective_root_prior_temperature,
                "minimum_value_improvement": self.config.minimum_value_improvement,
                "minimum_override_prior_ratio": self.config.minimum_override_prior_ratio,
                "minimum_score_improvement": self.config.minimum_score_improvement,
                "root_visit_budget": self.config.root_visit_budget,
                "root_extra_visits": self.config.root_extra_visits,
                "adaptive_root_contested_extra_visits": self.config.adaptive_root_contested_extra_visits,
                "adaptive_root_uncontested_extra_visits": self.config.adaptive_root_uncontested_extra_visits,
                "adaptive_root_policy_entropy_threshold": self.config.adaptive_root_policy_entropy_threshold,
                "adaptive_root_value_margin_threshold": self.config.adaptive_root_value_margin_threshold,
                "root_time_budget_ms": self.config.root_time_budget_ms,
                "root_opponent_action_scenarios": self.config.root_opponent_action_scenarios,
                "root_opponent_action_candidate_scenarios": self.config.root_opponent_action_candidate_scenarios,
                "leaf_rollout_rounds": self.config.leaf_rollout_rounds,
                "leaf_rollout_sampling": self.config.leaf_rollout_sampling,
                "belief_start_overrides": self.config.belief_start_overrides,
                "start_override_attempts": self.config.start_override_attempts,
                "belief_start_override_samples": self.config.belief_start_override_samples,
                "start_override_hp_fraction_tolerance": self.config.start_override_hp_fraction_tolerance,
                "opponent_legal_mask_mode": self.config.opponent_legal_mask_mode,
                "opponent_action_policies": dict(sorted(root_opponent_action_policies.items())),
                "foulplay_search_time_ms": self.config.search_time_ms,
                "allow_search_fallback": self.config.allow_search_fallback,
                "searches": root_searches,
                "fallbacks": root_fallbacks,
                "total_visits": root_total_visits,
                "opponent_action_scenarios_generated": root_scenarios_generated,
                "opponent_action_scenarios_skipped": root_scenarios_skipped,
                "opponent_action_scenarios_unsearched": root_scenarios_unsearched,
                "opponent_action_groups_generated": root_action_groups_generated,
                "opponent_action_groups_used": root_action_groups_used,
                "opponent_action_groups_skipped": root_action_groups_skipped,
                "opponent_action_groups_unsearched": root_action_groups_unsearched,
                "selected_prior_action_changes": root_selected_prior_action_changes,
                "pre_gate_prior_action_changes": root_pre_gate_prior_action_changes,
                "time_budget_exhaustions": root_time_budget_exhaustions,
                "start_override_sources_used": root_start_override_sources_used,
                "start_override_attempts_used": root_start_override_attempts_used,
                "start_override_duplicate_attempts": root_start_override_duplicate_attempts,
                "start_override_shared_samples": root_start_override_shared_samples,
                "start_override_shared_samples_accepted": root_start_override_shared_samples_accepted,
                "start_override_shared_samples_rejected": root_start_override_shared_samples_rejected,
            },
            # The journal's own header. Present in EVERY summary, including
            # `mode: "off"`, because an empty journal is otherwise ambiguous three
            # ways -- journaling off, journaling on with no fallback addresses, or a
            # producer too old to have the feature -- and a replay driver that
            # cannot tell those apart will report "not replayable" for all three.
            #
            # `recorded` vs `emitted` is the truncation, stated as a number. Under
            # `addressed` the gap is expected and large; under `full` any gap at all
            # is a bug.
            "opponent_journal": {
                "schema_version": OPPONENT_JOURNAL_SCHEMA_VERSION,
                "mode": self.config.opponent_journal,
                # Where the rows live, named here so a consumer never has to guess
                # that the header key and the row key differ.
                #
                # THE PREFIX IS THE CONTRACT: every per-game key this feature writes
                # is `entries_key` or `entries_key + "_" + <suffix>` --
                # `opponent_moves`, `opponent_moves_recorded`,
                # `opponent_moves_record_failures`. A consumer scans the row for that
                # prefix rather than being told each name, so a later sibling counter
                # does not need a new header field to be discoverable. Sibling keys
                # are omitted from a row when zero/empty; absent means zero.
                "entries_key": "opponent_moves",
                "recorded_decisions": sum(game.opponent_journal_recorded for game in self.games),
                "emitted_decisions": sum(len(game.opponent_journal) for game in self.games),
                "games_with_journal": sum(1 for game in self.games if game.opponent_journal),
                # Opponent decisions the recorder could not journal. Non-zero means
                # the journal is INCOMPLETE for those rounds and a replay of any
                # later round in that battle cannot be trusted.
                "record_failures": sum(game.opponent_journal_failures for game in self.games),
            },
            # The recorder's own header, present in EVERY summary for the same
            # reason the journal's is: an absent block and an empty one are three
            # different states (switched off, on with nothing to record, producer
            # too old), and a reader that cannot tell them apart will call all three
            # "no refusals". `health_reported: false` is the default and means
            # UNKNOWN.
            "refusal_recorder": (self.refusal_recorder or RefusalRecorderHealth()).to_dict(),
            "game_results": [game.to_dict() for game in self.games],
        }
        if self.value_leaf_provenance is not None:
            payload["value_leaf"] = dict(self.value_leaf_provenance)
        if policy_elapsed_values:
            payload["policy_timing"] = {
                "decision_count": len(policy_elapsed_values),
                "total_elapsed_seconds": sum(policy_elapsed_values),
                "average_elapsed_seconds": sum(policy_elapsed_values) / len(policy_elapsed_values),
                "p95_elapsed_seconds": _nearest_rank_percentile(policy_elapsed_values, percentile=0.95),
            }
        if root_puct_timings:
            payload["root_puct"]["timing"] = _aggregate_root_puct_timings(root_puct_timings).to_dict()
        if self.foulplay_random_seed_schedule is not None:
            payload["foulplay_random_seed_schedule"] = _foulplay_random_seed_schedule_payload(
                self.foulplay_random_seed_schedule
            )
        if elapsed_values:
            payload["root_puct"]["average_elapsed_seconds"] = sum(elapsed_values) / len(elapsed_values)
        if root_effective_total_visits:
            payload["root_puct"]["effective_total_visits"] = root_effective_total_visits
        if root_scenario_skip_categories:
            payload["root_puct"]["opponent_action_skip_categories"] = dict(
                sorted(root_scenario_skip_categories.items())
            )
        if root_missing_sampled_world_reason_categories:
            payload["root_puct"]["opponent_action_missing_sampled_world_reason_categories"] = dict(
                sorted(root_missing_sampled_world_reason_categories.items())
            )
        if root_replay_rejection_decision_rounds:
            payload["root_puct"]["opponent_action_replay_rejection_decision_rounds"] = dict(
                sorted(root_replay_rejection_decision_rounds.items(), key=lambda item: int(item[0]))
            )
        if root_replay_request_mismatch_decision_rounds:
            payload["root_puct"]["opponent_action_replay_request_mismatch_decision_rounds"] = dict(
                sorted(root_replay_request_mismatch_decision_rounds.items(), key=lambda item: int(item[0]))
            )
        if root_replay_request_mismatch_players:
            payload["root_puct"]["opponent_action_replay_request_mismatch_players"] = dict(
                sorted(root_replay_request_mismatch_players.items())
            )
        if root_replay_request_mismatch_shapes:
            payload["root_puct"]["opponent_action_replay_request_mismatch_shapes"] = dict(
                sorted(root_replay_request_mismatch_shapes.items())
            )
        if root_start_override_mismatch_decision_rounds:
            payload["root_puct"]["opponent_action_start_override_mismatch_decision_rounds"] = dict(
                sorted(root_start_override_mismatch_decision_rounds.items(), key=lambda item: int(item[0]))
            )
        if root_first_observation_mismatch_paths:
            payload["root_puct"]["opponent_action_first_observation_mismatch_paths"] = dict(
                sorted(root_first_observation_mismatch_paths.items())
            )
        if root_fallback_reasons:
            payload["root_puct"]["fallback_reasons"] = dict(sorted(root_fallback_reasons.items()))
        if root_fallback_categories:
            payload["root_puct"]["fallback_categories"] = dict(sorted(root_fallback_categories.items()))
        return payload


@dataclass(frozen=True)
class ControlledFoulPlayComparisonResult:
    config: ControlledFoulPlayConfig
    raw: ControlledFoulPlayBenchmarkResult | None
    root_puct: ControlledFoulPlayBenchmarkResult | None
    comparison_mode: str = "per-seed"
    opponent_crashes: tuple[ControlledFoulPlayOpponentCrash, ...] = ()

    @property
    def crashed_seed_count(self) -> int:
        return len({crash.seed for crash in self.opponent_crashes})

    @property
    def complete(self) -> bool:
        # Seeds abandoned to an opponent crash cannot complete; a run that accounted for every
        # requested seed (finished or crashed) is complete, with crashes reported as a caveat.
        return (
            self.raw is not None
            and self.root_puct is not None
            and self.raw.completed_games + self.crashed_seed_count >= self.raw.config.games
            and self.root_puct.completed_games + self.crashed_seed_count >= self.root_puct.config.games
        )

    @property
    def status(self) -> str:
        if self.raw is None:
            return "pending"
        return "complete" if self.complete else "partial"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "checkpoint": _checkpoint_path_label(self.config),
            "capture_driver": self.config.capture_driver,
            "audit_observation_schema": _audit_observation_schema_version(self.config),
            "format_id": self.config.format_id,
            "opponent_policy_id": "foul-play",
            "games": self.config.games,
            "seed_start": self.config.seed_start,
            "max_decision_rounds": self.config.max_decision_rounds,
            "foulplay_random_seed": self.config.resolved_foulplay_random_seed,
            "foulplay_random_seed_schedule": _comparison_foulplay_random_seed_schedule_payload(
                self.config,
                comparison_mode=self.comparison_mode,
                count=self.config.games,
            ),
            "comparison_mode": self.comparison_mode,
            "belief_set_source": self.config.belief_set_source_enabled(),
            "status": self.status,
            "complete": self.complete,
            "runs": {
                "raw": self.raw.to_dict() if self.raw is not None else None,
                "root_puct": self.root_puct.to_dict() if self.root_puct is not None else None,
            },
            "opponent_crashes": [crash.to_dict() for crash in self.opponent_crashes],
            "comparison": _comparison_readout(
                self.raw,
                self.root_puct,
                comparison_mode=self.comparison_mode,
                opponent_crashes=self.opponent_crashes,
            ),
        }


@dataclass(frozen=True)
class ControlledFoulPlayCaptureResult:
    """Persisted external-opponent capture plus the benchmark that generated it."""

    benchmark: ControlledFoulPlayBenchmarkResult
    output_path: Path
    pool_id: str
    checkpoint_sha256: str
    belief_set_source_hash: str | None
    observation_schema_version: str | None
    numeric_feature_count: int | None
    captured_games: int
    skipped_capped_games: int
    skipped_tied_games: int
    public_corpus_path: Path | None = None
    captured_public_decisions: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = self.benchmark.to_dict()
        payload["capture"] = {
            "out": str(self.output_path),
            "pool_id": self.pool_id,
            "sides": "p1-only",
            "policy_mode": "raw",
            "checkpoint_sha256": self.checkpoint_sha256,
            "belief_set_source_hash": self.belief_set_source_hash,
            "observation_schema_version": self.observation_schema_version,
            "numeric_feature_count": self.numeric_feature_count,
            "captured_games": self.captured_games,
            "skipped_capped_games": self.skipped_capped_games,
            "skipped_tied_games": self.skipped_tied_games,
            "foulplay_search_time_ms": self.benchmark.config.search_time_ms,
            "public_decision_corpus_out": (
                str(self.public_corpus_path) if self.public_corpus_path is not None else None
            ),
            "captured_public_decisions": self.captured_public_decisions,
        }
        return payload


@dataclass(frozen=True)
class ControlledFoulPlayCollisionSketchResult:
    """Compact public-only collision capture plus the benchmark that generated it."""

    benchmark: ControlledFoulPlayBenchmarkResult
    output_path: Path
    pool_id: str
    checkpoint_sha256: str | None
    belief_set_source_hash: str | None
    observation_schema_version: str | None
    captured_games: int
    skipped_capped_games: int
    skipped_tied_games: int
    captured_decisions: int
    resumed_decisions: int
    captured_new_decisions: int
    recovered_trailing_partial: bool
    protocol_signature_schema_version: str
    protocol_signatures: Mapping[str, int]
    protocol_signature_game_ids: Sequence[str]

    def to_dict(self) -> dict[str, Any]:
        payload = self.benchmark.to_dict()
        # A frequency-only public census lets the E/O/C inventory consume a
        # production-style capture without retaining raw protocol or requests.
        payload["protocol_signature_schema_version"] = self.protocol_signature_schema_version
        payload["protocol_signatures"] = dict(sorted(self.protocol_signatures.items()))
        payload["protocol_signature_game_ids"] = sorted(self.protocol_signature_game_ids)
        payload["collision_sketch_capture"] = {
            "out": str(self.output_path),
            "pool_id": self.pool_id,
            "sides": "p1-only",
            "policy_mode": "raw",
            "checkpoint_sha256": self.checkpoint_sha256,
            "capture_driver": self.benchmark.config.capture_driver,
            "audit_observation_schema": _audit_observation_schema_version(self.benchmark.config),
            "belief_set_source_hash": self.belief_set_source_hash,
            "observation_schema_version": self.observation_schema_version,
            "captured_games": self.captured_games,
            "skipped_capped_games": self.skipped_capped_games,
            "skipped_tied_games": self.skipped_tied_games,
            "captured_decisions": self.captured_decisions,
            "resumed_decisions": self.resumed_decisions,
            "captured_new_decisions": self.captured_new_decisions,
            "recovered_trailing_partial": self.recovered_trailing_partial,
            "foulplay_search_time_ms": self.benchmark.config.search_time_ms,
            "protocol_signature_schema_version": self.protocol_signature_schema_version,
            "protocol_signatures": dict(sorted(self.protocol_signatures.items())),
            "protocol_signature_game_ids": sorted(self.protocol_signature_game_ids),
        }
        return payload


def _comparison_readout(
    raw: ControlledFoulPlayBenchmarkResult | None,
    root_puct: ControlledFoulPlayBenchmarkResult | None,
    *,
    comparison_mode: str,
    opponent_crashes: tuple[ControlledFoulPlayOpponentCrash, ...] = (),
) -> dict[str, Any]:
    raw_by_seed = _games_by_seed(raw)
    search_by_seed = _games_by_seed(root_puct)
    matched_seeds = tuple(sorted(raw_by_seed.keys() & search_by_seed.keys()))
    raw_paired_wins = sum(1 for seed in matched_seeds if raw_by_seed[seed].pokezero_won)
    search_paired_wins = sum(1 for seed in matched_seeds if search_by_seed[seed].pokezero_won)
    raw_paired_score = sum(raw_by_seed[seed].outcome_score for seed in matched_seeds)
    search_paired_score = sum(search_by_seed[seed].outcome_score for seed in matched_seeds)
    both_won = sum(
        1
        for seed in matched_seeds
        if raw_by_seed[seed].pokezero_won and search_by_seed[seed].pokezero_won
    )
    raw_only_won = sum(
        1
        for seed in matched_seeds
        if raw_by_seed[seed].pokezero_won and not search_by_seed[seed].pokezero_won
    )
    root_puct_only_won = sum(
        1
        for seed in matched_seeds
        if search_by_seed[seed].pokezero_won and not raw_by_seed[seed].pokezero_won
    )
    neither_won = sum(
        1
        for seed in matched_seeds
        if not raw_by_seed[seed].pokezero_won and not search_by_seed[seed].pokezero_won
    )
    paired_games = len(matched_seeds)
    raw_completed_games = raw.completed_games if raw is not None else 0
    search_completed_games = root_puct.completed_games if root_puct is not None else 0
    raw_wins = raw.wins if raw is not None else 0
    search_wins = root_puct.wins if root_puct is not None else 0
    raw_score = raw.score if raw is not None else 0.0
    search_score = root_puct.score if root_puct is not None else 0.0

    crashed_seeds = sorted({crash.seed for crash in opponent_crashes})

    return {
        "sample_size": {
            "paired_games": paired_games,
            "minimum_strength_games": _MIN_STRENGTH_SAMPLE_GAMES,
            "status": "strength_sized" if paired_games >= _MIN_STRENGTH_SAMPLE_GAMES else "diagnostic_only",
        },
        "opponent_crashed_seeds": {
            "count": len(crashed_seeds),
            "seeds": crashed_seeds,
            "handling": "seed_excluded_from_paired_stats_and_aggregates",
        },
        "aggregate": {
            "analysis_method": "completed_prefix_marginal_rates",
            "raw": _rate_readout(raw_wins, raw_completed_games),
            "root_puct": _rate_readout(search_wins, search_completed_games),
            "root_puct_minus_raw_win_rate": _delta_rate(
                search_wins,
                search_completed_games,
                raw_wins,
                raw_completed_games,
                require_equal_games=True,
            ),
            "delta_interpretation": (
                "descriptive_only_when_both_prefixes_have_equal_nonzero_completed_games"
            ),
            "scored_outcomes": {
                "scoring": _OUTCOME_SCORING,
                "raw": _score_rate_readout(raw_score, raw_completed_games),
                "root_puct": _score_rate_readout(search_score, search_completed_games),
                "root_puct_minus_raw_score_rate": _delta_score_rate(
                    search_score,
                    search_completed_games,
                    raw_score,
                    raw_completed_games,
                    require_equal_games=True,
                ),
            },
        },
        "paired_by_seed": {
            "pairing_method": _pairing_method_for_comparison_mode(comparison_mode),
            "opponent_deterministic": False,
            "paired_counterfactual": False,
            "interval_method": "marginal_wilson_per_arm_not_paired_delta",
            "delta_interpretation": "descriptive_only",
            "games": paired_games,
            "raw": _rate_readout(raw_paired_wins, paired_games),
            "root_puct": _rate_readout(search_paired_wins, paired_games),
            "root_puct_minus_raw_win_rate": _delta_rate(
                search_paired_wins,
                paired_games,
                raw_paired_wins,
                paired_games,
            ),
            "scored_outcomes": {
                "scoring": _OUTCOME_SCORING,
                "raw": _score_rate_readout(raw_paired_score, paired_games),
                "root_puct": _score_rate_readout(search_paired_score, paired_games),
                "root_puct_minus_raw_score_rate": _delta_score_rate(
                    search_paired_score,
                    paired_games,
                    raw_paired_score,
                    paired_games,
                ),
            },
            "discordant_pairs": {
                "both_won": both_won,
                "raw_only_won": raw_only_won,
                "root_puct_only_won": root_puct_only_won,
                "neither_won": neither_won,
            },
            "first_seed": matched_seeds[0] if matched_seeds else None,
            "last_seed": matched_seeds[-1] if matched_seeds else None,
        },
    }


def _pairing_method_for_comparison_mode(comparison_mode: str) -> str:
    if comparison_mode == "per-seed":
        return "per_seed_shared_battlestream_seed_and_foulplay_start_seed"
    return "shared_battlestream_seed_only"


def _fallback_categories_from_reasons(
    reasons: Mapping[str, int],
    categories: Mapping[str, int],
) -> dict[str, int]:
    result = {str(category): int(count) for category, count in categories.items()}
    if result:
        return result
    for reason, count in reasons.items():
        category = root_puct_fallback_category(reason)
        result[category] = result.get(category, 0) + int(count)
    return result


def _merge_count_mapping(target: dict[str, int], source: object) -> None:
    if not isinstance(source, Mapping):
        return
    for key, value in source.items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        target[str(key)] = target.get(str(key), 0) + count


def _comparison_foulplay_random_seed_schedule_payload(
    config: ControlledFoulPlayConfig,
    *,
    comparison_mode: str,
    count: int,
) -> dict[str, Any]:
    if comparison_mode == "per-seed":
        return _foulplay_random_seed_schedule_payload(
            _per_seed_foulplay_random_seed_schedule(config, count=count)
        )
    return _foulplay_random_seed_schedule_payload((config.resolved_foulplay_random_seed,))


def _per_seed_foulplay_random_seed_schedule(
    config: ControlledFoulPlayConfig,
    *,
    count: int,
) -> tuple[int, ...]:
    return _per_seed_foulplay_random_seed_schedule_for_offsets(config, offsets=range(count))


def _per_seed_foulplay_random_seed_schedule_for_offsets(
    config: ControlledFoulPlayConfig,
    *,
    offsets: Iterable[int],
) -> tuple[int, ...]:
    return tuple(
        (
            config.foulplay_random_seed + offset
            if config.foulplay_random_seed is not None
            else config.seed_start + offset
        )
        for offset in offsets
    )


def _foulplay_random_seed_schedule_payload(seeds: tuple[int, ...]) -> dict[str, Any]:
    return {
        "count": len(seeds),
        "first_seed": seeds[0] if seeds else None,
        "last_seed": seeds[-1] if seeds else None,
        "mode": "constant" if len(set(seeds)) <= 1 else "per_game_incrementing",
        "seeds": list(seeds),
    }


def _games_by_seed(
    result: ControlledFoulPlayBenchmarkResult | None,
) -> dict[int, ControlledFoulPlayGameResult]:
    if result is None:
        return {}
    return {game.seed: game for game in result.games}


def _rate_readout(wins: int, games: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "games": games,
        "wins": wins,
        "win_rate": _rate(wins, games),
        "interval_method": "wilson_score_marginal_95",
    }
    if games:
        lower, upper = _wilson_interval(wins, games, z=_WILSON_95_Z)
        payload["wilson_95"] = {"lower": lower, "upper": upper}
    else:
        payload["wilson_95"] = None
    return payload


def _rate(wins: int, games: int) -> float:
    return wins / games if games else 0.0


_OUTCOME_SCORING = {
    "win": 1.0,
    "tie": 0.5,
    "capped": 0.5,
    "loss": 0.0,
}


def _score_rate_readout(score: float, games: int) -> dict[str, Any]:
    return {
        "games": games,
        "score": score,
        "score_rate": score / games if games else 0.0,
    }


def _nearest_rank_percentile(values: Sequence[float], *, percentile: float) -> float:
    """Return a deterministic nearest-rank percentile for a non-empty timing sample."""

    if not values:
        raise ValueError("percentile requires at least one value.")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1].")
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _delta_rate(
    first_wins: int,
    first_games: int,
    second_wins: int,
    second_games: int,
    *,
    require_equal_games: bool = False,
) -> float | None:
    if first_games <= 0 or second_games <= 0:
        return None
    if require_equal_games and first_games != second_games:
        return None
    return _rate(first_wins, first_games) - _rate(second_wins, second_games)


def _delta_score_rate(
    first_score: float,
    first_games: int,
    second_score: float,
    second_games: int,
    *,
    require_equal_games: bool = False,
) -> float | None:
    if first_games <= 0 or second_games <= 0:
        return None
    if require_equal_games and first_games != second_games:
        return None
    return (first_score / first_games) - (second_score / second_games)


def _wilson_interval(wins: int, games: int, *, z: float) -> tuple[float, float]:
    if games <= 0:
        return (0.0, 0.0)
    if z == 0.0:
        rate = wins / games
        return (rate, rate)
    p_hat = wins / games
    z_squared = z * z
    denominator = 1.0 + (z_squared / games)
    center = p_hat + (z_squared / (2.0 * games))
    adjustment = z * math.sqrt(((p_hat * (1.0 - p_hat)) + (z_squared / (4.0 * games))) / games)
    return (
        max(0.0, (center - adjustment) / denominator),
        min(1.0, (center + adjustment) / denominator),
    )


class FoulPlayProtocolError(RuntimeError):
    """Raised when the foul-play websocket client emits an unsupported protocol message."""


class FoulPlayProcessExitError(RuntimeError):
    """Raised when the external foul-play process exits before completing a protocol step."""

    def __init__(self, *, stage: str, returncode: int | None, log_tail: str) -> None:
        super().__init__(f"foul-play exited with status {returncode} before {stage}.\n{log_tail}")
        self.stage = stage
        self.returncode = returncode
        self.log_tail = log_tail


@dataclass
class _ProcessLogBuffer:
    stdout: list[str] = field(default_factory=list)
    stderr: list[str] = field(default_factory=list)

    def append_stdout(self, line: str) -> None:
        self.stdout.append(line)
        if len(self.stdout) > 200:
            del self.stdout[: len(self.stdout) - 200]

    def append_stderr(self, line: str) -> None:
        self.stderr.append(line)
        if len(self.stderr) > 200:
            del self.stderr[: len(self.stderr) - 200]

    def tail(self) -> str:
        parts = []
        if self.stderr:
            parts.append("stderr:\n" + "\n".join(self.stderr[-40:]))
        if self.stdout:
            parts.append("stdout:\n" + "\n".join(self.stdout[-40:]))
        return "\n\n".join(parts) or "(no foul-play output captured)"


class _FoulPlayWebsocketServer:
    def __init__(self, *, username: str, host: str) -> None:
        self.username = username
        self.host = host
        self.port: int | None = None
        self.websocket: Any = None
        self.server: Any = None
        self.challenge_queue: asyncio.Queue[str] = asyncio.Queue()
        self.choice_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

    @property
    def uri(self) -> str:
        if self.port is None:
            raise RuntimeError("server has not started.")
        return f"ws://{self.host}:{self.port}/showdown/websocket"

    async def start(self) -> None:
        import websockets

        self.server = await websockets.serve(self._handle_connection, self.host, 0, max_size=None)
        socket = self.server.sockets[0]
        self.port = int(socket.getsockname()[1])

    async def close(self) -> None:
        if self.websocket is not None:
            await self.websocket.close()
            self.websocket = None
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

    async def _handle_connection(self, websocket: Any) -> None:
        self.websocket = websocket
        try:
            await websocket.send("|challstr|1|pokezero-controlled")
            async for message in websocket:
                await self._handle_message(str(message))
        except Exception:
            # The caller monitors the foul-play process and will report its stderr/stdout. Avoid
            # leaking a noisy websocket traceback as the primary error.
            return
        finally:
            # A game-scoped client can disconnect after its successor has already connected.
            # Never let that stale handler clear the successor's active websocket.
            if self.websocket is websocket:
                self.websocket = None

    async def _handle_message(self, message: str) -> None:
        room, body = _split_outgoing_showdown_message(message)
        if room and (choice := _choice_body_from_outgoing_message(body)):
            await self.choice_queue.put((room, choice))
            return
        if body.startswith("/trn "):
            await self.send_global(f"|updateuser|{self.username}|1|0|")
            return
        if body.startswith("/challenge "):
            target = body[len("/challenge ") :].split(",", 1)[0].strip()
            await self.challenge_queue.put(target)
            return
        if body.startswith("/leave "):
            battle_id = body[len("/leave ") :].strip()
            await self.send_room_lines(battle_id, ["|deinit|"])
            return
        # /utm, /timer, chat, and /savereplay are accepted no-ops for this controlled harness.

    async def send_global(self, message: str) -> None:
        if self.websocket is None:
            raise FoulPlayProtocolError("foul-play websocket is not connected.")
        await self.websocket.send(message)

    async def send_room_lines(self, battle_id: str, lines: Sequence[str]) -> None:
        if self.websocket is None:
            raise FoulPlayProtocolError("foul-play websocket is not connected.")
        if not lines:
            return
        await self.websocket.send(f">{battle_id}\n" + "\n".join(lines))

    async def wait_for_challenge(self, *, expected_target: str, timeout_seconds: float = 30.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for foul-play challenge.")
            target = await asyncio.wait_for(self.challenge_queue.get(), timeout=remaining)
            if _showdown_id(target) == _showdown_id(expected_target):
                return

    async def wait_for_choice(self, *, battle_id: str, timeout_seconds: float = 120.0) -> str:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for foul-play choice.")
            room, choice = await asyncio.wait_for(self.choice_queue.get(), timeout=remaining)
            if room == battle_id:
                return choice


class _BattleBridge:
    def __init__(self, *, showdown_root: Path, node_binary: str) -> None:
        self.showdown_root = showdown_root
        self.node_binary = node_binary
        self.process: asyncio.subprocess.Process | None = None
        self.events: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()
        self.stderr_lines: list[str] = []
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self.process = await asyncio.create_subprocess_exec(
            self.node_binary,
            str(BRIDGE_PATH),
            "--showdown-root",
            str(self.showdown_root),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                "PATH": os.environ.get("PATH", ""),
                "POKEZERO_SHOWDOWN_ROOT": str(self.showdown_root),
            },
        )
        self._stdout_task = asyncio.create_task(self._drain_stdout())
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def close(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            if process.returncode is None:
                try:
                    await self.send({"type": "close"})
                    await asyncio.wait_for(process.wait(), timeout=1.0)
                except Exception:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
        finally:
            for task in (self._stdout_task, self._stderr_task):
                if task is not None:
                    task.cancel()
            self.process = None

    async def send(self, command: Mapping[str, Any]) -> None:
        if self.process is None or self.process.stdin is None or self.process.returncode is not None:
            raise RuntimeError(self._exit_message())
        self.process.stdin.write(json.dumps(command, separators=(",", ":")).encode("utf-8") + b"\n")
        await self.process.stdin.drain()

    async def next_event(self, *, timeout_seconds: float = 120.0) -> Mapping[str, Any]:
        event = await asyncio.wait_for(self.events.get(), timeout=timeout_seconds)
        if event.get("type") == "error":
            raise RuntimeError(str(event.get("message") or "BattleStream bridge error."))
        return event

    async def _drain_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        async for raw in self.process.stdout:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            self.events.put_nowait(json.loads(line))

    async def _drain_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        async for raw in self.process.stderr:
            self.stderr_lines.append(raw.decode("utf-8", errors="replace").rstrip())
            if len(self.stderr_lines) > 100:
                del self.stderr_lines[: len(self.stderr_lines) - 100]

    def _exit_message(self) -> str:
        if self.process is not None and self.process.returncode is not None:
            stderr = "\n".join(self.stderr_lines[-20:])
            suffix = f" Stderr:\n{stderr}" if stderr else ""
            return f"BattleStream bridge exited with status {self.process.returncode}.{suffix}"
        return "BattleStream bridge is not running."


@dataclass
class _ControlledBattleState:
    battle_id: str
    seed: int
    format_id: str
    public_lines: list[str] = field(default_factory=list)
    request_lines: dict[PlayerId, str] = field(default_factory=dict)
    trajectory: BattleTrajectory | None = None
    decisions: list[PolicyDecision] = field(default_factory=list)
    pokezero_decision_players: list[PlayerId] = field(default_factory=list)
    pokezero_submitted_choice_players: list[PlayerId] = field(default_factory=list)
    public_line_cursor: int = 0
    previous_requested_players: tuple[PlayerId, ...] = ()
    public_resolved_action_rounds: list[PublicResolvedActionRound] = field(default_factory=list)
    next_foulplay_rqid: int = 1
    foulplay_terminal_sent: bool = False
    # per-decision request snapshots (both seats), for omniscient trait capture; append-only.
    request_history: list[tuple[PlayerId, str]] = field(default_factory=list)
    # Opponent moves as submitted, in round order; append-only. Recorded in FULL
    # under both "addressed" and "full" -- which rounds "addressed" wants is not
    # known until the game ends and the moves cannot be recovered afterwards -- and
    # narrowed when the game result is built. Empty under "off".
    opponent_journal: list[OpponentJournalEntry] = field(default_factory=list)
    # Opponent decisions the recorder raised on. See the recording site.
    opponent_journal_failures: int = 0

    def all_lines(self) -> list[str]:
        return [*self.public_lines, *self.request_lines.values()]


async def run_controlled_foulplay_benchmark(
    config: ControlledFoulPlayConfig,
    *,
    progress_callback: ControlledFoulPlayProgressCallback | None = None,
    trajectory_callback: ControlledFoulPlayTrajectoryCallback | None = None,
) -> ControlledFoulPlayBenchmarkResult:
    """Run PokeZero vs foul-play with a known BattleStream seed and context-aware policy."""

    _validate_external_paths(config)
    if config.capture_driver == "random-legal":
        observation_spec = observation_spec_for_schema(OBSERVATION_SCHEMA_VERSION_V3)
        vocab = gen3_category_vocabulary(
            config.showdown_root,
            include_turn_merged=True,
        )
        return await _run_controlled_foulplay_games(
            config,
            policy=RandomLegalPolicy(policy_id="audit-random-legal"),
            policy_id="audit-random-legal",
            vocab=vocab,
            dex=load_showdown_dex_cached(config.showdown_root),
            observation_spec=observation_spec,
            feature_masks=DEFAULT_OBSERVATION_FEATURE_MASKS,
            checkpoint_sha256=None,
            progress_callback=progress_callback,
            trajectory_callback=trajectory_callback,
        )

    checkpoint = config.checkpoint
    if checkpoint is None:
        raise AssertionError("checkpoint driver validation must reject a missing checkpoint")
    checkpoint_sha256 = _checkpoint_sha256(config)
    model, result = load_transformer_checkpoint(checkpoint, map_location=config.device)
    value_model, value_result = model, result
    value_leaf_provenance: Mapping[str, object] | None = None
    if config.value_checkpoint is not None:
        value_model, value_result = load_transformer_checkpoint(
            config.value_checkpoint,
            map_location=config.device,
        )
        value_leaf_provenance = require_compatible_transformer_value_checkpoint(
            policy_checkpoint=checkpoint,
            policy_result=result,
            value_checkpoint=config.value_checkpoint,
            value_result=value_result,
        )
    _warn_on_belief_provenance_mismatch(config, result)
    policy_id = str(result.model_config.policy_id)
    # Schema + widths from the checkpoint's stamped provenance (dual-schema resolution): a v2
    # checkpoint is probed under the v2 encode, a v2.1 checkpoint under v2.1.
    observation_spec = observation_spec_from_model_config(result.model_config)
    # The vocabulary axis latches from the CHECKPOINT. It previously latched only with the
    # schema (review MED-2) — which token FAMILIES exist (turn-merged or not) — while the
    # enumeration ORDER inside them still came from the build. Order is what indexes the
    # embedding, so that half-latch left the silent failure open: a token added to the build
    # since training renumbers every token after it, and merged labels resolve to rows the
    # model learned as other values rather than OOV-hashing loudly.
    vocab = category_vocab_from_model_config(result.model_config, config.showdown_root)
    dex = load_showdown_dex_cached(config.showdown_root)
    # Encode-time feature masks come FROM the checkpoint's stamped provenance (never the
    # defaults): a K=32 / stats-off / exact-state-off arm must be probed on observations
    # encoded exactly as trained — the mask-axis twin of the #492 belief-source mismatch.
    feature_masks = feature_masks_from_model_config(result.model_config)
    env_config = LocalShowdownConfig(
        showdown_root=config.showdown_root,
        node_binary=config.node_binary,
        observation_spec=observation_spec,
        category_vocab=vocab,
        feature_masks=feature_masks,
    )
    rollout_config = RolloutConfig(
        max_decision_rounds=config.max_decision_rounds,
        format_id=config.format_id,
    )
    policy = _build_policy(
        config=config,
        model=model,
        result=result,
        value_model=value_model,
        value_result=value_result,
        env_config=env_config,
        rollout_config=rollout_config,
        policy_id=policy_id,
    )
    benchmark_policy_id = policy.policy_id if hasattr(policy, "policy_id") else policy_id
    return await _run_controlled_foulplay_games(
        config,
        policy=policy,
        policy_id=benchmark_policy_id,
        vocab=vocab,
        dex=dex,
        observation_spec=observation_spec,
        feature_masks=feature_masks,
        checkpoint_sha256=checkpoint_sha256,
        value_leaf_provenance=value_leaf_provenance,
        progress_callback=progress_callback,
        trajectory_callback=trajectory_callback,
    )


async def _run_controlled_foulplay_games(
    config: ControlledFoulPlayConfig,
    *,
    policy: Policy,
    policy_id: str,
    vocab: CategoryVocabulary,
    dex: ShowdownDex,
    observation_spec: Any,
    feature_masks: ObservationFeatureMasks,
    checkpoint_sha256: str | None,
    value_leaf_provenance: Mapping[str, object] | None = None,
    progress_callback: ControlledFoulPlayProgressCallback | None = None,
    trajectory_callback: ControlledFoulPlayTrajectoryCallback | None = None,
) -> ControlledFoulPlayBenchmarkResult:
    """Run a preconstructed legal policy through the shared FoulPlay bridge."""

    foulplay_random_seed_schedule = _per_seed_foulplay_random_seed_schedule(config, count=config.games)

    server = _FoulPlayWebsocketServer(username=config.foulplay_username, host=config.websocket_host)
    bridge = _BattleBridge(showdown_root=config.showdown_root, node_binary=config.node_binary)
    game_results: list[ControlledFoulPlayGameResult] = []
    # ONE recorder for the whole run: the bridge reuses a single policy across every
    # seed, and records are partitioned back per game by battle_id.
    refusal_capture = _RefusalCapture(
        policy, enabled=config.record_refusals, games=config.games
    )
    try:
        await server.start()
        await bridge.start()
        for offset in range(config.games):
            seed = config.seed_start + offset
            # Foul-play's websocket cleanup can terminate its process after a completed game.
            # Give every controlled battle an isolated client process so a completed result is
            # never reclassified as a failure while waiting for the next challenge.
            game_config = replace(
                config,
                foulplay_random_seed=foulplay_random_seed_schedule[offset],
            )
            foulplay_process = await _spawn_foulplay(game_config, server.uri, run_count=1)
            foulplay_logs = _ProcessLogBuffer()
            foulplay_log_tasks = [
                asyncio.create_task(_drain_process_stream(foulplay_process.stdout, foulplay_logs.append_stdout)),
                asyncio.create_task(_drain_process_stream(foulplay_process.stderr, foulplay_logs.append_stderr)),
            ]
            try:
                await _wait_for_foulplay_challenge_or_exit(
                    server=server,
                    expected_target=config.pokezero_username,
                    process=foulplay_process,
                    logs=foulplay_logs,
                )
                game_results.append(
                    await _run_single_game(
                        config=config,
                        bridge=bridge,
                        server=server,
                        policy=policy,
                        vocab=vocab,
                        dex=dex,
                        observation_spec=observation_spec,
                        feature_masks=feature_masks,
                        seed=seed,
                        foulplay_process=foulplay_process,
                        foulplay_logs=foulplay_logs,
                        trajectory_callback=trajectory_callback,
                        refusal_capture=refusal_capture,
                    )
                )
            finally:
                await _stop_foulplay_process(foulplay_process, foulplay_log_tasks)
            if progress_callback is not None:
                progress_callback(
                    ControlledFoulPlayBenchmarkResult(
                        config=config,
                        policy_id=policy_id,
                        games=tuple(game_results),
                        checkpoint_sha256=checkpoint_sha256,
                        foulplay_random_seed_schedule=foulplay_random_seed_schedule[: len(game_results)],
                        value_leaf_provenance=value_leaf_provenance,
                        policy_stats=_engine_policy_stats(policy, config.policy_mode),
                        # On the PARTIAL result too: `--summary-out` rewrites the whole
                        # document every game, so a run killed mid-way leaves this
                        # progress write as the only artifact. Reporting health only at
                        # the end would make every abandoned run read as unknown.
                        refusal_recorder=refusal_capture.health(),
                    )
                )
    finally:
        # EACH teardown in its own `finally`, so one raising cannot skip the next.
        # Flat statements here meant a `bridge.close()` that raised skipped BOTH
        # `server.close()` and the detach below. Leaking the detach is the worse of
        # the two and is invisible: `_Hook.remove` never empties `recorders`, so the
        # wrappers stay installed on a policy that outlives this call -- the
        # comparison runner's second arm, or a caller that reuses the policy -- and
        # every later `_fallback` fans out to N orphaned recorders that accumulate
        # for the life of the process while nothing reads them.
        try:
            await bridge.close()
        finally:
            try:
                await server.close()
            finally:
                # Records that never reached a game row are counted before the
                # health is read, so the reconciliation identity holds.
                refusal_capture.account_for_unrowed_records()
                refusal_capture.detach()

    return ControlledFoulPlayBenchmarkResult(
        config=config,
        policy_id=policy_id,
        games=tuple(game_results),
        checkpoint_sha256=checkpoint_sha256,
        foulplay_random_seed_schedule=foulplay_random_seed_schedule[: len(game_results)],
        value_leaf_provenance=value_leaf_provenance,
        policy_stats=_engine_policy_stats(policy, config.policy_mode),
        refusal_recorder=refusal_capture.health(),
    )


async def capture_controlled_foulplay_rollouts(
    config: ControlledFoulPlayConfig,
    *,
    out_path: Path,
    pool_id: str = "controlled-foulplay",
    progress_callback: ControlledFoulPlayProgressCallback | None = None,
    capture_progress_callback: ControlledFoulPlayCaptureProgressCallback | None = None,
    public_corpus_out: Path | None = None,
    append_public_corpus: bool = False,
) -> ControlledFoulPlayCaptureResult:
    """Capture raw-policy p1 trajectories from a deterministic foul-play seed band.

    Each completed game is appended and flushed before the next seed begins, so
    a process failure preserves all previously captured rollouts. The output
    path must be new; callers resume with a fresh seed band rather than risk
    silently mixing duplicate seeds into a frozen evaluation corpus.
    """

    if config.policy_mode != "raw":
        raise ValueError("controlled foul-play rollout capture requires policy_mode='raw'.")
    if config.capture_driver != "checkpoint":
        raise ValueError("controlled foul-play rollout capture requires the checkpoint capture driver.")
    if config.pokezero_player != "p1":
        raise ValueError("controlled foul-play rollout capture requires pokezero_player='p1'.")
    if config.opponent_legal_mask_mode != "hidden":
        raise ValueError("public decision corpus capture requires opponent_legal_mask_mode='hidden'.")
    if not pool_id.strip():
        raise ValueError("pool_id must be non-empty.")
    if out_path.exists():
        raise FileExistsError(f"capture output already exists: {out_path}")
    if public_corpus_out is not None and public_corpus_out == out_path:
        raise ValueError("public decision corpus output must differ from rollout output.")

    source = _resolved_belief_set_source(config)
    belief_set_source_hash = source.metadata.source_hash if source is not None else None
    checkpoint_sha256 = _checkpoint_sha256(config)
    if checkpoint_sha256 is None:
        raise AssertionError("checkpoint rollout capture requires checkpoint provenance")
    handle = None
    captured_games = 0
    skipped_capped_games = 0
    skipped_tied_games = 0
    observation_schema_version: str | None = None
    numeric_feature_count: int | None = None
    previous_capture_time = time.monotonic()
    public_corpus_writer: PublicDecisionCorpusWriter | None = None
    captured_public_decisions = 0
    if public_corpus_out is not None:
        public_corpus_writer = PublicDecisionCorpusWriter(
            public_corpus_out,
            manifest=public_corpus_manifest(
                checkpoint_sha256=checkpoint_sha256,
                belief_set_source_hash=belief_set_source_hash,
                capture_config=_public_corpus_capture_config(config),
            ),
            append=append_public_corpus,
        )

    def progress_payload(*, status: str) -> dict[str, Any]:
        return {
            "status": status,
            "capture": {
                "out": str(out_path),
                "pool_id": pool_id,
                "sides": "p1-only",
                "policy_mode": "raw",
                "checkpoint_sha256": checkpoint_sha256,
                "belief_set_source_hash": belief_set_source_hash,
                "observation_schema_version": observation_schema_version,
                "numeric_feature_count": numeric_feature_count,
                "captured_games": captured_games,
                "skipped_capped_games": skipped_capped_games,
                "skipped_tied_games": skipped_tied_games,
                "foulplay_search_time_ms": config.search_time_ms,
                "public_decision_corpus_out": str(public_corpus_out) if public_corpus_out is not None else None,
                "captured_public_decisions": captured_public_decisions,
            },
        }

    def emit_progress() -> None:
        if capture_progress_callback is not None:
            capture_progress_callback(progress_payload(status="running"))

    def capture(trajectory: BattleTrajectory) -> None:
        nonlocal handle, captured_games, skipped_capped_games, skipped_tied_games
        nonlocal observation_schema_version, numeric_feature_count, previous_capture_time, captured_public_decisions
        if trajectory.terminal is None:
            raise RuntimeError("controlled foul-play capture requires a terminal trajectory")
        if trajectory.terminal.winner is None:
            if trajectory.terminal.capped:
                skipped_capped_games += 1
            else:
                skipped_tied_games += 1
            emit_progress()
            return
        p1_trajectory = _p1_capture_trajectory(trajectory, pool_id=pool_id)
        if not p1_trajectory.steps:
            raise RuntimeError("controlled foul-play capture produced no p1 decision steps")
        observation = p1_trajectory.steps[0].observation
        if observation_schema_version is None:
            observation_schema_version = observation.schema_version
            numeric_feature_count = len(observation.numeric_features[0]) if observation.numeric_features else 0
        elif observation.schema_version != observation_schema_version or (
            observation.numeric_features
            and len(observation.numeric_features[0]) != numeric_feature_count
        ):
            raise RuntimeError("controlled foul-play capture observation schema drifted within one pool")
        if handle is None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            handle = out_path.open("x", encoding="utf-8")
        now = time.monotonic()
        record = RolloutRecord(
            battle_id=p1_trajectory.battle_id,
            seed=p1_trajectory.seed,
            format_id=p1_trajectory.format_id,
            policy_ids={"p1": f"neural:{_checkpoint_path_label(config)}", "p2": "foul-play"},
            decision_round_count=len(p1_trajectory.steps),
            elapsed_seconds=now - previous_capture_time,
            terminal=p1_trajectory.terminal,
            trajectory=p1_trajectory,
            belief_set_source_hash=belief_set_source_hash,
        )
        write_rollout_record(handle, record)
        handle.flush()
        os.fsync(handle.fileno())
        if public_corpus_writer is not None:
            captured_public_decisions += public_corpus_writer.append_trajectory(trajectory, acting_player="p1")
        previous_capture_time = now
        captured_games += 1
        emit_progress()

    try:
        benchmark = await run_controlled_foulplay_benchmark(
            config,
            progress_callback=progress_callback,
            trajectory_callback=capture,
        )
    finally:
        if handle is not None:
            handle.close()
        if public_corpus_writer is not None:
            public_corpus_writer.close()
    result = ControlledFoulPlayCaptureResult(
        benchmark=benchmark,
        output_path=out_path,
        pool_id=pool_id,
        checkpoint_sha256=checkpoint_sha256,
        belief_set_source_hash=belief_set_source_hash,
        observation_schema_version=observation_schema_version,
        numeric_feature_count=numeric_feature_count,
        captured_games=captured_games,
        skipped_capped_games=skipped_capped_games,
        skipped_tied_games=skipped_tied_games,
        public_corpus_path=public_corpus_out,
        captured_public_decisions=captured_public_decisions,
    )
    if capture_progress_callback is not None:
        capture_progress_callback({"status": "complete", **result.to_dict()})
    return result


async def capture_controlled_foulplay_collision_sketch(
    config: ControlledFoulPlayConfig,
    *,
    out_path: Path,
    pool_id: str = "controlled-foulplay-collision",
    capture_progress_callback: ControlledFoulPlayCaptureProgressCallback | None = None,
    initial_protocol_signatures: Mapping[str, int] | None = None,
    initial_protocol_signature_game_ids: Iterable[str] = (),
) -> ControlledFoulPlayCollisionSketchResult:
    """Capture compact public collision evidence without retaining model tensors.

    The resulting sketch has only input/public fingerprints and deterministic
    replay locators. It is intentionally not a training rollout or a full
    public-decision corpus, so large coverage samples remain storage-bounded.
    """

    if config.policy_mode != "raw":
        raise ValueError("collision sketch capture requires policy_mode='raw'.")
    if config.pokezero_player != "p1":
        raise ValueError("collision sketch capture requires pokezero_player='p1'.")
    if config.opponent_legal_mask_mode != "hidden":
        raise ValueError("collision sketch capture requires opponent_legal_mask_mode='hidden'.")
    if not pool_id.strip():
        raise ValueError("pool_id must be non-empty.")

    source = _resolved_belief_set_source(config)
    belief_set_source_hash = source.metadata.source_hash if source is not None else None
    checkpoint_sha256 = _checkpoint_sha256(config)
    capture_manifest = public_corpus_manifest(
        checkpoint_sha256=_capture_driver_identity(config, checkpoint_sha256=checkpoint_sha256),
        belief_set_source_hash=belief_set_source_hash,
        capture_config=_public_corpus_capture_config(config),
    )
    # A retry replays the same deterministic seed band. The writer keeps prior
    # valid rows and only appends decision locators that the replay had not yet
    # reached, so a transient pod loss cannot discard a large sketch shard.
    writer = CollisionSketchWriter(
        out_path,
        manifest=collision_sketch_manifest(capture_manifest=capture_manifest),
        resume=True,
    )
    captured_games = 0
    skipped_capped_games = 0
    skipped_tied_games = 0
    captured_new_decisions = 0
    observation_schema_version: str | None = None
    protocol_signatures = _validated_protocol_signature_counts(initial_protocol_signatures or {})
    protocol_signature_game_ids = _validated_protocol_signature_game_ids(initial_protocol_signature_game_ids)

    def progress_payload(*, status: str) -> dict[str, Any]:
        return {
            "status": status,
            "protocol_signature_schema_version": PROTOCOL_SIGNATURE_SCHEMA_VERSION,
            "protocol_signatures": dict(sorted(protocol_signatures.items())),
            "protocol_signature_game_ids": sorted(protocol_signature_game_ids),
            "collision_sketch_capture": {
                "out": str(out_path),
                "pool_id": pool_id,
                "sides": "p1-only",
                "policy_mode": "raw",
                "capture_driver": config.capture_driver,
                "audit_observation_schema": _audit_observation_schema_version(config),
                "checkpoint_sha256": checkpoint_sha256,
                "belief_set_source_hash": belief_set_source_hash,
                "observation_schema_version": observation_schema_version,
                "captured_games": captured_games,
                "skipped_capped_games": skipped_capped_games,
                "skipped_tied_games": skipped_tied_games,
                "captured_decisions": writer.record_count,
                "resumed_decisions": writer.resumed_record_count,
                "captured_new_decisions": captured_new_decisions,
                "recovered_trailing_partial": writer.recovered_trailing_partial,
                "foulplay_search_time_ms": config.search_time_ms,
                "protocol_signature_schema_version": PROTOCOL_SIGNATURE_SCHEMA_VERSION,
                "protocol_signatures": dict(sorted(protocol_signatures.items())),
                "protocol_signature_game_ids": sorted(protocol_signature_game_ids),
            },
        }

    def capture(trajectory: BattleTrajectory) -> None:
        nonlocal captured_games, skipped_capped_games, skipped_tied_games
        nonlocal captured_new_decisions, observation_schema_version
        if trajectory.terminal is None:
            raise RuntimeError("collision sketch capture requires a terminal trajectory")
        signature_schema = trajectory.metadata.get("protocol_signature_schema_version")
        signature_counts = trajectory.metadata.get("protocol_signatures")
        if signature_schema != PROTOCOL_SIGNATURE_SCHEMA_VERSION or not isinstance(signature_counts, Mapping):
            raise RuntimeError("collision sketch capture is missing public protocol-signature census metadata")
        validated_signature_counts = _validated_protocol_signature_counts(signature_counts)
        signature_game_id = _protocol_signature_game_id(
            pool_id=pool_id,
            showdown_seed=trajectory.seed,
            foulplay_seed=(
                config.foulplay_random_seed + (trajectory.seed - config.seed_start)
                if config.foulplay_random_seed is not None
                else trajectory.seed
            ),
        )
        if signature_game_id not in protocol_signature_game_ids:
            protocol_signatures.update(validated_signature_counts)
            protocol_signature_game_ids.add(signature_game_id)
        if trajectory.terminal.winner is None:
            if trajectory.terminal.capped:
                skipped_capped_games += 1
            else:
                skipped_tied_games += 1
            if capture_progress_callback is not None:
                capture_progress_callback(progress_payload(status="running"))
            return
        records = public_decision_records_from_trajectory(trajectory, acting_player="p1")
        if not records:
            raise RuntimeError("collision sketch capture produced no p1 decision records")
        current_schema = records[0].observation.schema_version
        if observation_schema_version is None:
            observation_schema_version = current_schema
        elif observation_schema_version != current_schema:
            raise RuntimeError("collision sketch capture observation schema drifted within one pool")
        foulplay_seed = (
            config.foulplay_random_seed + (trajectory.seed - config.seed_start)
            if config.foulplay_random_seed is not None
            else trajectory.seed
        )
        captured_new_decisions += writer.append_trajectory(records, foulplay_random_seed=foulplay_seed)
        captured_games += 1
        if capture_progress_callback is not None:
            capture_progress_callback(progress_payload(status="running"))

    try:
        # The writer has already fsynced its manifest. Persist matching empty
        # census state before the first game so an early interruption remains
        # safely resumable instead of leaving an orphaned manifest.
        if capture_progress_callback is not None:
            capture_progress_callback(progress_payload(status="running"))
        benchmark = await run_controlled_foulplay_benchmark(config, trajectory_callback=capture)
        writer.complete()
    finally:
        writer.close()
    result = ControlledFoulPlayCollisionSketchResult(
        benchmark=benchmark,
        output_path=out_path,
        pool_id=pool_id,
        checkpoint_sha256=checkpoint_sha256,
        belief_set_source_hash=belief_set_source_hash,
        observation_schema_version=observation_schema_version,
        captured_games=captured_games,
        skipped_capped_games=skipped_capped_games,
        skipped_tied_games=skipped_tied_games,
        captured_decisions=writer.record_count,
        resumed_decisions=writer.resumed_record_count,
        captured_new_decisions=captured_new_decisions,
        recovered_trailing_partial=writer.recovered_trailing_partial,
        protocol_signature_schema_version=PROTOCOL_SIGNATURE_SCHEMA_VERSION,
        protocol_signatures=dict(protocol_signatures),
        protocol_signature_game_ids=sorted(protocol_signature_game_ids),
    )
    if capture_progress_callback is not None:
        capture_progress_callback({"status": "complete", **result.to_dict()})
    return result


def _validated_protocol_signature_counts(value: Mapping[str, Any]) -> Counter[str]:
    """Return validated count-only protocol census data for resumable capture."""

    counts: Counter[str] = Counter()
    for signature, count in value.items():
        if not isinstance(signature, str) or not isinstance(count, int) or count < 0:
            raise RuntimeError("collision sketch capture has invalid public protocol-signature counts")
        counts[signature] += count
    return counts


def _validated_protocol_signature_game_ids(values: Iterable[str]) -> set[str]:
    """Return opaque, fixed-width game identifiers used only to de-duplicate retries."""

    game_ids: set[str] = set()
    for game_id in values:
        if (
            not isinstance(game_id, str)
            or len(game_id) != 64
            or any(char not in "0123456789abcdef" for char in game_id)
        ):
            raise RuntimeError("collision sketch capture has invalid protocol-signature game identifiers")
        game_ids.add(game_id)
    return game_ids


def _protocol_signature_game_id(*, pool_id: str, showdown_seed: int, foulplay_seed: int) -> str:
    """Return an opaque id so retry-safe census state need not retain raw game identifiers."""

    payload = f"{pool_id}\0{showdown_seed}\0{foulplay_seed}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _p1_capture_trajectory(trajectory: BattleTrajectory, *, pool_id: str) -> BattleTrajectory:
    """Keep only PokeZero's public/request-known decisions for external value evaluation."""

    if trajectory.terminal is None:
        raise ValueError("controlled foul-play trajectory has no terminal state.")
    src_metadata = {k: v for k, v in dict(trajectory.metadata).items()
                    # defensively drop any opt-in omniscient trait-capture keys: this artifact is
                    # p1-only/opponent-hidden and must never carry both-sides protocol/requests.
                    if k not in ("omniscient_protocol", "request_history")}
    captured = BattleTrajectory(
        battle_id=trajectory.battle_id,
        format_id=trajectory.format_id,
        seed=trajectory.seed,
        metadata={
            **src_metadata,
            "capture": "controlled-foulplay/raw",
            "pool": pool_id,
            "sides": "p1-only",
        },
    )
    for step in trajectory.steps:
        if step.player_id == "p1":
            captured.append(step)
    captured.record_terminal(trajectory.terminal)
    return captured


async def run_controlled_foulplay_comparison(
    config: ControlledFoulPlayConfig,
    *,
    comparison_mode: str = "per-seed",
    progress_callback: ControlledFoulPlayComparisonProgressCallback | None = None,
) -> ControlledFoulPlayComparisonResult:
    """Run raw checkpoint and root-PUCT against foul-play over the same seed band."""

    if comparison_mode not in _COMPARISON_MODES:
        raise ValueError(f"comparison_mode must be one of {sorted(_COMPARISON_MODES)!r}.")

    if comparison_mode == "per-seed":
        return await _run_controlled_foulplay_comparison_per_seed(
            config,
            progress_callback=progress_callback,
        )
    return await _run_controlled_foulplay_comparison_per_arm(
        config,
        progress_callback=progress_callback,
    )


async def _run_controlled_foulplay_comparison_per_arm(
    config: ControlledFoulPlayConfig,
    *,
    progress_callback: ControlledFoulPlayComparisonProgressCallback | None = None,
) -> ControlledFoulPlayComparisonResult:
    raw_result: ControlledFoulPlayBenchmarkResult | None = None
    root_puct_result: ControlledFoulPlayBenchmarkResult | None = None

    def emit_progress() -> None:
        if progress_callback is None:
            return
        progress_callback(
            ControlledFoulPlayComparisonResult(
                config=config,
                raw=raw_result,
                root_puct=root_puct_result,
                comparison_mode="per-arm",
            )
        )

    def raw_progress(result: ControlledFoulPlayBenchmarkResult) -> None:
        nonlocal raw_result
        raw_result = result
        emit_progress()

    def root_puct_progress(result: ControlledFoulPlayBenchmarkResult) -> None:
        nonlocal root_puct_result
        root_puct_result = result
        emit_progress()

    raw_result = await run_controlled_foulplay_benchmark(
        replace(config, policy_mode="raw"),
        progress_callback=raw_progress,
    )
    root_puct_result = await run_controlled_foulplay_benchmark(
        replace(config, policy_mode="root-puct"),
        progress_callback=root_puct_progress,
    )
    return ControlledFoulPlayComparisonResult(
        config=config,
        raw=raw_result,
        root_puct=root_puct_result,
        comparison_mode="per-arm",
    )


async def _run_controlled_foulplay_comparison_per_seed(
    config: ControlledFoulPlayConfig,
    *,
    progress_callback: ControlledFoulPlayComparisonProgressCallback | None = None,
) -> ControlledFoulPlayComparisonResult:
    raw_games: list[ControlledFoulPlayGameResult] = []
    root_puct_games: list[ControlledFoulPlayGameResult] = []
    raw_offsets: list[int] = []
    root_puct_offsets: list[int] = []
    raw_policy_id: str | None = None
    root_puct_policy_id: str | None = None
    checkpoint_sha256 = _checkpoint_sha256(config)
    raw_value_leaf_provenance: Mapping[str, object] | None = None
    root_puct_value_leaf_provenance: Mapping[str, object] | None = None
    opponent_crashes: list[ControlledFoulPlayOpponentCrash] = []

    def raw_result() -> ControlledFoulPlayBenchmarkResult | None:
        if raw_policy_id is None:
            return None
        return ControlledFoulPlayBenchmarkResult(
            config=replace(config, policy_mode="raw"),
            policy_id=raw_policy_id,
            games=tuple(raw_games),
            checkpoint_sha256=checkpoint_sha256,
            foulplay_random_seed_schedule=_per_seed_foulplay_random_seed_schedule_for_offsets(
                config,
                offsets=raw_offsets,
            ),
            value_leaf_provenance=raw_value_leaf_provenance,
        )

    def root_puct_result() -> ControlledFoulPlayBenchmarkResult | None:
        if root_puct_policy_id is None:
            return None
        return ControlledFoulPlayBenchmarkResult(
            config=replace(config, policy_mode="root-puct"),
            policy_id=root_puct_policy_id,
            games=tuple(root_puct_games),
            checkpoint_sha256=checkpoint_sha256,
            foulplay_random_seed_schedule=_per_seed_foulplay_random_seed_schedule_for_offsets(
                config,
                offsets=root_puct_offsets,
            ),
            value_leaf_provenance=root_puct_value_leaf_provenance,
        )

    def emit_progress() -> None:
        if progress_callback is None:
            return
        progress_callback(
            ControlledFoulPlayComparisonResult(
                config=config,
                raw=raw_result(),
                root_puct=root_puct_result(),
                comparison_mode="per-seed",
                opponent_crashes=tuple(opponent_crashes),
            )
        )

    for offset in range(config.games):
        seed = config.seed_start + offset
        single_config = _single_seed_comparison_config(config, seed=seed, offset=offset)

        raw_single, raw_crash = await _run_single_seed_comparison_arm(
            replace(single_config, policy_mode="raw"),
            seed=seed,
        )
        if raw_crash is not None:
            opponent_crashes.append(raw_crash)
            emit_progress()
            continue
        assert raw_single is not None
        raw_policy_id = raw_single.policy_id
        raw_value_leaf_provenance = raw_single.value_leaf_provenance
        raw_games.extend(raw_single.games)
        raw_offsets.extend([offset] * len(raw_single.games))
        emit_progress()

        root_puct_single, root_puct_crash = await _run_single_seed_comparison_arm(
            replace(single_config, policy_mode="root-puct"),
            seed=seed,
        )
        if root_puct_crash is not None:
            opponent_crashes.append(root_puct_crash)
            # Drop the raw arm's games for this seed so both arms stay symmetric and the
            # crashed seed is fully excluded from paired stats and aggregates.
            if raw_single.games:
                del raw_games[-len(raw_single.games) :]
                del raw_offsets[-len(raw_single.games) :]
            emit_progress()
            continue
        assert root_puct_single is not None
        root_puct_policy_id = root_puct_single.policy_id
        root_puct_value_leaf_provenance = root_puct_single.value_leaf_provenance
        root_puct_games.extend(root_puct_single.games)
        root_puct_offsets.extend([offset] * len(root_puct_single.games))
        emit_progress()

    return ControlledFoulPlayComparisonResult(
        config=config,
        raw=raw_result(),
        root_puct=root_puct_result(),
        comparison_mode="per-seed",
        opponent_crashes=tuple(opponent_crashes),
    )


async def _run_single_seed_comparison_arm(
    single_config: ControlledFoulPlayConfig,
    *,
    seed: int,
) -> tuple[ControlledFoulPlayBenchmarkResult | None, ControlledFoulPlayOpponentCrash | None]:
    """Run one arm of a paired seed, retrying opponent crashes up to opponent_crash_retries times."""

    attempts = 0
    while True:
        attempts += 1
        try:
            return await run_controlled_foulplay_benchmark(single_config), None
        except FoulPlayProcessExitError as error:
            if attempts <= single_config.opponent_crash_retries:
                continue
            return None, ControlledFoulPlayOpponentCrash(
                seed=seed,
                policy_mode=single_config.policy_mode,
                returncode=error.returncode,
                attempts=attempts,
                stage=error.stage,
                stderr_tail=error.log_tail,
            )


def _single_seed_comparison_config(
    config: ControlledFoulPlayConfig,
    *,
    seed: int,
    offset: int,
) -> ControlledFoulPlayConfig:
    foulplay_seed = config.foulplay_random_seed + offset if config.foulplay_random_seed is not None else seed
    return replace(
        config,
        games=1,
        seed_start=seed,
        foulplay_random_seed=foulplay_seed,
    )


def _validate_external_paths(config: ControlledFoulPlayConfig) -> None:
    if config.capture_driver == "checkpoint":
        if config.checkpoint is None:
            raise AssertionError("checkpoint driver validation must reject a missing checkpoint")
        if not config.checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {config.checkpoint}")
    if config.value_checkpoint is not None and not config.value_checkpoint.exists():
        raise FileNotFoundError(f"value checkpoint not found: {config.value_checkpoint}")
    if not (config.showdown_root / "dist" / "sim" / "index.js").exists():
        raise FileNotFoundError(
            f"built Pokemon Showdown simulator not found under {config.showdown_root}; "
            "set --showdown-root to a built checkout."
        )
    if config.belief_set_source_enabled() and not (
        config.showdown_root / "dist" / "data" / "random-battles" / "gen3" / "teams.js"
    ).exists():
        raise FileNotFoundError(
            "belief set source is enabled but the built Gen 3 randbat generator is missing at "
            f"{config.showdown_root}/dist/data/random-battles/gen3/teams.js; run `node build` in "
            "the Showdown checkout or disable via --belief-set-source off."
        )
    if not (config.foulplay_root / "run.py").exists():
        raise FileNotFoundError(
            f"foul-play checkout not found at {config.foulplay_root}; initialize third_party/foul-play "
            "or pass --foulplay-root."
        )
    if not config.resolved_foulplay_python.exists():
        raise FileNotFoundError(
            f"foul-play Python not found at {config.resolved_foulplay_python}; run "
            "scripts/setup_foulplay_eval.sh or pass --foulplay-python."
        )


def _build_policy(
    *,
    config: ControlledFoulPlayConfig,
    model: Any,
    result: Any,
    value_model: Any,
    value_result: Any,
    env_config: LocalShowdownConfig,
    rollout_config: RolloutConfig,
    policy_id: str,
) -> Policy:
    checkpoint = config.checkpoint
    if checkpoint is None:
        raise AssertionError("root-PUCT policy construction requires a checkpoint")
    raw_checkpoint = str(checkpoint.resolve(strict=False))
    raw_checkpoint_sha256 = _sha256_file(checkpoint) if checkpoint.is_file() else None

    def raw_policy(
        policy_id_override: str | None = None,
        *,
        deterministic: bool = True,
        inference_timing: TransformerInferenceTimingAccumulator | None = None,
    ) -> TransformerSoftmaxPolicy:
        return TransformerSoftmaxPolicy(
            model=model,
            result=result,
            deterministic=deterministic,
            sampling_temperature=config.temperature,
            device=config.device,
            policy_id=policy_id_override,
            checkpoint_path=raw_checkpoint,
            weights_sha256=raw_checkpoint_sha256,
            inference_timing=inference_timing,
        )

    if config.policy_mode == "raw":
        return raw_policy(policy_id)

    if config.policy_mode == "engine-mcts":
        # Native Rust MCTS over sampled belief worlds with in-crate TorchScript
        # leaves. Telemetry (including the per-phase encode/model/tree walls)
        # lives on the returned policy's .stats, so a strength row can carry the
        # same measurements its timing row did.
        from .engine_search import EngineMctsConfig, EngineMctsPolicy

        return EngineMctsPolicy(
            dex=load_showdown_dex_cached(config.showdown_root),
            # Always attached, NOT env-gated. The candidate-set source is what
            # the belief sampler draws determinized opponent teams from, so
            # engine-mcts cannot construct a single world without it -- passing
            # None is not a degraded mode, it is an AttributeError on the first
            # decision (determinization._gen3_randbat_belief_start_override_result
            # calls set_source.supports). The env flip point governs whether the
            # belief FEATURES are in the observation, which is a separate
            # question from whether the searcher can sample worlds at all.
            set_source=load_gen3_randbat_source_cached(config.showdown_root),
            policy_id=f"{policy_id}+engine-mcts-d{config.engine_depth}-s{config.engine_sims}",
            config=EngineMctsConfig(
                leaf_eval="model",
                checkpoint_path=str(config.checkpoint),
                model_path=str(config.engine_model_path),
                tables_path=str(config.engine_tables_path),
                model_device=config.device or "cpu",
                worlds=config.engine_worlds,
                search_sims=config.engine_sims,
                search_batch=config.engine_batch,
                search_depth=config.engine_depth,
                c_puct=config.engine_c_puct,
                model_priors=config.engine_model_priors,
                use_opponent_priors=config.engine_opponent_priors,
            ),
        )

    search_policy_id = f"{policy_id}+root-puct"
    # One accumulator spans the prior, opponent-prior, and value closures for
    # each root decision. RootPUCTSearchPolicy snapshots it before/after search
    # so W2 can report encode and forward sub-slices without affecting policy.
    inference_timing = TransformerInferenceTimingAccumulator()

    def value_fn(history: tuple[PokeZeroObservationV0, ...]) -> float:
        return evaluate_transformer_observation_value(
            model=value_model,
            result=value_result,
            observations=history,
            device=config.device,
            timing=inference_timing,
        )

    def prior_fn(history: tuple[PokeZeroObservationV0, ...]) -> tuple[float, ...]:
        return evaluate_transformer_action_priors(
            model=model,
            result=result,
            observations=history,
            temperature=1.0,
            device=config.device,
            timing=inference_timing,
        )

    def opponent_prior_fn(history: tuple[PokeZeroObservationV0, ...]) -> tuple[float, ...]:
        return evaluate_transformer_opponent_action_priors(
            model=model,
            result=result,
            observations=history,
            temperature=config.temperature,
            device=config.device,
            timing=inference_timing,
        )

    scenario_planner = None
    if config.root_opponent_action_candidate_scenarios > 1:
        scenario_planner = prior_top_k_opponent_action_scenario_planner(
            opponent_prior_fn,
            scenario_count=config.root_opponent_action_candidate_scenarios,
        )

    leaf_rollout_policy_factory = None
    if config.leaf_rollout_rounds:
        leaf_rollout_policy_factory = lambda player_id: raw_policy(
            f"{search_policy_id}-leaf-{player_id}",
            deterministic=not config.leaf_rollout_sampling,
            inference_timing=inference_timing,
        )

    start_override_planner = None
    if config.belief_start_overrides:
        set_source = load_gen3_randbat_source_cached(config.showdown_root)
        start_override_planner = gen3_randbat_belief_start_override_planner(set_source)

    root_visit_budget_selector = config.root_visit_budget_selector()
    return RootPUCTSearchPolicy(
        env_factory=lambda: LocalShowdownEnv(env_config),
        rollout_config=rollout_config,
        value_fn=value_fn,
        prior_fn=prior_fn,
        opponent_action_planner=greedy_opponent_action_planner(opponent_prior_fn),
        opponent_action_scenario_planner=scenario_planner,
        fallback_policy=raw_policy(
            f"{search_policy_id}-fallback",
            inference_timing=inference_timing,
        ),
        allow_fallback=config.allow_search_fallback,
        policy_id=search_policy_id,
        checkpoint_path=raw_checkpoint,
        weights_sha256=raw_checkpoint_sha256,
        neural_timing_snapshot=lambda: inference_timing.snapshot().to_dict(),
        cpuct=config.cpuct,
        selection_mode=config.selection_mode,
        root_prior_temperature=config.effective_root_prior_temperature,
        minimum_value_improvement=config.minimum_value_improvement,
        minimum_override_prior_ratio=config.minimum_override_prior_ratio,
        minimum_score_improvement=config.minimum_score_improvement,
        root_visit_budget=config.root_visit_budget,
        root_visit_budget_selector=root_visit_budget_selector,
        root_time_budget_seconds=(
            None if config.root_time_budget_ms is None else config.root_time_budget_ms / 1000.0
        ),
        max_opponent_action_scenarios=config.root_opponent_action_scenarios,
        leaf_rollout_decision_rounds=config.leaf_rollout_rounds,
        leaf_rollout_policy_factory=leaf_rollout_policy_factory,
        start_override_planner=start_override_planner,
        start_override_attempts=config.start_override_attempts,
        start_override_samples_per_scenario=config.belief_start_override_samples,
        start_override_hp_fraction_tolerance=config.start_override_hp_fraction_tolerance,
        leaf_rollout_metadata={
            "root_puct_leaf_rollout_opponent_policy": "checkpoint",
            "root_puct_leaf_rollout_sampling": config.leaf_rollout_sampling,
        }
        if config.leaf_rollout_rounds
        else {},
    )


async def _spawn_foulplay(
    config: ControlledFoulPlayConfig,
    websocket_uri: str,
    *,
    run_count: int | None = None,
) -> asyncio.subprocess.Process:
    env = _foulplay_env(config)
    return await asyncio.create_subprocess_exec(
        *_foulplay_command(config, websocket_uri, run_count=run_count),
        cwd=str(config.foulplay_root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )


def _foulplay_env(config: ControlledFoulPlayConfig) -> dict[str, str]:
    seed = config.resolved_foulplay_random_seed
    return {
        **os.environ,
        "FOULPLAY_LOCAL_NOSEC": "1",
        "PYTHONPATH": str(config.foulplay_root),
        "POKEZERO_FOULPLAY_RANDOM_SEED": str(seed),
        "PYTHONHASHSEED": str(seed % (2**32)),
    }


def _foulplay_command(
    config: ControlledFoulPlayConfig,
    websocket_uri: str,
    *,
    run_count: int | None = None,
) -> tuple[str, ...]:
    resolved_run_count = config.games if run_count is None else run_count
    if resolved_run_count <= 0:
        raise ValueError("run_count must be positive when set.")
    run_path = str(config.foulplay_root / "run.py")
    seed_wrapper = (
        "import os, random, runpy, sys; "
        "random.seed(int(os.environ['POKEZERO_FOULPLAY_RANDOM_SEED'])); "
        "script = sys.argv[1]; "
        "sys.argv = sys.argv[1:]; "
        "runpy.run_path(script, run_name='__main__')"
    )
    return (
        str(config.resolved_foulplay_python),
        "-c",
        seed_wrapper,
        run_path,
        "--websocket-uri",
        websocket_uri,
        "--ps-username",
        config.foulplay_username,
        "--bot-mode",
        "challenge_user",
        "--user-to-challenge",
        config.pokezero_username,
        "--pokemon-format",
        config.format_id,
        "--run-count",
        str(resolved_run_count),
        "--search-time-ms",
        str(config.search_time_ms),
    )


async def _drain_process_stream(
    stream: asyncio.StreamReader | None,
    append: Any,
) -> None:
    if stream is None:
        return
    async for raw in stream:
        append(raw.decode("utf-8", errors="replace").rstrip())


async def _stop_foulplay_process(
    process: asyncio.subprocess.Process,
    log_tasks: Sequence[asyncio.Task[None]],
) -> None:
    """Stop one game-scoped foul-play client and drain its background readers."""

    if process.returncode is None:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
    for task in log_tasks:
        task.cancel()
    if log_tasks:
        await asyncio.gather(*log_tasks, return_exceptions=True)


async def _run_single_game(
    *,
    config: ControlledFoulPlayConfig,
    bridge: _BattleBridge,
    server: _FoulPlayWebsocketServer,
    policy: Policy,
    vocab: CategoryVocabulary,
    dex: ShowdownDex,
    observation_spec: Any,
    feature_masks: "ObservationFeatureMasks" = DEFAULT_OBSERVATION_FEATURE_MASKS,
    seed: int,
    foulplay_process: asyncio.subprocess.Process,
    foulplay_logs: _ProcessLogBuffer,
    trajectory_callback: ControlledFoulPlayTrajectoryCallback | None = None,
    refusal_capture: "_RefusalCapture | None" = None,
) -> ControlledFoulPlayGameResult:
    battle_id = f"{DEFAULT_BATTLE_ID_PREFIX}-{seed}"
    state = _ControlledBattleState(
        battle_id=battle_id,
        seed=seed,
        format_id=config.format_id,
        trajectory=BattleTrajectory(
            battle_id=battle_id,
            format_id=config.format_id,
            seed=seed,
            metadata={
                "opponent_policy_id": "foul-play",
                "controlled_foulplay_bridge": True,
                "pokezero_player": config.pokezero_player,
                "foulplay_player": config.foulplay_player,
            },
        ),
    )
    await server.send_room_lines(
        battle_id,
        ["|init|battle", f"|title|{config.pokezero_username} vs. {config.foulplay_username}"],
    )
    await bridge.send(
        {
            "type": "start",
            "battleId": battle_id,
            "formatid": config.format_id,
            "seed": showdown_seed_from_int(seed),
            "players": {
                config.pokezero_player: config.pokezero_username,
                config.foulplay_player: config.foulplay_username,
            },
        }
    )

    requested_players: tuple[PlayerId, ...] = ()
    decision_round = 0
    terminal: TerminalState | None = None

    while terminal is None:
        event = await bridge.next_event()
        if event.get("battleId") != battle_id:
            continue
        event_type = event.get("type")
        if event_type == "stream":
            await _handle_stream_event(state, server, event, config=config)
            terminal = _terminal_from_public_lines(state.public_lines, config)
            continue
        if event_type == "ready":
            requested_players = tuple(str(player) for player in event.get("requested") or ())
            if not requested_players:
                continue
            # Let protocol output from the final permitted choice settle before deciding this
            # battle exceeded the cap. A terminal win/tie follows that choice as a later event.
            if decision_round >= config.max_decision_rounds:
                terminal = TerminalState(winner=None, turn_count=config.max_decision_rounds, capped=True)
                break
            terminal = await _handle_decision_boundary(
                config=config,
                bridge=bridge,
                server=server,
                state=state,
                policy=policy,
                vocab=vocab,
                dex=dex,
                observation_spec=observation_spec,
                feature_masks=feature_masks,
                decision_round=decision_round,
                requested_players=requested_players,
                foulplay_process=foulplay_process,
                foulplay_logs=foulplay_logs,
            )
            decision_round += 1
            continue
        if event_type == "terminal":
            terminal = _terminal_from_public_lines(state.public_lines, config) or TerminalState(
                winner=None,
                turn_count=decision_round,
            )
            break

    await _notify_foulplay_terminal(
        state=state,
        server=server,
        terminal=terminal,
        config=config,
    )
    winner_name = _winner_name(terminal, config)
    if state.trajectory is not None:
        state.trajectory.record_terminal(terminal)
    elapsed = [
        float(decision.metadata["root_puct_elapsed_seconds"])
        for decision in state.decisions
        if (
            decision.metadata.get("policy_family") == "root-puct-search"
            and not decision.metadata.get("root_puct_fallback")
            and "root_puct_elapsed_seconds" in decision.metadata
        )
    ]
    policy_elapsed = tuple(
        float(decision.metadata["policy_elapsed_seconds"])
        for decision in state.decisions
        if "policy_elapsed_seconds" in decision.metadata
    )
    root_searches = sum(
        1
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
        and not decision.metadata.get("root_puct_fallback")
    )
    root_fallbacks = sum(1 for decision in state.decisions if decision.metadata.get("root_puct_fallback"))
    engine_decisions = sum(
        1 for decision in state.decisions if "engine_mcts" in (decision.metadata or {})
    )
    engine_fallback_reasons: dict[str, int] = {}
    for decision in state.decisions:
        block = (decision.metadata or {}).get("engine_mcts") or {}
        reason = block.get("fallback") if isinstance(block, Mapping) else None
        if reason:
            engine_fallback_reasons[str(reason)] = engine_fallback_reasons.get(str(reason), 0) + 1
    engine_fallbacks = sum(engine_fallback_reasons.values())
    root_fallback_reasons: dict[str, int] = {}
    root_fallback_categories: dict[str, int] = {}
    root_opponent_action_policies: dict[str, int] = {}
    for decision in state.decisions:
        if not decision.metadata.get("root_puct_fallback"):
            continue
        reason = str(decision.metadata.get("root_puct_fallback_reason") or "unknown")
        root_fallback_reasons[reason] = root_fallback_reasons.get(reason, 0) + 1
        category = str(
            decision.metadata.get("root_puct_fallback_category")
            or root_puct_fallback_category(reason)
        )
        root_fallback_categories[category] = root_fallback_categories.get(category, 0) + 1
    for decision in state.decisions:
        if (
            decision.metadata.get("policy_family") != "root-puct-search"
            or decision.metadata.get("root_puct_fallback")
        ):
            continue
        planner_id = decision.metadata.get("root_puct_opponent_action_policy")
        if isinstance(planner_id, str) and planner_id:
            root_opponent_action_policies[planner_id] = root_opponent_action_policies.get(planner_id, 0) + 1
    root_total_visits = sum(
        int(decision.metadata.get("root_puct_total_visits") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
        and not decision.metadata.get("root_puct_fallback")
    )
    root_effective_total_visits = sum(
        int(decision.metadata.get("root_puct_effective_total_visits") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
        and not decision.metadata.get("root_puct_fallback")
    )
    root_scenarios_generated = sum(
        int(decision.metadata.get("root_puct_opponent_action_scenarios_generated") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_scenarios_skipped = sum(
        int(decision.metadata.get("root_puct_opponent_action_scenarios_skipped") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_scenarios_unsearched = sum(
        int(decision.metadata.get("root_puct_opponent_action_scenarios_unsearched") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_scenario_skip_categories: dict[str, int] = {}
    root_missing_sampled_world_reason_categories: dict[str, int] = {}
    root_replay_rejection_decision_rounds: dict[str, int] = {}
    root_replay_request_mismatch_decision_rounds: dict[str, int] = {}
    root_replay_request_mismatch_players: dict[str, int] = {}
    root_replay_request_mismatch_shapes: dict[str, int] = {}
    root_start_override_mismatch_decision_rounds: dict[str, int] = {}
    root_first_observation_mismatch_paths: dict[str, int] = {}
    for decision in state.decisions:
        if decision.metadata.get("policy_family") != "root-puct-search":
            continue
        _merge_count_mapping(
            root_scenario_skip_categories,
            decision.metadata.get("root_puct_opponent_action_skip_categories"),
        )
        _merge_count_mapping(
            root_missing_sampled_world_reason_categories,
            sanitize_root_puct_missing_sampled_world_reason_categories(
                decision.metadata.get("root_puct_opponent_action_missing_sampled_world_reason_categories")
            ),
        )
        _merge_count_mapping(
            root_replay_rejection_decision_rounds,
            decision.metadata.get("root_puct_opponent_action_replay_rejection_decision_rounds"),
        )
        _merge_count_mapping(
            root_replay_request_mismatch_decision_rounds,
            decision.metadata.get("root_puct_opponent_action_replay_request_mismatch_decision_rounds"),
        )
        _merge_count_mapping(
            root_replay_request_mismatch_players,
            decision.metadata.get("root_puct_opponent_action_replay_request_mismatch_players"),
        )
        _merge_count_mapping(
            root_replay_request_mismatch_shapes,
            decision.metadata.get("root_puct_opponent_action_replay_request_mismatch_shapes"),
        )
        _merge_count_mapping(
            root_start_override_mismatch_decision_rounds,
            decision.metadata.get("root_puct_opponent_action_start_override_mismatch_decision_rounds"),
        )
        _merge_count_mapping(
            root_first_observation_mismatch_paths,
            decision.metadata.get("root_puct_opponent_action_first_observation_mismatch_paths"),
        )
    root_action_groups_generated = sum(
        int(decision.metadata.get("root_puct_opponent_action_groups_generated") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_action_groups_used = sum(
        int(decision.metadata.get("root_puct_opponent_action_groups_used") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_action_groups_skipped = sum(
        int(decision.metadata.get("root_puct_opponent_action_groups_skipped") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_action_groups_unsearched = sum(
        int(decision.metadata.get("root_puct_opponent_action_groups_unsearched") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_selected_prior_action_changes = sum(
        1
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
        and not decision.metadata.get("root_puct_fallback")
        and decision.metadata.get("root_puct_selected_changed_prior_action")
    )
    root_pre_gate_prior_action_changes = sum(
        1
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
        and not decision.metadata.get("root_puct_fallback")
        and decision.metadata.get("root_puct_pre_gate_changed_prior_action")
    )
    root_time_budget_exhaustions = sum(
        1
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
        and not decision.metadata.get("root_puct_fallback")
        and decision.metadata.get("root_puct_time_budget_exhausted")
    )
    root_start_override_sources_used = sum(
        int(decision.metadata.get("root_puct_start_override_sources_used") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
        and not decision.metadata.get("root_puct_fallback")
    )
    root_start_override_attempts_used = sum(
        int(decision.metadata.get("root_puct_start_override_attempts_used") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_start_override_duplicate_attempts = sum(
        int(decision.metadata.get("root_puct_start_override_duplicate_attempts") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_start_override_shared_samples = sum(
        int(decision.metadata.get("root_puct_start_override_shared_samples") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_start_override_shared_samples_accepted = sum(
        int(decision.metadata.get("root_puct_start_override_shared_samples_accepted") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_start_override_shared_samples_rejected = sum(
        int(decision.metadata.get("root_puct_start_override_shared_samples_rejected") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_start_override_direct_materializations = sum(
        int(decision.metadata.get("root_puct_start_override_direct_materializations") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_start_override_replay_materializations = sum(
        int(decision.metadata.get("root_puct_start_override_replay_materializations") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_prior_action_change_details = _root_puct_prior_action_change_details(state.decisions)
    if trajectory_callback is not None:
        if state.trajectory.terminal is None:
            raise RuntimeError("controlled foul-play capture requires a terminal trajectory")
        state.trajectory.metadata = {
            **dict(state.trajectory.metadata),
            "public_resolved_action_rounds": [
                round_.to_dict() for round_ in state.public_resolved_action_rounds
            ],
            "protocol_signature_schema_version": PROTOCOL_SIGNATURE_SCHEMA_VERSION,
            "protocol_signatures": dict(sorted(protocol_signature_counts(state.public_lines).items())),
        }
        # Opt-in omniscient stash for trait capture (trait_foulplay.py). Gated on an env flag so it
        # never contaminates the default p1-only/opponent-hidden capture path, which spreads this
        # metadata into its artifacts (see _p1_capture_trajectory).
        if os.environ.get("POKEZERO_TRAIT_CAPTURE"):
            state.trajectory.metadata = {
                **dict(state.trajectory.metadata),
                "omniscient_protocol": list(state.public_lines),
                "request_history": [[seat, line] for seat, line in state.request_history],
            }
        trajectory_callback(state.trajectory)
    return ControlledFoulPlayGameResult(
        battle_id=battle_id,
        seed=seed,
        winner=winner_name,
        pokezero_won=winner_name == config.pokezero_username,
        tied=terminal.winner is None and not terminal.capped,
        capped=terminal.capped,
        decision_rounds=decision_round,
        pokezero_decisions=len(state.decisions),
        root_puct_searches=root_searches,
        root_puct_fallbacks=root_fallbacks,
        engine_mcts_decisions=engine_decisions,
        engine_mcts_fallbacks=engine_fallbacks,
        engine_mcts_fallback_reasons=dict(engine_fallback_reasons),
        root_puct_opponent_action_policies=root_opponent_action_policies,
        root_puct_total_visits=root_total_visits,
        root_puct_effective_total_visits=root_effective_total_visits,
        root_puct_opponent_action_scenarios_generated=root_scenarios_generated,
        root_puct_opponent_action_scenarios_skipped=root_scenarios_skipped,
        root_puct_opponent_action_scenarios_unsearched=root_scenarios_unsearched,
        root_puct_opponent_action_skip_categories=root_scenario_skip_categories,
        root_puct_opponent_action_missing_sampled_world_reason_categories=(
            root_missing_sampled_world_reason_categories
        ),
        root_puct_opponent_action_replay_rejection_decision_rounds=(
            root_replay_rejection_decision_rounds
        ),
        root_puct_opponent_action_replay_request_mismatch_decision_rounds=(
            root_replay_request_mismatch_decision_rounds
        ),
        root_puct_opponent_action_replay_request_mismatch_players=(
            root_replay_request_mismatch_players
        ),
        root_puct_opponent_action_replay_request_mismatch_shapes=(
            root_replay_request_mismatch_shapes
        ),
        root_puct_opponent_action_start_override_mismatch_decision_rounds=(
            root_start_override_mismatch_decision_rounds
        ),
        root_puct_opponent_action_first_observation_mismatch_paths=root_first_observation_mismatch_paths,
        root_puct_opponent_action_groups_generated=root_action_groups_generated,
        root_puct_opponent_action_groups_used=root_action_groups_used,
        root_puct_opponent_action_groups_skipped=root_action_groups_skipped,
        root_puct_opponent_action_groups_unsearched=root_action_groups_unsearched,
        root_puct_selected_prior_action_changes=root_selected_prior_action_changes,
        root_puct_pre_gate_prior_action_changes=root_pre_gate_prior_action_changes,
        root_puct_time_budget_exhaustions=root_time_budget_exhaustions,
        root_puct_start_override_sources_used=root_start_override_sources_used,
        root_puct_start_override_attempts_used=root_start_override_attempts_used,
        root_puct_start_override_duplicate_attempts=root_start_override_duplicate_attempts,
        root_puct_start_override_shared_samples=root_start_override_shared_samples,
        root_puct_start_override_shared_samples_accepted=root_start_override_shared_samples_accepted,
        root_puct_start_override_shared_samples_rejected=root_start_override_shared_samples_rejected,
        root_puct_start_override_direct_materializations=root_start_override_direct_materializations,
        root_puct_start_override_replay_materializations=root_start_override_replay_materializations,
        root_puct_prior_action_change_details=root_prior_action_change_details,
        root_puct_fallback_reasons=root_fallback_reasons,
        root_puct_fallback_categories=root_fallback_categories,
        root_puct_average_elapsed_seconds=(sum(elapsed) / len(elapsed) if elapsed else None),
        policy_elapsed_seconds=policy_elapsed,
        root_puct_timings=tuple(
            timing
            for decision in state.decisions
            if (
                timing := _root_puct_timing_from_metadata(decision.metadata)
            ) is not None
        ),
        pokezero_decision_players=tuple(state.pokezero_decision_players),
        pokezero_submitted_choice_players=tuple(state.pokezero_submitted_choice_players),
        opponent_journal=_opponent_journal_for_result(
            state.opponent_journal,
            mode=config.opponent_journal,
            policy=policy,
            battle_id=battle_id,
            seat=config.pokezero_player,
        ),
        opponent_journal_recorded=len(state.opponent_journal),
        opponent_journal_failures=state.opponent_journal_failures,
        # Filtered on the REAL battle_id, so a run-scoped recorder still files each
        # refusal against the battle it happened in.
        refusal_records=(
            refusal_capture.records_for(battle_id) if refusal_capture is not None else ()
        ),
    )


def _last_addressed_round(policy: Any, battle_id: str, seat: PlayerId | None = None) -> int | None:
    """Highest round of ``(battle_id, seat)`` that filed a fallback ADDRESS, else None.

    Reads ``policy.stats.fallback_samples`` -- the same store the address reader
    consumes -- rather than a second bridge-side tally, so the journal cannot cover
    a different set of rounds than the addresses it exists to serve.

    ``seat`` is the ACTING seat, matching ``FallbackAddress.seat`` (``context.player_id``,
    i.e. OUR seat), not the opponent seat the journal entries carry. Filtering on it
    completes the address locator ``(battle_id, round, seat)`` rather than half of it.
    INERT in the bridge today: ``pokezero_player`` is fixed for a whole invocation and
    ``battle_id`` embeds a per-invocation seed, so no foreign-seat entry can reach a
    live call. Kept because the locator is the contract, and because a caller that
    ever shares one policy across seats -- the shape `foulplay_paired_eval` already
    has at the process level -- would otherwise silently over-extend the prefix. It is
    tested directly rather than through the bridge, where it cannot fire.

    None (not 0) when the policy has no address store at all. Only engine-mcts keeps
    one; the raw and root-puct arms produce no ``fallback_samples`` either, so there
    is no address in those shards for a journal to make replayable. Distinguishing
    "no store" from "store, zero hits" matters only to the caller's telemetry, and
    both correctly yield an empty journal.
    """
    samples = getattr(getattr(policy, "stats", None), "fallback_samples", None)
    if not isinstance(samples, Mapping):
        return None
    last: int | None = None
    for entries in samples.values():
        if not isinstance(entries, (list, tuple)):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping) or entry.get("battle_id") != battle_id:
                continue
            if seat is not None and entry.get("seat") != seat:
                continue
            round_index = entry.get("round")
            if isinstance(round_index, bool) or not isinstance(round_index, int):
                continue
            last = round_index if last is None else max(last, round_index)
    return last


def _opponent_journal_for_result(
    journal: Sequence[OpponentJournalEntry],
    *,
    mode: str,
    policy: Any,
    battle_id: str,
    seat: PlayerId | None = None,
) -> tuple[OpponentJournalEntry, ...]:
    """Narrow a fully-recorded journal to what the configured mode emits."""
    if mode == "off" or not journal:
        return ()
    if mode == "full":
        return tuple(journal)
    # "addressed": the PREFIX a replay of this battle's last address needs. A prefix,
    # not the single round, because the live root fold is advanced incrementally per
    # decision and never refolded from a log -- see the module block.
    #
    # Inclusive of the addressed round: the opponent's move at round R is submitted
    # simultaneously with ours, so a driver that wants to step past the address
    # rather than stop on it already has what it needs.
    last = _last_addressed_round(policy, battle_id, seat)
    if last is None:
        return ()
    return tuple(entry for entry in journal if entry.round <= last)


def _root_puct_timing_from_metadata(
    metadata: Mapping[str, Any],
) -> Mapping[str, float | int] | None:
    """Normalize a decision timing map before writing it to a FoulPlay artifact."""

    payload = metadata.get("root_puct_timing")
    if not isinstance(payload, Mapping):
        return None
    normalized: dict[str, float | int] = {}
    for name in _ROOT_PUCT_TIMING_FIELD_NAMES:
        value = payload.get(name, 0)
        if name.endswith("_count"):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return None
            normalized[name] = value
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return None
        normalized[name] = float(value)
    return RootPUCTSearchTiming(**normalized).to_dict()


def _aggregate_root_puct_timings(
    timings: Sequence[Mapping[str, float | int]],
) -> RootPUCTSearchTiming:
    """Aggregate persisted timing records without summing derived residual fields."""

    return RootPUCTSearchTiming.aggregate(
        tuple(
            RootPUCTSearchTiming(
                **{
                    name: timing[name]
                    for name in _ROOT_PUCT_TIMING_FIELD_NAMES
                }
            )
            for timing in timings
        )
    )


def _root_puct_prior_action_change_details(
    decisions: Sequence[PolicyDecision],
) -> tuple[Mapping[str, Any], ...]:
    details: list[dict[str, Any]] = []
    for decision_index, decision in enumerate(decisions):
        metadata = decision.metadata
        if metadata.get("policy_family") != "root-puct-search":
            continue
        if metadata.get("root_puct_fallback"):
            continue
        if not (
            metadata.get("root_puct_selected_changed_prior_action")
            or metadata.get("root_puct_pre_gate_changed_prior_action")
        ):
            continue
        details.append(
            {
                "decision_index": decision_index,
                "selected_action": decision.action_index,
                "search_action": _optional_int(metadata.get("root_puct_search_action")),
                "prior_action": _optional_int(metadata.get("root_puct_prior_action")),
                "selected_changed_prior_action": bool(metadata.get("root_puct_selected_changed_prior_action")),
                "pre_gate_changed_prior_action": bool(metadata.get("root_puct_pre_gate_changed_prior_action")),
                "selected_value": _optional_float(metadata.get("root_puct_selected_value")),
                "search_value": _optional_float(metadata.get("root_puct_search_action_value")),
                "prior_value": _optional_float(metadata.get("root_puct_prior_value")),
                "selected_score": _optional_float(metadata.get("root_puct_selected_score")),
                "search_score": _optional_float(metadata.get("root_puct_search_action_score")),
                "prior_score": _optional_float(metadata.get("root_puct_prior_score")),
                "selected_action_prior": _optional_float(metadata.get("root_puct_selected_action_prior")),
                "search_action_prior": _optional_float(metadata.get("root_puct_search_action_prior")),
                "prior_action_prior": _optional_float(metadata.get("root_puct_prior_action_prior")),
                "selected_visits": _optional_int(metadata.get("root_puct_selected_action_visits")),
                "search_visits": _optional_int(metadata.get("root_puct_search_action_visits")),
                "prior_visits": _optional_int(metadata.get("root_puct_prior_action_visits")),
                "value_gate_used": bool(metadata.get("root_puct_value_gate_used", False)),
                "prior_ratio_gate_used": bool(metadata.get("root_puct_prior_ratio_gate_used", False)),
                "minimum_override_prior_ratio": _optional_float(
                    metadata.get("root_puct_minimum_override_prior_ratio")
                ),
                "prior_ratio_gate_required_prior": _optional_float(
                    metadata.get("root_puct_prior_ratio_gate_required_prior")
                ),
                "score_gate_used": bool(metadata.get("root_puct_score_gate_used", False)),
                "minimum_score_improvement": _optional_float(
                    metadata.get("root_puct_minimum_score_improvement")
                ),
                "score_gate_required_score": _optional_float(
                    metadata.get("root_puct_score_gate_required_score")
                ),
            }
        )
    return tuple(details)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


async def _handle_stream_event(
    state: _ControlledBattleState,
    server: _FoulPlayWebsocketServer,
    event: Mapping[str, Any],
    *,
    config: ControlledFoulPlayConfig,
) -> None:
    stream = event.get("stream")
    raw_lines = event.get("lines")
    if not isinstance(stream, str) or not isinstance(raw_lines, list):
        raise RuntimeError(f"malformed BattleStream event: {event!r}")
    lines = [str(line) for line in raw_lines if str(line)]
    if stream == "omniscient":
        state.public_lines.extend(lines)
    elif stream in {"p1", "p2"}:
        for line in lines:
            if line.startswith("|request|"):
                state.request_lines[stream] = line
                state.request_history.append((stream, line))
        if stream == config.foulplay_player:
            forwarded = [_line_for_foulplay(state, line) for line in lines]
            for chunk in _line_chunks_safe_for_foulplay(forwarded):
                await server.send_room_lines(state.battle_id, chunk)
            if any(_is_terminal_protocol_line(line) for line in forwarded):
                state.foulplay_terminal_sent = True


def _capture_resolved_public_action_round(
    state: _ControlledBattleState,
    decision_round: int,
) -> None:
    """Project completed protocol events to public action IDs for the prior round.

    This only reads the public BattleStream transcript. It deliberately does
    not inspect either request, a FoulPlay choice string, or an opponent
    observation to recover a request-local action slot.
    """

    lines = tuple(state.public_lines[state.public_line_cursor :])
    state.public_line_cursor = len(state.public_lines)
    if decision_round == 0:
        return
    action_round = public_action_round_from_protocol_lines(
        lines,
        turn_index=decision_round - 1,
        requested_players=state.previous_requested_players,
    )
    state.public_resolved_action_rounds.append(action_round)
    if state.trajectory is not None:
        append_public_action_round(state.trajectory, action_round)


async def _notify_foulplay_terminal(
    *,
    state: _ControlledBattleState,
    server: _FoulPlayWebsocketServer,
    terminal: TerminalState,
    config: ControlledFoulPlayConfig,
) -> None:
    if state.foulplay_terminal_sent:
        return
    line = _terminal_line_for_foulplay(terminal, config)
    await server.send_room_lines(state.battle_id, [line])
    state.foulplay_terminal_sent = True


def _terminal_line_for_foulplay(
    terminal: TerminalState,
    config: ControlledFoulPlayConfig,
) -> str:
    winner = _winner_name(terminal, config)
    if winner is None:
        return "|tie|"
    return f"|win|{winner}"


def _is_terminal_protocol_line(line: str) -> bool:
    return line.startswith("|win|") or line == "|tie" or line.startswith("|tie|")


def _line_for_foulplay(state: _ControlledBattleState, line: str) -> str:
    if not line.startswith("|request|"):
        return line
    payload = json.loads(line[len("|request|") :])
    if isinstance(payload, dict) and "rqid" not in payload:
        payload = dict(payload)
        payload["rqid"] = state.next_foulplay_rqid
        state.next_foulplay_rqid += 1
        return "|request|" + json.dumps(payload, separators=(",", ":"))
    return line


def _line_chunks_safe_for_foulplay(lines: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Filter and chunk BattleStream lines into messages foul-play can parse.

    foul-play uses the first pipe-delimited command in a websocket message to decide how to parse
    the whole block. BattleStream can put metadata before ``|player|`` or ``|request|`` in the same
    chunk, so force those parser-sensitive lines to the front of their own messages.
    """

    safe_lines = tuple(
        line
        for line in lines
        if line and line != "|" and not line.startswith("|t:|")
    )
    chunks: list[tuple[str, ...]] = []
    current: list[str] = []
    for line in safe_lines:
        if line.startswith("|player|") or line.startswith("|request|"):
            if current:
                chunks.append(tuple(current))
                current = []
            chunks.append((line,))
        else:
            current.append(line)
    if current:
        chunks.append(tuple(current))
    return tuple(chunks)


async def _handle_decision_boundary(
    *,
    config: ControlledFoulPlayConfig,
    bridge: _BattleBridge,
    server: _FoulPlayWebsocketServer,
    state: _ControlledBattleState,
    policy: Policy,
    vocab: CategoryVocabulary,
    dex: ShowdownDex,
    observation_spec: Any,
    feature_masks: "ObservationFeatureMasks" = DEFAULT_OBSERVATION_FEATURE_MASKS,
    decision_round: int,
    requested_players: tuple[PlayerId, ...],
    foulplay_process: asyncio.subprocess.Process,
    foulplay_logs: _ProcessLogBuffer,
) -> TerminalState | None:
    assert state.trajectory is not None
    pokezero_player = config.pokezero_player
    foulplay_player = config.foulplay_player
    _capture_resolved_public_action_round(state, decision_round)
    state.previous_requested_players = requested_players
    belief_set_source = _resolved_belief_set_source(config)
    player_states = {
        player: _player_state(
            state,
            player,
            set_source=belief_set_source,
            include_turn_merged=observation_spec.schema_version in TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS,
        )
        for player in requested_players
    }
    observations = {
        player: _observation_with_search_metadata(
            observation_from_player_state(
                player_states[player],
                category_vocab=vocab,
                spec=observation_spec,
                dex=dex,
                feature_masks=feature_masks,
            ),
            player_states[player],
        )
        for player in requested_players
    }
    choices: dict[PlayerId, str] = {}
    decisions: dict[PlayerId, PolicyDecision] = {}
    pokezero_context: PolicyContext | None = None
    if pokezero_player in requested_players:
        public_materialization_state = (
            _public_materialization_state(
                state,
                pokezero_player,
                set_source=belief_set_source,
            )
            # Capability, not identity: ANY policy that consumes a materialized
            # public state needs one. Gating on isinstance(RootPUCTSearchPolicy)
            # silently handed EngineMctsPolicy a None, so engine-MCTS fell back
            # to uniform-legal on every decision and played random moves while
            # reporting no error (0/20 vs the raw policy's 10/20).
            if getattr(policy, "requires_public_materialization_state", False)
            or isinstance(policy, RootPUCTSearchPolicy)
            else None
        )
        pokezero_context = PolicyContext(
            player_id=pokezero_player,
            decision_round_index=decision_round,
            battle_id=state.battle_id,
            format_id=config.format_id,
            seed=state.seed,
            observation=observations[pokezero_player],
            requested_players=requested_players,
            trajectory=state.trajectory,
            requested_legal_action_masks=_requested_legal_action_masks_for_context(
                observations,
                acting_player=pokezero_player,
                opponent_legal_mask_mode=config.opponent_legal_mask_mode,
            ),
            requested_observations=_requested_observations_for_context(
                observations,
                acting_player=pokezero_player,
                opponent_legal_mask_mode=config.opponent_legal_mask_mode,
            ),
            public_materialization_state=public_materialization_state,
        )
        # Match the local benchmark boundary: policy selection begins after the
        # observation/context are ready and ends at the returned decision.
        pokezero_choice_wall_start = time.perf_counter()
        policy_decision = await asyncio.to_thread(
            _select_policy_decision,
            policy,
            observations[pokezero_player],
            pokezero_context,
            seed=state.seed,
        )
        policy_elapsed_seconds = time.perf_counter() - pokezero_choice_wall_start
        choices[pokezero_player] = showdown_choice_for_action(
            player_states[pokezero_player],
            policy_decision.action_index,
        )
        decisions[pokezero_player] = replace(
            policy_decision,
            metadata={
                **dict(policy_decision.metadata),
                "policy_elapsed_seconds": policy_elapsed_seconds,
            },
        )
        # Capture the actual controller context that selected the decision. This is distinct from
        # a policy display id and remains useful when a checkpoint happens to use that same id.
        state.pokezero_decision_players.append(pokezero_context.player_id)
    if foulplay_player in requested_players:
        choice = await _wait_for_foulplay_choice_or_exit(
            server=server,
            battle_id=state.battle_id,
            process=foulplay_process,
            logs=foulplay_logs,
        )
        foulplay_action = action_index_from_choice_string(player_states[foulplay_player], choice)
        if foulplay_action is None:
            raise RuntimeError(f"unable to decode foul-play choice {choice!r}.")
        choices[foulplay_player] = choice
        decisions[foulplay_player] = PolicyDecision(
            action_index=foulplay_action,
            policy_id="foul-play",
            metadata={"raw_choice": choice},
        )
        # Journal AFTER the decode and BEFORE the submit, reading only values that
        # already exist. Nothing here is passed to a policy, to the searcher, or to
        # the BattleStream, and no RNG is drawn -- the pokezero decision above has
        # already been selected and submitted into `choices`, so this cannot reorder
        # or perturb it.
        #
        # ISOLATED, because this is TELEMETRY on the live decision path and it is ON
        # BY DEFAULT. Every input is already validated by the lines above, so the
        # except is not expected to fire -- but the failure mode it guards is losing
        # a battle (and, in the paired-eval harness, forfeiting a scored seed band)
        # to a bug in a field nobody scores. The count is reported per game and
        # summed into the journal header, so a silently short journal is visible as a
        # number: an unrecorded round makes every later round of that battle
        # unreplayable, and that must not be inferred from a gap.
        if config.opponent_journal != "off":
            try:
                state.opponent_journal.append(
                    OpponentJournalEntry(
                        round=decision_round,
                        seat=foulplay_player,
                        choice=choice,
                        action=foulplay_action,
                        request_sha256=_request_digest(state.request_lines.get(foulplay_player)),
                    )
                )
            except Exception:  # noqa: BLE001 -- count, never fail the battle
                state.opponent_journal_failures += 1

    for player in requested_players:
        decision = decisions.get(player)
        if decision is None:
            continue
        state.trajectory.append(
            TrajectoryStep(
                player_id=player,
                turn_index=decision_round,
                observation=observations[player],
                legal_action_mask=tuple(observations[player].legal_action_mask),
                action_index=decision.action_index,
                metadata={"policy_id": decision.policy_id, **dict(decision.metadata)},
            )
        )
        if player == pokezero_player:
            state.decisions.append(decision)

    await bridge.send({"type": "choices", "battleId": state.battle_id, "choices": choices})
    if pokezero_context is not None and pokezero_context.player_id in choices:
        state.pokezero_submitted_choice_players.append(pokezero_context.player_id)
    return None


def _observation_with_search_metadata(
    observation: PokeZeroObservationV0,
    state: PlayerRelativeBattleState,
) -> PokeZeroObservationV0:
    return replace(
        observation,
        metadata={
            **dict(observation.metadata),
            "belief_view": state.belief_view.to_overlay_payload(),
        },
    )


async def _wait_for_foulplay_choice_or_exit(
    *,
    server: _FoulPlayWebsocketServer,
    battle_id: str,
    process: asyncio.subprocess.Process,
    logs: _ProcessLogBuffer,
) -> str:
    if process.returncode is not None:
        raise FoulPlayProcessExitError(stage="choosing", returncode=process.returncode, log_tail=logs.tail())
    choice_task = asyncio.create_task(server.wait_for_choice(battle_id=battle_id))
    process_task = asyncio.create_task(process.wait())
    try:
        done, pending = await asyncio.wait(
            {choice_task, process_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if choice_task in done:
            return choice_task.result()
        raise FoulPlayProcessExitError(
            stage="choosing",
            returncode=process.returncode,
            log_tail=logs.tail(),
        )
    finally:
        for task in (choice_task, process_task):
            if not task.done():
                task.cancel()


async def _wait_for_foulplay_challenge_or_exit(
    *,
    server: _FoulPlayWebsocketServer,
    expected_target: str,
    process: asyncio.subprocess.Process,
    logs: _ProcessLogBuffer,
) -> None:
    if process.returncode is not None:
        raise FoulPlayProcessExitError(stage="challenging", returncode=process.returncode, log_tail=logs.tail())
    challenge_task = asyncio.create_task(server.wait_for_challenge(expected_target=expected_target))
    process_task = asyncio.create_task(process.wait())
    try:
        done, pending = await asyncio.wait(
            {challenge_task, process_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if challenge_task in done:
            challenge_task.result()
            return
        raise FoulPlayProcessExitError(
            stage="challenging",
            returncode=process.returncode,
            log_tail=logs.tail(),
        )
    finally:
        for task in (challenge_task, process_task):
            if not task.done():
                task.cancel()


def _select_policy_decision(
    policy: Policy,
    observation: PokeZeroObservationV0,
    context: PolicyContext,
    *,
    seed: int,
) -> PolicyDecision:
    rng = random.Random(f"{seed}:{context.player_id}:{context.decision_round_index}")
    selector = getattr(policy, "select_action_with_context", None)
    if callable(selector):
        return selector(context, rng=rng)
    return policy.select_action(observation, rng=rng)


def _requested_legal_action_masks_for_context(
    observations: Mapping[PlayerId, PokeZeroObservationV0],
    *,
    acting_player: PlayerId,
    opponent_legal_mask_mode: str,
) -> dict[PlayerId, tuple[bool, ...]]:
    masks: dict[PlayerId, tuple[bool, ...]] = {}
    for player, observation in observations.items():
        if player != acting_player and opponent_legal_mask_mode == "hidden":
            continue
        masks[player] = tuple(observation.legal_action_mask)
    return masks


def _requested_observations_for_context(
    observations: Mapping[PlayerId, PokeZeroObservationV0],
    *,
    acting_player: PlayerId,
    opponent_legal_mask_mode: str,
) -> dict[PlayerId, PokeZeroObservationV0]:
    return {
        player: observation
        for player, observation in observations.items()
        if player == acting_player or opponent_legal_mask_mode != "hidden"
    }


def _resolved_belief_set_source(config: ControlledFoulPlayConfig):
    """Candidate-set source for belief views, matching the training-side gate.

    The loader is process-cached, so resolving per decision is cheap. Returning None keeps the
    revealed-facts-only behavior.
    """
    if not config.belief_set_source_enabled():
        return None
    return load_gen3_randbat_source_cached(config.showdown_root)


_PROVENANCE_WARNINGS_EMITTED: set[tuple[str, str | None, str | None]] = set()


def _warn_on_belief_provenance_mismatch(config: ControlledFoulPlayConfig, result: Any) -> None:
    """Warn when benchmark observation conditions differ from the checkpoint's training provenance.

    Non-fatal: legacy checkpoints record no provenance, and a deliberate ablation run is
    legitimate — but a silent mismatch systematically distorts strength/value reads. Emitted once
    per (checkpoint, condition) per process: per-seed comparison mode re-enters the benchmark for
    every arm of every pair, and hundreds of identical lines would bury real diagnostics.
    """
    recorded = getattr(result, "belief_set_source_hash", None)
    current = _resolved_belief_set_source(config)
    current_hash = current.metadata.source_hash if current is not None else None
    if recorded == current_hash:
        return
    key = (_checkpoint_path_label(config) or f"audit-driver:{config.capture_driver}", recorded, current_hash)
    if key in _PROVENANCE_WARNINGS_EMITTED:
        return
    _PROVENANCE_WARNINGS_EMITTED.add(key)
    benchmark_side = f"enabled ({current_hash[:12]})" if current_hash else "disabled"
    if recorded is None:
        detail = (
            "checkpoint records no belief provenance (legacy or source-off training) "
            f"while the benchmark side is {benchmark_side}"
        )
    elif current_hash is None:
        detail = f"checkpoint trained with candidate-set source {recorded[:12]} but the benchmark runs with it disabled"
    else:
        detail = f"checkpoint provenance {recorded[:12]} != benchmark source {current_hash[:12]}"
    print(f"warning: belief set-source mismatch — {detail}.", file=sys.stderr)


def _player_state(
    state: _ControlledBattleState,
    player: PlayerId,
    *,
    set_source=None,
    include_turn_merged: bool = False,
) -> PlayerRelativeBattleState:
    replay = parse_showdown_replay(
        state.all_lines(),
        battle_id=state.battle_id,
        complete_prefix=True,
        hp_visibility={"p1": "exact", "p2": "exact"},
    )
    return normalize_for_player(
        replay,
        player_id=player,
        configured_showdown_slot=player,
        format_id=state.format_id,
        set_source=set_source,
        include_turn_merged=include_turn_merged,
    )


def _public_materialization_state(
    state: _ControlledBattleState,
    player: PlayerId,
    *,
    set_source=None,
) -> PublicBattleMaterializationState:
    """Build a direct-search source from public protocol plus the actor's own request.

    The controlled bridge stores both request streams so it can drive the external opponent, but
    this boundary intentionally parses only the omniscient/public transcript and admits exactly
    one private payload: the PokeZero player's current request.
    """

    request_line = state.request_lines.get(player)
    if request_line is None or not request_line.startswith("|request|"):
        raise RuntimeError(f"missing current request for direct materialization player {player!r}.")
    request = json.loads(request_line[len("|request|") :])
    if not isinstance(request, Mapping):
        raise RuntimeError("direct materialization request must be a JSON object.")
    # The controlled bridge has both request streams to drive FoulPlay, but direct search may
    # carry only the acting player's historical requests. They retain PP for a Pokemon that was
    # active earlier and is now benched; the opponent's request payload remains excluded.
    actor_requests: list[Mapping[str, Any]] = []
    for request_player, historical_line in state.request_history:
        if request_player != player or not historical_line.startswith("|request|"):
            continue
        historical_request = json.loads(historical_line[len("|request|") :])
        if isinstance(historical_request, Mapping):
            actor_requests.append(historical_request)
    public_replay = parse_showdown_replay(
        state.public_lines,
        battle_id=state.battle_id,
        complete_prefix=True,
        hp_visibility={"p1": "exact", "p2": "exact"},
    )
    belief_engine = PublicBattleBeliefEngine.from_events(
        public_replay.public_events,
        format_id=state.format_id,
        set_source=set_source,
    )
    return PublicBattleMaterializationState(
        player_id=player,
        format_id=state.format_id,
        observation_format_id=state.format_id,
        replay=public_replay,
        belief_engine=belief_engine,
        self_request=json.loads(json.dumps(request, separators=(",", ":"))),
        self_move_states=actor_move_states_from_request_history(
            actor_requests,
            initial_request=actor_requests[0] if actor_requests else request,
        ),
        self_initial_request=json.loads(
            json.dumps(actor_requests[0] if actor_requests else request, separators=(",", ":"))
        ),
    )


def _terminal_from_public_lines(
    lines: Sequence[str],
    config: ControlledFoulPlayConfig,
) -> TerminalState | None:
    turn = 0
    winner: PlayerId | None = None
    for line in lines:
        if line.startswith("|turn|"):
            try:
                turn = int(line.split("|", 2)[2])
            except (IndexError, ValueError):
                pass
        elif line.startswith("|win|"):
            winner_name = line.split("|", 2)[2] if len(line.split("|", 2)) >= 3 else ""
            if winner_name == config.pokezero_username:
                winner = config.pokezero_player
            elif winner_name == config.foulplay_username:
                winner = config.foulplay_player
            return TerminalState(winner=winner, turn_count=turn)
        elif line == "|tie" or line.startswith("|tie|"):
            return TerminalState(winner=None, turn_count=turn)
    return None


def _winner_name(terminal: TerminalState, config: ControlledFoulPlayConfig) -> str | None:
    if terminal.winner == config.pokezero_player:
        return config.pokezero_username
    if terminal.winner == config.foulplay_player:
        return config.foulplay_username
    return None


def _split_outgoing_showdown_message(message: str) -> tuple[str, str]:
    if "|" not in message:
        return "", message.strip()
    room, body = message.split("|", 1)
    return room.strip(), body.strip()


def _choice_body_from_outgoing_message(body: str) -> str | None:
    command = body.split("|", 1)[0].strip()
    if command.startswith("/choose "):
        return command[len("/choose ") :].strip()
    if command.startswith("/switch "):
        return f"switch {command[len('/switch ') :].strip()}"
    return None


def _showdown_id(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_path_label(config: ControlledFoulPlayConfig) -> str | None:
    """Return a path only when a trained checkpoint actually drove the game."""

    return str(config.checkpoint) if config.checkpoint is not None else None


def _checkpoint_sha256(config: ControlledFoulPlayConfig) -> str | None:
    checkpoint = config.checkpoint
    return _sha256_file(checkpoint) if checkpoint is not None and checkpoint.is_file() else None


def _audit_observation_schema_version(config: ControlledFoulPlayConfig) -> str | None:
    if config.audit_observation_schema is None:
        return None
    return observation_schema_version_from_choice(config.audit_observation_schema)


def _capture_driver_identity(config: ControlledFoulPlayConfig, *, checkpoint_sha256: str | None) -> str:
    """Stable capture provenance when no checkpoint weights exist."""

    return checkpoint_sha256 if checkpoint_sha256 is not None else f"audit-driver:{config.capture_driver}"


def _public_corpus_capture_config(config: ControlledFoulPlayConfig) -> dict[str, Any]:
    """Return only the stable public-corpus capture conditions.

    Seed bands are intentionally omitted so separate controlled runs can append
    to one checkpoint/config-homogeneous corpus without weakening provenance.
    """

    return {
        "capture_mode": f"controlled-foulplay/{config.capture_driver}",
        "capture_driver": config.capture_driver,
        "audit_observation_schema": _audit_observation_schema_version(config),
        "format_id": config.format_id,
        "policy_mode": config.policy_mode,
        "max_decision_rounds": config.max_decision_rounds,
        "foulplay_search_time_ms": config.search_time_ms,
        "belief_set_source_enabled": config.belief_set_source_enabled(),
        "opponent_legal_mask_mode": "hidden",
        "root_dirichlet_alpha": None,
        "root_noise_enabled": False,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a controlled BattleStream benchmark: PokeZero policy vs external foul-play.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Transformer checkpoint path.")
    parser.add_argument(
        "--value-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional compatible checkpoint used only for root-PUCT leaf values. "
            "Use a frozen calibrated copy while --checkpoint supplies raw policy priors."
        ),
    )
    parser.add_argument(
        "--showdown-root",
        type=Path,
        default=Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT", "")) if os.environ.get("POKEZERO_SHOWDOWN_ROOT") else None,
        help="Built Pokemon Showdown checkout root, or POKEZERO_SHOWDOWN_ROOT.",
    )
    parser.add_argument("--foulplay-root", type=Path, default=DEFAULT_FOULPLAY_ROOT, help="foul-play checkout path.")
    parser.add_argument("--foulplay-python", type=Path, default=None, help="Python executable for foul-play.")
    parser.add_argument("--games", type=int, default=1, help="Number of games.")
    parser.add_argument("--seed-start", type=int, default=1, help="First deterministic BattleStream seed.")
    parser.add_argument(
        "--foulplay-random-seed",
        type=int,
        default=None,
        help=(
            "Seed for foul-play's Python random/hash startup state. Defaults to --seed-start. "
            "This controls foul-play's random stream but does not make wall-clock MCTS fully deterministic."
        ),
    )
    parser.add_argument("--search-time-ms", type=int, default=1000, help="foul-play search time per move.")
    parser.add_argument("--max-decision-rounds", type=int, default=250, help="Decision-round cap.")
    parser.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    parser.add_argument(
        "--policy-mode", choices=("raw", "root-puct", "engine-mcts"), default="root-puct",
        help="raw = the checkpoint's own argmax; root-puct = the Python root search; "
             "engine-mcts = the native Rust search over sampled belief worlds.",
    )
    # engine-mcts axes. Every one changes search semantics or wall time, so the
    # config treats them as a frozen contract -- surfaced here rather than
    # defaulted silently, so a run's config_id is reconstructable from argv.
    parser.add_argument("--engine-model-path", type=Path, default=None,
                        help="TorchScript trace for the in-crate leaf evaluator.")
    parser.add_argument("--engine-tables-path", type=Path, default=None,
                        help="Encoder tables JSON matching the checkpoint's contract.")
    parser.add_argument("--engine-depth", type=int, default=4)
    parser.add_argument("--engine-sims", type=int, default=1024)
    parser.add_argument("--engine-batch", type=int, default=64)
    parser.add_argument("--engine-worlds", type=int, default=4)
    parser.add_argument("--engine-c-puct", type=float, default=1.4)
    parser.add_argument("--no-engine-model-priors", action="store_true",
                        help="Disable model priors in the native search (default: enabled).")
    parser.add_argument("--engine-opponent-priors", action="store_true",
                        help="Seed the OPPONENT seat's priors from the checkpoint's "
                             "opponent action head (default: uniform, as every recorded "
                             "result was produced).")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, mps.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Checkpoint policy softmax temperature.")
    parser.add_argument("--cpuct", type=float, default=1.25, help="Root PUCT exploration constant.")
    parser.add_argument(
        "--selection-mode",
        choices=("puct", "value", "visits"),
        default="visits",
        help=(
            "Root search candidate selection rule. Defaults to 'visits', which uses PUCT's "
            "exploration term for traversal but selects the most-visited root action. 'puct' "
            "selects by final Q+U score and should be treated as diagnostic."
        ),
    )
    parser.add_argument(
        "--root-prior-temperature",
        type=float,
        default=None,
        help=(
            "Temperature applied only to root-PUCT action priors. Defaults to --temperature, "
            "while opponent-action priors and fallback policy continue using --temperature."
        ),
    )
    parser.add_argument(
        "--minimum-value-improvement",
        type=float,
        default=None,
        help=(
            "Require the search-selected action to beat the prior-best action by this value margin; "
            "otherwise use the prior-best action."
        ),
    )
    parser.add_argument(
        "--minimum-override-prior-ratio",
        type=float,
        default=None,
        help=(
            "When search would override the checkpoint prior's greedy legal action, require the "
            "selected action prior to be at least this fraction of the prior-best action prior. "
            "A value of 1.0 only allows max-prior ties to override."
        ),
    )
    parser.add_argument(
        "--minimum-score-improvement",
        type=float,
        default=None,
        help=(
            "When search would override the checkpoint prior's greedy legal action, require the "
            "selected action's root-PUCT score to be at least this much higher than the prior-best "
            "action's score. Use 0.0 to reject lower-score overrides."
        ),
    )
    parser.add_argument(
        "--root-visit-budget",
        type=int,
        default=16,
        help=(
            "Root visits per opponent-action scenario; defaults to 16. "
            "With multiple scenarios, total decision visits scale by the searched scenario count."
        ),
    )
    parser.add_argument(
        "--root-extra-visits",
        type=int,
        default=None,
        help=(
            "Fixed visits added after the mandatory legal-action sweep. Mutually exclusive "
            "with adaptive root budgeting; use 0 for the sweep-only arm."
        ),
    )
    parser.add_argument(
        "--adaptive-root-contested-extra-visits",
        type=int,
        default=None,
        help="Extra post-sweep visits for contested decisions; enables adaptive root budgeting.",
    )
    parser.add_argument(
        "--adaptive-root-uncontested-extra-visits",
        type=int,
        default=0,
        help="Extra post-sweep visits for uncontested decisions when adaptive budgeting is enabled.",
    )
    parser.add_argument(
        "--adaptive-root-policy-entropy-threshold",
        type=float,
        default=None,
        help="Mark a decision contested when normalized legal-action policy entropy reaches this value.",
    )
    parser.add_argument(
        "--adaptive-root-value-margin-threshold",
        type=float,
        default=None,
        help="Mark a decision contested when the initial top-two leaf-value margin is at most this value.",
    )
    parser.add_argument(
        "--root-time-budget-ms",
        type=int,
        default=None,
        help=(
            "PokeZero-side wall-clock budget for extra post-sweep root visits. With multiple "
            "opponent-action scenarios, each scenario receives the remaining decision budget at "
            "the time it is searched. The mandatory initial legal-action sweep is always completed "
            "and can exceed the configured budget. Time-bounded searches clear the legacy "
            "--root-visit-budget cap."
        ),
    )
    parser.add_argument(
        "--root-opponent-action-scenarios",
        type=int,
        default=1,
        help="Number of checkpoint-prior opponent root-action scenarios to average.",
    )
    parser.add_argument(
        "--root-opponent-action-candidate-scenarios",
        type=int,
        default=ACTION_COUNT,
        help=(
            "Number of checkpoint-prior opponent root-action candidates to try while searching "
            "for replay-legal scenarios. Defaults to the full action space; when the opponent "
            "legal mask is hidden, exchangeable switch slots are collapsed into one summed switch "
            "candidate before this cap is applied. The search stops after "
            "--root-opponent-action-scenarios legal scenarios are accepted."
        ),
    )
    parser.add_argument(
        "--leaf-rollout-rounds",
        type=int,
        default=0,
        help="Decision rounds to continue each root branch before leaf value evaluation.",
    )
    parser.add_argument(
        "--leaf-rollout-sampling",
        action="store_true",
        help="Use sampled checkpoint policies, rather than greedy policies, inside leaf rollouts.",
    )
    parser.add_argument(
        "--belief-start-overrides",
        action="store_true",
        help=(
            "Sample public Gen 3 randbat belief into complete custom-game branch starts for "
            "root-PUCT replay search. This is hidden-info safe but experimental."
        ),
    )
    parser.add_argument(
        "--start-override-attempts",
        type=int,
        default=DEFAULT_START_OVERRIDE_ATTEMPTS,
        help=(
            "Replay-consistency attempts per opponent-action scenario when a start-override "
            "planner is enabled. Defaults to 10 to borrow the randbat determinization "
            "recipe's rejection-sampling budget; "
            "lower values are useful for fast smoke diagnostics."
        ),
    )
    parser.add_argument(
        "--belief-start-override-samples",
        type=int,
        default=1,
        help=(
            "Belief start-override samples to average per accepted opponent-action scenario. "
            "Requires --belief-start-overrides. Values above 1 split each opponent-action "
            "scenario across multiple sampled hidden worlds without increasing the accepted "
            "opponent-action cap, increasing search cost."
        ),
    )
    parser.add_argument(
        "--start-override-hp-fraction-tolerance",
        type=float,
        default=0.02,
        help=(
            "Allowed branch-point HP-fraction drift when validating sampled start overrides. "
            "Only self/opponent Pokemon HP-fraction numeric cells use this tolerance; request "
            "shape, legal mask, action candidates, categorical state, status, and all other "
            "numeric features remain exact."
        ),
    )
    parser.add_argument(
        "--opponent-legal-mask-mode",
        choices=("hidden", "privileged"),
        default="hidden",
        help=(
            "Whether root opponent-action planning withholds the opponent's private legal mask "
            "(hidden, default) or uses it as a privileged benchmark safety guard."
        ),
    )
    parser.add_argument(
        "--belief-set-source",
        choices=("env", "on", "off"),
        default="env",
        help=(
            "Candidate-set source for player-relative belief views: 'on'/'off' override the "
            "POKEZERO_BELIEF_SET_SOURCE env gate (default 'env'). Training runs with the gate "
            "enabled must benchmark with it enabled or the net sees ablated observations."
        ),
    )
    parser.add_argument(
        "--opponent-journal",
        choices=OPPONENT_JOURNAL_MODES,
        default="addressed",
        help=(
            "Record the opponent's submitted move per decision round, so a future "
            "replay driver can feed recorded moves back instead of re-running "
            "foul-play (whose search is not reproducible). No replayer exists yet. "
            "'addressed' (default) records only battles that filed a fallback "
            "address, as a prefix through the last such round; 'full' records every "
            "battle and is several times larger; 'off' records nothing. See the "
            "OPPONENT-MOVE JOURNAL block in foulplay_bridge.py for the size and "
            "coverage measurements behind the default."
        ),
    )
    parser.add_argument(
        "--no-refusal-records",
        action="store_true",
        help=(
            "Switch OFF the #1180 refusal recorder, which is on by default. On, every "
            "fallback decision the engine policy takes is written into --summary-out "
            "with the per-decision state that produced it: the world-failure classes "
            "that fired on THAT decision, the worlds spent, the engine's proposed "
            "choices, the request's legal set in the engine's vocabulary and their "
            "disagreement. Off, the summary keeps only the address, which for a "
            "bridge shard cannot be replayed. Measured cost at the era-64 cell-D "
            "fallback rate: ~1.3 us per decision (2.5e-7 of a 5.02 s decision "
            "boundary) and +5.1% summary bytes; see the REFUSAL RECORDER block in "
            "foulplay_bridge.py for both measurements and their parameters."
        ),
    )
    parser.add_argument(
        "--no-search-fallback",
        action="store_true",
        help="Raise on search failure instead of falling back to the raw checkpoint action.",
    )
    parser.add_argument("--node-binary", default="node", help="Node executable for BattleStream bridge.")
    parser.add_argument("--pokezero-username", default="PokeZeroBot")
    parser.add_argument("--foulplay-username", default="FoulPlayBot")
    parser.add_argument(
        "--pokezero-player",
        choices=("p1", "p2"),
        default="p1",
        help="Showdown seat controlled by PokeZero; use both seats for mirrored evaluation.",
    )
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional JSON result path.")
    parser.add_argument("--json", action="store_true", help="Print JSON result.")
    return parser


def build_comparison_arg_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    _remove_optional_argument(parser, "--policy-mode")
    parser.set_defaults(policy_mode="root-puct")
    parser.add_argument(
        "--comparison-mode",
        choices=tuple(sorted(_COMPARISON_MODES)),
        default="per-seed",
        help=(
            "Comparison execution order. 'per-seed' runs raw and root-PUCT for each seed before "
            "advancing, restarting foul-play with a matching per-seed startup seed and producing "
            "paired partial progress earlier. 'per-arm' preserves the older raw-all-then-root-PUCT "
            "order and is mainly useful when process startup overhead dominates."
        ),
    )
    parser.add_argument(
        "--opponent-crash-retries",
        type=int,
        default=1,
        help=(
            "Times to retry a seed's arm after foul-play exits before finishing the game "
            "(per-seed mode). A seed that still crashes is recorded as an opponent crash, "
            "excluded from paired stats, and the run continues. Use 0 to disable retries."
        ),
    )
    parser.add_argument(
        "--progress-interval-games",
        type=int,
        default=None,
        help=(
            "Emit paired-game liveness progress to stderr every N completed games and at completion. "
            "This does not change the persisted comparison result."
        ),
    )
    parser.description = (
        "Run paired controlled BattleStream benchmarks: raw checkpoint and root-PUCT "
        "against external foul-play over the same seed band."
    )
    parser.epilog = "The comparison runner always runs both raw and root-puct policy modes."
    return parser


def _remove_optional_argument(parser: argparse.ArgumentParser, option: str) -> None:
    for action in tuple(parser._actions):
        if option not in action.option_strings:
            continue
        parser._remove_action(action)
        for group in parser._action_groups:
            if action in group._group_actions:
                group._group_actions.remove(action)
        for option_string in action.option_strings:
            parser._option_string_actions.pop(option_string, None)
        return
    raise AssertionError(f"parser option not found: {option}")


def _config_from_args(
    args: argparse.Namespace,
    *,
    policy_mode: str | None = None,
) -> ControlledFoulPlayConfig:
    return ControlledFoulPlayConfig(
        checkpoint=args.checkpoint,
        showdown_root=args.showdown_root,
        value_checkpoint=args.value_checkpoint,
        foulplay_root=args.foulplay_root,
        foulplay_python=args.foulplay_python,
        games=args.games,
        seed_start=args.seed_start,
        foulplay_random_seed=args.foulplay_random_seed,
        search_time_ms=args.search_time_ms,
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
        policy_mode=policy_mode if policy_mode is not None else args.policy_mode,
        engine_model_path=getattr(args, "engine_model_path", None),
        engine_tables_path=getattr(args, "engine_tables_path", None),
        engine_depth=getattr(args, "engine_depth", 4),
        engine_sims=getattr(args, "engine_sims", 1024),
        engine_batch=getattr(args, "engine_batch", 64),
        engine_worlds=getattr(args, "engine_worlds", 4),
        engine_c_puct=getattr(args, "engine_c_puct", 1.4),
        engine_model_priors=not getattr(args, "no_engine_model_priors", False),
        engine_opponent_priors=getattr(args, "engine_opponent_priors", False),
        device=args.device,
        temperature=args.temperature,
        cpuct=args.cpuct,
        selection_mode=args.selection_mode,
        root_prior_temperature=args.root_prior_temperature,
        minimum_value_improvement=args.minimum_value_improvement,
        minimum_override_prior_ratio=args.minimum_override_prior_ratio,
        minimum_score_improvement=args.minimum_score_improvement,
        root_visit_budget=args.root_visit_budget,
        root_extra_visits=args.root_extra_visits,
        adaptive_root_contested_extra_visits=args.adaptive_root_contested_extra_visits,
        adaptive_root_uncontested_extra_visits=args.adaptive_root_uncontested_extra_visits,
        adaptive_root_policy_entropy_threshold=args.adaptive_root_policy_entropy_threshold,
        adaptive_root_value_margin_threshold=args.adaptive_root_value_margin_threshold,
        root_time_budget_ms=args.root_time_budget_ms,
        root_opponent_action_scenarios=args.root_opponent_action_scenarios,
        root_opponent_action_candidate_scenarios=args.root_opponent_action_candidate_scenarios,
        leaf_rollout_rounds=args.leaf_rollout_rounds,
        leaf_rollout_sampling=args.leaf_rollout_sampling,
        belief_start_overrides=args.belief_start_overrides,
        start_override_attempts=args.start_override_attempts,
        belief_start_override_samples=args.belief_start_override_samples,
        start_override_hp_fraction_tolerance=args.start_override_hp_fraction_tolerance,
        opponent_legal_mask_mode=args.opponent_legal_mask_mode,
        opponent_crash_retries=getattr(args, "opponent_crash_retries", 1),
        belief_set_source={"env": None, "on": True, "off": False}[getattr(args, "belief_set_source", "env")],
        allow_search_fallback=not args.no_search_fallback,
        node_binary=args.node_binary,
        pokezero_username=args.pokezero_username,
        foulplay_username=args.foulplay_username,
        pokezero_player=args.pokezero_player,
        capture_driver=getattr(args, "capture_driver", "checkpoint"),
        audit_observation_schema=getattr(args, "observation_schema", None),
        opponent_journal=getattr(args, "opponent_journal", "addressed"),
        record_refusals=not getattr(args, "no_refusal_records", False),
    )


def _controlled_foulplay_comparison_progress_callback(
    interval_games: int,
) -> Callable[..., None]:
    """Emit paired FoulPlay progress without changing comparison artifacts."""

    if interval_games <= 0:
        raise ValueError("comparison progress interval must be positive.")
    last_reported: tuple[int, int] | None = None

    def emit(result: ControlledFoulPlayComparisonResult, *, force: bool = False) -> None:
        nonlocal last_reported
        raw_completed = result.raw.completed_games if result.raw is not None else 0
        root_puct_completed = result.root_puct.completed_games if result.root_puct is not None else 0
        paired_completed = min(raw_completed, root_puct_completed)
        opponent_crash_count = len(result.opponent_crashes)
        state = (paired_completed, opponent_crash_count)
        if state == last_reported:
            return
        if not force and (paired_completed == 0 or paired_completed % interval_games != 0):
            return
        payload = {
            "comparison_mode": result.comparison_mode,
            "games_completed": paired_completed,
            "games_total": result.config.games,
            "opponent_crash_count": opponent_crash_count,
        }
        print(
            f"controlled_foulplay_comparison_progress: {json.dumps(payload, sort_keys=True)}",
            file=sys.stderr,
            flush=True,
        )
        last_reported = state

    return emit


async def async_main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.showdown_root is None:
        parser.error("--showdown-root is required, or set POKEZERO_SHOWDOWN_ROOT.")
    config = _config_from_args(args)

    def write_progress(result: ControlledFoulPlayBenchmarkResult) -> None:
        if args.summary_out is not None:
            _write_json(args.summary_out, result.to_dict())

    result = await run_controlled_foulplay_benchmark(
        config,
        progress_callback=write_progress if args.summary_out is not None else None,
    )
    payload = result.to_dict()
    if args.summary_out is not None:
        _write_json(args.summary_out, payload)
        print(f"controlled_foulplay_summary: {args.summary_out}", file=sys.stderr)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"RESULT: {result.policy_id} won {result.wins}/{result.completed_games} "
            f"vs foul-play ({result.win_rate:.1%})"
        )
        root = payload["root_puct"]
        if isinstance(root, Mapping) and root.get("searches"):
            print(
                "root-puct: "
                f"searches={root.get('searches')} fallbacks={root.get('fallbacks')} "
                f"avg_elapsed={root.get('average_elapsed_seconds', 'n/a')}"
            )
    return 0


async def async_comparison_main(argv: Sequence[str] | None = None) -> int:
    parser = build_comparison_arg_parser()
    args = parser.parse_args(argv)
    if args.showdown_root is None:
        parser.error("--showdown-root is required, or set POKEZERO_SHOWDOWN_ROOT.")
    if args.progress_interval_games is not None and args.progress_interval_games <= 0:
        parser.error("--progress-interval-games must be positive when set.")
    config = _config_from_args(args, policy_mode="root-puct")
    emit_progress = (
        _controlled_foulplay_comparison_progress_callback(args.progress_interval_games)
        if args.progress_interval_games is not None
        else None
    )

    def write_progress(result: ControlledFoulPlayComparisonResult) -> None:
        if args.summary_out is not None:
            _write_json(args.summary_out, result.to_dict())
        if emit_progress is not None:
            emit_progress(result)

    result = await run_controlled_foulplay_comparison(
        config,
        comparison_mode=args.comparison_mode,
        progress_callback=write_progress if args.summary_out is not None or emit_progress is not None else None,
    )
    if emit_progress is not None:
        emit_progress(result, force=True)
    payload = result.to_dict()
    if args.summary_out is not None:
        _write_json(args.summary_out, payload)
        print(f"controlled_foulplay_comparison_summary: {args.summary_out}", file=sys.stderr)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        comparison = payload["comparison"]
        paired = comparison["paired_by_seed"] if isinstance(comparison, Mapping) else {}
        raw = paired.get("raw", {}) if isinstance(paired, Mapping) else {}
        root_puct = paired.get("root_puct", {}) if isinstance(paired, Mapping) else {}
        sample = comparison["sample_size"] if isinstance(comparison, Mapping) else {}
        result_label = (
            "DIAGNOSTIC RESULT"
            if isinstance(sample, Mapping) and sample.get("status") == "diagnostic_only"
            else "RESULT"
        )
        delta = paired.get("root_puct_minus_raw_win_rate") if isinstance(paired, Mapping) else None
        delta_text = "n/a" if delta is None else f"{float(delta):.1%}"
        print(
            f"{result_label}: root-PUCT "
            f"{int(root_puct.get('wins', 0))}/{int(root_puct.get('games', 0))} "
            "vs raw "
            f"{int(raw.get('wins', 0))}/{int(raw.get('games', 0))} "
            f"on paired foul-play seeds ({args.comparison_mode}) "
            f"(descriptive_delta={delta_text})"
        )
        if isinstance(sample, Mapping) and sample.get("status") == "diagnostic_only":
            print(
                "sample-size: diagnostic_only "
                f"({sample.get('paired_games')}/{sample.get('minimum_strength_games')} paired games)"
            )
        crashed = comparison.get("opponent_crashed_seeds") if isinstance(comparison, Mapping) else None
        if isinstance(crashed, Mapping) and crashed.get("count"):
            print(
                f"opponent-crashes: {crashed.get('count')} seed(s) excluded because foul-play "
                f"exited early: {crashed.get('seeds')}"
            )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


def comparison_main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(async_comparison_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
