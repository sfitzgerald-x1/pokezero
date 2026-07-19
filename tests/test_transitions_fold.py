"""Differential prefix-closure proof for the incremental fold state (track B).

The engine-swap plan's schema-v2 prerequisite (docs/test_time_search_plan_v3.md,
"Schema v2"; component verdicts in docs/fold_closure_probe.md): the production
transition/tendency fold must be (state + slice)-closed. The proof here is the
differential itself — real LocalShowdownEnv games (random gen3randombattle seeds
plus every curated scenario game, which deterministically exercise the risky
components: Pursuit interception, Baton Pass boundaries, RestTalk collapse,
Explosion double-faints, Truant/recharge, Transform, screens, sand + Shedinja),
checked at EVERY decision boundary, for BOTH perspectives, on every
observation-visible product:

- the merged transition-token tail (what the v2.2 encode reads, ``[-budget:]``),
- the per-action token tail (the v2/v2.1 encode surface + annotation substrate),
- both stream totals (the attention-mask fill),
- the full TendencyStats dataclass,
- and, in the annotated battery, the Tier-2/investment annotation join plus the
  full-stream pinned surfaces the v2.1/v2.2 encode derives.

Batch arm: ``extract_transition_products`` over the captured replay snapshot —
exactly what production runs per observe. Incremental arm: ``FoldState.advance``
over the inter-boundary raw-line slices. Serialization round-trips happen
MID-GAME inside the differential (serialize -> canonical JSON -> resume ->
continue advancing), so payload fidelity is proven by the same equality chain.

Synthetic (no-Showdown) batteries cover line-by-line advancing against every
batch prefix (mid-chunk boundary cuts included), the ``|t:|`` filter, and
payload determinism. Skip gates match the other live-sim suites.
"""

import json
import os
import random
import shutil
import time
import unittest
from pathlib import Path

from pokezero.showdown import _normalize_identifier, parse_showdown_replay
from pokezero.transitions import TOKEN_KIND_MOVE
from pokezero.transitions_fold import (
    DEFAULT_ACTION_TAIL_LIMIT,
    DEFAULT_MERGED_TAIL_LIMIT,
    FoldState,
)
from pokezero.turn_merged import extract_transition_products


def _integration_root() -> Path | None:
    from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT

    root = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)
    if not (root / "dist" / "sim" / "index.js").exists():
        return None
    if shutil.which("node") is None:
        return None
    return root


def _batch_products(replay, slot):
    return extract_transition_products(replay, perspective_slot=slot)


class _DifferentialHarness:
    """Advances per-perspective FoldStates alongside a live env and asserts equality."""

    def __init__(
        self,
        test: unittest.TestCase,
        *,
        serialize_at_boundary: int | None = None,
        compare_pure_batch: bool = True,
    ):
        self.test = test
        self.states = {
            "p1": FoldState.initial(perspective_slot="p1"),
            "p2": FoldState.initial(perspective_slot="p2"),
        }
        self.prev_lines: tuple[str, ...] = ()
        self.boundaries = 0
        self.checks = 0
        self.serialize_at_boundary = serialize_at_boundary
        # The annotated battery layers tracker overlays onto the fold states, so its
        # products are compared against the ANNOTATED env state (in the caller's
        # on_boundary hook), not against the pure batch fold.
        self.compare_pure_batch = compare_pure_batch

    def check_boundary(self, replay) -> None:
        raw = tuple(event.raw_line for event in replay.public_events)
        self.test.assertEqual(
            raw[: len(self.prev_lines)],
            self.prev_lines,
            "public event stream is not append-only across boundaries",
        )
        slice_ = raw[len(self.prev_lines) :]
        self.prev_lines = raw
        self.boundaries += 1
        for slot in ("p1", "p2"):
            self.states[slot], products = self.states[slot].advance(slice_)
            if not self.compare_pure_batch:
                continue
            batch_tokens, batch_merged, batch_tendencies = _batch_products(replay, slot)
            self.test.assertEqual(products.transition_token_total, len(batch_tokens))
            self.test.assertEqual(
                products.transition_tokens, batch_tokens[-DEFAULT_ACTION_TAIL_LIMIT:]
            )
            self.test.assertEqual(products.turn_merged_total, len(batch_merged))
            self.test.assertEqual(
                products.turn_merged_tokens, batch_merged[-DEFAULT_MERGED_TAIL_LIMIT:]
            )
            self.test.assertEqual(products.tendency_stats, batch_tendencies)
            self.checks += 1
        if self.serialize_at_boundary is not None and self.boundaries == self.serialize_at_boundary:
            self.round_trip()

    def round_trip(self) -> None:
        """Mid-game serialize -> canonical JSON -> resume; later boundaries prove convergence."""
        for slot in ("p1", "p2"):
            payload = self.states[slot].to_payload()
            canonical = json.dumps(payload, sort_keys=True)
            resumed = FoldState.from_payload(json.loads(canonical))
            self.test.assertEqual(
                json.dumps(resumed.to_payload(), sort_keys=True),
                canonical,
                "fold-state payload round-trip is not deterministic",
            )
            self.states[slot] = resumed


def _drive_random_game(test, env, seed, harness, *, max_rounds=400, on_boundary=None):
    rng = random.Random(seed)
    env.reset(seed=seed)
    rounds = 0
    while rounds < max_rounds and env.terminal() is None:
        requested = env.requested_players()
        if not requested:
            break
        replay = env.public_materialization_state(requested[0]).replay
        harness.check_boundary(replay)
        actions = {}
        for player in requested:
            state = env._state_for_player(player)
            if on_boundary is not None:
                on_boundary(player, state)
            legal = [index for index, ok in enumerate(state.legal_action_mask) if ok]
            actions[player] = rng.choice(legal)
        env.step(actions)
        rounds += 1


@unittest.skipUnless(_integration_root() is not None, "requires built Showdown checkout and node")
class FoldClosureRandomGamesTest(unittest.TestCase):
    """The core differential: 10 random gen3randombattle games, every boundary."""

    GAMES = 10

    def test_prefix_closure_over_random_games(self) -> None:
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv

        env = LocalShowdownEnv(
            LocalShowdownConfig(showdown_root=_integration_root(), set_belief_source=False)
        )
        total_boundaries = 0
        total_checks = 0
        try:
            for seed in range(9001, 9001 + self.GAMES):
                harness = _DifferentialHarness(self, serialize_at_boundary=7)
                _drive_random_game(self, env, seed, harness)
                self.assertGreater(harness.boundaries, 5, f"seed {seed}: game too short to prove anything")
                total_boundaries += harness.boundaries
                total_checks += harness.checks
        finally:
            env.close()
        print(
            f"\n[fold-closure] random games: {self.GAMES} games, "
            f"{total_boundaries} boundaries, {total_checks} differential checks (all equal)"
        )


@unittest.skipUnless(_integration_root() is not None, "requires built Showdown checkout and node")
class FoldClosureScenarioGamesTest(unittest.TestCase):
    """The curated edge-case scenarios (Pursuit / Baton Pass / RestTalk / Explosion /
    Truant / Transform / recharge / screens / sand+Shedinja / toxic stall)."""

    def test_prefix_closure_over_scenario_games(self) -> None:
        from pokezero.golden_corpus_scenarios import (
            ScriptedPreferencePolicy,
            _scenario_override,
            scenario_specs,
        )
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv

        env = LocalShowdownEnv(
            LocalShowdownConfig(showdown_root=_integration_root(), set_belief_source=False)
        )
        total_boundaries = 0
        total_checks = 0
        names = []
        try:
            for spec in scenario_specs():
                override = _scenario_override(spec)
                env.reset_with_start_override(seed=spec.seed, start_override=override)
                policies = {
                    "p1": ScriptedPreferencePolicy(spec.p1_prefs),
                    "p2": ScriptedPreferencePolicy(spec.p2_prefs),
                }
                harness = _DifferentialHarness(self, serialize_at_boundary=3)
                rng = random.Random(spec.seed)
                rounds = 0
                while rounds < spec.max_decision_rounds and env.terminal() is None:
                    requested = env.requested_players()
                    if not requested:
                        break
                    replay = env.public_materialization_state(requested[0]).replay
                    harness.check_boundary(replay)
                    actions = {}
                    for player in requested:
                        observation = env.observe(player)
                        actions[player] = policies[player].select_action(
                            observation, rng=rng
                        ).action_index
                    env.step(actions)
                    rounds += 1
                self.assertGreater(harness.boundaries, 0, f"scenario {spec.name} produced no boundaries")
                names.append(f"{spec.name}:{harness.boundaries}")
                total_boundaries += harness.boundaries
                total_checks += harness.checks
        finally:
            env.close()
        print(
            f"\n[fold-closure] scenarios: {len(names)} games, {total_boundaries} boundaries, "
            f"{total_checks} differential checks (all equal) [{', '.join(names)}]"
        )


@unittest.skipUnless(_integration_root() is not None, "requires built Showdown checkout and node")
class FoldAnnotatedSurfaceTest(unittest.TestCase):
    """The Tier-2 annotation join + pinned surfaces, against the live trackers.

    The env runs the production annotation stack (Tier2LiveTracker +
    InvestmentLiveTracker + annotate_turn_merged_tokens); the incremental arm
    receives the tracker conclusions as a per-index overlay (exactly what the
    trackers hold) and must reproduce the annotated per-action tail, the annotated
    merged tail, and the full-stream pinned reductions the encoder derives
    (showdown.py tier2_cb_pinned_species / tier2_investment_pinned).
    """

    GAMES = 3

    def test_annotated_products_match(self) -> None:
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
        from pokezero.observation import ObservationFeatureMasks

        env = LocalShowdownEnv(
            LocalShowdownConfig(
                showdown_root=_integration_root(),
                set_belief_source=True,
                feature_masks=ObservationFeatureMasks(tier2_investment=True),
            )
        )
        try:
            if not env.tier2_residuals_active():
                self.skipTest("belief set source unavailable; Tier-2 trackers inactive")
            total_boundaries = 0
            annotated_indices = 0
            for seed in range(17001, 17001 + self.GAMES):
                harness = _DifferentialHarness(
                    self, serialize_at_boundary=9, compare_pure_batch=False
                )

                def on_boundary(player, state):
                    nonlocal annotated_indices
                    overlay = {
                        index: (
                            token.residual,
                            token.residual_valid,
                            token.cb_bit,
                            token.investment,
                        )
                        for index, token in enumerate(state.transition_tokens)
                        if token.residual is not None
                        or token.residual_valid
                        or token.cb_bit
                        or token.investment
                    }
                    annotated_indices = max(annotated_indices, len(overlay))
                    harness.states[player] = harness.states[player].apply_annotations(overlay)
                    products = harness.states[player].products()
                    self.assertEqual(
                        products.transition_tokens,
                        state.transition_tokens[-DEFAULT_ACTION_TAIL_LIMIT:],
                    )
                    self.assertEqual(
                        products.turn_merged_tokens,
                        state.turn_merged_tokens[-DEFAULT_MERGED_TAIL_LIMIT:],
                    )
                    self.assertEqual(products.tendency_stats, state.tendency_stats)
                    # The encoder's full-stream pinned reductions (showdown.py).
                    opponent = state.perspective.opponent_showdown_slot
                    expected_cb = frozenset(
                        _normalize_identifier(token.actor_species)
                        for token in state.transition_tokens
                        if token.cb_bit
                        and token.kind == TOKEN_KIND_MOVE
                        and token.actor_slot == opponent
                    )
                    self.assertEqual(products.cb_pinned_species, expected_cb)
                    expected_investment = {}
                    for token in state.transition_tokens:
                        if (
                            token.investment
                            and token.kind == TOKEN_KIND_MOVE
                            and token.actor_slot == state.perspective.showdown_slot
                            and token.defender_species
                        ):
                            expected_investment[
                                _normalize_identifier(token.defender_species)
                            ] = token.investment
                    self.assertEqual(dict(products.investment_pinned), expected_investment)

                _drive_random_game(self, env, seed, harness, on_boundary=on_boundary)
                total_boundaries += harness.boundaries
        finally:
            env.close()
        print(
            f"\n[fold-closure] annotated: {self.GAMES} games, {total_boundaries} boundaries, "
            f"max overlay size {annotated_indices} (all annotated surfaces equal)"
        )


# ---------------------------------------------------------------------------
# Synthetic batteries (no Showdown needed).
# ---------------------------------------------------------------------------

_SYNTHETIC_LINES = (
    "|player|p1|Alice|1",
    "|player|p2|Bob|2",
    "|t:|1700000001",
    "|switch|p1a: Tyranitar|Tyranitar, L74, M|100/100",
    "|switch|p2a: Starmie|Starmie, L77|100/100",
    "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar",
    "|turn|1",
    "|move|p2a: Starmie|Hydro Pump|p1a: Tyranitar",
    "|-supereffective|p1a: Tyranitar",
    "|-damage|p1a: Tyranitar|20/100",
    "|move|p1a: Tyranitar|Crunch|p2a: Starmie",
    "|-supereffective|p2a: Starmie",
    "|-damage|p2a: Starmie|45/100",
    "|",
    "|-weather|Sandstorm|[upkeep]",
    "|-damage|p2a: Starmie|39/100|[from] Sandstorm",
    "|upkeep",
    "|turn|2",
    "|t:|1700000002",
    "|-activate|p2a: Starmie|move: Pursuit",
    "|move|p1a: Tyranitar|Pursuit|p2a: Starmie",
    "|-damage|p2a: Starmie|11/100",
    "|switch|p2a: Skarmory|Skarmory, L79|100/100",
    "|",
    "|-weather|Sandstorm|[upkeep]",
    "|upkeep",
    "|turn|3",
    "|cant|p1a: Tyranitar|par",
    "|move|p2a: Skarmory|Spikes|p1a: Tyranitar",
    "|-sidestart|p1: Alice|Spikes",
    "|",
    "|-weather|Sandstorm|[upkeep]",
    "|upkeep",
    "|turn|4",
    "|move|p2a: Skarmory|Drill Peck|p1a: Tyranitar",
    "|-resisted|p1a: Tyranitar",
    "|-damage|p1a: Tyranitar|12/100",
    "|move|p1a: Tyranitar|Rock Slide|p2a: Skarmory",
    "|-damage|p2a: Skarmory|55/100",
    "|",
    "|win|Bob",
)


class FoldSyntheticClosureTest(unittest.TestCase):
    """Line-by-line advance vs batch fold on EVERY prefix — mid-chunk cuts included."""

    def test_every_prefix_matches_batch(self) -> None:
        for slot in ("p1", "p2"):
            state = FoldState.initial(perspective_slot=slot)
            for prefix_len in range(len(_SYNTHETIC_LINES) + 1):
                if prefix_len:
                    state, products = state.advance(_SYNTHETIC_LINES[prefix_len - 1 : prefix_len])
                else:
                    products = state.products()
                replay = parse_showdown_replay(
                    list(_SYNTHETIC_LINES[:prefix_len]), battle_id="fold-synthetic"
                )
                batch_tokens, batch_merged, batch_tendencies = _batch_products(replay, slot)
                self.assertEqual(products.transition_token_total, len(batch_tokens))
                self.assertEqual(products.transition_tokens, batch_tokens)
                self.assertEqual(products.turn_merged_total, len(batch_merged))
                self.assertEqual(products.turn_merged_tokens, batch_merged)
                self.assertEqual(products.tendency_stats, batch_tendencies)

    def test_pursuit_intercept_flagged_incrementally(self) -> None:
        state = FoldState.initial(perspective_slot="p1")
        state, products = state.advance(_SYNTHETIC_LINES)
        pursuit = [token for token in products.transition_tokens if token.action == "pursuit"]
        self.assertEqual(len(pursuit), 1)
        self.assertTrue(pursuit[0].pursuit_intercept)
        # And the tendency counter saw it (the opponent's doubled Pursuit, p2 view).
        p2 = FoldState.initial(perspective_slot="p2")
        _, p2_products = p2.advance(_SYNTHETIC_LINES)
        self.assertEqual(p2_products.tendency_stats.pursuit_intercept_predict_count, 1)

    def test_wall_clock_lines_are_filtered(self) -> None:
        without_t = tuple(line for line in _SYNTHETIC_LINES if not line.startswith("|t:|"))
        state_a, products_a = FoldState.initial(perspective_slot="p1").advance(_SYNTHETIC_LINES)
        state_b, products_b = FoldState.initial(perspective_slot="p1").advance(without_t)
        self.assertEqual(products_a, products_b)
        self.assertEqual(
            json.dumps(state_a.to_payload(), sort_keys=True),
            json.dumps(state_b.to_payload(), sort_keys=True),
        )

    def test_advance_is_pure(self) -> None:
        state = FoldState.initial(perspective_slot="p1")
        before = json.dumps(state.to_payload(), sort_keys=True)
        state.advance(_SYNTHETIC_LINES)
        self.assertEqual(json.dumps(state.to_payload(), sort_keys=True), before)


class FoldSerializationTest(unittest.TestCase):
    def test_round_trip_mid_game_and_resume(self) -> None:
        cut = 21  # mid-turn-2, open window in flight
        state = FoldState.initial(perspective_slot="p1")
        state, _ = state.advance(_SYNTHETIC_LINES[:cut])
        payload = state.to_payload()
        canonical = json.dumps(payload, sort_keys=True)
        resumed = FoldState.from_payload(json.loads(canonical))
        self.assertEqual(json.dumps(resumed.to_payload(), sort_keys=True), canonical)
        # Resume must converge identically with a never-serialized twin.
        twin, twin_products = state.advance(_SYNTHETIC_LINES[cut:])
        resumed, resumed_products = resumed.advance(_SYNTHETIC_LINES[cut:])
        self.assertEqual(resumed_products, twin_products)
        self.assertEqual(
            json.dumps(resumed.to_payload(), sort_keys=True),
            json.dumps(twin.to_payload(), sort_keys=True),
        )

    def test_payload_is_json_safe(self) -> None:
        state, _ = FoldState.initial(perspective_slot="p2").advance(_SYNTHETIC_LINES)
        encoded = json.dumps(state.to_payload(), sort_keys=True)
        self.assertIsInstance(encoded, str)
        decoded = FoldState.from_payload(json.loads(encoded))
        self.assertEqual(decoded.products(), state.products())

    def test_annotation_overlay_round_trips(self) -> None:
        state, products = FoldState.initial(perspective_slot="p1").advance(_SYNTHETIC_LINES)
        # Annotate the (only) opponent Drill Peck strike index with a residual + CB bit.
        target = next(
            index
            for index, token in enumerate(products.transition_tokens)
            if token.action == "drillpeck"
        )
        state = state.apply_annotations({target: (-0.05, True, True, 0.0)})
        annotated = state.products()
        token = annotated.transition_tokens[target]
        self.assertEqual(token.residual, -0.05)
        self.assertTrue(token.residual_valid)
        self.assertTrue(token.cb_bit)
        self.assertEqual(annotated.cb_pinned_species, frozenset({"skarmory"}))
        merged_with_bit = [
            merged
            for merged in annotated.turn_merged_tokens
            for sub in (merged.first, merged.second)
            if sub.cb_bit
        ]
        self.assertEqual(len(merged_with_bit), 1)
        canonical = json.dumps(state.to_payload(), sort_keys=True)
        resumed = FoldState.from_payload(json.loads(canonical))
        self.assertEqual(resumed.products(), annotated)
        # Re-application of the same values is idempotent; changed values raise.
        resumed.apply_annotations({target: (-0.05, True, True, 0.0)})
        with self.assertRaises(ValueError):
            resumed.apply_annotations({target: (0.10, True, True, 0.0)})


@unittest.skipUnless(
    _integration_root() is not None and os.environ.get("POKEZERO_FOLD_PERF"),
    "perf note: set POKEZERO_FOLD_PERF=1 (requires built Showdown checkout and node)",
)
class FoldPerfNote(unittest.TestCase):
    """Stretch measurement: batch-per-observe vs incremental advance over one game."""

    def test_measure(self) -> None:
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv

        env = LocalShowdownEnv(
            LocalShowdownConfig(showdown_root=_integration_root(), set_belief_source=False)
        )
        prefixes = []
        try:
            rng = random.Random(9001)
            env.reset(seed=9001)
            rounds = 0
            while rounds < 400 and env.terminal() is None:
                requested = env.requested_players()
                if not requested:
                    break
                replay = env.public_materialization_state(requested[0]).replay
                prefixes.append(replay)
                actions = {}
                for player in requested:
                    state = env._state_for_player(player)
                    legal = [index for index, ok in enumerate(state.legal_action_mask) if ok]
                    actions[player] = rng.choice(legal)
                env.step(actions)
                rounds += 1
        finally:
            env.close()

        batch_start = time.perf_counter()
        for replay in prefixes:
            _batch_products(replay, "p1")
        batch_seconds = time.perf_counter() - batch_start

        raw_prefixes = [tuple(e.raw_line for e in replay.public_events) for replay in prefixes]
        pure_start = time.perf_counter()
        state = FoldState.initial(perspective_slot="p1")
        prev = 0
        for raw in raw_prefixes:
            state, _ = state.advance(raw[prev:])
            prev = len(raw)
        pure_seconds = time.perf_counter() - pure_start

        inplace_start = time.perf_counter()
        state = FoldState.initial(perspective_slot="p1")
        prev = 0
        for raw in raw_prefixes:
            state.advance_in_place(raw[prev:])
            state.products()
            prev = len(raw)
        inplace_seconds = time.perf_counter() - inplace_start

        payload_bytes = len(json.dumps(state.to_payload(), sort_keys=True).encode())
        print(
            f"\n[fold-perf] boundaries={len(prefixes)} lines={prev} "
            f"batch_per_observe={batch_seconds * 1e3:.1f}ms "
            f"incremental_pure={pure_seconds * 1e3:.1f}ms "
            f"incremental_in_place={inplace_seconds * 1e3:.1f}ms "
            f"final_payload={payload_bytes}B"
        )


if __name__ == "__main__":
    unittest.main()
