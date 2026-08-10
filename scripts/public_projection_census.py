#!/usr/bin/env python3
"""Direction-2 census: every searched world's public projection vs the observed log.

Fallback-burndown plan 4, sequencing item 4. Direction 1
(``scripts/truth_differential_census.py``) asks whether the TRUE world
constructs and kills guards that refuse too much. This asks whether the worlds
search actually used are worlds the opponent could still believe in, and kills
relaxations that answer wrongly. Per PLAN section 3 a guard change is correct
iff BOTH directions hold, so this runner deliberately consumes the SAME census
plan file, the same shard split and the same identity-witness apparatus as
direction 1 -- it imports them rather than restating them.

Two arms, two different claims, reported separately and never summed:

``state``   Every world the sampler produced, at every decision, projected back
            into public protocol facts and compared with the observed record.
            Costs nothing beyond the driver's own search: the worlds arrive
            through ``EngineMctsPolicy``'s ``world_observer`` hook.

``render``  The renderer's own projection. Takes the joint action that was
            ACTUALLY played, renders the branch set from the TRUE world through
            ``pokezero_search.branch_events``, and requires the transition
            Showdown took to lie in the rendered support. This is the only arm
            that can see a renderer-side relaxation (#1211's shape), and it is
            SAMPLED: ``--render-every N`` evaluates one boundary in N, so its
            coverage is a floor and is published as one.

Build requirement, non-negotiable and inherited from direction 1: the crate must
be built ``--features model``. Without it the abort gate at
``rust/pokezero-search/src/tree.rs`` is ``allow(dead_code)``, and while direction
2 does not read the abort channel directly, a census whose two directions ran on
different builds is not one census. torch 2.11.0 or 2.12.1; never 2.12.0 (removed
``at::Tensor::align_as``) and never 2.13.0.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Direction 1's runner is the source of the plan, the shard split, the
# neutral-cwd witness child and the model-feature gate. Imported, not copied:
# two census runners that disagree about which games are in the block are two
# blocks, and PLAN section 3's "a guard change is correct iff BOTH directions
# hold" is only meaningful on one.
from truth_differential_census import (  # noqa: E402
    _default_showdown_root,
    _shard_games,
    require_model_feature,
)


def _neutral_cwd_witness(env: Mapping[str, str]) -> dict[str, Any]:
    """Re-resolve THIS module's identity witness in a child spawned from `/`.

    Direction 1's `_neutral_cwd_witness` runs `python -m pokezero.truth_differential`,
    which witnesses ITS module and knows nothing about this one. Borrowing it made
    every shard report four permanent "mismatches" -- `public_projection_file`,
    `public_projection_present`, `public_projection_axis_count` and the
    `source_sha256` map -- against a child that had simply never been asked. A
    witness that always reports a mismatch trains its reader to ignore it, which
    is worse than not having one. Same shape, same neutral cwd, same environment;
    only the entrypoint differs.
    """

    import subprocess  # noqa: PLC0415

    try:
        completed = subprocess.run(
            [sys.executable, "-B", "-m", "pokezero.public_projection"],
            cwd="/",
            env=dict(env),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"}
    if completed.returncode != 0:
        return {"error": f"exit {completed.returncode}", "stderr": completed.stderr[-2000:]}
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        return {"error": f"unparseable witness: {error}", "stdout": completed.stdout[-2000:]}

OBSERVATION_FORMAT_ID = "gen3randombattle"


# --- forcings ------------------------------------------------------------------
#
# An instrument that cannot report failure will report success. Direction 1
# carries `--force construct|abort|unmapped|unmapped-persistent`; direction 2's
# equivalent corrupts the WORLD after construction and before projection, one
# axis at a time, so that every forcing must produce a counted, ATTRIBUTED
# mismatch on exactly the axis it targets and silence on the others. A forcing
# that lights up three axes is as useless as one that lights up none: it would
# prove the comparator reacts, not that it discriminates.


def _spec_sides(world: Any) -> tuple[str, str]:
    """(engine side attribute for p1, for p2)."""

    return world.slot_sides["p1"], world.slot_sides["p2"]


def _replace_side(world: Any, side_key: str, **changes: Any) -> Any:
    """Return the world's BattleSpec with one side replaced."""

    import dataclasses as dc

    side = getattr(world.spec, side_key)
    return dc.replace(world.spec, **{side_key: dc.replace(side, **changes)})


def _replace_active(world: Any, side_key: str, **changes: Any) -> Any:
    import dataclasses as dc

    side = getattr(world.spec, side_key)
    party = list(side.pokemon)
    index = int(side.active_index)
    party[index] = dc.replace(party[index], **changes)
    return _replace_side(world, side_key, pokemon=tuple(party))


def _forcing_state_hp(world: Any) -> tuple[str, Any]:
    side_key = world.slot_sides["p1"]
    active = getattr(world.spec, side_key).pokemon[
        int(getattr(world.spec, side_key).active_index)
    ]
    hp = int(active.hp)
    return "state-hp", _replace_active(
        world, side_key, hp=hp - 1 if hp > 1 else hp + 1
    )


def _forcing_state_pp(world: Any) -> tuple[str, Any]:
    import dataclasses as dc

    side_key = world.slot_sides["p1"]
    side = getattr(world.spec, side_key)
    active = side.pokemon[int(side.active_index)]
    moves = list(active.moves)
    if not moves:
        return "state-pp:noop", world.spec
    moves[0] = dc.replace(moves[0], pp=max(0, int(moves[0].pp) - 1))
    return "state-pp", _replace_active(world, side_key, moves=tuple(moves))


def _forcing_state_disabled(world: Any) -> tuple[str, Any]:
    import dataclasses as dc

    side_key = world.slot_sides["p1"]
    side = getattr(world.spec, side_key)
    active = side.pokemon[int(side.active_index)]
    moves = list(active.moves)
    if not moves:
        return "state-disabled:noop", world.spec
    moves[0] = dc.replace(moves[0], disabled=not bool(moves[0].disabled))
    return "state-disabled", _replace_active(world, side_key, moves=tuple(moves))


def _forcing_state_move(world: Any) -> tuple[str, Any]:
    """Swap a move id -- fires `self_move_set` on the acting seat and
    `opponent_revealed_moves` on the other whenever that move was revealed."""

    import dataclasses as dc

    spec = world.spec
    for side_key in ("side_one", "side_two"):
        side = getattr(spec, side_key)
        active = side.pokemon[int(side.active_index)]
        moves = list(active.moves)
        if not moves:
            continue
        replacement = "splash" if str(moves[0].id) != "splash" else "tackle"
        moves[0] = dc.replace(moves[0], id=replacement)
        party = list(side.pokemon)
        party[int(side.active_index)] = dc.replace(active, moves=tuple(moves))
        spec = dc.replace(spec, **{side_key: dc.replace(side, pokemon=tuple(party))})
    return "state-move", spec


def _forcing_state_toxic(world: Any) -> tuple[str, Any]:
    import dataclasses as dc

    spec = world.spec
    for side_key in ("side_one", "side_two"):
        side = getattr(spec, side_key)
        conditions = dict(side.side_conditions)
        conditions["toxic_count"] = int(conditions.get("toxic_count", 0)) + 1
        spec = dc.replace(spec, **{side_key: dc.replace(side, side_conditions=conditions)})
    return "state-toxic", spec


def _forcing_state_boost(world: Any) -> tuple[str, Any]:
    side_key = world.slot_sides["p1"]
    side = getattr(world.spec, side_key)
    boosts = dict(side.boosts)
    boosts["attack"] = int(boosts.get("attack", 0)) + 1
    return "state-boost", _replace_side(world, side_key, boosts=boosts)


def _forcing_state_item(world: Any) -> tuple[str, Any]:
    import dataclasses as dc

    spec = world.spec
    for side_key in ("side_one", "side_two"):
        side = getattr(spec, side_key)
        party = list(side.pokemon)
        for index, mon in enumerate(party):
            item = "leftovers" if str(mon.item or "") != "leftovers" else "lumberry"
            party[index] = dc.replace(mon, item=item)
        spec = dc.replace(spec, **{side_key: dc.replace(side, pokemon=tuple(party))})
    return "state-item", spec


def _forcing_state_status(world: Any) -> tuple[str, Any]:
    side_key = world.slot_sides["p1"]
    side = getattr(world.spec, side_key)
    active = side.pokemon[int(side.active_index)]
    status = "burn" if str(active.status) != "burn" else "paralyze"
    return "state-status", _replace_active(world, side_key, status=status)


def _forcing_state_weather(world: Any) -> tuple[str, Any]:
    import dataclasses as dc

    weather = "sand" if str(world.spec.weather) != "sand" else "rain"
    return "state-weather", dc.replace(
        world.spec, weather=weather, weather_turns_remaining=5
    )


def _forcing_state_species(world: Any) -> tuple[str, Any]:
    """Rewrite BOTH actives' species -- the opponent's active is always revealed,
    so this must fire `opponent_revealed_species` as well as `self_party_species`."""

    import dataclasses as dc

    spec = world.spec
    for side_key in ("side_one", "side_two"):
        side = getattr(spec, side_key)
        party = list(side.pokemon)
        index = int(side.active_index)
        replacement = "ditto" if str(party[index].id) != "ditto" else "unown"
        party[index] = dc.replace(party[index], id=replacement)
        spec = dc.replace(spec, **{side_key: dc.replace(side, pokemon=tuple(party))})
    return "state-species", spec


def _forcing_state_ability(world: Any) -> tuple[str, Any]:
    import dataclasses as dc

    spec = world.spec
    for side_key in ("side_one", "side_two"):
        side = getattr(spec, side_key)
        party = list(side.pokemon)
        for index, mon in enumerate(party):
            ability = "levitate" if str(mon.ability or "") != "levitate" else "static"
            party[index] = dc.replace(mon, ability=ability, base_ability=ability)
        spec = dc.replace(spec, **{side_key: dc.replace(side, pokemon=tuple(party))})
    return "state-ability", spec


def _forcing_state_sidecond(world: Any) -> tuple[str, Any]:
    import dataclasses as dc

    spec = world.spec
    for side_key in ("side_one", "side_two"):
        side = getattr(spec, side_key)
        conditions = dict(side.side_conditions)
        conditions["spikes"] = 0 if int(conditions.get("spikes", 0)) else 1
        spec = dc.replace(spec, **{side_key: dc.replace(side, side_conditions=conditions)})
    return "state-sidecond", spec


FORCINGS = {
    "none": None,
    "state-hp": _forcing_state_hp,
    "state-pp": _forcing_state_pp,
    "state-disabled": _forcing_state_disabled,
    "state-move": _forcing_state_move,
    "state-toxic": _forcing_state_toxic,
    "state-boost": _forcing_state_boost,
    "state-item": _forcing_state_item,
    "state-status": _forcing_state_status,
    "state-weather": _forcing_state_weather,
    "state-species": _forcing_state_species,
    "state-ability": _forcing_state_ability,
    "state-sidecond": _forcing_state_sidecond,
}


def resolve_forcing(spec: str) -> Any:
    """Compose a comma-separated forcing spec into a world REBUILDER.

    A forcing rewrites the world's ``BattleSpec`` and REBUILDS the engine state
    from it. It does not poke the built ``poke_engine.State``: those pyo3 objects
    are immutable from Python, and the first version of this apparatus wrote to
    them, caught 312 ``AttributeError``s per game and reported a clean zero
    mismatches while doing so -- an instrument test that tested nothing and said
    so only in a field nobody was reading. The rebuild is honest and is the same
    call the constructor makes.

    Unknown tokens raise rather than degrade to `none`: a typo that silently
    disarms the instrument test is exactly the shape of a harness that reports
    success while measuring nothing.
    """

    modes = [part.strip() for part in str(spec).split(",") if part.strip()]
    chosen = [FORCINGS[mode] for mode in modes if _known(mode)]
    chosen = [fn for fn in chosen if fn is not None]
    if not chosen:
        return None

    def composed(world: Any, state: Any) -> tuple[str, Any, Any]:
        import dataclasses as dc

        from pokezero.poke_engine_adapter import build_poke_engine_state

        labels = []
        spec = world.spec
        for fn in chosen:
            label, spec = fn(dc.replace(world, spec=spec))
            labels.append(label)
        forced_world = dc.replace(world, spec=spec)
        return "+".join(labels), forced_world, build_poke_engine_state(spec)

    return composed


def _known(mode: str) -> bool:
    if mode not in FORCINGS:
        raise SystemExit(f"unknown --force mode {mode!r}; known: {sorted(FORCINGS)}")
    return True


# --- the render arm -------------------------------------------------------------


class RenderArm:
    """Deferred render-vs-observed comparison on the TRUE world.

    A render can only be compared once the log has said what happened, and that
    is one decision boundary later. So a boundary is PREPARED when both seats
    have chosen (the true world, the joint action, the fold consumed count) and
    EVALUATED at the next decision in the same battle, against exactly the lines
    appended in between.

    Single-seat boundaries (a forced replacement) are never prepared: the engine
    resolves a joint action and there is no second choice to supply. They are
    counted so the denominator is legible rather than quietly smaller.
    """

    def __init__(
        self,
        *,
        set_source: Any,
        dex: Any,
        every: int,
        records_by_key: dict[tuple[str, int], Any],
    ) -> None:
        self._set_source = set_source
        self._dex = dex
        self.every = max(0, int(every))
        self.records_by_key = records_by_key
        self.counts: Counter[str] = Counter()
        self.errors: list[str] = []
        self._pending: dict[str, dict[str, Any]] = {}
        self._boundary: dict[tuple[str, int], dict[str, Any]] = {}
        self._prepared_index = 0

    # -- seat side ----------------------------------------------------------

    def note_decision(self, context: Any, decision: Any, override: Any) -> None:
        if self.every == 0:
            return
        battle = str(getattr(context, "battle_id", "?"))
        round_index = getattr(context, "decision_round_index", None)
        if round_index is None:
            return
        key = (battle, int(round_index))
        slot = str(getattr(context, "player_id", "?"))
        entry = self._boundary.setdefault(key, {})
        entry[slot] = {
            "context": context,
            "action_index": int(getattr(decision, "action_index", -1)),
            "override": override,
        }
        if len(entry) == 2:
            self._prepare(key, entry)
            self._boundary.pop(key, None)
        # Bound the map: a battle that never completes a boundary must not grow
        # it without limit.
        if len(self._boundary) > 8:
            for stale in sorted(self._boundary)[:-4]:
                self._boundary.pop(stale, None)
                self.counts["skipped:single_seat_boundary"] += 1

    def _prepare(self, key: tuple[str, int], entry: Mapping[str, Any]) -> None:
        self._prepared_index += 1
        if self._prepared_index % self.every != 0:
            self.counts["skipped:sampling"] += 1
            return
        from pokezero.engine_world import world_battle_spec
        from pokezero.poke_engine_adapter import build_poke_engine_state
        from pokezero.engine_world import unpack_team

        context = entry["p1"]["context"]
        override = entry["p1"]["override"]
        if override is None:
            self.counts["skipped:no_truth_override"] += 1
            return
        try:
            world = world_battle_spec(
                context.public_materialization_state, override, dex=self._dex
            )
            state = build_poke_engine_state(world.spec)
        except Exception as error:  # noqa: BLE001
            self.counts[f"skipped:construction:{type(error).__name__}"] += 1
            return

        from engine_transition_differential import (  # noqa: PLC0415
            UnmappableChoice,
            engine_choice_for_action,
        )

        sides = {
            "p1": state.side_one if world.slot_sides["p1"] == "side_one" else state.side_two,
            "p2": state.side_one if world.slot_sides["p2"] == "side_one" else state.side_two,
        }
        choices: dict[str, str] = {}
        for slot in ("p1", "p2"):
            metadata = getattr(entry[slot]["context"].observation, "metadata", None)
            candidates = (metadata or {}).get("action_candidates") or []
            try:
                choices[slot] = engine_choice_for_action(
                    action_index=entry[slot]["action_index"],
                    candidates=candidates,
                    engine_side=sides[slot],
                )
            except UnmappableChoice as error:
                self.counts[f"skipped:unmappable_choice:{error.reason}"] += 1
                return
            except Exception as error:  # noqa: BLE001
                self.counts[f"skipped:choice:{type(error).__name__}"] += 1
                return

        replay = context.public_materialization_state.replay
        party_display: dict[str, list[str]] = {}
        for slot in ("p1", "p2"):
            packed = override.player_teams.get(slot)
            party_display[slot] = [str(mon.species) for mon in unpack_team(packed)]

        from pokezero.public_projection import (  # noqa: PLC0415
            _engine_turn_features,
            pre_state_summary,
        )

        self._pending[str(getattr(context, "battle_id", "?"))] = {
            "key": key,
            "state_string": state.to_string(),
            "slot_sides": dict(world.slot_sides),
            "party_display": party_display,
            "turn": int(getattr(replay, "turn_number", 0) or 0),
            "choices": choices,
            "consumed": len(replay.public_events),
            "pre_features": _engine_turn_features(state, world.slot_sides),
            "pre_summary": pre_state_summary(state, world.slot_sides),
        }
        self.counts["prepared"] += 1

    # -- evaluation ----------------------------------------------------------

    def evaluate_ready(self, context: Any) -> None:
        battle = str(getattr(context, "battle_id", "?"))
        pending = self._pending.get(battle)
        if pending is None:
            return
        replay = getattr(
            getattr(context, "public_materialization_state", None), "replay", None
        )
        if replay is None:
            return
        events = replay.public_events
        if len(events) <= pending["consumed"]:
            return  # nothing happened yet; the other seat is still being asked
        self._pending.pop(battle, None)
        step_lines = [event.raw_line for event in events[pending["consumed"] :]]
        # A PARTIAL TURN is not comparable. The engine enumerates a whole turn
        # including its end-of-turn segment; if the next request arrived mid-turn
        # (a forced replacement after a faint) the observed lines are a prefix of
        # that, and matching a prefix against a complete branch reports the
        # harness's own truncation as a renderer defect. `|turn|` is the marker
        # that the turn resolved and a new one began.
        if not any(line.startswith("|turn|") for line in step_lines):
            self.counts["skipped:partial_turn"] += 1
            return
        from pokezero.public_projection import render_projection_mismatch

        try:
            mismatches, diagnostics = render_projection_mismatch(
                state_string=pending["state_string"],
                slot_sides=pending["slot_sides"],
                party_display=pending["party_display"],
                turn=pending["turn"],
                choices=pending["choices"],
                observed_lines=step_lines,
                pre_features=pending["pre_features"],
                pre_summary=pending["pre_summary"],
            )
        except Exception as error:  # noqa: BLE001
            self.counts[f"error:{type(error).__name__}"] += 1
            self.errors.append(f"{battle}: {type(error).__name__}: {error}")
            return
        if diagnostics.get("render_error"):
            self.counts["error:branch_events"] += 1
        self.counts["evaluated"] += 1
        record = self.records_by_key.get(pending["key"])
        payload = {
            "axes": [m.axis for m in mismatches],
            "predicates": [m.predicate for m in mismatches],
            "detail": [m.detail for m in mismatches][:2],
            "diagnostics": diagnostics,
            "choices": pending["choices"],
            "observed_lines": step_lines[:24] if mismatches else [],
        }
        if mismatches:
            self.counts["mismatched"] += 1
            for mismatch in mismatches:
                self.counts[f"axis:{mismatch.axis}"] += 1
        else:
            self.counts["matched"] += 1
        if record is not None:
            record.render = payload
        else:
            self.counts["orphan_render_record"] += 1


# --- the shard runner ------------------------------------------------------------


def run_shard(args: argparse.Namespace) -> int:
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.engine_search import (
        EngineMctsConfig,
        EngineMctsPolicy,
        EngineSearchFallbackWarning,
        EnvTier2AnnotationSource,
    )
    from pokezero.env import BattleStartOverride
    from pokezero.local_showdown import (
        LocalShowdownConfig,
        LocalShowdownEnv,
        env_config_from_checkpoint_provenance,
    )
    from pokezero.neural_policy import (
        category_vocab_from_model_config,
        feature_masks_from_model_config,
        load_transformer_model_config,
        observation_spec_from_model_config,
    )
    from pokezero.public_projection import (
        DecisionProjectionRecord,
        WorldObserver,
        aggregate_projection_records,
        identity_witness,
    )
    from pokezero.randbat import load_gen3_randbat_source_cached
    from pokezero.rollout import RolloutConfig, continue_rollout_from_current_state
    from pokezero.truth_differential import TruthWorldBuilder

    warnings.simplefilter("ignore", EngineSearchFallbackWarning)

    witness = identity_witness()
    child_witness = _neutral_cwd_witness(os.environ)
    require_model_feature(witness)
    _WITNESS_DIFF_KEYS = (
        "pokezero_file",
        "engine_search_file",
        "public_projection_file",
        "pokezero_search_file",
        "pokezero_search_so_sha256",
        "pokezero_search_model_feature",
        "public_projection_present",
        "public_projection_axis_count",
        "engine_search_world_observer_hook",
        "source_sha256",
        "torch_version",
    )
    mismatches = {
        key: [witness.get(key), child_witness.get(key)]
        for key in _WITNESS_DIFF_KEYS
        if witness.get(key) != child_witness.get(key)
    }
    if not witness.get("engine_search_world_observer_hook"):
        raise SystemExit(
            "REFUSING TO RUN: the loaded engine_search has no world_observer hook, so the "
            "state comparator would see zero worlds and report a clean zero. That is the "
            "'silence is not success' failure this oracle exists to prevent."
        )

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8")) if args.plan else None
    dex = load_showdown_dex_cached(args.showdown_root)
    set_source = load_gen3_randbat_source_cached(args.showdown_root)
    if plan is not None and plan["source_hash"] != set_source.metadata.source_hash:
        raise SystemExit(
            f"plan source_hash {plan['source_hash']} != loaded {set_source.metadata.source_hash}"
        )

    env_config = LocalShowdownConfig(
        showdown_root=Path(args.showdown_root), set_belief_source=True
    )
    model_config = load_transformer_model_config(str(args.checkpoint))
    env_config = env_config_from_checkpoint_provenance(
        env_config,
        feature_masks_from_model_config(model_config),
        required_specs=observation_spec_from_model_config(model_config),
        required_vocabs=category_vocab_from_model_config(
            model_config, env_config.resolved_showdown_root()
        ),
        context="public-projection differential census",
    )
    env = LocalShowdownEnv(env_config)
    annotation_source = EnvTier2AnnotationSource(env)

    driver_config = EngineMctsConfig(
        leaf_eval=args.driver_leaf_eval,
        model_path=args.model_path if args.driver_leaf_eval == "model" else None,
        checkpoint_path=args.checkpoint if args.driver_leaf_eval == "model" else None,
        tables_path=args.tables if args.driver_leaf_eval == "model" else None,
        model_device="cpu",
        worlds=args.driver_worlds,
        sample_retry_factor=args.sample_retry_factor,
        search_sims=args.driver_sims,
        search_batch=args.driver_batch,
        search_depth=args.driver_depth,
    )

    records: list[DecisionProjectionRecord] = []
    exemplars: dict[str, Any] = {}
    forcing = resolve_forcing(args.force)
    observers: dict[str, WorldObserver] = {}
    policies: dict[str, Any] = {}
    for seat in ("p1", "p2"):
        observer = WorldObserver(
            arm="driver",
            records=records,
            exemplar_store=exemplars,
            forcing=forcing,
        )
        observers[seat] = observer
        policies[seat] = EngineMctsPolicy(
            dex=dex,
            set_source=set_source,
            annotation_source=annotation_source,
            config=driver_config,
            policy_id=f"ppc-driver-{seat}",
            world_observer=observer,
        )

    builder = TruthWorldBuilder(env, set_source=set_source)
    records_by_key: dict[tuple[str, int], Any] = {}
    render_arm = RenderArm(
        set_source=set_source,
        dex=dex,
        every=args.render_every,
        records_by_key=records_by_key,
    )

    class _Seat:
        """Drives the game with the production policy; measures around it."""

        def __init__(self, seat: str) -> None:
            self.seat = seat
            self.inner = policies[seat]

        @property
        def policy_id(self) -> str:
            return self.inner.policy_id

        @property
        def stats(self) -> Any:
            return self.inner.stats

        def select_action(self, observation: Any, *, rng: Any) -> Any:
            return self.inner.select_action(observation, rng=rng)

        def select_action_with_context(self, context: Any, *, rng: Any) -> Any:
            render_arm.evaluate_ready(context)
            before = len(records)
            decision = self.inner.select_action_with_context(context, rng=rng)
            for record in records[before:]:
                records_by_key[(record.battle_id, record.round)] = record
            override, _failure = builder.override_for(context)
            render_arm.note_decision(context, decision, override)
            return decision

    seats = {seat: _Seat(seat) for seat in ("p1", "p2")}

    games = _shard_games(plan, args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    per_game: list[dict[str, Any]] = []

    def dump() -> None:
        payload = {
            "schema": "public-projection-census-shard/v1",
            "config": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
            },
            "driver_config": dataclasses.asdict(driver_config),
            "identity_witness": witness,
            "identity_witness_child_neutral_cwd": child_witness,
            "identity_witness_mismatches": mismatches,
            "plan_source_hash": plan["source_hash"] if plan else None,
            "wall_seconds": time.perf_counter() - started,
            "per_game": per_game,
            "forced_labels": {
                seat: dict(observer.forced_labels) for seat, observer in observers.items()
            },
            "instrument_errors": sorted(
                {error for observer in observers.values() for error in observer.errors}
            )[:20],
            "instrument_error_count": sum(
                len(observer.errors) for observer in observers.values()
            ),
            "render_counts": dict(render_arm.counts),
            "render_errors": render_arm.errors[:10],
            "driver_stats": {
                seat: policy.stats.to_dict() for seat, policy in policies.items()
            },
            "summary": aggregate_projection_records(records),
            "records": [record.to_dict() for record in records],
        }
        out_path.write_text(
            json.dumps(payload, indent=1, sort_keys=True, default=str), encoding="utf-8"
        )

    try:
        for index, game in enumerate(games):
            seed = int(game["seed"])
            battle_id = f"ppc-{args.tag}-{seed}"
            for observer in observers.values():
                observer.seed = seed
            builder.reset()
            before = len(records)
            game_started = time.perf_counter()
            try:
                if game.get("packed"):
                    env.reset_with_start_override(
                        seed=seed,
                        start_override=BattleStartOverride(
                            player_teams=dict(game["packed"]),
                            observation_format_id=OBSERVATION_FORMAT_ID,
                        ),
                    )
                else:
                    env.reset(seed=seed, format_id=OBSERVATION_FORMAT_ID)
                continue_rollout_from_current_state(
                    env=env,
                    policies={"p1": seats["p1"], "p2": seats["p2"]},
                    config=RolloutConfig(
                        max_decision_rounds=args.max_rounds,
                        format_id=OBSERVATION_FORMAT_ID,
                    ),
                    seed=seed,
                    battle_id=battle_id,
                    reset_policies=False,
                )
                ok, error = True, None
            except Exception as exc:  # noqa: BLE001 - one bad game must not lose the shard
                ok, error = False, f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - game_started
            per_game.append(
                {
                    "game_index": game.get("game_index", index),
                    "seed": seed,
                    "battle_id": battle_id,
                    "ok": ok,
                    "error": error,
                    "decisions": len(records) - before,
                    "wall_seconds": round(elapsed, 3),
                }
            )
            print(
                f"[{args.tag}] game {index + 1}/{len(games)} seed={seed} ok={ok} "
                f"decisions+={len(records) - before} {elapsed:.1f}s "
                f"total={time.perf_counter() - started:.0f}s",
                flush=True,
            )
            if index % args.dump_every == 0 or index == len(games) - 1:
                dump()
            if args.limit_decisions and len(records) >= args.limit_decisions:
                print(f"[{args.tag}] decision limit reached", flush=True)
                break
    finally:
        dump()
        close = getattr(env, "close", None)
        if callable(close):
            close()

    summary = aggregate_projection_records(records)
    print("=" * 78)
    print(f"wrote {out_path}")
    print(json.dumps({k: v for k, v in summary.items() if k != "predicates"}, indent=2)[:4000])
    print("render:", json.dumps(dict(render_arm.counts), indent=1, sort_keys=True)[:2000])
    return 0


# --- report ------------------------------------------------------------------


def merge_shards(paths: Iterable[Path]) -> dict[str, Any]:
    from pokezero.public_projection import aggregate_projection_records

    records: list[Any] = []
    shards: list[dict[str, Any]] = []
    render_counts: Counter[str] = Counter()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(payload.get("records") or [])
        for key, count in (payload.get("render_counts") or {}).items():
            render_counts[str(key)] += int(count)
        shards.append(
            {
                "path": str(path),
                "wall_seconds": payload.get("wall_seconds"),
                "instrument_error_count": payload.get("instrument_error_count"),
                "instrument_errors": (payload.get("instrument_errors") or [])[:10],
                "identity_witness_mismatches": payload.get("identity_witness_mismatches"),
                "identity_witness": payload.get("identity_witness"),
                "forced_labels": payload.get("forced_labels"),
                "games_ok": sum(1 for g in payload.get("per_game") or [] if g.get("ok")),
                "games_failed": sum(1 for g in payload.get("per_game") or [] if not g.get("ok")),
                "driver_config": payload.get("driver_config"),
            }
        )
    summary = aggregate_projection_records(records)
    summary["shards"] = shards
    summary["shard_count"] = len(shards)
    summary["render_counts"] = dict(render_counts.most_common())
    summary["instrument_error_total"] = sum(
        int(shard.get("instrument_error_count") or 0) for shard in shards
    )
    summary["games_ok"] = sum(int(shard["games_ok"]) for shard in shards)
    summary["games_failed"] = sum(int(shard["games_failed"]) for shard in shards)
    return summary


def render_queue(summary: Mapping[str, Any], *, commands: Sequence[str], title: str) -> str:
    """One table per UNIT. WORLDS, DECISIONS and BOUNDARIES are never co-ranked."""

    lines: list[str] = [f"# {title}", ""]
    lines.append("## Commands that produced every figure below")
    lines.append("")
    lines.append("```sh")
    lines.extend(commands)
    lines.append("```")
    lines.append("")
    lines.append("## The number (state arm)")
    lines.append("")
    lines.append("| quantity | unit | value |")
    lines.append("|---|---|---|")
    lines.append(
        "| projection-mismatched worlds | WORLDS | "
        f"{summary.get('projection_mismatched_worlds')} |"
    )
    lines.append(f"| worlds projected | WORLDS | {summary.get('worlds_projected')} |")
    lines.append(
        "| world mismatch rate | WORLDS | "
        f"{_pct(summary.get('projection_world_mismatch_rate'))} |"
    )
    lines.append(
        "| projection-mismatched decisions | DECISIONS | "
        f"{summary.get('projection_mismatched_decisions')} |"
    )
    lines.append(f"| decisions seen | DECISIONS | {summary.get('decisions_seen')} |")
    lines.append(
        "| decision mismatch rate | DECISIONS | "
        f"{_pct(summary.get('projection_decision_mismatch_rate'))} |"
    )
    lines.append(
        f"| distinct open predicates | PREDICATES | {summary.get('distinct_open_predicates')} |"
    )
    lines.append(f"| battles | BATTLES | {summary.get('battles')} |")
    lines.append(
        f"| instrument errors | EVENTS | {summary.get('instrument_error_total')} |"
    )
    lines.append("")
    lines.append("## The number (render arm) — a SEPARATE unit, never summed with the above")
    lines.append("")
    lines.append("| quantity | unit | value |")
    lines.append("|---|---|---|")
    lines.append(
        "| boundaries compared | BOUNDARIES | "
        f"{summary.get('render_boundaries_compared')} |"
    )
    lines.append(
        "| mismatched boundaries | BOUNDARIES | "
        f"{summary.get('render_mismatched_boundaries')} |"
    )
    lines.append(
        f"| render mismatch rate | BOUNDARIES | {_pct(summary.get('render_mismatch_rate'))} |"
    )
    for axis, count in (summary.get("render_axis_boundaries") or {}).items():
        lines.append(f"| `{axis}` | BOUNDARIES | {count} |")
    # ROWS, in their own block and under their own unit. Emitting one figure per
    # axis and labelling it BOUNDARIES is what overstated
    # `render_post_state_disagreement` by 3.5x: it emits one row per (branch,
    # slot, field), so 80 rows sat in a BOUNDARIES column over 23 boundaries.
    rows_by_axis = summary.get("render_axis_rows") or {}
    for axis, count in rows_by_axis.items():
        lines.append(f"| `{axis}` | ROWS | {count} |")
    if rows_by_axis:
        lines.append("")
        lines.append(
            "ROWS and BOUNDARIES above are DIFFERENT UNITS and neither is a "
            "renderer defect rate. One boundary carries one "
            "`render_unmatched_transition` row but as many "
            "`render_post_state_disagreement` rows as it has disagreeing "
            "(branch, slot, field) triples, so only the BOUNDARIES column is "
            "comparable with `boundaries compared`."
        )
    lines.append("")
    lines.append("### Render-arm disposition (why a boundary was not compared)")
    lines.append("")
    lines.append("| bucket | BOUNDARIES |")
    lines.append("|---|---|")
    for key, count in (summary.get("render_counts") or {}).items():
        # `axis:*` are per-ROW shard counters, not dispositions, and this table's
        # header says BOUNDARIES. Carrying them here is the second place the same
        # 80 was labelled BOUNDARIES. The axis figures are published above, in
        # both units, from the per-boundary records rather than from a
        # pre-aggregated counter that cannot distinguish them.
        if str(key).startswith("axis:"):
            continue
        lines.append(f"| `{key}` | {count} |")
    if not summary.get("render_counts"):
        lines.append("| *(render arm not run)* | 0 |")
    lines.append("")
    lines.append("## The queue: every public fact a searched world contradicted")
    lines.append("")
    lines.append(
        "`worlds` counts WORLDS whose projection carried this predicate; `decisions` "
        "counts DECISIONS on which at least one searched world did. They are different "
        "units and are not comparable to each other or to the render table."
    )
    lines.append("")
    lines.append("| # | predicate | axis | worlds | decisions |")
    lines.append("|---|---|---|---|---|")
    for index, row in enumerate(summary.get("predicates") or [], start=1):
        lines.append(
            f"| {index} | `{row['predicate']}` | {row['axis']} | {row['worlds']} | "
            f"{row['decisions']} |"
        )
    if not summary.get("predicates"):
        lines.append("| - | *(empty)* | - | 0 | 0 |")
    lines.append("")
    lines.append("## Exemplars (one per predicate, first occurrence)")
    lines.append("")
    for row in summary.get("predicates") or []:
        lines.append(f"### `{row['predicate']}`")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(row.get("exemplar") or {}, indent=1, sort_keys=True))
        lines.append("```")
        lines.append("")
    lines.append("## Identity witness (per shard, from the LOADED module)")
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(
            [
                {
                    "path": shard["path"],
                    "witness": shard.get("identity_witness"),
                    "mismatches_vs_neutral_cwd_child": shard.get(
                        "identity_witness_mismatches"
                    ),
                    "forced_labels": shard.get("forced_labels"),
                }
                for shard in (summary.get("shards") or [])[:2]
            ],
            indent=1,
            sort_keys=True,
        )
    )
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _pct(value: Any) -> str:
    if value is None:
        return "UNMEASURED (no denominator)"
    return f"{value:.4%}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="run", choices=("run", "report", "queue", "witness"))
    parser.add_argument("--showdown-root", default=None)
    parser.add_argument("--out", default="-")
    parser.add_argument("--plan")
    parser.add_argument("--shard", help="i/N")
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--limit-decisions", type=int, default=0)
    parser.add_argument("--control-games", type=int, default=0)
    parser.add_argument("--control-seed-base", type=int, default=9_900_000)
    parser.add_argument("--tag", default="ppc")
    parser.add_argument("--max-rounds", type=int, default=250)
    parser.add_argument("--dump-every", type=int, default=5)
    parser.add_argument("--model-path")
    parser.add_argument("--checkpoint")
    parser.add_argument("--tables")
    parser.add_argument(
        "--driver-leaf-eval", default="hp_fraction_crate",
        choices=("hp_fraction_crate", "model"),
    )
    parser.add_argument("--driver-worlds", type=int, default=8)
    parser.add_argument("--driver-sims", type=int, default=256)
    parser.add_argument("--driver-batch", type=int, default=16)
    parser.add_argument("--driver-depth", type=int, default=4)
    parser.add_argument("--sample-retry-factor", type=int, default=4)
    parser.add_argument(
        "--render-every", type=int, default=0,
        help="evaluate one prepared boundary in N through the render comparator; "
             "0 disables the arm. Coverage is a FLOOR and is published as one.",
    )
    parser.add_argument(
        "--force", default="none",
        help="comma-separated: " + "|".join(sorted(FORCINGS)),
    )
    parser.add_argument("--shards", nargs="*", default=[])
    parser.add_argument("--summary")
    parser.add_argument("--queue-title", default="Public-projection differential census")
    parser.add_argument("--command", action="append", default=[])
    args = parser.parse_args(argv)
    if args.showdown_root is None:
        args.showdown_root = _default_showdown_root()

    if args.mode == "witness":
        from pokezero.public_projection import identity_witness

        print(
            json.dumps(
                {
                    "in_process": identity_witness(),
                    "child_neutral_cwd": _neutral_cwd_witness(os.environ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.mode == "report":
        summary = merge_shards(Path(path) for path in args.shards)
        text = json.dumps(summary, indent=1, sort_keys=True)
        if args.out == "-":
            print(text)
        else:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"wrote {args.out}")
        return 0

    if args.mode == "queue":
        if args.summary:
            summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
        else:
            summary = merge_shards(Path(path) for path in args.shards)
        text = render_queue(summary, commands=args.command, title=args.queue_title)
        if args.out == "-":
            print(text)
        else:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"wrote {args.out}")
        return 0

    for required in ("model_path", "checkpoint", "tables"):
        if not getattr(args, required):
            raise SystemExit(f"--{required.replace('_', '-')} is required in run mode")
    if args.out == "-":
        raise SystemExit("--out is required in run mode")
    return run_shard(args)


if __name__ == "__main__":
    sys.exit(main())
