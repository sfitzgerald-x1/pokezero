"""Large-scale ORACLE DIFFERENTIAL + INVARIANT suite over the production
observation encoder (the whole PUBLIC surface), analogous to
``leaf_vs_reality.py`` but for the LIVE self-play path.

WHAT IT DOES
------------
Drives a corpus of gen3randombattle self-play games through the production
encoder (``LocalShowdownEnv.observe`` -> ``observation_from_player_state``).
For EVERY decision it takes the omniscient serialized simulator state
(``LocalShowdownEnv.snapshot()`` -> ``State.serializeBattle`` in the vendored
Node engine) as an INDEPENDENT oracle and asserts, column by column, that each
PUBLIC observation cell equals what the omniscient battle state says it should
be. The oracle is computed by a completely separate codebase (the JS engine's
serialization) from the one under audit (the Python protocol parser + belief
engine), so divergences surface genuine encoder defects rather than parser
self-agreement.

This is the same differential idea that caught the toxic-stage delta, widened
to the entire public surface: HP, active/present/fainted, level, species base
stats, own actual stats, stat-stage boosts, toxic ramp, sleep counter, status,
hazards, screens, weather, turn, plus the legal-action mask, plus an
"impossible observation" invariant suite.

PUBLIC vs BELIEF. The observation is built from Showdown's OMNISCIENT stream
(``local_showdown.py`` ``_apply_event`` appends the ``omniscient`` stream), so
"public" HP here is the EXACT omniscient fraction for BOTH sides (not the
percentage a real opponent sees) -- that is the trainer's design and the oracle
matches it exactly. The BELIEF / uncertainty columns (candidate sets, possible
moves/items/abilities, expected stats) are intentionally uncertain and are NOT
asserted for exact equality here; they get the lighter Part C checks.

CONVENTIONS (verified against the vendored engine + encoder source, cited):
  * toxic_stage: encoder = Showdown ``statusState.stage`` + 1. Showdown's
    ``data/conditions.ts`` tox increments stage in the END-OF-TURN residual;
    the encoder counts the |turn| line (showdown.py ~L818-821, "each turn ...
    escalates") so at a PRE-residual decision it already holds the NEXT tick's
    multiplier. Consistently +1 (accepted convention, not a bug).
  * sleep_turns: encoder tracks ELAPSED turns asleep = startTime - remaining
    (snapshot ``statusState.time`` is turns REMAINING). Accepted.
  Both carry an expected +/-1 turn-boundary timing skew (snapshot is
  post-residual, encoder counts the |turn| line).

CLASSIFICATION per divergence:
  REAL     encoder != true public state (gates the exit code)
  ACCEPTED documented approximation / convention (cited)
  ORACLE   the oracle itself cannot see the datum (excluded from rates)

Usage:
    PYTHONPATH=src python scripts/oracle_differential.py --games 200 [--seed S]
        [--max-steps 100] [--json report.json] [--verbose]

Read-only: never mutates production code and never steps with a real policy
(uniform-random legal actions drive coverage).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pokezero import showdown as S  # noqa: E402
from pokezero.dex import load_showdown_dex_cached, normalize_id  # noqa: E402
from pokezero.local_showdown import (  # noqa: E402
    DEFAULT_SHOWDOWN_ROOT,
    LocalShowdownConfig,
    LocalShowdownEnv,
)
from pokezero.randbat import canonical_gen3_randbat_species_id  # noqa: E402
from pokezero.randbat_vocab import gen3_category_vocabulary  # noqa: E402

TOL = 1e-6

# ---- column -> family + expected classification -----------------------------
BASE_SLOTS = {
    "hp": S.NUMERIC_BASE_HP, "atk": S.NUMERIC_BASE_ATK, "def": S.NUMERIC_BASE_DEF,
    "spa": S.NUMERIC_BASE_SPA, "spd": S.NUMERIC_BASE_SPD, "spe": S.NUMERIC_BASE_SPE,
}
ACT_SLOTS = {
    "hp": S.NUMERIC_ACTUAL_HP, "atk": S.NUMERIC_ACTUAL_ATK, "def": S.NUMERIC_ACTUAL_DEF,
    "spa": S.NUMERIC_ACTUAL_SPA, "spd": S.NUMERIC_ACTUAL_SPD, "spe": S.NUMERIC_ACTUAL_SPE,
}
BOOST_SLOTS = {
    "atk": S.NUMERIC_BOOST_ATK, "def": S.NUMERIC_BOOST_DEF, "spa": S.NUMERIC_BOOST_SPA,
    "spd": S.NUMERIC_BOOST_SPD, "spe": S.NUMERIC_BOOST_SPE,
}


class _SchemaNumericRow:
    """Semantic ``NUMERIC_*`` access over one schema-specific physical row."""

    def __init__(self, observation, token: int) -> None:
        self._observation = observation
        self._token = token

    def __getitem__(self, legacy_index: int) -> float:
        physical_index = S.numeric_index_for_schema(
            self._observation.schema_version, legacy_index
        )
        return self._observation.numeric_features[self._token][physical_index]


class _SchemaNumericRows:
    """Two-dimensional semantic view over an observation's numeric tensor."""

    def __init__(self, observation) -> None:
        self._observation = observation

    def __getitem__(self, token: int) -> _SchemaNumericRow:
        return _SchemaNumericRow(self._observation, token)


def _numeric_if_present(observation, token: int, legacy_index: int) -> float | None:
    try:
        return _SchemaNumericRow(observation, token)[legacy_index]
    except ValueError:
        return None


class Acc:
    """Per-family accounting: checks, divergences, worst delta, samples."""

    def __init__(self) -> None:
        self.fam: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"n": 0, "div": 0, "maxd": 0.0, "samples": []}
        )
        self.real_signatures: dict[str, dict[str, Any]] = {}
        self.invariant_violations: dict[str, dict[str, Any]] = {}

    def check(self, family: str, enc: float, orc: float | None, ctx: Mapping[str, Any], tol: float = TOL) -> None:
        r = self.fam[family]
        r["n"] += 1
        if orc is None:
            return
        d = abs(enc - orc)
        if d > tol:
            r["div"] += 1
            r["maxd"] = max(r["maxd"], d)
            if len(r["samples"]) < 6:
                r["samples"].append({"enc": round(enc, 6), "orc": round(orc, 6), **ctx})

    def real_bug(self, sig: str, ctx: Mapping[str, Any]) -> None:
        rec = self.real_signatures.setdefault(sig, {"count": 0, "samples": []})
        rec["count"] += 1
        if len(rec["samples"]) < 6:
            rec["samples"].append(dict(ctx))

    def invariant(self, name: str, ctx: Mapping[str, Any]) -> None:
        rec = self.invariant_violations.setdefault(name, {"count": 0, "samples": []})
        rec["count"] += 1
        if len(rec["samples"]) < 6:
            rec["samples"].append(dict(ctx))


class Oracle:
    def __init__(self, root: Path) -> None:
        self.dex = load_showdown_dex_cached(root)
        self.vocab = gen3_category_vocabulary(root, include_turn_merged=True)

    def canon(self, x: str) -> str:
        n = normalize_id(x)
        return canonical_gen3_randbat_species_id(n) or n

    def species_of_mon(self, m: Mapping[str, Any]) -> str:
        st = m.get("set") or {}
        if st.get("speciesId"):
            return self.canon(st["speciesId"])
        d = m.get("details")
        if d:
            return self.canon(d.split(",")[0])
        return self.canon(str(m.get("species")))

    def level_of_mon(self, m: Mapping[str, Any]) -> int:
        for part in (m.get("details") or "").split(","):
            p = part.strip()
            if p.startswith("L") and p[1:].isdigit():
                return int(p[1:])
        st = m.get("set") or {}
        return st["level"] if isinstance(st.get("level"), int) else 100

    def base_stats(self, species: str) -> dict[str, float] | None:
        info = S._species_info_base_fallback(self.dex, species)
        if info is None:
            return None
        out = {}
        for k in BASE_SLOTS:
            v = info.base_stats.get(k)
            out[k] = min(1.0, float(v) / 200.0) if v else 0.0
        return out

    def status_id(self, status: str) -> int:
        return self.vocab.encode(f"status:{status or 'none'}")


def audit_side(acc: Acc, orc: Oracle, obs, team, offset: int, role: str, side: Mapping[str, Any]) -> None:
    mons = {orc.species_of_mon(m): m for m in side["pokemon"]}
    nf = _SchemaNumericRows(obs)
    cats = obs.categorical_ids
    for i, member in enumerate(team[:6]):
        tok = offset + i
        sp = orc.canon(member.species)
        m = mons.get(sp)
        if m is None:
            acc.check(f"identity/unmapped_{role}", 1.0, 0.0, {"sp": sp, "avail": list(mons.keys())})
            continue
        hp = m.get("hp") or 0
        maxhp = m.get("maxhp") or 1
        fainted = bool(m.get("fainted")) or hp == 0
        active = bool(m.get("isActive"))
        vols = m.get("volatiles") or {}
        ss = m.get("statusState") or {}
        st = m.get("status") or ""
        ctx = {"role": role, "sp": sp, "active": active, "hp": hp, "maxhp": maxhp, "fainted": fainted}

        # HP FRACTION (exact omniscient, both sides; clamp; fainted -> 0)
        hp_frac = 0.0 if fainted else max(0.0, min(1.0, hp / maxhp))
        acc.check("hp_fraction", nf[tok][S.NUMERIC_HP_FRACTION], hp_frac, ctx)

        # ACTIVE. A fainted-but-not-yet-replaced mon still occupies the request's
        # active slot (encoder is faithful to the request); only a NON-fainted
        # active mismatch is a real defect.
        enc_active = nf[tok][S.NUMERIC_ACTIVE]
        if not fainted:
            acc.check("active", enc_active, 1.0 if active else 0.0, ctx)
            if abs(enc_active - (1.0 if active else 0.0)) > TOL:
                acc.real_bug("active_flag_wrong_nonfainted", {**ctx, "enc": enc_active})
        else:
            acc.check("active_fainted(accepted)", enc_active, enc_active, ctx)  # informational

        # PRESENT / LEGAL(fainted)
        acc.check("present", nf[tok][S.NUMERIC_PRESENT], 1.0, ctx)
        acc.check("legal_switchable(fainted)", nf[tok][S.NUMERIC_LEGAL], 0.0 if fainted else 1.0, ctx)

        # LEVEL
        lvl = orc.level_of_mon(m)
        acc.check("level", nf[tok][S.NUMERIC_LEVEL], min(1.0, lvl / 100.0), {**ctx, "lvl": lvl})

        # SPECIES BASE STATS (dex-derived). A live Transform rewrites the encoded identity
        # to the copied species by design, so that *active transform state* has no direct
        # base-stat oracle. A normal or fainted Ditto is still checked: this catches stale
        # copied identity after the simulator has ended Transform.
        is_transformed = "transform" in vols
        if not is_transformed:
            bexp = orc.base_stats(sp)
            if bexp:
                for k, slot in BASE_SLOTS.items():
                    acc.check(f"base_stat/{k}", nf[tok][slot], bexp[k], {**ctx, "stat": k})

        # OWN ACTUAL STATS (self only; hp = maxhp). A FAINTED self mon reports its
        # condition as "0 fnt", which carries no max-HP denominator, so the encoder
        # legitimately drops actual HP (0) -- exclude fainted from this oracle.
        if role == "self" and not is_transformed and not fainted:
            stored = m.get("storedStats") or {}
            for k, slot in ACT_SLOTS.items():
                v = maxhp if k == "hp" else stored.get(k)
                if v:
                    acc.check(f"actual_stat/{k}", nf[tok][slot], min(1.0, float(v) / 714.0), {**ctx, "stat": k})

        # BOOSTS + TOXIC (active only). Transform also copies the target's public boost
        # stages, so those remain oracle-checkable even though base stats are not.
        if active:
            boosts = m.get("boosts") or {}
            for k, slot in BOOST_SLOTS.items():
                stg = boosts.get(k) or 0
                acc.check(f"boost/{k}", nf[tok][slot], max(-1.0, min(1.0, stg / 6.0)), {**ctx, "stat": k, "stage": stg})
            if st == "tox":
                snap_stage = ss.get("stage", 0)
                enc_stage = round(nf[tok][S.NUMERIC_TOXIC_STAGE] * 15)
                # Accepted convention: encoder = snapshot stage + 1 (pre-residual
                # next-tick multiplier), with an occasional 0 at turn boundaries.
                off = enc_stage - snap_stage
                acc.fam["toxic_stage(offset accepted +1)"]["n"] += 1
                if off not in (0, 1):
                    acc.fam["toxic_stage(offset accepted +1)"]["div"] += 1
                    acc.real_bug("toxic_stage_offset_out_of_band", {**ctx, "enc_stage": enc_stage, "snap_stage": snap_stage, "off": off})

        # SLEEP counter (any token). Encoder = elapsed = startTime - remaining.
        if st == "slp":
            time_left = ss.get("time")
            start = ss.get("startTime")
            enc_sleep = round(nf[tok][S.NUMERIC_SLEEP_TURNS] * 5)
            if isinstance(time_left, int) and isinstance(start, int):
                elapsed = start - time_left
                acc.fam["sleep_turns(elapsed)"]["n"] += 1
                if abs(enc_sleep - elapsed) > 1:  # allow +/-1 turn-boundary skew
                    acc.fam["sleep_turns(elapsed)"]["div"] += 1
                    acc.real_bug("sleep_turns_out_of_band", {**ctx, "enc": enc_sleep, "elapsed": elapsed, "time": time_left, "start": start})

        # STATUS categorical. Self side reads the request condition ("0 fnt" -> none);
        # opponent reads belief.status which can retain the PRE-FAINT status on a
        # fainted mon -- a real (low-severity) divergence from the true public state.
        enc_status = cats[tok][S.CATEGORY_SECONDARY]
        true_status = "none" if fainted else (st or "none")
        exp_id = orc.status_id(true_status)
        acc.fam[f"status_cat/{role}"]["n"] += 1
        if enc_status != exp_id:
            acc.fam[f"status_cat/{role}"]["div"] += 1
            if fainted:
                # Fainted mon: Showdown clears status on faint but emits no explicit
                # cure line, so the belief keeps the last-known status. Low severity
                # (the mon is inert: hp 0, legal 0).
                acc.real_bug("fainted_mon_retains_stale_status(low-sev)", {**ctx, "enc_id": enc_status})
            else:
                # ALIVE mon with the wrong status = an impossible observation. The
                # dominant cause is a Natural-Cure switch-out cure (public
                # -curestatus) being misattributed to the replacement mon, leaving
                # the outgoing mon's status_on_exit stale and restored on re-entry.
                acc.real_bug("alive_mon_stale_status_after_public_cure", {**ctx, "enc_id": enc_status, "exp_id": exp_id})

        # ---- INVARIANTS (Part B) ----
        run_invariants(acc, obs, tok, m, role, sp, fainted, active, st, ss, vols)


def run_invariants(acc, obs, tok, m, role, sp, fainted, active, st, ss, vols) -> None:
    nf = _SchemaNumericRow(obs, tok)
    ctx = {"role": role, "sp": sp}
    hpf = nf[S.NUMERIC_HP_FRACTION]
    # bounds
    if not (0.0 - TOL <= hpf <= 1.0 + TOL):
        acc.invariant("bounds/hp_in_0_1", {**ctx, "hp": hpf})
    for k, slot in BOOST_SLOTS.items():
        if not (-1.0 - TOL <= nf[slot] <= 1.0 + TOL):
            acc.invariant("bounds/boost_in_-1_1", {**ctx, "stat": k, "v": nf[slot]})
    if not (0.0 - TOL <= nf[S.NUMERIC_TOXIC_STAGE] <= 1.0 + TOL):
        acc.invariant("bounds/toxic_in_0_1", {**ctx, "v": nf[S.NUMERIC_TOXIC_STAGE]})
    # consistency: status==tox => toxic_stage>=1 (on active)
    if active and st == "tox" and nf[S.NUMERIC_TOXIC_STAGE] < 1 / 15.0 - TOL:
        acc.invariant("consistency/tox_implies_stage_ge_1", {**ctx, "stage": nf[S.NUMERIC_TOXIC_STAGE]})
    # consistency: fainted => hp==0
    if fainted and hpf > TOL:
        acc.invariant("consistency/fainted_implies_hp0", {**ctx, "hp": hpf})
    # consistency: fainted => no active volatiles surfaced
    if fainted and active:
        acc.invariant("consistency/fainted_not_active(accepted_window)", {**ctx})
    # no-placeholder: present & alive => hp>0 and level>0
    if not fainted:
        if hpf <= TOL:
            acc.invariant("placeholder/alive_hp_zero", {**ctx, "hp": hpf})
        if nf[S.NUMERIC_LEVEL] <= TOL:
            acc.invariant("placeholder/alive_level_zero", {**ctx})
    # opponent revealed-move PP fraction bounds + candidate_set sanity
    if role == "opp":
        for k in range(S.BELIEF_MOVE_BUCKET_COUNT):
            v = nf[S.NUMERIC_OPP_MOVE_PP_OFFSET + k]
            if not (0.0 - TOL <= v <= 1.0 + TOL):
                acc.invariant("bounds/opp_pp_in_0_1", {**ctx, "bucket": k, "v": v})


def audit_field(acc, orc, obs, battle, self_slot, opp_slot, turn) -> None:
    nf = _SchemaNumericRow(obs, S.FIELD_TOKEN_OFFSET)
    side_by_id = {s["id"]: s for s in battle["sides"]}
    acc.check("field/turn", nf[S.NUMERIC_TURN_COUNT], min(1.0, turn / 1000.0), {"turn": turn})

    def counts(side):
        sc = side.get("sideConditions") or {}
        spikes = sc.get("spikes", {}).get("layers", 0) if "spikes" in sc else 0
        screens = (1 if "reflect" in sc else 0) + (1 if "lightscreen" in sc else 0)
        return spikes, screens

    ss = side_by_id.get(self_slot)
    os_ = side_by_id.get(opp_slot)
    if ss:
        sp, scr = counts(ss)
        acc.check("field/self_hazards", nf[S.NUMERIC_SELF_HAZARDS], min(1.0, sp / 3.0), {"spikes": sp})
        encoded_screens = _numeric_if_present(
            obs, S.FIELD_TOKEN_OFFSET, S.NUMERIC_SELF_SCREENS
        )
        if encoded_screens is not None:
            acc.check(
                "field/self_screens", encoded_screens, min(1.0, scr / 2.0), {"scr": scr}
            )
    if os_:
        sp, scr = counts(os_)
        acc.check("field/opp_hazards", nf[S.NUMERIC_OPP_HAZARDS], min(1.0, sp / 3.0), {"spikes": sp})
        encoded_screens = _numeric_if_present(
            obs, S.FIELD_TOKEN_OFFSET, S.NUMERIC_OPP_SCREENS
        )
        if encoded_screens is not None:
            acc.check(
                "field/opp_screens", encoded_screens, min(1.0, scr / 2.0), {"scr": scr}
            )
    # WEATHER categorical (only asserted when weather is up)
    w = normalize_id((battle.get("field") or {}).get("weather") or "")
    if w:
        enc_w = obs.categorical_ids[S.FIELD_TOKEN_OFFSET][S.CATEGORY_SECONDARY]
        exp_w = orc.vocab.encode(f"weather:{w}")
        acc.fam["field/weather_cat"]["n"] += 1
        if enc_w != exp_w:
            acc.fam["field/weather_cat"]["div"] += 1
            acc.real_bug("weather_cat_mismatch", {"true_w": w, "enc": enc_w, "exp": exp_w})


def audit_legal_mask(acc, obs, state, side: Mapping[str, Any]) -> None:
    """Legal-action-mask invariants vs the omniscient side state.

    Legal moves must map to real active moves with PP>0; legal switch actions
    must map to ALIVE benched mons; at least one action is legal at a live
    decision. (The mask itself is Showdown-authored via the request; these are
    cheap structural guards on the 9-action projection.)"""
    mask = obs.legal_action_mask
    active_mon = next((m for m in side["pokemon"] if m.get("isActive")), None)
    alive_bench = sum(
        1 for m in side["pokemon"]
        if not m.get("isActive") and not (m.get("fainted") or (m.get("hp") or 0) == 0)
    )
    n_switch = sum(1 for i in range(S.MOVE_ACTION_COUNT, S.ACTION_COUNT) if mask[i])
    ctx = {"n_switch": n_switch, "alive_bench": alive_bench}
    if not any(mask):
        acc.invariant("legal_mask/no_action_at_live_decision", ctx)
    if n_switch > alive_bench:
        acc.invariant("legal_mask/more_switches_than_alive_bench", ctx)
    # legal move slots must be within the active mon's declared moves and PP>0.
    # EXCEPTION: when every move is at 0 PP the request forces STRUGGLE into move
    # slot 0 (a legal, usable move), so the snapshot's 0-PP moveSlots no longer map
    # to the request's move list -- skip the PP check in that Struggle situation.
    if active_mon is not None:
        slots = active_mon.get("moveSlots") or []
        struggle_forced = bool(slots) and all((s.get("pp") or 0) <= 0 for s in slots)
        for i in range(S.MOVE_ACTION_COUNT):
            if mask[i]:
                if i >= len(slots):
                    acc.invariant("legal_mask/move_slot_out_of_range", {"slot": i, "nmoves": len(slots)})
                elif not struggle_forced and (slots[i].get("pp") or 0) <= 0 and not slots[i].get("disabled"):
                    acc.invariant("legal_mask/legal_move_zero_pp", {"slot": i, "move": slots[i].get("id")})


def audit_belief_partc(acc, orc, obs, state, side: Mapping[str, Any]) -> None:
    """Part C (light): belief candidate-universe sanity on the OPPONENT.

    Regression guard for the candidate-universe over-pruning fix (#757): every
    REVEALED opponent mon must keep candidate_set_count >= 1, and no truly-owned
    opponent move may be pruned from the possible-move set (the true set stays
    in-universe). Also records candidate_set_count per (game, species) so the
    caller can assert monotone non-increase across the game."""
    beliefs = state.belief_view.opponent_by_species()
    mons = {orc.species_of_mon(m): m for m in side["pokemon"]}
    nf = _SchemaNumericRows(obs)
    for i, member in enumerate(state.opponent_team[:6]):
        tok = S.OPPONENT_POKEMON_TOKEN_OFFSET + i
        present = nf[tok][S.NUMERIC_PRESENT] > 0.5
        if not present:
            continue
        sp = orc.canon(member.species)
        m = mons.get(sp)
        if m is None or m.get("fainted") or (m.get("hp") or 0) == 0:
            continue
        if sp == "ditto" or "transform" in (m.get("volatiles") or {}):
            # Transformed Ditto's true moveSlots are the COPIED mon's moves; the
            # belief tracks Ditto's own set -> not an over-pruning signal.
            continue
        belief = beliefs.get(normalize_id(member.species))
        possible = set()
        if belief is not None:
            possible = {normalize_id(x) for x in getattr(belief, "possible_moves", ())}
            possible |= {normalize_id(x) for x in getattr(belief, "revealed_moves", ())}
        if not possible:
            # Set source off / not yet enumerated: nothing to check for this mon.
            acc.fam["partC/belief_inert"]["n"] += 1
            continue
        acc.fam["partC/belief_active"]["n"] += 1
        # OFF-SCRIPT (regression guard for #757): the TRUE opponent moveset must stay
        # in-universe -- every real move must be reachable in the candidate set.
        # Hidden Power serializes with the GENERIC id ("hiddenpower") in the sim
        # moveSlots but the belief tracks the TYPED variant ("hiddenpowerfire"...),
        # so accept any "hiddenpower*" as covering the generic entry.
        hp_covered = any(mv.startswith("hiddenpower") for mv in possible)
        true_moves = {normalize_id((ms.get("id") or "")) for ms in (m.get("moveSlots") or [])}
        true_moves.discard("")
        offscript = {mv for mv in (true_moves - possible)
                     if not (mv == "hiddenpower" and hp_covered)}
        if offscript:
            acc.invariant("partC/true_move_pruned_offscript", {"sp": sp, "offscript": sorted(offscript)})
            acc.real_bug("belief_over_pruned_true_move", {"sp": sp, "offscript": sorted(offscript)})


def run(games: int, seed0: int, max_steps: int, belief_source: bool) -> tuple[Acc, dict]:
    orc = Oracle(DEFAULT_SHOWDOWN_ROOT)
    acc = Acc()
    stats = {"games": 0, "decisions": 0, "belief_source": belief_source}
    # Production training runs the belief candidate-universe machinery on (env var);
    # enable it so Part C exercises the real candidate sets and possible-move surface.
    env = LocalShowdownEnv(LocalShowdownConfig(set_belief_source=belief_source))
    try:
        for g in range(games):
            env.reset(seed=seed0 + g)
            rng = random.Random(seed0 + g)
            steps = 0
            last_cset: dict[tuple, float] = {}
            mono_seen: set[tuple] = set()
            stats["games"] += 1
            while env.terminal() is None and steps < max_steps:
                requested = env.requested_players()
                if not requested:
                    break
                try:
                    snap = env.snapshot()
                except Exception:
                    break
                battle = snap.bridge_snapshot["battle"]
                turn = battle.get("turn") or 0
                side_by_id = {s["id"]: s for s in battle["sides"]}
                for p in requested:
                    try:
                        obs = env.observe(p)
                        state = env._state_for_player(p)
                    except Exception:
                        continue
                    stats["decisions"] += 1
                    slot = obs.perspective.showdown_slot
                    oppslot = obs.perspective.opponent_showdown_slot
                    if side_by_id.get(slot):
                        audit_side(acc, orc, obs, state.self_team, S.SELF_POKEMON_TOKEN_OFFSET, "self", side_by_id[slot])
                        audit_legal_mask(acc, obs, state, side_by_id[slot])
                    if side_by_id.get(oppslot):
                        audit_side(acc, orc, obs, state.opponent_team, S.OPPONENT_POKEMON_TOKEN_OFFSET, "opp", side_by_id[oppslot])
                        audit_belief_partc(acc, orc, obs, state, side_by_id[oppslot])
                        # Monotonicity of candidate_set_count. Key by (OBSERVER, species):
                        # BOTH teams can carry the same species, so a species-only key
                        # would cross-contaminate p1's-opponent vs p2's-opponent (two
                        # different mons) and fire false violations. Recorded once per
                        # (game, observer, species) -- a re-widening persists for many
                        # decisions and would otherwise inflate the count.
                        for i, member in enumerate(state.opponent_team[:6]):
                            tok = S.OPPONENT_POKEMON_TOKEN_OFFSET + i
                            if _SchemaNumericRow(obs, tok)[S.NUMERIC_PRESENT] <= 0.5:
                                continue
                            spk = orc.canon(member.species)
                            if spk == "ditto":  # Transform legitimately reopens variants
                                continue
                            key = (g, slot, spk)
                            cset = _SchemaNumericRow(obs, tok)[S.NUMERIC_CANDIDATE_SET_COUNT]
                            floor = last_cset.get(key)
                            if floor is not None and cset > floor + TOL and key not in mono_seen:
                                mono_seen.add(key)
                                acc.invariant(
                                    "monotonicity/candidate_set_count_increased",
                                    {"game": g, "observer": slot, "sp": spk,
                                     "floor": floor, "now": cset},
                                )
                            last_cset[key] = min(cset, floor if floor is not None else cset)
                    audit_field(acc, orc, obs, battle, slot, oppslot, turn)
                actions = {}
                ok = True
                for p in requested:
                    mask = env.legal_actions(p)
                    legal = [i for i, b in enumerate(mask) if b]
                    if not legal:
                        ok = False
                        break
                    actions[p] = rng.choice(legal)
                if not ok:
                    break
                try:
                    env.step(actions)
                except Exception:
                    break
                steps += 1
    finally:
        env.close()
    return acc, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-belief-source", action="store_true",
                    help="disable the opponent candidate-set source (Part C becomes inert)")
    a = ap.parse_args()

    acc, stats = run(a.games, a.seed, a.max_steps, belief_source=not a.no_belief_source)

    print(f"games={stats['games']}  decisions={stats['decisions']}  belief_source={stats['belief_source']}")
    print("\n== PART A: per-column-family differential ==")
    real_family_div = 0
    for fam in sorted(acc.fam):
        r = acc.fam[fam]
        rate = (r["div"] / r["n"]) if r["n"] else 0.0
        flag = ""
        accepted = "accepted" in fam or "offset" in fam or "elapsed" in fam or "fainted" in fam
        if r["div"] and not accepted:
            flag = "  <<< DIVERGENCE"
            real_family_div += r["div"]
        print(f"  {fam:40s} n={r['n']:7d} div={r['div']:6d} rate={rate:.5f} maxd={r['maxd']:.4f}{flag}")
        if a.verbose and r["div"] and r["samples"]:
            for s in r["samples"][:4]:
                print("        ", json.dumps(s, default=str))

    print("\n== REAL-BUG signatures ==")
    if not acc.real_signatures:
        print("  (none)")
    for sig, rec in sorted(acc.real_signatures.items(), key=lambda kv: -kv[1]["count"]):
        print(f"  [{rec['count']:6d}] {sig}")
        for s in rec["samples"][:3]:
            print("        ", json.dumps(s, default=str))

    print("\n== PART B: invariant violations ==")
    if not acc.invariant_violations:
        print("  (none)")
    for name, rec in sorted(acc.invariant_violations.items(), key=lambda kv: -kv[1]["count"]):
        print(f"  [{rec['count']:6d}] {name}")
        for s in rec["samples"][:3]:
            print("        ", json.dumps(s, default=str))

    if a.json:
        Path(a.json).write_text(json.dumps({
            "stats": stats,
            "families": {k: {kk: vv for kk, vv in v.items() if kk != "samples"} | {"samples": v["samples"]} for k, v in acc.fam.items()},
            "real_signatures": acc.real_signatures,
            "invariant_violations": acc.invariant_violations,
        }, indent=1, default=str))
        print(f"\nwrote {a.json}")

    # Exit code: only genuine REAL-bug signatures gate (accepted conventions do not).
    gating = {k: v for k, v in acc.real_signatures.items()
              if k not in ("fainted_mon_retains_stale_status",)}
    return 1 if gating else 0


if __name__ == "__main__":
    raise SystemExit(main())
