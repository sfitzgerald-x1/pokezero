#!/usr/bin/env python
"""Replay C15's fixed WHY sample plan without changing the certification readout.

The certification sweep's two remaining WHAT-level shapes are deliberately not
reclassified by family name.  This runner loads the sample plan committed
before measurement, finds those exact retained rows, and records the evidence
needed for a causal adjudication:

* the live protocol's event order;
* every branch instruction emitted by the patched engine;
* the engine's legal direct-damage rolls; and
* three controlled, state-local counterfactuals where replay made a mechanism
  testable rather than merely plausible.

It reads the retained sweep archive only.  It neither modifies the archive nor
rewrites ``cert_sweep_readout.py`` or the living divergence ledger.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import poke_engine

from engine_transition_differential import _split_components, damage_components, legal_roll_damages
from triage_roll_components import triage_row


FAMILIES = ("CAND_unresolved_magnitude", "CAND_same_turn_stat_event_gap")

# These are row-level conclusions, not a replacement for the shape-level
# families.  A shape is generalized only where every sampled row supports it.
ADJUDICATIONS: dict[tuple[int, int], dict[str, str]] = {
    (2000261, 31): {
        "verdict": "WHAT-level engine base-damage candidate",
        "lane": "engine candidate; no patch locus licensed",
        "why": "Calm Mind is applied before the hit, but the observed 20 remains below the post-boost legal 21..25 set. The pre-state-stat hypothesis is refuted.",
        "prediction": "refuted H-D2",
    },
    (2000298, 23): {
        "verdict": "switch-choice matcher limitation",
        "lane": "instrument; underlying magnitude remains WHAT-level",
        "why": "The joint action switches Dewgong in before Ice Beam. calculate_damage reports the outgoing-state range (149..354), unrelated to the branch's Damage 10, so its legal-set rejection cannot adjudicate the observed 8.",
        "prediction": "partial H-A",
    },
    (2000431, 32): {
        "verdict": "WHAT-level direct-damage candidate",
        "lane": "engine candidate; no patch locus licensed",
        "why": "The observed Ice Beam damage 5 is outside the engine's 6..16 roll set; no same-turn event or patch-42/43/44 signature is present.",
        "prediction": "partial H-B",
    },
    (2000561, 67): {
        "verdict": "switch-choice matcher limitation",
        "lane": "instrument; underlying magnitude remains WHAT-level",
        "why": "The defender switches before Fire Punch. The direct-damage API prices the pre-switch state while the retained branch prices the incoming Dewgong, so the reported legal set is not evidence about the observed 20.",
        "prediction": "partial H-A",
    },
    (2100079, 7): {
        "verdict": "WHAT-level engine base-damage candidate",
        "lane": "engine candidate; no patch locus licensed",
        "why": "Calm Mind is applied before Ice Beam, but observed 17 is below the post-boost legal 18..22 set. This is not a pre-state-stat phase failure.",
        "prediction": "refuted H-D2",
    },
    (2201005, 55): {
        "verdict": "engine dynamic-HP timing defect",
        "lane": "engine",
        "why": "Crunch lowers Dodrio from 147 to 39 before Flail. The engine branch prices Flail from 147 HP (max 99); the controlled 39-HP state yields max 121 and admits the observed 110.",
        "prediction": "confirmed H-B",
    },
    (2300040, 84): {
        "verdict": "roll-inherited capped residual",
        "lane": "instrument / documented comparison limit",
        "why": "Ice Punch's observed 58 is in the engine legal roll set. The subsequent Leftovers and Leech Seed/Liquid Ooze amounts cap from that preceding roll, so the apparent component mismatch is inherited pricing rather than a new engine mechanism.",
        "prediction": "confirmed H-C",
    },
    (2300552, 117): {
        "verdict": "event-aware legal-set omission",
        "lane": "instrument",
        "why": "The branch applies Clefable's Calm Mind before Fire Blast. Recomputing after that boost gives legal 21..25, which includes the observed 21; the pre-state matcher range was 24..29.",
        "prediction": "confirmed H-D1",
    },
    (2400156, 29): {
        "verdict": "WHAT-level direct-damage candidate",
        "lane": "engine candidate; no patch locus licensed",
        "why": "Flamethrower's observed 31 is below the engine legal 32..38 range. No same-turn stat or known patch signature explains the one-point base gap.",
        "prediction": "partial H-B",
    },
    (2400451, 56): {
        "verdict": "engine Forecast weather-expiry timing defect",
        "lane": "engine",
        "why": "Showdown uses Return while Castform is still Water, then ends Rain at upkeep. The engine branch changes Castform to Normal before Damage, raising Return from the Water-state max 73 to the Normal-state max 109.",
        "prediction": "confirmed H-B",
    },
    (2401127, 54): {
        "verdict": "WHAT-level dynamic type-effect candidate",
        "lane": "engine candidate; no patch locus licensed",
        "why": "The retained branch emits Color Change before Return, but the observed 45 remains outside the branch's damage support. The evidence rejects a simple event-order narration without identifying the remaining type-effect calculation locus.",
        "prediction": "partial H-B",
    },
    (2500120, 60): {
        "verdict": "misbucketed switch-in magnitude",
        "lane": "instrument classification; underlying magnitude remains WHAT-level",
        "why": "Grumpig switches in before Flamethrower; the only status event is a possible burn after damage. This row does not contain a pre-hit boost/status event and cannot support the stat-gap WHY.",
        "prediction": "refuted H-D2",
    },
    (2500576, 7): {
        "verdict": "WHAT-level direct-damage candidate",
        "lane": "engine candidate; no patch locus licensed",
        "why": "Ice Punch's observed 32 is below the engine legal 33..39 range in a no-switch, no-event state. The sample supports a magnitude discrepancy but not a specific pipeline step.",
        "prediction": "partial H-B",
    },
    (2600362, 82): {
        "verdict": "legal-roll matcher accounting",
        "lane": "instrument",
        "why": "Observed Knock Off damage 21 is in the engine's enumerated 20..24 legal set. The divergence is matcher accounting, not an engine damage error.",
        "prediction": "confirmed H-A",
    },
    (2600657, 49): {
        "verdict": "misbucketed static magnitude",
        "lane": "engine candidate; no patch locus licensed",
        "why": "The protocol has Flamethrower followed by Dragon Claw with no boost, status, switch, or type event before the observed damage 41. It cannot support the stat-event hypothesis.",
        "prediction": "refuted H-D2",
    },
    (2601196, 46): {
        "verdict": "WHAT-level direct-damage candidate",
        "lane": "engine candidate; no patch locus licensed",
        "why": "Walrein's observed 6 from Ice Beam is outside the engine's 7..18 set, while the opposite Surf damage is legal. The sample is a localized magnitude candidate, not a family-wide WHY.",
        "prediction": "partial H-B",
    },
}

_EVENT_PREFIXES = (
    "|move|", "|switch|", "|replace|", "|-boost|", "|-unboost|", "|-damage|", "|-heal|",
    "|-start|", "|-end|", "|-activate|", "|-weather|", "|-ability|", "|-item|", "|-status|",
    "|-curestatus|", "|-immune|", "|-fail|", "|-prepare|", "|-crit|",
)


def _target_plan(prediction: Mapping[str, Any]) -> dict[tuple[int, int], str]:
    targets: dict[tuple[int, int], str] = {}
    for family, rows in (prediction.get("sample_plan") or {}).items():
        if family not in FAMILIES or not isinstance(rows, list):
            continue
        for row in rows:
            targets[(int(row["seed"]), int(row["step"]))] = family
    return targets


def _load_rows(archive: Path, targets: Mapping[tuple[int, int], str]) -> dict[tuple[int, int], dict[str, Any]]:
    found: dict[tuple[int, int], dict[str, Any]] = {}
    for path_str in sorted(glob.glob(str(archive / "cert_shard_*.jsonl"))):
        for line in Path(path_str).read_text().splitlines():
            record = json.loads(line)
            for row in record.get("repros") or []:
                key = (int(row.get("seed", -1)), int(row.get("step", -1)))
                if key in targets:
                    found[key] = dict(row)
    missing = sorted(set(targets) - set(found))
    if missing:
        raise ValueError(f"retained archive is missing fixed sample rows: {missing}")
    return found


def _engine_choices(row: Mapping[str, Any]) -> tuple[str, str]:
    choices = row["choices"]
    slots = row.get("slot_sides") or {"p1": "side_one", "p2": "side_two"}
    side_one = choices["p1"] if slots["p1"] == "side_one" else choices["p2"]
    side_two = choices["p2"] if slots["p2"] == "side_two" else choices["p1"]
    return str(side_one), str(side_two)


def _legal_rolls(state: Any, side_one: str, side_two: str) -> dict[str, Any]:
    bases: dict[str, list[int]] = {}
    values: set[int] = set()
    for first in (False, True):
        side_one_rolls, side_two_rolls = poke_engine.calculate_damage(
            state, side_one, side_two, first
        )
        flattened = [int(value) for value in list(side_one_rolls) + list(side_two_rolls)]
        bases["critical" if first else "noncritical"] = flattened
        values.update(legal_roll_damages(flattened))
    return {"bases": bases, "values": sorted(values)}


def _observed_components(row: Mapping[str, Any]) -> dict[str, list[list[Any]]]:
    pre = row["pre_features"]
    lines = [line for line in row.get("protocol") or [] if not line.startswith("|request|")]
    components = damage_components(lines, {"p1": int(pre["p1_hp"]), "p2": int(pre["p2_hp"])})
    return {
        slot: [[source, delta] for source, delta in _split_components(components[slot])[1]]
        for slot in ("p1", "p2")
    }


def _boost_replay(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Reprice a row after the in-branch stat boost, when one precedes damage."""

    key = (int(row["seed"]), int(row["step"]))
    if key not in {(2000261, 31), (2100079, 7), (2300552, 117)}:
        return None
    side_one, side_two = _engine_choices(row)
    state = poke_engine.State.from_string((row.get("engine_states") or [row["engine_state"]])[0])
    candidates = [
        branch for branch in poke_engine.generate_instructions(state, side_one, side_two)
        if "Boost" in str(branch.instruction_list) and "Damage" in str(branch.instruction_list)
        and str(branch.instruction_list).index("Boost") < str(branch.instruction_list).index("Damage")
    ]
    if not candidates:
        raise AssertionError(f"expected boost-before-damage branch for {key}")
    branch = max(candidates, key=lambda candidate: float(candidate.percentage))
    post = state.apply_instructions(branch)
    return {
        "branch": str(branch.instruction_list),
        "post_boosts": {
            "side_one_special_defense": int(post.side_one.special_defense_boost),
            "side_two_special_defense": int(post.side_two.special_defense_boost),
        },
        "post_event_legal_rolls": _legal_rolls(post, side_one, side_two),
    }


def _state_string_counterfactual(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Run the two state-local controls that isolate a dynamic engine dependency."""

    key = (int(row["seed"]), int(row["step"]))
    raw = (row.get("engine_states") or [row["engine_state"]])[0]
    side_one, side_two = _engine_choices(row)
    if key == (2201005, 55):
        old = "DODRIO,78,NORMAL,FLYING,NORMAL,FLYING,147,222"
        new = "DODRIO,78,NORMAL,FLYING,NORMAL,FLYING,39,222"
        label = "Flail after the recorded Crunch damage"
    elif key == (2400451, 56):
        old = "CASTFORM,90,WATER,TYPELESS,WATER,TYPELESS"
        new = "CASTFORM,90,NORMAL,TYPELESS,NORMAL,TYPELESS"
        label = "Return after a hypothetical pre-move Forecast expiry"
    else:
        return None
    if raw.count(old) != 1:
        raise AssertionError(f"counterfactual anchor is not unique for {key}")
    before = poke_engine.State.from_string(raw)
    after = poke_engine.State.from_string(raw.replace(old, new))
    return {
        "label": label,
        "before": _legal_rolls(before, side_one, side_two),
        "after": _legal_rolls(after, side_one, side_two),
    }


def _row_evidence(row: Mapping[str, Any], family: str) -> dict[str, Any]:
    key = (int(row["seed"]), int(row["step"]))
    side_one, side_two = _engine_choices(row)
    states = row.get("engine_states") or [row["engine_state"]]
    branches: list[dict[str, Any]] = []
    roll_sets: list[dict[str, Any]] = []
    for state_text in states:
        state = poke_engine.State.from_string(state_text)
        roll_sets.append(_legal_rolls(state, side_one, side_two))
        branches.extend(
            {
                "percentage": float(branch.percentage),
                "instructions": str(branch.instruction_list),
            }
            for branch in poke_engine.generate_instructions(state, side_one, side_two)
        )
    triage = triage_row(row, verbose=False)
    return {
        "seed": key[0],
        "step": key[1],
        "family": family,
        "choices": row["choices"],
        "engine_choices": {"side_one": side_one, "side_two": side_two},
        "protocol_events": [
            line for line in row.get("protocol") or [] if line.startswith(_EVENT_PREFIXES)
        ],
        "observed_roll_components": _observed_components(row),
        "pre_state_legal_rolls": roll_sets,
        "engine_branches": branches,
        "triage": {
            key: triage[key]
            for key in ("bucket", "findings", "observed_in_legal_set", "context")
        },
        "controlled_boost_replay": _boost_replay(row),
        "controlled_state_counterfactual": _state_string_counterfactual(row),
        "overlap": {
            "patches_42_44_or_world_row_overlap": False,
            "note": (
                "Pre-registered overlap check found zero (seed, step) matches against patches "
                "42-44, the recoil/substitute set, the incapacitation set, and world-lane rows. "
                "Seed 2000431 has a separate absorb row at step 19; this sample is step 32."
                if key == (2000431, 32)
                else "No row-level overlap with patches 42-44 or the current world lanes."
            ),
        },
        "adjudication": ADJUDICATIONS[key],
    }


def _prediction_score(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[str]] = {"confirmed": [], "partial": [], "refuted": []}
    for row in rows:
        label = str(row["adjudication"]["prediction"])
        bucket = "confirmed" if label.startswith("confirmed") else "partial" if label.startswith("partial") else "refuted"
        buckets[bucket].append(f"{row['seed']}/{row['step']}")
    return {"scored_rows": len(rows), **buckets}


def _markdown(result: Mapping[str, Any]) -> str:
    score = result["prediction_score"]
    lines = [
        "# C15 WHY adjudication: magnitude and same-turn-stat samples",
        "",
        "## Scope",
        "",
        "This is a replay-first adjudication of the 16 fixed samples registered before measurement. "
        "It does not relabel the certification sweep, modify the ledger, or claim a family-level WHY "
        "where samples disagree.",
        "",
        "## Prediction Score",
        "",
        f"- Confirmed: {len(score['confirmed'])}/{score['scored_rows']} ({', '.join(score['confirmed']) or 'none'})",
        f"- Partially supported: {len(score['partial'])}/{score['scored_rows']} ({', '.join(score['partial']) or 'none'})",
        f"- Refuted: {len(score['refuted'])}/{score['scored_rows']} ({', '.join(score['refuted']) or 'none'})",
        "",
        "The main correction is negative: the engine does apply a same-turn Calm Mind before the "
        "opposing hit. The broad pre-state-stat hypothesis is therefore refuted, not promoted to an engine patch.",
        "",
        "## Per-Row Verdicts",
        "",
        "| Family | Row | Verdict | Lane | Prediction |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in result["rows"]:
        adjudication = row["adjudication"]
        lines.append(
            f"| {row['family']} | {row['seed']}/{row['step']} | {adjudication['verdict']} | "
            f"{adjudication['lane']} | {adjudication['prediction']} |"
        )
    lines.extend(["", "## Generalization Boundary", ""])
    lines.extend([
        "- `CAND_same_turn_stat_event_gap` is not one mechanism: one sample is an event-aware legal-set omission, two are post-boost one-point candidates, and two are misbucketed non-stat rows.",
        "- `CAND_unresolved_magnitude` is not one mechanism: the samples include switch-choice matcher misuse, a roll-inherited capped residual, dynamic Flail HP timing, Forecast timing, and static magnitude candidates.",
        "- Only the row-level mechanisms above are adjudicated. The remaining unsampled rows stay WHAT-level until replay establishes a shared WHY.",
        "- No sampled row overlaps patches 42-44 or the active world-lane rows at the same `(seed, step)`; the one shared game is explicitly recorded in the JSON artifact.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "PYTHONPATH=src:scripts .venv/bin/python scripts/c15_why_adjudication.py \\",
        "  --archive <retained-sweep-archive> \\",
        "  --prediction reports/c15_why_magnitude_statgap_predictions.json \\",
        "  --out-json reports/c15_why_magnitude_statgap_results.json \\",
        "  --out-md reports/c15_why_magnitude_statgap_report.md",
        "```",
        "",
        "The JSON artifact retains every branch instruction, protocol event, legal-roll set, controlled probe, and per-row rationale.",
    ])
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args(argv)

    prediction = json.loads(args.prediction.read_text())
    targets = _target_plan(prediction)
    if len(targets) != 16:
        raise ValueError(f"expected the pre-registered 16-row sample plan, found {len(targets)}")
    rows_by_key = _load_rows(args.archive, targets)
    rows = [_row_evidence(rows_by_key[key], targets[key]) for key in sorted(targets)]
    result = {
        "schema": "c15-why-adjudication/1",
        "registered_before_measurement": bool(prediction.get("registered_before_measurement")),
        "sample_count": len(rows),
        "family_counts": {family: sum(row["family"] == family for row in rows) for family in FAMILIES},
        "prediction_score": _prediction_score(rows),
        "rows": rows,
    }
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    args.out_md.write_text(_markdown(result))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
