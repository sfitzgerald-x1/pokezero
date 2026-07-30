#!/usr/bin/env python
"""Replay C15's fixed WHY population without changing the certification readout.

The certification sweep's two remaining WHAT-level shapes are deliberately not
reclassified by family name.  This runner loads the sample plan committed
before measurement, finds those exact retained rows, and records the evidence
needed for a causal adjudication:

* the live protocol's event order;
* every branch instruction emitted by the patched engine;
* the engine's legal direct-damage rolls; and
* controlled, state-local counterfactuals where replay made a mechanism
  testable rather than merely plausible; and
* fixed-point base-power controls for the odd-power ability-modifier defect.

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
from pokezero.gen3_damage import Gen3DamageContext, apply_chain, gen3_damage_rolls
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
    (2001162, 120): {
        "verdict": "engine odd-base-power modifier rounding defect",
        "lane": "engine; Torrent fixed-point base-power follow-up",
        "why": "Mightyena's switch and Intimidate are noncausal for special Surf. Swampert is inside Torrent range: the Rust `*= 1.5` carries 95*1.5=142.5 into damage, while Showdown chainModify rounds half-down to 142. The fixed-point 132..156 set admits observed 132.",
        "prediction": "partial S3",
        "follow_up_locus": "third_party/poke-engine-src/src/gen3/abilities.rs ability_modify_attack_being_used: Torrent",
    },
    (2100079, 7): {
        "verdict": "WHAT-level engine base-damage candidate",
        "lane": "engine candidate; no patch locus licensed",
        "why": "Calm Mind is applied before Ice Beam, but observed 17 is below the post-boost legal 18..22 set. This is not a pre-state-stat phase failure.",
        "prediction": "refuted H-D2",
    },
    (2100482, 83): {
        "verdict": "engine odd-base-power modifier rounding defect",
        "lane": "engine; Torrent fixed-point base-power follow-up",
        "why": "Blastoise is inside Torrent range before Dustox switches in. Rust carries Surf's 142.5 base power; Showdown chainModify rounds it to 142. The fixed-point 102..121 set admits observed 102.",
        "prediction": "partial M5",
        "follow_up_locus": "third_party/poke-engine-src/src/gen3/abilities.rs ability_modify_attack_being_used: Torrent",
    },
    (2200369, 75): {
        "verdict": "engine odd-base-power modifier rounding defect",
        "lane": "engine; Thick Fat fixed-point base-power follow-up",
        "why": "Flamethrower into Thick Fat uses Rust 95/2=47.5 instead of Showdown's half-down chainModify result 47. The fixed-point 49..58 set admits observed 49.",
        "prediction": "partial M5",
        "follow_up_locus": "third_party/poke-engine-src/src/gen3/abilities.rs ability_modify_attack_against: Thick Fat",
    },
    (2201005, 55): {
        "verdict": "engine dynamic-HP timing defect",
        "lane": "engine",
        "why": "Crunch lowers Dodrio from 147 to 39 before Flail. The engine branch prices Flail from 147 HP (max 99); the controlled 39-HP state yields max 121 and admits the observed 110.",
        "prediction": "confirmed H-B",
        "follow_up_locus": "third_party/poke-engine-src/src/gen3/generate_instructions.rs before_move -> choice_effects::modify_choice FLAIL|REVERSAL; recompute dynamic BP after earlier same-turn damage",
    },
    (2300040, 84): {
        "verdict": "roll-inherited capped residual",
        "lane": "instrument / documented comparison limit",
        "why": "Ice Punch's observed 58 is in the engine legal roll set. The subsequent Leftovers and Leech Seed/Liquid Ooze amounts cap from that preceding roll, so the apparent component mismatch is inherited pricing rather than a new engine mechanism.",
        "prediction": "confirmed H-C",
        "follow_up_locus": "scripts/engine_transition_differential.py roll_components_agree capped_lethal and *_to_full handling; comparison-limit lane only",
    },
    (2300154, 80): {
        "verdict": "engine odd-base-power modifier rounding defect",
        "lane": "engine; Thick Fat fixed-point base-power follow-up",
        "why": "Exploud's Flamethrower into Thick Fat carries 47.5 base power in Rust instead of fixed-point 47. The fixed-point 32..38 set admits observed 32.",
        "prediction": "partial M5",
        "follow_up_locus": "third_party/poke-engine-src/src/gen3/abilities.rs ability_modify_attack_against: Thick Fat",
    },
    (2300552, 117): {
        "verdict": "event-aware legal-set omission",
        "lane": "instrument",
        "why": "The branch applies Clefable's Calm Mind before Fire Blast. Recomputing after that boost gives legal 21..25, which includes the observed 21; the pre-state matcher range was 24..29.",
        "prediction": "confirmed H-D1",
        "follow_up_locus": "scripts/engine_transition_differential.py evaluate_boundary_strict -> roll_components_agree; derive branch-event-aware legal rolls after same-turn stat changes",
    },
    (2400156, 29): {
        "verdict": "WHAT-level direct-damage candidate",
        "lane": "engine candidate; no patch locus licensed",
        "why": "Flamethrower's observed 31 is below the engine legal 32..38 range. No same-turn stat or known patch signature explains the one-point base gap.",
        "prediction": "partial H-B",
    },
    (2400140, 9): {
        "verdict": "engine odd-base-power modifier rounding defect",
        "lane": "engine; Thick Fat fixed-point base-power follow-up",
        "why": "Toxic happens after Magmar's Flamethrower and is noncausal. Thick Fat carries 47.5 base power in Rust instead of fixed-point 47; the fixed-point 43..51 set admits observed 43.",
        "prediction": "partial S3",
        "follow_up_locus": "third_party/poke-engine-src/src/gen3/abilities.rs ability_modify_attack_against: Thick Fat",
    },
    (2400172, 89): {
        "verdict": "engine odd-base-power modifier rounding defect",
        "lane": "engine; Thick Fat fixed-point base-power follow-up",
        "why": "Fire Punch into Thick Fat carries 75/2=37.5 in Rust instead of fixed-point 37. The fixed-point set after Grumpig's +4 SpA is 86..102 and admits observed 86.",
        "prediction": "partial M5",
        "follow_up_locus": "third_party/poke-engine-src/src/gen3/abilities.rs ability_modify_attack_against: Thick Fat",
    },
    (2400342, 78): {
        "verdict": "engine odd-base-power modifier rounding defect",
        "lane": "engine; Thick Fat fixed-point base-power follow-up",
        "why": "Toxic occurs after Volbeat's Ice Punch and is noncausal. Thick Fat carries 37.5 base power in Rust instead of fixed-point 37; the fixed-point +6 SpA set 92..109 admits observed 92.",
        "prediction": "partial S3",
        "follow_up_locus": "third_party/poke-engine-src/src/gen3/abilities.rs ability_modify_attack_against: Thick Fat",
    },
    (2400451, 56): {
        "verdict": "engine Forecast weather-expiry timing defect",
        "lane": "engine",
        "why": "Showdown uses Return while Castform is still Water, then ends Rain at upkeep. The engine branch changes Castform to Normal before Damage, raising Return from the Water-state max 73 to the Normal-state max 109.",
        "prediction": "confirmed H-B",
        "follow_up_locus": "third_party/poke-engine-src/src/gen3/abilities.rs update_forecast plus generate_instructions.rs weather-expiry call ordering; preserve move-time type until residual expiry",
    },
    (2401002, 8): {
        "verdict": "engine odd-base-power modifier rounding defect",
        "lane": "engine; Thick Fat fixed-point base-power follow-up",
        "why": "Pelipper's Ice Beam into Thick Fat carries 47.5 base power in Rust instead of fixed-point 47. The fixed-point 39..47 set admits observed 39.",
        "prediction": "partial M5",
        "follow_up_locus": "third_party/poke-engine-src/src/gen3/abilities.rs ability_modify_attack_against: Thick Fat",
    },
    (2401127, 54): {
        "verdict": "WHAT-level dynamic type-effect candidate",
        "lane": "engine candidate; no patch locus licensed",
        "why": "The retained branch emits Color Change before Return, but the observed 45 remains outside the branch's damage support. The evidence rejects a simple event-order narration without identifying the remaining type-effect calculation locus.",
        "prediction": "partial H-B",
    },
    (2401237, 14): {
        "verdict": "engine odd-base-power modifier rounding defect",
        "lane": "engine; Thick Fat fixed-point base-power follow-up",
        "why": "The switch changes the target to Dewgong, whose Thick Fat halves Fire Punch. Rust carries 37.5 base power instead of fixed-point 37; the post-switch fixed-point 21..25 set admits observed 21.",
        "prediction": "partial M5",
        "follow_up_locus": "third_party/poke-engine-src/src/gen3/abilities.rs ability_modify_attack_against: Thick Fat",
    },
    (2500120, 60): {
        "verdict": "misbucketed switch-in magnitude",
        "lane": "instrument classification; underlying magnitude remains WHAT-level",
        "why": "Grumpig switches in before Flamethrower; the only status event is a possible burn after damage. This row does not contain a pre-hit boost/status event and cannot support the stat-gap WHY.",
        "prediction": "refuted H-D2",
    },
    (2500151, 116): {
        "verdict": "engine odd-base-power modifier rounding defect",
        "lane": "engine; Thick Fat fixed-point base-power follow-up",
        "why": "Dewgong waking does not affect Togetic's Flamethrower. Thick Fat carries 47.5 base power in Rust instead of fixed-point 47; the fixed-point 32..38 set admits observed 32.",
        "prediction": "partial M5",
        "follow_up_locus": "third_party/poke-engine-src/src/gen3/abilities.rs ability_modify_attack_against: Thick Fat",
    },
    (2500297, 96): {
        "verdict": "engine odd-base-power modifier rounding defect",
        "lane": "engine; Thick Fat fixed-point base-power follow-up",
        "why": "Walrein's Ice Beam into Thick Fat carries 47.5 base power in Rust instead of fixed-point 47. The fixed-point STAB set 41..49 admits observed 41.",
        "prediction": "partial M5",
        "follow_up_locus": "third_party/poke-engine-src/src/gen3/abilities.rs ability_modify_attack_against: Thick Fat",
    },
    (2500576, 7): {
        "verdict": "WHAT-level direct-damage candidate",
        "lane": "engine candidate; no patch locus licensed",
        "why": "Ice Punch's observed 32 is below the engine legal 33..39 range in a no-switch, no-event state. The sample supports a magnitude discrepancy but not a specific pipeline step.",
        "prediction": "partial H-B",
    },
    (2501061, 96): {
        "verdict": "engine odd-base-power modifier rounding defect",
        "lane": "engine; Thick Fat fixed-point base-power follow-up",
        "why": "Snorlax waking does not change Thick Fat. Walrein's Ice Beam carries 47.5 base power in Rust instead of fixed-point 47; the fixed-point STAB set 41..49 admits observed 41.",
        "prediction": "partial M5",
        "follow_up_locus": "third_party/poke-engine-src/src/gen3/abilities.rs ability_modify_attack_against: Thick Fat",
    },
    (2600362, 82): {
        "verdict": "legal-roll matcher accounting",
        "lane": "instrument",
        "why": "Observed Knock Off damage 21 is in the engine's enumerated 20..24 legal set. The divergence is matcher accounting, not an engine damage error.",
        "prediction": "confirmed H-A",
        "follow_up_locus": "scripts/engine_transition_differential.py evaluate_boundary_strict pre-state legal-roll construction -> roll_components_agree; derive post-switch branch legality",
    },
    (2600510, 111): {
        "verdict": "engine odd-base-power modifier rounding defect",
        "lane": "engine; Thick Fat fixed-point base-power follow-up",
        "why": "Suicune's +4 SpA is already represented. Thick Fat carries Ice Beam's 47.5 base power in Rust instead of fixed-point 47; the fixed-point 67..79 set admits observed 67.",
        "prediction": "partial M5",
        "follow_up_locus": "third_party/poke-engine-src/src/gen3/abilities.rs ability_modify_attack_against: Thick Fat",
    },
    (2600535, 80): {
        "verdict": "documented Substitute-health comparison limit",
        "lane": "world/comparison limit; no engine patch",
        "why": "Public state does not expose remaining Substitute HP, so the sweep deliberately materializes Suicune's sub at the fresh maxhp/4 upper bound (67). Showdown both breaks the real sub and heals Parasect for 24, while engine branches at the approximated 67 HP either deal 57/heal 28 without breaking or 67/heal 33 and break. The mismatch is the documented hidden-state approximation, not a damage formula defect.",
        "prediction": "refuted M2/M4/M5",
        "follow_up_locus": "src/pokezero/engine_world.py _build_side_spec substitute_health = maxhp // 4; comparison-limit lane only",
    },
    (2600546, 22): {
        "verdict": "engine odd-base-power modifier rounding defect",
        "lane": "engine; Thick Fat fixed-point base-power follow-up",
        "why": "The switch-in freeze is after Ice Beam and noncausal. Hariyama's Thick Fat carries 47.5 base power in Rust instead of fixed-point 47; the post-switch fixed-point STAB set 51..61 admits observed 51.",
        "prediction": "partial S3",
        "follow_up_locus": "third_party/poke-engine-src/src/gen3/abilities.rs ability_modify_attack_against: Thick Fat",
    },
    (2600546, 25): {
        "verdict": "engine odd-base-power modifier rounding defect",
        "lane": "engine; Thick Fat fixed-point base-power follow-up",
        "why": "Piloswine's Ice Beam into Hariyama's Thick Fat carries 47.5 base power in Rust instead of fixed-point 47. The fixed-point STAB set 46..55 admits observed 46.",
        "prediction": "partial M5",
        "follow_up_locus": "third_party/poke-engine-src/src/gen3/abilities.rs ability_modify_attack_against: Thick Fat",
    },
    (2600657, 49): {
        "verdict": "misbucketed static magnitude",
        "lane": "engine candidate; no patch locus licensed",
        "why": "The protocol has Flamethrower followed by Dragon Claw with no boost, status, switch, or type event before the observed damage 41. It cannot support the stat-event hypothesis.",
        "prediction": "refuted H-D2",
    },
    (2600992, 21): {
        "verdict": "engine odd-base-power modifier rounding defect",
        "lane": "engine; Thick Fat fixed-point base-power follow-up",
        "why": "Glalie's Ice Beam is quarter-effective into Water/Ice Walrein. Thick Fat carries 47.5 base power in Rust instead of fixed-point 47; the fixed-point STAB/type set 9..11 admits observed 9.",
        "prediction": "partial M5",
        "follow_up_locus": "third_party/poke-engine-src/src/gen3/abilities.rs ability_modify_attack_against: Thick Fat",
    },
    (2601033, 129): {
        "verdict": "engine odd-base-power modifier rounding defect",
        "lane": "engine; Thick Fat fixed-point base-power follow-up",
        "why": "Whiscash's Ice Beam into Thick Fat carries 47.5 base power in Rust instead of fixed-point 47. The fixed-point 33..39 set admits observed 33.",
        "prediction": "partial M5",
        "follow_up_locus": "third_party/poke-engine-src/src/gen3/abilities.rs ability_modify_attack_against: Thick Fat",
    },
    (2601196, 25): {
        "verdict": "engine odd-base-power modifier rounding defect",
        "lane": "engine; Thick Fat fixed-point base-power follow-up",
        "why": "Porygon2 copies Walrein's Thick Fat on switch before Ice Beam. Rust then carries 47.5 base power instead of fixed-point 47; the fixed-point STAB set 41..49 admits observed 41.",
        "prediction": "partial M5",
        "follow_up_locus": "third_party/poke-engine-src/src/gen3/abilities.rs ability_modify_attack_against: Thick Fat",
    },
    (2601196, 46): {
        "verdict": "WHAT-level direct-damage candidate",
        "lane": "engine candidate; no patch locus licensed",
        "why": "Walrein's observed 6 from Ice Beam is outside the engine's 7..18 set, while the opposite Surf damage is legal. The sample is a localized magnitude candidate, not a family-wide WHY.",
        "prediction": "partial H-B",
    },
    (2700218, 151): {
        "verdict": "engine odd-base-power modifier rounding defect",
        "lane": "engine; Thick Fat fixed-point base-power follow-up",
        "why": "Jirachi's +5 SpA is already represented. Fire Punch into Thick Fat carries 37.5 base power in Rust instead of fixed-point 37; the fixed-point 85..101 set admits observed 85.",
        "prediction": "partial M5",
        "follow_up_locus": "third_party/poke-engine-src/src/gen3/abilities.rs ability_modify_attack_against: Thick Fat",
    },
    (2700355, 37): {
        "verdict": "engine odd-base-power modifier rounding defect",
        "lane": "engine; Thick Fat fixed-point base-power follow-up",
        "why": "Mew's Flamethrower into Grumpig's Thick Fat carries 47.5 base power in Rust instead of fixed-point 47. The fixed-point 20..24 set admits observed 20.",
        "prediction": "partial M5",
        "follow_up_locus": "third_party/poke-engine-src/src/gen3/abilities.rs ability_modify_attack_against: Thick Fat",
    },
}

STILL_WHAT = {
    (2000261, 31),
    (2000298, 23),
    (2000431, 32),
    (2000561, 67),
    (2100079, 7),
    (2400156, 29),
    (2401127, 54),
    (2500120, 60),
    (2500576, 7),
    (2600657, 49),
    (2601196, 46),
}
COMPARISON_LIMITS = {(2300040, 84), (2600535, 80)}
INSTRUMENT_DEFECTS = {(2300552, 117), (2600362, 82)}

_EVENT_PREFIXES = (
    "|move|", "|switch|", "|replace|", "|-boost|", "|-unboost|", "|-damage|", "|-heal|",
    "|-start|", "|-end|", "|-activate|", "|-weather|", "|-ability|", "|-item|", "|-status|",
    "|-curestatus|", "|-immune|", "|-fail|", "|-prepare|", "|-crit|",
)


# actor=1 means side one's move damages side two; actor=2 is the reverse.
# Every case was registered before replay as a M5/S3 alternative rather than
# inferred from its family name. The retained branch/source read established
# the shared odd-base-power fixed-point mechanism.
_BASE_POWER_PROBES: dict[tuple[int, int], dict[str, Any]] = {
    (2001162, 120): {"actor": 2, "move": "surf", "type": "WATER", "base_power": 95, "modifier": 1.5, "observed": 132, "switch_index": 3},
    (2100482, 83): {"actor": 2, "move": "surf", "type": "WATER", "base_power": 95, "modifier": 1.5, "observed": 102, "switch_index": 0},
    (2200369, 75): {"actor": 2, "move": "flamethrower", "type": "FIRE", "base_power": 95, "modifier": 0.5, "observed": 49},
    (2300154, 80): {"actor": 1, "move": "flamethrower", "type": "FIRE", "base_power": 95, "modifier": 0.5, "observed": 32},
    (2400140, 9): {"actor": 2, "move": "flamethrower", "type": "FIRE", "base_power": 95, "modifier": 0.5, "observed": 43},
    (2400172, 89): {"actor": 2, "move": "firepunch", "type": "FIRE", "base_power": 75, "modifier": 0.5, "observed": 86},
    (2400342, 78): {"actor": 2, "move": "icepunch", "type": "ICE", "base_power": 75, "modifier": 0.5, "observed": 92},
    (2401002, 8): {"actor": 1, "move": "icebeam", "type": "ICE", "base_power": 95, "modifier": 0.5, "observed": 39},
    (2401237, 14): {"actor": 2, "move": "firepunch", "type": "FIRE", "base_power": 75, "modifier": 0.5, "observed": 21, "switch_index": 5},
    (2500151, 116): {"actor": 2, "move": "flamethrower", "type": "FIRE", "base_power": 95, "modifier": 0.5, "observed": 32},
    (2500297, 96): {"actor": 1, "move": "icebeam", "type": "ICE", "base_power": 95, "modifier": 0.5, "observed": 41},
    (2501061, 96): {"actor": 1, "move": "icebeam", "type": "ICE", "base_power": 95, "modifier": 0.5, "observed": 41},
    (2600510, 111): {"actor": 1, "move": "icebeam", "type": "ICE", "base_power": 95, "modifier": 0.5, "observed": 67},
    (2600546, 22): {"actor": 2, "move": "icebeam", "type": "ICE", "base_power": 95, "modifier": 0.5, "observed": 51, "switch_index": 2},
    (2600546, 25): {"actor": 2, "move": "icebeam", "type": "ICE", "base_power": 95, "modifier": 0.5, "observed": 46},
    (2600992, 21): {"actor": 2, "move": "icebeam", "type": "ICE", "base_power": 95, "modifier": 0.5, "observed": 9, "effectiveness": 0.25},
    (2601033, 129): {"actor": 1, "move": "icebeam", "type": "ICE", "base_power": 95, "modifier": 0.5, "observed": 33},
    (2601196, 25): {"actor": 2, "move": "icebeam", "type": "ICE", "base_power": 95, "modifier": 0.5, "observed": 41, "switch_index": 5, "trace_thick_fat": True},
    (2700218, 151): {"actor": 1, "move": "firepunch", "type": "FIRE", "base_power": 75, "modifier": 0.5, "observed": 85},
    (2700355, 37): {"actor": 2, "move": "flamethrower", "type": "FIRE", "base_power": 95, "modifier": 0.5, "observed": 20},
}


def _target_plan(
    prediction: Mapping[str, Any], remainder_prediction: Mapping[str, Any]
) -> tuple[dict[tuple[int, int], str], set[tuple[int, int]], set[tuple[int, int]]]:
    targets: dict[tuple[int, int], str] = {}
    initial: set[tuple[int, int]] = set()
    for family, rows in (prediction.get("sample_plan") or {}).items():
        if family not in FAMILIES or not isinstance(rows, list):
            continue
        for row in rows:
            key = (int(row["seed"]), int(row["step"]))
            targets[key] = family
            initial.add(key)
    remainder: set[tuple[int, int]] = set()
    for row in remainder_prediction.get("predictions") or []:
        family = str(row["family"])
        if family not in FAMILIES:
            raise ValueError(f"unexpected remainder family: {family}")
        key = (int(row["seed"]), int(row["step"]))
        if key in targets:
            raise ValueError(f"remainder duplicates initial sample: {key}")
        targets[key] = family
        remainder.add(key)
    return targets, initial, remainder


def _state_with_side_one_active(raw: str, active_index: int) -> Any:
    chunks = raw.split("/")
    side = chunks[0].split("=")
    side[6] = str(active_index)
    chunks[0] = "=".join(side)
    return poke_engine.State.from_string("/".join(chunks))


def _replace_mon_field(raw: str, species: str, field: int, value: str) -> str:
    start = raw.index(f"{species},")
    end = raw.index("=", start)
    fields = raw[start:end].split(",")
    fields[field] = value
    return raw[:start] + ",".join(fields) + raw[end:]


def _fixed_point_base_power_probe(row: Mapping[str, Any]) -> dict[str, Any] | None:
    key = (int(row["seed"]), int(row["step"]))
    probe = _BASE_POWER_PROBES.get(key)
    if probe is None:
        return None

    raw = (row.get("engine_states") or [row["engine_state"]])[0]
    switch_index = probe.get("switch_index")
    state = (
        _state_with_side_one_active(raw, int(switch_index))
        if switch_index is not None
        else poke_engine.State.from_string(raw)
    )
    actor = int(probe["actor"])
    attacking_side = state.side_one if actor == 1 else state.side_two
    defending_side = state.side_two if actor == 1 else state.side_one
    attacker = attacking_side.pokemon[int(str(attacking_side.active_index))]
    defender = defending_side.pokemon[int(str(defending_side.active_index))]

    # Trace copies Thick Fat on switch before damage. The active-index control
    # bypasses switch instructions, so restore that one observed branch fact.
    if probe.get("trace_thick_fat"):
        adjusted = _replace_mon_field(state.to_string(), str(defender.id), 8, "THICKFAT")
        state = poke_engine.State.from_string(adjusted)
        defending_side = state.side_two if actor == 1 else state.side_one
        defender = defending_side.pokemon[int(str(defending_side.active_index))]

    side_one_choice, side_two_choice = _engine_choices(row)
    if switch_index is not None:
        switched = state.side_one.pokemon[int(str(state.side_one.active_index))]
        side_one_choice = str(switched.moves[0].id).lower()
    if actor == 1:
        side_one_choice = str(probe["move"])
    else:
        side_two_choice = str(probe["move"])
    side_one_bases, side_two_bases = poke_engine.calculate_damage(
        state, side_one_choice, side_two_choice, False
    )
    actor_bases = side_one_bases if actor == 1 else side_two_bases
    current_values = sorted(legal_roll_damages([int(value) for value in actor_bases]))

    if str(state.weather).upper() != "NONE":
        raise AssertionError(f"unexpected weather in fixed-point control for {key}")
    if int(defending_side.side_conditions.light_screen):
        raise AssertionError(f"unexpected Light Screen in fixed-point control for {key}")
    fixed_values = list(
        gen3_damage_rolls(
            Gen3DamageContext(
                level=int(attacker.level),
                base_power=int(probe["base_power"]),
                category="Special",
                attack=int(attacker.special_attack),
                defense=int(defender.special_defense),
                attack_boost=int(attacking_side.special_attack_boost),
                defense_boost=int(defending_side.special_defense_boost),
                base_power_mods=((float(probe["modifier"]), 1),),
                stab=str(probe["type"]) in {str(value) for value in attacker.types},
                effectiveness=float(probe.get("effectiveness", 1.0)),
            )
        )
    )
    observed = int(probe["observed"])
    if observed in current_values:
        raise AssertionError(f"expected current engine legal set to reject {key}: {observed}")
    if observed not in fixed_values:
        raise AssertionError(f"expected fixed-point legal set to admit {key}: {observed}")
    return {
        "move": probe["move"],
        "actor": actor,
        "attacker": str(attacker.id),
        "defender": str(defender.id),
        "observed": observed,
        "current_engine_legal_rolls": current_values,
        "fixed_point_legal_rolls": fixed_values,
        "observed_in_current_engine_set": False,
        "observed_in_fixed_point_set": True,
        "base_power": int(probe["base_power"]),
        "modifier": float(probe["modifier"]),
        "rust_float_base_power": float(probe["base_power"]) * float(probe["modifier"]),
        "showdown_fixed_point_base_power": apply_chain(
            int(probe["base_power"]), ((float(probe["modifier"]), 1),)
        ),
        "source_loci": [
            "third_party/poke-engine-src/src/gen3/abilities.rs",
            "pokemon-showdown/data/mods/gen4/abilities.ts (inherited by gen3)",
        ],
    }


def _substitute_health_probe(row: Mapping[str, Any]) -> dict[str, Any] | None:
    key = (int(row["seed"]), int(row["step"]))
    if key != (2600535, 80):
        return None
    state = poke_engine.State.from_string((row.get("engine_states") or [row["engine_state"]])[0])
    branches = list(poke_engine.generate_instructions(state, *_engine_choices(row)))
    return {
        "materialized_substitute_health": int(state.side_one.substitute_health),
        "materialized_active_maxhp": int(
            state.side_one.pokemon[int(str(state.side_one.active_index))].maxhp
        ),
        "materializer_rule": "fresh maxhp/4 upper bound because remaining Substitute HP is not public",
        "observed_protocol": [
            line
            for line in row.get("protocol") or []
            if line.startswith(("|-end|", "|-heal|"))
        ],
        "engine_substitute_and_drain_branches": [
            str(branch.instruction_list)
            for branch in branches
            if "DamageSubstitute" in str(branch.instruction_list)
        ],
        "source_locus": "src/pokezero/engine_world.py approximate_substitute_health",
    }


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
    fixed_point_probe = _fixed_point_base_power_probe(row)
    adjudication = dict(ADJUDICATIONS[key])
    if fixed_point_probe is not None:
        # The remainder rubric only defined "partial" for an unresolved WHAT or
        # a mechanism without an exact locus. These rows resolved an alternative
        # hypothesis more strongly than that, while the first-listed hypothesis
        # was not established. Preserve the frozen rubric by leaving them
        # unscored instead of silently widening "partial" after measurement.
        hypothesis = str(adjudication["prediction"]).split()[-1]
        adjudication["prediction"] = f"unscored resolution-gap {hypothesis}"
    if key in STILL_WHAT:
        adjudication["why_status"] = "still_WHAT"
    elif key in COMPARISON_LIMITS:
        adjudication["why_status"] = "comparison_limit"
    elif key in INSTRUMENT_DEFECTS:
        adjudication["why_status"] = "confirmed_instrument_defect"
    else:
        adjudication["why_status"] = "confirmed_engine_defect"
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
        "fixed_point_base_power_probe": fixed_point_probe,
        "substitute_health_probe": _substitute_health_probe(row),
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
        "adjudication": adjudication,
    }


def _prediction_score(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[str]] = {
        "confirmed": [],
        "partial": [],
        "refuted": [],
        "unscored_resolution_gap": [],
    }
    for row in rows:
        label = str(row["adjudication"]["prediction"])
        if label.startswith("confirmed"):
            bucket = "confirmed"
        elif label.startswith("partial"):
            bucket = "partial"
        elif label.startswith("refuted"):
            bucket = "refuted"
        elif label.startswith("unscored resolution-gap"):
            bucket = "unscored_resolution_gap"
        else:
            raise ValueError(f"unknown prediction outcome: {label}")
        buckets[bucket].append(f"{row['seed']}/{row['step']}")
    scored_rows = sum(len(buckets[name]) for name in ("confirmed", "partial", "refuted"))
    return {"registered_rows": len(rows), "scored_rows": scored_rows, **buckets}


def _markdown(result: Mapping[str, Any]) -> str:
    score = result["prediction_score"]
    registrations = result["prediction_score_by_registration"]
    initial = registrations["initial_16"]
    remainder = registrations["remaining_21"]
    unresolved = [
        f"{row['seed']}/{row['step']}"
        for row in result["rows"]
        if row["adjudication"]["why_status"] == "still_WHAT"
    ]
    lines = [
        "# C15 WHY adjudication: full magnitude and same-turn-stat population",
        "",
        "## Scope",
        "",
        "This is a replay-first adjudication of all 37 rows in the two registered C15 populations: "
        "28 `CAND_unresolved_magnitude` rows and 9 `CAND_same_turn_stat_event_gap` rows. "
        "The original 16-row sample and its refutations are preserved; the exact 21-row complement "
        "was separately preregistered before any remaining repro was opened. This artifact does not "
        "relabel the certification sweep or modify the living ledger.",
        "",
        "**Coverage: 37/37 (100%).**",
        "",
        "## Prediction Score",
        "",
        f"- Rule-scored rows: {score['scored_rows']}/{score['registered_rows']}.",
        f"- Confirmed: {len(score['confirmed'])} ({', '.join(score['confirmed']) or 'none'})",
        f"- Partially supported: {len(score['partial'])} ({', '.join(score['partial']) or 'none'})",
        f"- Refuted: {len(score['refuted'])} ({', '.join(score['refuted']) or 'none'})",
        f"- Unscored resolution gap: {len(score['unscored_resolution_gap'])} "
        f"({', '.join(score['unscored_resolution_gap']) or 'none'}).",
        "",
        f"- Preserved initial score: {len(initial['confirmed'])} confirmed, "
        f"{len(initial['partial'])} partial, {len(initial['refuted'])} refuted, "
        f"{len(initial['unscored_resolution_gap'])} unscored.",
        f"- Preregistered remainder: {remainder['scored_rows']}/{remainder['registered_rows']} "
        f"rule-scored ({len(remainder['confirmed'])} confirmed, "
        f"{len(remainder['partial'])} partial, {len(remainder['refuted'])} refuted); "
        f"{len(remainder['unscored_resolution_gap'])} unscored resolution gaps.",
        "",
        "The frozen remainder rubric did not define a score for an alternative hypothesis that "
        "resolved to a confirmed defect with an exact locus: it allowed `partial` only while the "
        "row remained WHAT-level or lacked a locus, and `confirmed` only for the first-listed "
        "mechanism. The 20 fixed-point rows are therefore reported as unscored rather than "
        "retroactively widening the rubric.",
        "",
        "The first sample's broad same-turn pre-state-stat hypothesis remains refuted. The remainder "
        "instead found one shared source mechanism across 20 rows: odd base powers modified by "
        "Torrent or Thick Fat are carried as `.5` floats in Rust, while Showdown's inherited "
        "`chainModify` rounds half-down before the damage formula.",
        "",
        "## Population Readout",
        "",
        f"- Confirmed engine defects: {result['why_status_counts']['confirmed_engine_defect']}/37.",
        f"- Confirmed instrument defects: {result['why_status_counts']['confirmed_instrument_defect']}/37.",
        f"- Documented comparison limits: {result['why_status_counts']['comparison_limit']}/37.",
        f"- Still WHAT-level: {result['why_status_counts']['still_WHAT']}/37 "
        f"({', '.join(unresolved)}).",
        "",
        "## Per-Row Verdicts",
        "",
        "| Family | Row | WHY status | Verdict | Lane | Prediction |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in result["rows"]:
        adjudication = row["adjudication"]
        lines.append(
            f"| {row['family']} | {row['seed']}/{row['step']} | {adjudication['why_status']} | "
            f"{adjudication['verdict']} | "
            f"{adjudication['lane']} | {adjudication['prediction']} |"
        )
    lines.extend(["", "## Generalization Boundary", ""])
    lines.extend([
        "- The odd-base-power finding generalizes only to the 20 replayed rows whose exact fixed-point control admits the observation while the current engine set rejects it. It does not absorb the other 17 rows.",
        "- `CAND_same_turn_stat_event_gap` is not a mechanism: its rows include fixed-point base-power defects, an event-aware matcher omission, post-boost WHAT candidates, and noncausal/misbucketed events.",
        "- `CAND_unresolved_magnitude` is also mixed: switch comparison limits, capped residuals, hidden Substitute HP, dynamic HP/weather timing, fixed-point base-power defects, matcher accounting, and 11 still-WHAT rows.",
        "- `2600535/80` is a comparison limit. Public state omits remaining Substitute HP; the documented maxhp/4 materialization cannot reproduce both the observed sub break and drain heal.",
        "- No row overlaps patches 42-44 or active world-lane rows at the same `(seed, step)`; the shared seed 2000431 is explicitly recorded as a different step.",
        "",
        "## Banked Follow-Ups",
        "",
        "- Engine fixed-point: replace Torrent's floating `*= 1.5` in `abilities.rs::ability_modify_attack_being_used` and Thick Fat's `/= 2.0` in `abilities.rs::ability_modify_attack_against`; audit sibling Blaze, Overgrow, and Swarm arms.",
        "- Engine timing: inspect `generate_instructions.rs::before_move -> choice_effects::modify_choice` for Flail/Reversal BP after earlier same-turn damage (`2201005/55`), and `abilities.rs::update_forecast` plus the weather-expiry call ordering for `2400451/56`.",
        "- Instrument: in `engine_transition_differential.py::evaluate_boundary_strict -> roll_components_agree`, derive event-aware legal rolls after same-turn stat changes (`2300552/117`) and post-switch branch legality (`2600362/82`).",
        "- Comparison limits: keep capped residual handling in `roll_components_agree` (`2300040/84`) and `_build_side_spec`'s `substitute_health = maxhp // 4` approximation (`2600535/80`) explicit; do not turn hidden Substitute HP into a deterministic engine patch.",
        "- Remaining WHY: carry the 11 exact unresolved identities above into a focused source lane rather than inferring from family or ratio.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "PYTHONPATH=src:scripts .venv/bin/python scripts/c15_why_adjudication.py \\",
        "  --archive <retained-sweep-archive> \\",
        "  --prediction reports/c15_why_magnitude_statgap_predictions.json \\",
        "  --remainder-prediction reports/c15_why_magnitude_statgap_remainder_predictions.json \\",
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
    parser.add_argument("--remainder-prediction", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args(argv)

    prediction = json.loads(args.prediction.read_text())
    remainder_prediction = json.loads(args.remainder_prediction.read_text())
    targets, initial_keys, remainder_keys = _target_plan(prediction, remainder_prediction)
    if len(initial_keys) != 16:
        raise ValueError(f"expected the pre-registered 16-row sample plan, found {len(initial_keys)}")
    if len(remainder_keys) != 21:
        raise ValueError(f"expected the pre-registered 21-row remainder, found {len(remainder_keys)}")
    if len(targets) != 37 or len(ADJUDICATIONS) != 37:
        raise ValueError(
            f"expected complete 37-row population/adjudication, found {len(targets)}/{len(ADJUDICATIONS)}"
        )
    expected_family_counts = {
        "CAND_unresolved_magnitude": 28,
        "CAND_same_turn_stat_event_gap": 9,
    }
    actual_family_counts = {
        family: sum(target == family for target in targets.values()) for family in FAMILIES
    }
    if actual_family_counts != expected_family_counts:
        raise ValueError(f"unexpected family population: {actual_family_counts}")
    rows_by_key = _load_rows(args.archive, targets)
    rows = [_row_evidence(rows_by_key[key], targets[key]) for key in sorted(targets)]
    rows_by_identity = {(int(row["seed"]), int(row["step"])): row for row in rows}
    initial_rows = [rows_by_identity[key] for key in sorted(initial_keys)]
    remainder_rows = [rows_by_identity[key] for key in sorted(remainder_keys)]
    why_status_counts = {
        status: sum(row["adjudication"]["why_status"] == status for row in rows)
        for status in (
            "confirmed_engine_defect",
            "confirmed_instrument_defect",
            "comparison_limit",
            "still_WHAT",
        )
    }
    result = {
        "schema": "c15-why-adjudication/3",
        "registered_before_measurement": {
            "initial_16": bool(prediction.get("registered_before_measurement")),
            "remaining_21": bool(
                remainder_prediction.get("registered_before_remaining_measurement")
            ),
        },
        "population_count": len(rows),
        "coverage": "37/37",
        "family_counts": actual_family_counts,
        "why_status_counts": why_status_counts,
        "prediction_score": _prediction_score(rows),
        "prediction_score_by_registration": {
            "initial_16": _prediction_score(initial_rows),
            "remaining_21": _prediction_score(remainder_rows),
        },
        "rows": rows,
    }
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    args.out_md.write_text(_markdown(result))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
