#!/usr/bin/env python
"""Showdown ground-truth gate for the gen3 switch-out / Protect fidelity pins.

Companion to ``scripts/rapidspin_differential.py``. Where that script drives BOTH
the sim and the patched ``poke_engine`` Python wheel, this one records only the
**Showdown** half: the engine half is asserted natively by
``rust/pokezero-search/tests/gen3_switch_fidelity.rs`` (``cargo test --test
gen3_switch_fidelity``), which reads the same vendored, gen3-patched engine the
search crate links and therefore needs no wheel build. Run both to close the loop.

Scenarios (all gen3 Custom Game, real Node sim via ``pokezero.showdown_fixture``):

  spinprotect   : Rapid Spin into Protect -> hazards STAY (no ``-sideend``).
  spinconnect   : Rapid Spin connecting  -> hazards CLEARED (``-sideend`` Spikes).
  batonpass     : Perish Song then Baton Pass -> the count rides the pass and the
                  RECEIVER hits ``perish0`` and faints. This is the divergence
                  ``third_party/poke-engine-gen3-batonpass-perish.patch`` fixes:
                  upstream retained only Substitute and Leech Seed across a pass.
  batonpasscontrol : same line without Perish Song -> the receiver survives.
  leechseed     : a seeded Pokemon switches out -> Leech Seed stops ticking
                  (``Pokemon.clearVolatile()``); the no-switch control keeps ticking.
  partialtrap   : the TRAPPER switches out -> the victim stops taking partial-trap
                  damage (``partiallytrapped.onResidual`` frees it once the source
                  is no longer active); the no-switch control keeps taking it.

``leechseed`` and ``partialtrap`` depend on a 90%/85% accurate SETUP move, so they
only assert on seeds where the setup actually landed and require at least one such
seed. Everything else is deterministic.

Usage:
    .venv/bin/python scripts/gen3_switch_differential.py \
        --showdown-root /Users/scott/workspace/pokerena/vendor/pokemon-showdown
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokezero.local_showdown import LocalShowdownConfig
from pokezero.showdown_fixture import FixturePokemon, run_multi_turn_fixture


# --- curated gen3 Custom Game sets ------------------------------------------

def _forretress():  # spinner
    return FixturePokemon(species="Forretress", ability="Sturdy", item="Leftovers",
                          moves=("Rapid Spin", "Spikes", "Toxic", "Protect"))


def _skarmory():  # hazard setter / Protect user
    return FixturePokemon(species="Skarmory", ability="Keen Eye", item="Leftovers",
                          moves=("Spikes", "Protect", "Toxic", "Drill Peck"))


def _smeargle():  # Baton Pass + Perish Song carrier
    return FixturePokemon(species="Smeargle", ability="Technician", item="None",
                          moves=("Baton Pass", "Perish Song", "Splash"))


def _snorlax():  # Baton Pass receiver
    return FixturePokemon(species="Snorlax", ability="Immunity", item="None",
                          moves=("Splash", "Body Slam"))


def _misdreavus():  # inert opponent (Levitate, no residual interference)
    return FixturePokemon(species="Misdreavus", ability="Levitate", item="None",
                          moves=("Splash", "Confuse Ray"))


def _cacturne():  # Leech Seed setter
    return FixturePokemon(species="Cacturne", ability="Sand Veil", item="None",
                          moves=("Leech Seed", "Splash"))


def _ninetales():  # partial-trap setter
    return FixturePokemon(species="Ninetales", ability="Flash Fire", item="None",
                          moves=("Fire Spin", "Splash"))


def _blissey():  # trap/seed victim
    return FixturePokemon(species="Blissey", ability="Natural Cure", item="None",
                          moves=("Splash", "Soft-Boiled"))


def _has(lines, needle: str) -> bool:
    return any(needle in line for line in lines)


def _spikes_cleared(lines) -> bool:
    """True if the SPINNER's own Spikes were removed (``-sideend`` on p2's side)."""
    return any(
        line.startswith("|-sideend|p2") and "Spikes" in line for line in lines
    )


def _residual_from(lines, source: str, seat: str) -> bool:
    """True if `seat` took residual damage attributed to `source` in `lines`."""
    return any(
        line.startswith(f"|-damage|{seat}") and "[from]" in line and source in line
        for line in lines
    )


# --- scenario specs ---------------------------------------------------------
# `expect` maps fact-name -> ground truth. `setup_step`/`setup_landed` gate the
# scenarios whose setup move can miss.

def _spec(name):
    if name == "spinprotect":
        # p1 Skarmory lays Spikes on p2's side; p2 Forretress spins into Protect,
        # so its OWN Spikes must stay.
        return dict(
            p1=[_skarmory()], p2=[_forretress()],
            turns=[("move spikes", "move toxic"), ("move protect", "move rapidspin")],
            measured=1, setup_step=None, setup_landed=None,
            facts=lambda L: {"spikes_cleared": _spikes_cleared(L)},
            expect={"spikes_cleared": False},
            landmark=lambda L: _has(L, "|-activate|") and _has(L, "Protect"),
            landmark_desc="Protect activated")
    if name == "spinconnect":
        return dict(
            p1=[_skarmory()], p2=[_forretress()],
            turns=[("move spikes", "move toxic"), ("move toxic", "move rapidspin")],
            measured=1, setup_step=None, setup_landed=None,
            facts=lambda L: {"spikes_cleared": _spikes_cleared(L)},
            expect={"spikes_cleared": True},
            landmark=lambda L: _has(L, "|move|p2a: Forretress|Rapid Spin"),
            landmark_desc="Rapid Spin used")
    if name == "batonpass":
        # Smeargle sings Perish Song, then Baton Passes into Snorlax. Counts down
        # on the RECEIVER: perish0 + faint four boundaries later.
        return dict(
            p1=[_smeargle(), _snorlax()], p2=[_misdreavus()],
            turns=[("move perishsong", "move splash"), ("move batonpass", "move splash"),
                   ("switch 2", None), ("move splash", "move splash"),
                   ("move splash", "move splash"), ("move splash", "move splash")],
            measured=None, setup_step=None, setup_landed=None,
            facts=lambda L: {
                "receiver_perished": _has(L, "|-start|p1a: Snorlax|perish0")
                                     and _has(L, "|faint|p1a: Snorlax"),
            },
            expect={"receiver_perished": True},
            landmark=lambda L: _has(L, "[from] Baton Pass"),
            landmark_desc="Baton Pass resolved")
    if name == "batonpasscontrol":
        return dict(
            p1=[_smeargle(), _snorlax()], p2=[_misdreavus()],
            turns=[("move splash", "move splash"), ("move batonpass", "move splash"),
                   ("switch 2", None), ("move splash", "move splash"),
                   ("move splash", "move splash"), ("move splash", "move splash")],
            measured=None, setup_step=None, setup_landed=None,
            facts=lambda L: {
                "receiver_perished": _has(L, "|-start|p1a: Snorlax|perish0")
                                     and _has(L, "|faint|p1a: Snorlax"),
            },
            expect={"receiver_perished": False},
            landmark=lambda L: _has(L, "[from] Baton Pass"),
            landmark_desc="Baton Pass resolved")
    if name == "leechseed":
        # p1 seeds p2's Blissey; p2 switches out on the measured turn -> no tick.
        return dict(
            p1=[_cacturne()], p2=[_blissey(), _misdreavus()],
            turns=[("move leechseed", "move splash"), ("move splash", "switch 2")],
            measured=1, setup_step=0,
            setup_landed=lambda L: _has(L, "|-start|") and _has(L, "move: Leech Seed")
                                   and not _has(L, "[miss]"),
            facts=lambda L: {"victim_ticked": _residual_from(L, "Leech Seed", "p2a")},
            expect={"victim_ticked": False},
            landmark=lambda L: True, landmark_desc="")
    if name == "leechseedcontrol":
        return dict(
            p1=[_cacturne()], p2=[_blissey(), _misdreavus()],
            turns=[("move leechseed", "move splash"), ("move splash", "move splash")],
            measured=1, setup_step=0,
            setup_landed=lambda L: _has(L, "|-start|") and _has(L, "move: Leech Seed")
                                   and not _has(L, "[miss]"),
            facts=lambda L: {"victim_ticked": _residual_from(L, "Leech Seed", "p2a")},
            expect={"victim_ticked": True},
            landmark=lambda L: True, landmark_desc="")
    if name == "partialtrap":
        # p1 Ninetales traps p2's Blissey, then the TRAPPER leaves -> victim freed.
        return dict(
            p1=[_ninetales(), _cacturne()], p2=[_blissey()],
            turns=[("move firespin", "move splash"), ("switch 2", "move splash")],
            measured=1, setup_step=0,
            setup_landed=lambda L: _has(L, "|-activate|") and _has(L, "move: Fire Spin")
                                   and not _has(L, "[miss]"),
            facts=lambda L: {"victim_ticked": _residual_from(L, "Fire Spin", "p2a")},
            expect={"victim_ticked": False},
            landmark=lambda L: True, landmark_desc="")
    if name == "partialtrapcontrol":
        return dict(
            p1=[_ninetales(), _cacturne()], p2=[_blissey()],
            turns=[("move firespin", "move splash"), ("move splash", "move splash")],
            measured=1, setup_step=0,
            setup_landed=lambda L: _has(L, "|-activate|") and _has(L, "move: Fire Spin")
                                   and not _has(L, "[miss]"),
            facts=lambda L: {"victim_ticked": _residual_from(L, "Fire Spin", "p2a")},
            expect={"victim_ticked": True},
            landmark=lambda L: True, landmark_desc="")
    raise ValueError(name)


SCENARIOS = ("spinprotect", "spinconnect", "batonpass", "batonpasscontrol",
             "leechseed", "leechseedcontrol", "partialtrap", "partialtrapcontrol")


def run_scenario(name, seeds, config) -> tuple[bool, list[str]]:
    spec = _spec(name)
    notes: list[str] = []
    asserted = 0
    for seed in seeds:
        result = run_multi_turn_fixture(
            p1_team=spec["p1"], p2_team=spec["p2"], turns=spec["turns"],
            seed=seed, config=config,
        )
        steps = result.steps
        if spec["setup_step"] is not None:
            setup_lines = steps[spec["setup_step"]].protocol_lines
            if not spec["setup_landed"](setup_lines):
                notes.append(f"  seed {seed}: setup missed, skipped")
                continue
        if spec["measured"] is None:
            lines = [line for step in steps for line in step.protocol_lines]
        else:
            lines = list(steps[spec["measured"]].protocol_lines)
        if not spec["landmark"](lines):
            return False, notes + [
                f"  seed {seed}: landmark not observed ({spec['landmark_desc']})"
            ]
        facts = spec["facts"](lines)
        for key, want in spec["expect"].items():
            if facts[key] != want:
                return False, notes + [
                    f"  seed {seed}: {key}={facts[key]!r}, ground truth {want!r}"
                ]
        asserted += 1
    if asserted == 0:
        return False, notes + ["  no seed produced a landed setup — inconclusive"]
    notes.append(f"  asserted on {asserted}/{len(seeds)} seed(s)")
    return True, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--showdown-root", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003])
    parser.add_argument("--only", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    args = parser.parse_args()

    config = LocalShowdownConfig(showdown_root=args.showdown_root)
    failures = []
    for name in args.only:
        ok, notes = run_scenario(name, args.seeds, config)
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        for note in notes:
            print(note)
        if not ok:
            failures.append(name)
    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print(f"all {len(args.only)} scenario(s) match gen3 Showdown ground truth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
