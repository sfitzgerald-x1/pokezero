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
  spikesNlayer(s) : a 461 max HP Snorlax switching into 1/2/3 layers of Spikes
                  lands on exactly 404/385/346 — 1/8, 1/6 and 1/4 of max HP,
                  floored. ``spikesminimum`` is the other end of the same
                  ``clampIntRange(damage, 1)``: one layer FAINTS a 1 HP Shedinja.
                  Both are the divergence
                  ``third_party/poke-engine-gen3-spikes-layers.patch`` fixes;
                  upstream dealt ``maxhp * layers / 8``.
  faintresiduals / faintresidualsdeferred : a Pokemon that faints mid-turn defers
                  the WHOLE end-of-turn residual block past its forced
                  replacement — the faint ply's protocol ends at ``|faint|`` and
                  the block (sandstorm upkeep, the incoming Pokemon's sandstorm
                  damage, the survivor's Leftovers heal) lands in the NEXT ply,
                  after ``|switch|``. The divergence
                  ``third_party/poke-engine-gen3-residual-defer-on-faint.patch``
                  fixes; ``faintresidualscontrol`` is the same line with no faint.
  confusionduration : Confuse Ray, then the victim attacks every turn -> confusion
                  ENDS, after between one and four turns that carried a self-hit
                  roll, and never activates again. This is the divergence
                  ``third_party/poke-engine-gen3-confusion-duration.patch`` fixes:
                  upstream had no expiry path at all, so search saw a permanent
                  50%-per-turn self-hit. Showdown rolls
                  ``time = this.random(2, 6)`` once at ``addVolatile`` and gen3's
                  ``onBeforeMove`` (the gen4 mod's) decrements it before the
                  self-hit roll, so ``-activate|...|confusion`` fires ``time - 1``
                  times -- uniform on {1,2,3,4} -- and then ``-end``.
  confusiondurationcontrol : same line without Confuse Ray -> no confusion at all.
  confusionbatonpass : a confused Smeargle Baton Passes into Snorlax -> the
                  RECEIVER is confused, and burns the passer's REMAINING duration
                  (``copyVolatileFrom`` shallow-clones every volatile lacking
                  ``noCopy``; ``confusion`` has none anywhere in gen3's chain, and
                  ``onStart`` never re-runs so there is no fresh roll).
  confusionbatonpasscontrol : same line with an ORDINARY switch -> the receiver is
                  clean, because ``Pokemon.clearVolatile()`` blanks the volatile
                  table on a normal switch-out.
  transform     : Ditto Transforms into Machamp and then USES one of Machamp's
                  moves. This is the divergence
                  ``third_party/poke-engine-gen3-transform.patch`` fixes: upstream
                  poke-engine defines ``Choices::TRANSFORM`` in the MOVES table and
                  nowhere else, so clicking Transform produced no state change at
                  all.
  transformcontrol : the same line without Transform -> Ditto never acquires
                  Machamp's moves.
  transformsub  : the target is behind a Substitute -> Transform still lands. gen3
                  inherits gen4's ``bypasssub`` flag and the ``substitute``
                  bail-out inside ``transformInto`` is gated on ``gen >= 5``.
  transformmirror : Ditto vs Ditto, both Transform -> the second one ``-fail``s
                  (``pokemon.transformed && this.battle.gen >= 2``).
  transformrevert : Transform, switch out, switch back, Transform again ->
                  ``Pokemon.clearVolatile()`` restored ``baseMoveSlots`` and the
                  base species, so Ditto has its own move again.

``leechseed`` and ``partialtrap`` depend on a 90%/85% accurate SETUP move, so they
only assert on seeds where the setup actually landed and require at least one such
seed. ``confusionbatonpass`` is gated the same way, on two counts: the passer's
confusion check on the Baton Pass turn can snap out (nothing left to carry, caught
by ``setup_landed``) or self-hit (the pass never happens, so the scripted
force-switch boundary never arrives -- ``tolerate_desync`` turns that strict
boundary error into a skip rather than a crash). Everything else is deterministic.

Usage:
    .venv/bin/python scripts/gen3_switch_differential.py \
        --showdown-root /Users/scott/workspace/pokerena/vendor/pokemon-showdown
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownError
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


def _blissey_confusable():  # confusion victim: no Own Tempo, no Substitute, no recovery
    return FixturePokemon(species="Blissey", ability="Natural Cure", item="None",
                          moves=("Splash",))


def _smeargle_bp():  # Baton Pass carrier, no Perish Song (confusion is the payload)
    return FixturePokemon(species="Smeargle", ability="Technician", item="None",
                          moves=("Baton Pass", "Splash"))


def _cacturne():  # Leech Seed setter
    return FixturePokemon(species="Cacturne", ability="Sand Veil", item="None",
                          moves=("Leech Seed", "Splash"))


def _ninetales():  # partial-trap setter
    return FixturePokemon(species="Ninetales", ability="Flash Fire", item="None",
                          moves=("Fire Spin", "Splash"))


def _blissey():  # trap/seed victim
    return FixturePokemon(species="Blissey", ability="Natural Cure", item="None",
                          moves=("Splash", "Soft-Boiled"))


def _spikes_skarmory():  # hazard setter with an inert filler move
    return FixturePokemon(species="Skarmory", ability="Keen Eye", item="None",
                          moves=("Spikes", "Splash"))


def _snorlax_hazard_victim():  # 461 max HP, grounded: exact-HP hazard target
    return FixturePokemon(species="Snorlax", ability="Immunity", item="None",
                          moves=("Splash",))


def _shedinja():  # 1 max HP: the Spikes minimum-damage clamp
    return FixturePokemon(species="Shedinja", ability="Wonder Guard", item="None",
                          moves=("Splash",))


def _sand_tyranitar():  # permanent gen3 sandstorm + a Leftovers residual to watch
    return FixturePokemon(species="Tyranitar", ability="Sand Stream", item="Leftovers",
                          moves=("Crunch", "Splash"))


def _abra():  # 191 max HP: dies to Crunch on any roll, chips Tyranitar meanwhile
    return FixturePokemon(species="Abra", ability="Synchronize", item="None",
                          moves=("Night Shade", "Splash"))


def _ditto():  # the only gen3 randbats Transform carrier with a one-move set
    # Splash is a Custom Game convenience so the control arm has something to do;
    # the randbats set is movepool ["transform"], Limber, level 100.
    return FixturePokemon(species="Ditto", ability="Limber", item="None",
                          moves=("Transform", "Splash"))


def _machamp():  # copy target: distinct species, types, stats, ability and moves
    return FixturePokemon(species="Machamp", ability="Guts", item="None",
                          moves=("Bulk Up", "Cross Chop", "Splash"))


def _machamp_sub():  # copy target that hides behind a Substitute first
    return FixturePokemon(species="Machamp", ability="Guts", item="None",
                          moves=("Substitute", "Splash"))


def _has(lines, needle: str) -> bool:
    return any(needle in line for line in lines)


def _spikes_cleared(lines) -> bool:
    """True if the SPINNER's own Spikes were removed (``-sideend`` on p2's side)."""
    return any(
        line.startswith("|-sideend|p2") and "Spikes" in line for line in lines
    )


def _count(lines, prefix: str) -> int:
    return sum(1 for line in lines if line.startswith(prefix))


def _activates_after_end(lines, seat: str) -> bool:
    """True if `seat` took another confusion check after ``-end|...|confusion``."""
    end = f"|-end|{seat}|confusion"
    activate = f"|-activate|{seat}|confusion"
    seen_end = False
    for line in lines:
        if line.startswith(end):
            seen_end = True
        elif seen_end and line.startswith(activate):
            return True
    return False


def _residual_from(lines, source: str, seat: str) -> bool:
    """True if `seat` took residual damage attributed to `source` in `lines`."""
    return any(
        line.startswith(f"|-damage|{seat}") and "[from]" in line and source in line
        for line in lines
    )


_SAND_UPKEEP = "|-weather|Sandstorm|[upkeep]"


def _spikes_landing_hp(lines):
    """The ``cur/max`` (or ``0 fnt``) p2 lands on after its Spikes hit."""
    for line in lines:
        if line.startswith("|-damage|p2a") and "[from] Spikes" in line:
            return line.split("|")[3]
    return None


def _residual_block_ran(lines) -> bool:
    """True if the end-of-turn residual block resolved inside ``lines``.

    ``-weather ... [upkeep]`` is the field-level residual and is emitted
    unconditionally whenever the block runs, so its presence brackets the WHOLE
    block — not just the parts that happen to damage or heal someone.
    """
    return any(line.startswith(_SAND_UPKEEP) for line in lines)


def _index_of(lines, prefix: str):
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            return index
    return None


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
    if name in ("spikes1layer", "spikes2layers", "spikes3layers"):
        # p1 Skarmory stacks `layers` of Spikes on p2's side, then p2 sends in a
        # 461 max HP Snorlax. Showdown's gen4 Spikes condition (which gen3
        # inherits) deals `[0, 3, 4, 6][layers] * maxhp / 24` floored, i.e. 57 /
        # 76 / 115 HP -> 404 / 385 / 346. The engine dealt `maxhp * layers / 8`,
        # so it only agreed at ONE layer and over-damaged by 1.5x at two and
        # three (poke-engine-gen3-spikes-layers.patch).
        layers = {"spikes1layer": 1, "spikes2layers": 2, "spikes3layers": 3}[name]
        landing = {1: "404/461", 2: "385/461", 3: "346/461"}[layers]
        return dict(
            p1=[_spikes_skarmory()], p2=[_blissey(), _snorlax_hazard_victim()],
            turns=[("move spikes", "move splash")] * layers
                  + [("move splash", "switch 2")],
            measured=layers, setup_step=None, setup_landed=None,
            facts=lambda L: {"landing_hp": _spikes_landing_hp(L)},
            expect={"landing_hp": landing},
            landmark=lambda L: _has(L, "|switch|p2a: Snorlax"),
            landmark_desc="Snorlax switched into the hazard")
    if name == "spikesminimum":
        # The other end of the same formula: `clampIntRange(damage, 1)` floors
        # the hit at 1 HP, so one layer FAINTS a 1 max HP Shedinja. The engine's
        # integer `maxhp * layers / 8` truncated to zero and let it walk in free.
        return dict(
            p1=[_spikes_skarmory()], p2=[_blissey(), _shedinja()],
            turns=[("move spikes", "move splash"), ("move splash", "switch 2")],
            measured=1, setup_step=None, setup_landed=None,
            facts=lambda L: {"landing_hp": _spikes_landing_hp(L),
                             "fainted": _has(L, "|faint|p2a: Shedinja")},
            expect={"landing_hp": "0 fnt", "fainted": True},
            landmark=lambda L: _has(L, "|switch|p2a: Shedinja"),
            landmark_desc="Shedinja switched into the hazard")

    # --- residual deferral across a forced replacement ----------------------
    # Shared line for the three scenarios below: permanent Sand Stream sandstorm
    # plus a Leftovers heal on p1, so BOTH seats have a residual to watch. Abra
    # chips Tyranitar with fixed-damage Night Shade and dies to Crunch on every
    # damage roll, which keeps the faint on the ply the script expects.
    _faint_p1 = [_sand_tyranitar()]
    _faint_p2 = [_abra(), _blissey()]
    _faint_turns = [("move splash", "move nightshade"),
                    ("move crunch", "move nightshade"),
                    (None, "switch 2")]
    if name == "faintresiduals":
        # Measured on the FAINT ply: `runAction` sees the pending switch flag,
        # issues a `switch` request and returns with the queued `residual`
        # action untouched, so the protocol block ends at `|faint|` — no
        # `-weather ... [upkeep]`, no `|upkeep|`, no `|turn|`.
        return dict(
            p1=_faint_p1, p2=_faint_p2, turns=_faint_turns,
            measured=1, setup_step=None, setup_landed=None,
            facts=lambda L: {"residual_block_ran": _residual_block_ran(L),
                             "survivor_healed": _has(L, "[from] item: Leftovers")},
            expect={"residual_block_ran": False, "survivor_healed": False},
            landmark=lambda L: _has(L, "|faint|p2a: Abra"),
            landmark_desc="the victim fainted mid-turn")
    if name == "faintresidualsdeferred":
        # Measured on the REPLACEMENT ply: the deferred block runs here, after
        # the switch, and applies to the Pokemon that just came in.
        return dict(
            p1=_faint_p1, p2=_faint_p2, turns=_faint_turns,
            measured=2, setup_step=None, setup_landed=None,
            facts=lambda L: {
                "residual_block_ran": _residual_block_ran(L),
                "replacement_took_sandstorm": _residual_from(L, "Sandstorm", "p2a"),
                "survivor_healed": _has(L, "[from] item: Leftovers"),
                "switch_precedes_residuals":
                    _index_of(L, "|switch|p2a") is not None
                    and _index_of(L, _SAND_UPKEEP) is not None
                    and _index_of(L, "|switch|p2a") < _index_of(L, _SAND_UPKEEP),
            },
            expect={"residual_block_ran": True,
                    "replacement_took_sandstorm": True,
                    "survivor_healed": True,
                    "switch_precedes_residuals": True},
            landmark=lambda L: _has(L, "|switch|p2a: Blissey"),
            landmark_desc="the replacement came in")
    if name == "faintresidualscontrol":
        # Same line without the faint: the block runs on the move ply itself,
        # which is what makes its absence above a deferral rather than a drop.
        return dict(
            p1=_faint_p1, p2=_faint_p2,
            turns=[("move splash", "move nightshade"),
                   ("move splash", "move nightshade")],
            measured=1, setup_step=None, setup_landed=None,
            facts=lambda L: {"residual_block_ran": _residual_block_ran(L),
                             "victim_took_sandstorm": _residual_from(L, "Sandstorm", "p2a"),
                             "survivor_healed": _has(L, "[from] item: Leftovers")},
            expect={"residual_block_ran": True,
                    "victim_took_sandstorm": True,
                    "survivor_healed": True},
            landmark=lambda L: not _has(L, "|faint|"),
            landmark_desc="nobody fainted")
    if name == "confusionduration":
        # p1 Misdreavus (faster) confuses p2 Blissey, which then attacks every
        # turn for seven more boundaries. Showdown emits one `-activate` per turn
        # that carried a self-hit roll and then exactly one `-end`; a permanent
        # confusion would still be activating at the end of the script.
        return dict(
            p1=[_misdreavus()], p2=[_blissey_confusable()],
            turns=[("move confuseray", "move splash")]
                  + [("move splash", "move splash")] * 7,
            measured=None, setup_step=None, setup_landed=None,
            facts=lambda L: {
                "risk_turns": _count(L, "|-activate|p2a: Blissey|confusion"),
                "ends": _count(L, "|-end|p2a: Blissey|confusion"),
                "activates_after_end": _activates_after_end(L, "p2a: Blissey"),
            },
            # `risk_turns` is `time - 1` for `time ~ Uniform{2,3,4,5}`; the
            # per-seed value is checked against the window by run_scenario's
            # `expect_in` rather than pinned to one number.
            expect={"ends": 1, "activates_after_end": False},
            expect_in={"risk_turns": (1, 2, 3, 4)},
            landmark=lambda L: _has(L, "|-start|p2a: Blissey|confusion"),
            landmark_desc="confusion applied")
    if name == "confusiondurationcontrol":
        return dict(
            p1=[_misdreavus()], p2=[_blissey_confusable()],
            turns=[("move splash", "move splash")] * 8,
            measured=None, setup_step=None, setup_landed=None,
            facts=lambda L: {"any_confusion": _has(L, "confusion")},
            expect={"any_confusion": False},
            landmark=lambda L: True, landmark_desc="")
    if name == "confusionbatonpass":
        # p2 Misdreavus (faster) confuses p1 Smeargle, which Baton Passes into
        # Snorlax. The volatile rides the pass, so the RECEIVER shows the rest of
        # the duration: `-activate` and/or `-end` under Snorlax's name.
        return dict(
            p1=[_smeargle_bp(), _snorlax()], p2=[_misdreavus()],
            turns=[("move splash", "move confuseray"), ("move batonpass", "move splash"),
                   ("switch 2", None)] + [("move splash", "move splash")] * 6,
            measured=None,
            # The pass turn's own confusion check may snap the passer out, leaving
            # nothing to carry; `-activate` on that step means it is still on.
            setup_step=1,
            setup_landed=lambda L: _has(L, "|-activate|p1a: Smeargle|confusion"),
            tolerate_desync=True,
            seeds=(1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009),
            facts=lambda L: {
                "receiver_confused": _has(L, "|-activate|p1a: Snorlax|confusion")
                                     or _has(L, "|-end|p1a: Snorlax|confusion"),
            },
            expect={"receiver_confused": True},
            landmark=lambda L: _has(L, "[from] Baton Pass"),
            landmark_desc="Baton Pass resolved")
    if name == "confusionbatonpasscontrol":
        # Same line, ORDINARY switch: `Pokemon.clearVolatile()` drops confusion, so
        # the receiver is clean. A switch also cannot be aborted by a self-hit, so
        # this control needs neither the desync tolerance nor the seed widening.
        return dict(
            p1=[_smeargle_bp(), _snorlax()], p2=[_misdreavus()],
            turns=[("move splash", "move confuseray"), ("switch 2", "move splash")]
                  + [("move splash", "move splash")] * 6,
            measured=None, setup_step=0,
            setup_landed=lambda L: _has(L, "|-start|p1a: Smeargle|confusion"),
            facts=lambda L: {
                "receiver_confused": _has(L, "|-activate|p1a: Snorlax|confusion")
                                     or _has(L, "|-end|p1a: Snorlax|confusion"),
            },
            expect={"receiver_confused": False},
            landmark=lambda L: True, landmark_desc="")
    if name == "transform":
        # Ditto copies Machamp and then USES one of Machamp's moves, which is only
        # possible if the moveset was really copied into Ditto's slots. This is the
        # divergence `poke-engine-gen3-transform.patch` fixes: upstream defines
        # Choices::TRANSFORM in the MOVES table and nowhere else, so clicking
        # Transform changed nothing at all.
        return dict(
            p1=[_ditto()], p2=[_machamp()],
            turns=[("move transform", "move splash"), ("move crosschop", "move splash")],
            measured=None, setup_step=None, setup_landed=None,
            facts=lambda L: {
                "transformed": _has(L, "|-transform|p1a: Ditto|p2a: Machamp"),
                "copied_move_used": _has(L, "|move|p1a: Ditto|Cross Chop"),
            },
            expect={"transformed": True, "copied_move_used": True},
            landmark=lambda L: _has(L, "|move|p1a: Ditto|Transform"),
            landmark_desc="Transform used")
    if name == "transformcontrol":
        # Same teams, same target, Ditto simply does not Transform: it must never
        # acquire Machamp's moves on its own.
        return dict(
            p1=[_ditto()], p2=[_machamp()],
            turns=[("move splash", "move splash"), ("move splash", "move splash")],
            measured=None, setup_step=None, setup_landed=None,
            facts=lambda L: {
                "transformed": _has(L, "|-transform|"),
                "copied_move_used": _has(L, "|move|p1a: Ditto|Cross Chop"),
            },
            expect={"transformed": False, "copied_move_used": False},
            landmark=lambda L: _has(L, "|move|p1a: Ditto|Splash"),
            landmark_desc="Ditto acted")
    if name == "transformsub":
        # gen3 inherits gen4's `transform: { flags: { bypasssub, ... } }` and the
        # `volatiles['substitute']` bail-out in `transformInto` is gated on
        # `gen >= 5`, so a Substitute does NOT stop Transform here.
        return dict(
            p1=[_ditto()], p2=[_machamp_sub()],
            turns=[("move transform", "move substitute"), ("move splash", "move splash")],
            measured=None, setup_step=None, setup_landed=None,
            facts=lambda L: {
                "transformed": _has(L, "|-transform|p1a: Ditto|p2a: Machamp"),
            },
            expect={"transformed": True},
            landmark=lambda L: _has(L, "|-start|p2a: Machamp|Substitute"),
            landmark_desc="Substitute was up")
    if name == "transformmirror":
        # `pokemon.transformed && this.battle.gen >= 2` -> the SECOND Transform in a
        # Ditto mirror fails outright. Speed is tied so either seat may be the one
        # that fails; the assertion is order-independent.
        return dict(
            p1=[_ditto()], p2=[_ditto()],
            turns=[("move transform", "move transform")],
            measured=None, setup_step=None, setup_landed=None,
            facts=lambda L: {
                "one_transformed": _has(L, "|-transform|"),
                "other_failed": _has(L, "|-fail|p1a: Ditto") or _has(L, "|-fail|p2a: Ditto"),
            },
            expect={"one_transformed": True, "other_failed": True},
            landmark=lambda L: _has(L, "|move|p1a: Ditto|Transform"),
            landmark_desc="Transform used")
    if name == "transformrevert":
        # `Pokemon.clearVolatile()` ends with `setSpecies(this.baseSpecies)` and
        # restores `baseMoveSlots`, so a Ditto that switches out and back has its
        # own single move again — clicking `move transform` on the last boundary is
        # only a legal choice at all if the revert happened.
        return dict(
            p1=[_ditto(), _snorlax()], p2=[_machamp()],
            turns=[("move transform", "move splash"), ("switch 2", "move splash"),
                   ("switch 2", "move splash"), ("move transform", "move splash")],
            measured=3, setup_step=None, setup_landed=None,
            facts=lambda L: {
                "retransformed": _has(L, "|move|p1a: Ditto|Transform")
                                 and _has(L, "|-transform|p1a: Ditto|p2a: Machamp"),
            },
            expect={"retransformed": True},
            landmark=lambda L: True, landmark_desc="")
    raise ValueError(name)


SCENARIOS = ("spinprotect", "spinconnect", "batonpass", "batonpasscontrol",
             "leechseed", "leechseedcontrol", "partialtrap", "partialtrapcontrol",
             "spikes1layer", "spikes2layers", "spikes3layers", "spikesminimum",
             "faintresiduals", "faintresidualsdeferred", "faintresidualscontrol",
             "confusionduration", "confusiondurationcontrol",
             "confusionbatonpass", "confusionbatonpasscontrol",
             "transform", "transformcontrol", "transformsub", "transformmirror",
             "transformrevert")


def run_scenario(name, seeds, config) -> tuple[bool, list[str]]:
    spec = _spec(name)
    notes: list[str] = []
    asserted = 0
    seeds = spec.get("seeds") or seeds
    for seed in seeds:
        try:
            result = run_multi_turn_fixture(
                p1_team=spec["p1"], p2_team=spec["p2"], turns=spec["turns"],
                seed=seed, config=config,
            )
        except LocalShowdownError as exc:
            # A scripted boundary that never arrives is a desynchronized
            # trajectory, which the fixture refuses to paper over. For scenarios
            # whose setup move can be aborted mid-script (a confusion self-hit
            # eating a Baton Pass), that is the same "setup missed" case the
            # setup_landed gate handles, just surfaced one layer earlier.
            if not spec.get("tolerate_desync"):
                raise
            notes.append(f"  seed {seed}: setup aborted mid-script, skipped ({exc})")
            continue
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
        for key, allowed in spec.get("expect_in", {}).items():
            if facts[key] not in allowed:
                return False, notes + [
                    f"  seed {seed}: {key}={facts[key]!r}, ground truth one of {allowed!r}"
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
