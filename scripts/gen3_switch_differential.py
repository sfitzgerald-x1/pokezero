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
  meanlook / spiderweb : the target is trapped -- ``-activate|...|trapped`` and a
                  standing request with ``trapped: true`` and no switch slots.
                  The divergence
                  ``third_party/poke-engine-gen3-move-trapping.patch`` fixes:
                  upstream defined all three trapping moves as pure no-ops.
                  ``meanlookcontrol`` is the same line without the move.
  meanlookghost : gen3 has NO Ghost trapping immunity (it arrives in gen6), so a
                  Misdreavus is trapped like anything else.
  meanlookprotect / meanlooksub : Protect blocks the trap (gen3 inherits the gen5
                  mod's protect flag on Mean Look and Block) and a Substitute
                  makes it outright ``-fail`` (no bypasssub flag).
  meanlooktrapperleaves : the trap is a LINKED volatile, so the trapper switching
                  out runs ``removeLinkedVolatiles`` and frees the victim.
  meanlooktrapperbatonpass / ...freed : but a BATON PASS by the trapper does not.
                  ``copyVolatileFrom`` moves ``trapper`` to the receiver, DELETES
                  the old trapper's link, and only then runs ``clearVolatile()``,
                  which finds nothing left to release; the victim's link is
                  re-pointed, so the trap changes OWNER and the RECEIVER's later
                  departure is what frees the victim. This is the line 2 of the 3
                  gen3 randbats Ariados sets are built around (Spider Web + Baton
                  Pass).
  meanlookbatonpass : gen3/gen4 alone re-declare ``trapped`` with
                  ``noCopy: false``, so the trap rides a Baton Pass and the
                  RECEIVER is still stuck.
  perishladderfirsttick / perishladder : the Perish Song ladder announces
                  ``perish3`` in the residual block of the move's OWN turn, then
                  ``perish2``, ``perish1``, and ``perish0`` + ``|faint|`` on the
                  FOURTH block, with both seats replacing at one shared boundary.
                  These are a VERDICT pin rather than a fix -- the engine already
                  matches -- and they exist because the ladder only lines up
                  while the end-of-turn block runs exactly once on every ply.

  toxicladder   : a 651 max HP Blissey (651 % 16 == 11) ticks 40/80/120/160/200,
                  i.e. ``floor(maxhp/16) * stage``. This is the divergence
                  ``third_party/poke-engine-gen3-residual-rounding.patch`` fixes:
                  upstream computed ``floor(maxhp * stage / 16)`` — flooring
                  AFTER the multiply — and dealt 40/81/122/162/203.
                  ``toxicladdercontrol`` runs the same ladder on a 656 max HP
                  Blissey, where 16 divides max HP and BOTH orderings agree.
  sandminimum   : sandstorm's ``onWeather`` is ``this.damage(baseMaxhp / 16)`` and
                  every ``damage()`` runs through ``clampIntRange(damage, 1)``, so
                  a 1 max HP Shedinja takes 1 and FAINTS. Upstream truncated to
                  zero and left it standing in the sand forever.

  whirlwindprotect / roarprotect : gen3 phazing IS blocked by Protect —
                  ``-activate <target> Protect`` and no ``|drag|``. This is the
                  divergence ``third_party/poke-engine-gen3-phaze-protect.patch``
                  fixes: gen3 inherits gen4's flag override
                  (``{ protect: 1, mirror: 1, bypasssub: 1, metronome: 1 }``) and
                  upstream carried no protect flag, so the target was dragged out
                  through its own Protect.
  whirlwinddrag : the no-regression control — the same turn WITHOUT Protect must
                  still drag.
  whirlwindsub  : ``bypasssub`` is in the gen3 flag set, so a Substitute does not
                  stop the drag.

  flailladder / reversalladder : gen3 declares its OWN base-power ladder for both
                  moves (``data/mods/gen3/moves.ts:273`` and ``:496``) —
                  ``ratio = floor(hp * 48 / maxhp)`` clamped to >= 1, then
                  200/150/100/80/40/20 — and does NOT inherit gen4's 64-scale.
                  Sandstorm chips the attacker a fixed 16 HP a turn, so repeating
                  the move walks the ladder and its damage must climb. This is
                  the divergence
                  ``third_party/poke-engine-gen3-variable-bp.patch`` fixes:
                  upstream Flail was INERT (no arm, no base power) and Reversal
                  used rounded float ratios that misclassify the boundary bands.
  flailladdercontrol : the same line with a FIXED-power move (Body Slam), whose
                  damage must NOT climb — separating "the ladder works" from
                  "the attacker is taking chip-damage variance".

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
import re
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


def _smeargle_encorer():  # spe 75: encores Blissey BEFORE it moves (no duration bump)
    return FixturePokemon(species="Smeargle", ability="Technician", item="None",
                          moves=("Encore", "Splash"))


def _pp_faker():  # Fake Out on a fast body: a 100%-deterministic immobilized turn
    return FixturePokemon(species="Misdreavus", ability="Levitate", item="None",
                          moves=("Fake Out", "Splash"))


def _pp_grinder():  # sole move, 5 base PP -> 8 with PP Ups, harmless
    return FixturePokemon(species="Smeargle", ability="Technician", item="None",
                          moves=("Mind Reader",))


def _solarbeam_user():  # Exeggutor: the species from the release-damage repro
    return FixturePokemon(species="Exeggutor", ability="Chlorophyll", item="None",
                          moves=("Solar Beam", "Sunny Day", "Sandstorm", "Splash"))


def _solarbeam_wall():  # 651 HP, survives a full-power beam so the number is readable
    return FixturePokemon(species="Blissey", ability="Natural Cure", item="None",
                          moves=("Splash",))


def _lock_tank():  # Steel/Ground wall: survives eight Sky Attacks without fainting
    return FixturePokemon(species="Steelix", ability="Sturdy", item="None",
                          moves=("Splash",))


def _lock_charger():  # Sky Attack: two-turn, 5 base PP -> 8 with PP Ups, sole move
    return FixturePokemon(species="Smeargle", ability="Technician", item="None",
                          moves=("Sky Attack",))


def _lastmove_encorer():  # spe 85: out-speeds Smeargle, so Encore resolves first
    return FixturePokemon(species="Misdreavus", ability="Levitate", item="None",
                          moves=("Encore", "Splash", "Thunder Wave", "Confuse Ray"))


def _lastmove_faker():  # Fake Out on a FAST body, so the flinched target never acts
    return FixturePokemon(species="Misdreavus", ability="Levitate", item="None",
                          moves=("Fake Out", "Encore", "Splash"))


def _lastmove_victim():  # two distinguishable moves: Splash is "used", Tackle is "attempted"
    return FixturePokemon(species="Smeargle", ability="Technician", item="None",
                          moves=("Splash", "Tackle"))


def _misdreavus_encorer():  # spe 85: out-speeds Smeargle, so the Encore attempt lands first
    return FixturePokemon(species="Misdreavus", ability="Levitate", item="None",
                          moves=("Encore", "Splash"))


def _encore_target_smeargle():  # ordinary encorable move
    return FixturePokemon(species="Smeargle", ability="Technician", item="None",
                          moves=("Splash",))


def _mindreader_smeargle():  # 5 base PP -> 8 with PP Ups; sole move, so Struggle after 8
    return FixturePokemon(species="Smeargle", ability="Technician", item="None",
                          moves=("Mind Reader",))


def _mirrormove_smeargle():  # gen3 failencore flag, no PP grind needed
    return FixturePokemon(species="Smeargle", ability="Technician", item="None",
                          moves=("Mirror Move",))


def _shuckle_encorer():  # spe 5: encores Blissey AFTER it moves (onStart bumps duration)
    return FixturePokemon(species="Shuckle", ability="Sturdy", item="None",
                          moves=("Encore", "Splash"))


def _blissey_encore_victim():  # spe 55, two moves so the lock is a real restriction
    return FixturePokemon(species="Blissey", ability="Natural Cure", item="None",
                          moves=("Splash", "Soft-Boiled"))


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


def _tosser():  # Blissey carries Seismic Toss on 17 of the pool's 393 sets
    return FixturePokemon(species="Blissey", ability="Natural Cure", item="None",
                          moves=("Seismic Toss", "Body Slam", "Splash"))


def _sub_user(species: str = "Snorlax"):
    """Substitute users sized on either side of a 100 HP Seismic Toss.

    Snorlax's 461 max HP gives a 115 HP substitute, which survives the hit;
    Dodrio's 261 gives 65, which breaks. Neither is Ghost, so the immunity arm
    stays out of the way.
    """

    ability = "Run Away" if species == "Dodrio" else "Immunity"
    return FixturePokemon(species=species, ability=ability, item="None",
                          moves=("Substitute", "Splash"))


def _ghost_wall():
    return FixturePokemon(species="Gengar", ability="Levitate", item="None",
                          moves=("Splash",))


def _flailer(move: str = "Flail"):
    """261 max HP Dodrio — the pool's own Flail user. One 48th is 5.4 HP, so a
    sandstorm's 16 HP/turn walks the ladder from 20 BP to 200 BP in ~16 turns."""

    return FixturePokemon(species="Dodrio", ability="Run Away", item="None",
                          moves=(move, "Splash"))


def _sand_anvil(move: str):
    """Sets permanent gen3 sandstorm and soaks the ladder move under test.

    Sand Stream chips the Flail user 16 HP a turn with no accuracy roll, which is
    what walks the ladder deterministically. The anvil has to RESIST the move
    being laddered, or the line ends before the climb finishes — so it is chosen
    per move: Rock/Dark against Flail's Normal (and against the control's Body
    Slam), Poison/Flying against Reversal's Fighting, which it takes at 0.25x.
    Sand Stream rides a non-native species in the Reversal case deliberately:
    Custom Game does not validate abilities, and only the weather matters here.
    """

    if move == "Reversal":
        return FixturePokemon(species="Crobat", ability="Sand Stream", item="None",
                              moves=("Splash",), evs={"hp": 252})
    return FixturePokemon(species="Tyranitar", ability="Sand Stream", item="None",
                          moves=("Splash",), evs={"hp": 252})


def _phazer():  # Skarmory carries Whirlwind AND Protect on its own randbats set
    return FixturePokemon(species="Skarmory", ability="Keen Eye", item="None",
                          moves=("Whirlwind", "Roar", "Splash"))


def _phaze_target():  # can Protect, sub, or simply stand there
    return FixturePokemon(species="Snorlax", ability="Immunity", item="None",
                          moves=("Protect", "Substitute", "Splash"))


def _toxic_user():  # lays the ladder, then idles while it climbs
    return FixturePokemon(species="Umbreon", ability="Synchronize", item="None",
                          moves=("Toxic", "Splash"))


def _toxic_ladder_control_victim():
    """The same Blissey at 656 max HP, where 16 divides max HP exactly.

    Level-100 HP is ``2*base + IV + floor(EV/4) + 110``; Blissey's base 255 with
    31 IVs and 20 HP EVs gives 510 + 31 + 5 + 110 = 656, and 656 % 16 == 0. Both
    roundings agree there, so this control proves the fixture is really reading
    the ladder rather than the assertion being loose.
    """

    return FixturePokemon(species="Blissey", ability="Natural Cure", item="None",
                          moves=("Splash",), evs={"hp": 20})


def _shedinja():  # 1 max HP: the Spikes minimum-damage clamp
    return FixturePokemon(species="Shedinja", ability="Wonder Guard", item="None",
                          moves=("Splash",))


def _sand_tyranitar():  # permanent gen3 sandstorm + a Leftovers residual to watch
    return FixturePokemon(species="Tyranitar", ability="Sand Stream", item="Leftovers",
                          moves=("Crunch", "Splash"))


def _umbreon():  # Mean Look user (the pool's user is Misdreavus; any is fine here)
    return FixturePokemon(species="Umbreon", ability="Synchronize", item="None",
                          moves=("Mean Look", "Splash"))


def _ariados():  # Spider Web user, straight out of the gen3 randbats pool
    return FixturePokemon(species="Ariados", ability="Insomnia", item="None",
                          moves=("Spider Web", "Splash"))


def _trap_victim():  # can also answer with Protect / Substitute
    return FixturePokemon(species="Snorlax", ability="Immunity", item="None",
                          moves=("Splash", "Protect", "Substitute"))


def _ariados_batonpass():  # the pool's own Spider Web + Baton Pass set
    return FixturePokemon(species="Ariados", ability="Insomnia", item="None",
                          moves=("Spider Web", "Baton Pass", "Splash"))


def _smeargle_receiver():  # takes the pass, and the trap's ownership with it
    return FixturePokemon(species="Smeargle", ability="Technician", item="None",
                          moves=("Splash",))


def _misdreavus_batonpass():  # Ghost trap victim that can pass the trap on
    return FixturePokemon(species="Misdreavus", ability="Levitate", item="None",
                          moves=("Splash", "Baton Pass"))


def _misdreavus_perishsong():  # the pool's Perish Song carrier
    return FixturePokemon(species="Misdreavus", ability="Levitate", item="None",
                          moves=("Perish Song", "Splash"))


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


def _solarbeam_release_damage(lines) -> int:
    """HP lost to the Solar Beam RELEASE hit, ignoring weather chip.

    The release is the `-damage` line with no `[from]` tag; sandstorm upkeep
    damage carries `[from] Sandstorm`, so tracking the most recent reported HP
    and differencing only at the untagged line isolates the beam itself.
    """
    current = None
    for line in lines:
        m = re.search(r"\|-damage\|p2a: Blissey\|(\d+)/(\d+)", line)
        if m:
            after, maxhp = int(m.group(1)), int(m.group(2))
            if "[from]" not in line:
                # In clear weather and sun the release is the FIRST hp line of
                # the battle, so there is no earlier reading to difference
                # against -- full health is the baseline.
                return (current if current is not None else maxhp) - after
            current = after
            continue
        m = re.search(r"\|-heal\|p2a: Blissey\|(\d+)/(\d+)", line)
        if m:
            current = int(m.group(1))
        elif line.startswith("|switch|p2a: Blissey"):
            m = re.search(r"(\d+)/(\d+)", line)
            if m:
                current = int(m.group(1))
    return 0


def _has(lines, needle: str) -> bool:
    return any(needle in line for line in lines)


def _spikes_cleared(lines) -> bool:
    """True if the SPINNER's own Spikes were removed (``-sideend`` on p2's side)."""
    return any(
        line.startswith("|-sideend|p2") and "Spikes" in line for line in lines
    )


def _encore_span(lines, seat: str) -> int:
    """Turns from ``-start ... Encore`` to ``-end ... Encore`` inclusive, 0 if absent.

    ``lines`` is the flattened per-step protocol, so a step boundary is counted by
    tracking which step each marker fell in. Showdown emits the ``-end`` in the
    residual phase of the last locked turn, so this span is exactly the value the
    duration model has to reproduce.
    """
    start = end = None
    step = 0
    for line in lines:
        if line.startswith("|turn|"):
            step += 1
        elif line.startswith(f"|-start|{seat}|Encore"):
            start = step
        elif line.startswith(f"|-end|{seat}|Encore"):
            end = step
    if start is None or end is None:
        return 0
    return end - start + 1


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


def _seat_trapped(requests, seat: str):
    """Whether ``seat``'s standing request reports the hard move-trap.

    Showdown answers a trapped seat with ``active[0].trapped: true`` and offers
    it no switch slots at all; an untrapped seat carries no such key. ``None``
    means the seat has no move request at this boundary (it is waiting on the
    other seat's forced switch), which every caller treats as "not trapped".
    """
    request = requests.get(seat)
    if not isinstance(request, dict):
        return False
    active = request.get("active")
    if not active:
        return False
    return bool(active[0].get("trapped"))


def _toxic_ladder_hp(lines, maxhp: int):
    """The ``cur`` HP p2 lands on after each ``[from] psn`` tick, in order."""

    ladder = []
    for line in lines:
        if not (line.startswith("|-damage|p2a") and "[from] psn" in line):
            continue
        condition = line.split("|")[3]
        if condition.startswith("0 fnt"):
            ladder.append(0)
            continue
        current = condition.split(" ")[0].split("/")[0]
        ladder.append(int(current))
    return ladder


def _damage_climbs(lines, seat: str, *, factor: float) -> bool:
    """True if `seat`'s LAST attack-damage reading is `factor`x its first.

    Reads only plain `-damage` lines (no `[from]` tag), so residual chip on the
    same seat cannot be mistaken for the attack itself.
    """

    hits = []
    hp = None
    for line in lines:
        if line.startswith(f"|switch|{seat}") or line.startswith(f"|drag|{seat}"):
            hp = int(line.split("|")[4].split(" ")[0].split("/")[0])
            continue
        if not line.startswith(f"|-damage|{seat}"):
            continue
        condition = line.split("|")[3]
        current = 0 if condition.startswith("0 fnt") else int(condition.split(" ")[0].split("/")[0])
        if hp is not None and "[from]" not in line:
            hits.append(hp - current)
        hp = current
    if len(hits) < 2:
        return False
    return hits[-1] >= hits[0] * factor


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

    # --- move-trapping (Mean Look / Spider Web) -----------------------------
    # p1 traps p2's lead. The trap is announced ONCE with `-activate ... trapped`
    # and never mentioned again, so every "still trapped?" fact below is read off
    # the switch options in p2's standing request instead of the protocol.
    _trapped = lambda R: {"p2_trapped": _seat_trapped(R, "p2")}
    if name in ("meanlook", "spiderweb"):
        return dict(
            p1=[_umbreon() if name == "meanlook" else _ariados(), _blissey()],
            p2=[_trap_victim(), _blissey()],
            turns=[("move meanlook" if name == "meanlook" else "move spiderweb",
                    "move splash")],
            measured=0, setup_step=None, setup_landed=None,
            facts=lambda L: {"announced": _has(L, "|-activate|p2a: Snorlax|trapped")},
            request_facts=_trapped,
            expect={"announced": True, "p2_trapped": True},
            landmark=lambda L: _has(L, "|move|p1a:"),
            landmark_desc="the trapping move was used")
    if name == "meanlookcontrol":
        return dict(
            p1=[_umbreon(), _blissey()], p2=[_trap_victim(), _blissey()],
            turns=[("move splash", "move splash")],
            measured=0, setup_step=None, setup_landed=None,
            facts=lambda L: {"announced": _has(L, "|-activate|p2a: Snorlax|trapped")},
            request_facts=_trapped,
            expect={"announced": False, "p2_trapped": False},
            landmark=lambda L: True, landmark_desc="")
    if name == "meanlookghost":
        # Gen 3 has no Ghost trapping immunity — that arrives in gen6, and gen5's
        # typechart override drops the `trapped: 3` entry gen3 would inherit.
        return dict(
            p1=[_umbreon(), _blissey()], p2=[_misdreavus_batonpass(), _blissey()],
            turns=[("move meanlook", "move splash")],
            measured=0, setup_step=None, setup_landed=None,
            facts=lambda L: {"announced": _has(L, "|-activate|p2a: Misdreavus|trapped")},
            request_facts=_trapped,
            expect={"announced": True, "p2_trapped": True},
            landmark=lambda L: _has(L, "|move|p1a: Umbreon|Mean Look"),
            landmark_desc="Mean Look was used on the Ghost")
    if name == "meanlookprotect":
        # gen3 inherits the gen5 mod's protect flag on Mean Look.
        return dict(
            p1=[_umbreon(), _blissey()], p2=[_trap_victim(), _blissey()],
            turns=[("move meanlook", "move protect")],
            measured=0, setup_step=None, setup_landed=None,
            facts=lambda L: {"announced": _has(L, "|-activate|p2a: Snorlax|trapped")},
            request_facts=_trapped,
            expect={"announced": False, "p2_trapped": False},
            landmark=lambda L: _has(L, "|-activate|p2a: Snorlax|Protect"),
            landmark_desc="Protect activated")
    if name == "meanlooksub":
        # No bypasssub flag, so the Substitute makes the move outright fail.
        return dict(
            p1=[_umbreon(), _blissey()], p2=[_trap_victim(), _blissey()],
            turns=[("move splash", "move substitute"), ("move meanlook", "move splash")],
            measured=1, setup_step=None, setup_landed=None,
            facts=lambda L: {"announced": _has(L, "|-activate|p2a: Snorlax|trapped"),
                             "failed": _has(L, "|-fail|p1a: Umbreon")},
            request_facts=_trapped,
            expect={"announced": False, "failed": True, "p2_trapped": False},
            landmark=lambda L: _has(L, "|move|p1a: Umbreon|Mean Look"),
            landmark_desc="Mean Look was used into the Substitute")
    if name == "meanlooktrapperleaves":
        # `addVolatile('trapped', source, move, 'trapper')` LINKS the volatiles,
        # so the trapper leaving runs removeLinkedVolatiles and frees the victim.
        return dict(
            p1=[_umbreon(), _blissey()], p2=[_trap_victim(), _blissey()],
            turns=[("move meanlook", "move splash"), ("switch 2", "move splash")],
            measured=1, setup_step=None, setup_landed=None,
            facts=lambda L: {"trapper_left": _has(L, "|switch|p1a: Blissey")},
            request_facts=_trapped,
            expect={"trapper_left": True, "p2_trapped": False},
            landmark=lambda L: True, landmark_desc="")
    if name in ("meanlooktrapperbatonpass", "meanlooktrapperbatonpassfreed"):
        # The line the randbats set is built around: Ariados webs, then Baton
        # Passes. The trap does NOT break — `copyVolatileFrom` moves `trapper` to
        # the receiver (gen3 inherits gen4's `noCopy: false` for BOTH halves of
        # the link), DELETES the old trapper's link, and only then runs
        # `clearVolatile()`, which finds nothing left to release. The victim's
        # link is re-pointed, so the trap changes OWNER — and the receiver's own
        # departure is what finally frees the victim (measured step 3).
        freed = name.endswith("freed")
        return dict(
            p1=[_ariados_batonpass(), _smeargle_receiver(), _blissey()],
            p2=[_trap_victim(), _blissey()],
            turns=[("move spiderweb", "move splash"),
                   ("move batonpass", "move splash"),
                   ("switch 2", None)]
                  + ([("switch 3", "move splash")] if freed else []),
            measured=3 if freed else 2, setup_step=None, setup_landed=None,
            facts=lambda L: {
                "receiver_in": _has(L, "|switch|p1a: Blissey" if freed
                                    else "|switch|p1a: Smeargle"),
            },
            request_facts=_trapped,
            expect={"receiver_in": True, "p2_trapped": not freed},
            landmark=lambda L: True, landmark_desc="")
    if name == "meanlookbatonpass":
        # gen3/gen4 alone re-declare `trapped` with `noCopy: false`, so the trap
        # rides the pass and the RECEIVER is still stuck.
        return dict(
            p1=[_umbreon(), _blissey()], p2=[_misdreavus_batonpass(), _blissey()],
            turns=[("move meanlook", "move splash"), ("move splash", "move batonpass"),
                   (None, "switch 2")],
            measured=2, setup_step=None, setup_landed=None,
            facts=lambda L: {"receiver_in": _has(L, "|switch|p2a: Blissey")},
            request_facts=_trapped,
            expect={"receiver_in": True, "p2_trapped": True},
            landmark=lambda L: _has(L, "[from] Baton Pass"),
            landmark_desc="Baton Pass resolved")

    # --- Perish Song ladder (verdict, no engine change) ---------------------
    # Perish Song on turn 1; the residual block of that SAME turn announces
    # perish3, then perish2, perish1, and perish0 + faint on the FOURTH block.
    _perish_p1 = [_misdreavus_perishsong(), _blissey()]
    _perish_p2 = [_snorlax_hazard_victim(), _blissey()]
    _perish_turns = [("move perishsong", "move splash"),
                     ("move splash", "move splash"),
                     ("move splash", "move splash"),
                     ("move splash", "move splash"),
                     ("switch 2", "switch 2")]
    if name == "perishladderfirsttick":
        return dict(
            p1=_perish_p1, p2=_perish_p2, turns=_perish_turns,
            measured=0, setup_step=None, setup_landed=None,
            facts=lambda L: {"announced": _has(L, "|-start|p1a: Misdreavus|perish3")
                                          and _has(L, "|-start|p2a: Snorlax|perish3"),
                             "fainted": _has(L, "|faint|")},
            expect={"announced": True, "fainted": False},
            landmark=lambda L: _has(L, "|-fieldactivate|move: Perish Song"),
            landmark_desc="Perish Song resolved")
    if name in ("solarbeamrelease", "solarbeamcontrol"):
        # Solar Beam is the ONLY charge move in the gen3 randbats pool (of the 17
        # dex moves flagged `charge: 1`), so this is the whole reachable surface of
        # the mid-charge state that world construction was missing.
        #
        # The protocol is the contract both new halves key off: the charge is
        # `|move|...||[still]` + `|-prepare|`, and the release is a SECOND `|move|`
        # tagged `[from] lockedmove` — one click, two announced turns. The control
        # is the same Pokemon in SUN, and it is more interesting than expected:
        # Showdown still emits `|move|...||[still]` AND `|-prepare|` there, then
        # fires in the SAME turn via `|-anim|` + damage, with no second `|move|`
        # line. So `-prepare` alone does NOT mean "is charging" — the discriminator
        # is whether a `[from] lockedmove` release follows or an `-anim` lands on
        # the spot. That is exactly what the parser keys off, and getting it wrong
        # leaves a sunny Solar Beam user holding a phantom charge.
        sun = name == "solarbeamcontrol"
        eggy = FixturePokemon(species="Exeggutor", ability="Chlorophyll", item="None",
                              moves=("Solar Beam", "Sunny Day"))
        lax = FixturePokemon(species="Snorlax", ability="Immunity", item="None",
                             moves=("Splash",))
        turns = ([("move sunnyday", "move splash")] if sun else []) + [
            ("move solarbeam", "move splash"), ("move solarbeam", "move splash"),
        ]
        return dict(
            p1=[eggy], p2=[lax],
            turns=turns, measured=None, setup_step=None, setup_landed=None,
            facts=lambda L: {
                "prepared": _has(L, "|-prepare|p1a: Exeggutor|Solar Beam"),
                "released_from_lockedmove": _has(L, "[from] lockedmove"),
                "fired_on_the_spot": _has(L, "|-anim|p1a: Exeggutor|Solar Beam"),
                "damaged": _has(L, "|-damage|p2a: Snorlax"),
            },
            # `-prepare` is announced either way; the two paths differ in HOW the
            # beam lands, which is the fact world construction has to read.
            expect={"prepared": True,
                    "released_from_lockedmove": not sun,
                    "fired_on_the_spot": sun,
                    "damaged": True},
            landmark=lambda L: _has(L, "|move|p1a: Exeggutor|Solar Beam"),
            landmark_desc="Solar Beam used")
    if name == "perishladder":
        return dict(
            p1=_perish_p1, p2=_perish_p2, turns=_perish_turns,
            measured=3, setup_step=None, setup_landed=None,
            facts=lambda L: {"perish0": _has(L, "|-start|p1a: Misdreavus|perish0")
                                        and _has(L, "|-start|p2a: Snorlax|perish0"),
                             "both_fainted": _has(L, "|faint|p1a: Misdreavus")
                                             and _has(L, "|faint|p2a: Snorlax")},
            expect={"perish0": True, "both_fainted": True},
            landmark=lambda L: True, landmark_desc="")
    if name in ("encoreduration", "encoreoutlivesshortest", "encoredurationslow",
                "encoredurationcontrol"):
        # p2 Blissey (spe 55) is the victim. Smeargle (spe 75) encores it BEFORE
        # it moves; Shuckle (spe 5) encores it AFTER. `encore.onStart` bumps the
        # duration in the second case (`if (!queue.willMove(target)) duration++`),
        # so the victim is locked for the same `duration` turns either way and the
        # `-end` lands one step later for the slow encorer. Both windows are
        # asserted, which is what pins the compensation.
        #
        # The victim is scripted onto the locked move throughout: Showdown's
        # `onDisableMove` marks every other move disabled, and the fixture's
        # choice validation refuses an unavailable choice outright — itself a
        # standing proof of the lock, which the engine side pins natively in
        # rust/pokezero-search/tests/gen3_encore_fidelity.rs.
        slow = name == "encoredurationslow"
        control = name == "encoredurationcontrol"
        opener = "move splash" if control else "move encore"
        turns = [("move splash", "move splash"), (opener, "move splash")]
        turns += [("move splash", "move splash")] * 9
        spec = dict(
            p1=[_shuckle_encorer() if slow else _smeargle_encorer()],
            p2=[_blissey_encore_victim()],
            turns=turns, measured=None, setup_step=None, setup_landed=None,
            facts=lambda L: {"span": _encore_span(L, "p2a: Blissey")},
            landmark=lambda L: control or _has(L, "|-start|p2a: Blissey|Encore"),
            landmark_desc="Encore applied")
        if control:
            # No Encore at all -> `_encore_span` reports 0, and nothing may end.
            spec["expect"] = {"span": 0}
            return spec
        spec["expect"] = {}
        if slow:
            # duration + 1: the victim had already moved when Encore landed.
            spec["expect_in"] = {"span": (4, 5, 6, 7)}
        elif name == "encoreoutlivesshortest":
            # Seeds chosen so EVERY one runs past the 3-turn floor, covering 4, 5
            # and 6. A model that always ended at the minimum roll passes
            # `encoreduration` on three of its four seeds but fails every one of
            # these.
            spec["expect_in"] = {"span": (4, 5, 6)}
            spec["seeds"] = (1003, 1004, 1005, 1006)
        else:
            # `this.random(3, 7)` -> uniform {3,4,5,6}. The default seeds happen
            # to cover both ends of the window (3, 3, 3, 6).
            spec["expect_in"] = {"span": (3, 4, 5, 6)}
        return spec
    if name in ("encorefailstruggle", "encorefailnolastmove", "encorefailmirrormove",
                "encoreappliescontrol"):
        # Encore's APPLICATION-failure set. Misdreavus (spe 85) out-speeds
        # Smeargle (spe 75), so p1 always gets to try Encore before p2 acts.
        #
        # encorefailstruggle: Smeargle's ONLY move is Mind Reader (5 base PP ->
        #   8 with PP Ups), so eight turns exhaust it and it is forced onto
        #   Struggle. Struggle carries `failencore` in data/mods/gen3/moves.ts
        #   AND is absent from `moveSlots`, so both of Showdown's arms reject it;
        #   the engine's slot-index representation only reproduces the flag arm,
        #   which is sufficient because either alone fails the move.
        # encorefailnolastmove: Encore on turn 1 from the faster side — the
        #   target has not moved yet and `lastMove` is null.
        # encorefailmirrormove: Mirror Move also carries gen3's `failencore`, and
        #   unlike Struggle it needs no PP grind, so it isolates the flag arm from
        #   the not-in-moveSlots arm.
        # encoreappliescontrol: the same line against an ordinary move lands.
        control = name == "encoreappliescontrol"
        if name == "encorefailstruggle":
            victim = _mindreader_smeargle()
            turns = [("move splash", "move mindreader")] * 8
            turns += [("move splash", "move struggle"), ("move encore", "move struggle")]
        elif name == "encorefailnolastmove":
            victim = _encore_target_smeargle()
            turns = [("move encore", "move splash")]
        elif name == "encorefailmirrormove":
            victim = _mirrormove_smeargle()
            turns = [("move splash", "move mirrormove"), ("move encore", "move mirrormove")]
        else:
            victim = _encore_target_smeargle()
            turns = [("move splash", "move splash"), ("move encore", "move splash")]
        return dict(
            p1=[_misdreavus_encorer()], p2=[victim], turns=turns,
            measured=None, setup_step=None, setup_landed=None,
            facts=lambda L: {"encore_applied": _has(L, "|-start|p2a: Smeargle|Encore")},
            expect={"encore_applied": control},
            # Every non-control line must actually SHOW the failure, so a scenario
            # that silently stopped reaching the Encore attempt cannot pass.
            landmark=lambda L: (_has(L, "|-start|p2a: Smeargle|Encore") if control
                               else _has(L, "|-fail|p1a: Misdreavus")),
            landmark_desc="Encore applied" if control else "Encore failed")
    # --- residual rounding (toxic ladder / weather minimum) -----------------
    if name in ("toxicladder", "toxicladdercontrol"):
        # Blissey's 651 max HP is the sharp case: 651 % 16 == 11, so
        # floor(651/16) * stage (Showdown) and floor(651 * stage / 16) (upstream)
        # diverge from stage 2 onward — 40/80/120/160/200 vs 40/81/122/162/203.
        # The control is a 656 max HP Snorlax-class target chosen so that
        # 656 % 16 == 0, where BOTH orderings agree: it proves the fixture reads
        # the ladder correctly rather than the assertion being loose.
        divisible = name.endswith("control")
        victim = _toxic_ladder_control_victim() if divisible else _blissey()
        maxhp = 656 if divisible else 651
        per_stage = maxhp // 16
        ladder = [maxhp - per_stage * sum(range(1, n + 1)) for n in range(1, 6)]
        return dict(
            p1=[_toxic_user()], p2=[victim],
            turns=[("move toxic", "move splash")]
                  + [("move splash", "move splash")] * 4,
            measured=None, setup_step=0,
            setup_landed=lambda L: _has(L, "|-status|p2a") and not _has(L, "[miss]"),
            facts=lambda L: {"ladder": _toxic_ladder_hp(L, maxhp)},
            expect={"ladder": ladder},
            landmark=lambda L: _has(L, "|-status|p2a"),
            landmark_desc="Toxic landed")
    if name == "sandminimum":
        # Sandstorm's onWeather is this.damage(baseMaxhp / 16), and every
        # damage() runs through clampIntRange(damage, 1) — so a 1 max HP
        # Shedinja takes 1 and FAINTS. Upstream truncated to zero and left it
        # standing in the sand forever.
        return dict(
            p1=[_sand_tyranitar()], p2=[_shedinja(), _blissey()],
            turns=[("move splash", "move splash")],
            measured=0, setup_step=None, setup_landed=None,
            facts=lambda L: {
                "chipped": _residual_from(L, "Sandstorm", "p2a"),
                "fainted": _has(L, "|faint|p2a: Shedinja"),
            },
            expect={"chipped": True, "fainted": True},
            landmark=lambda L: _has(L, _SAND_UPKEEP),
            landmark_desc="the sandstorm upkeep ran")
    if name in ("lastmoveparaencore", "lastmoveconfusionencore",
                "lastmoveflinchencore", "lastmoveexecutedcontrol"):
        # Does an immobilized turn count as "using" the move? Showdown records
        # lastMove in moveUsed(), which the BeforeMove gate short-circuits, so it
        # does not -- and Encore then locks the EARLIER move instead.
        #
        # The victim uses Splash on turn 1 (so lastMove = Splash), then attempts
        # Tackle while immobilized on turn 2, then p1 Encores on turn 3. If the
        # immobilized attempt had counted, Encore would lock Tackle and the
        # scripted "move splash" on turn 4 would be an unavailable choice -- so
        # the seeds where the immobilization did NOT land desync, and are skipped
        # exactly like any other missed setup.
        if name == "lastmoveflinchencore":
            # Fake Out flinches on turn 1, so the victim never moves at all and
            # has NO last move: Encore fails outright rather than retargeting.
            return dict(
                p1=[_lastmove_faker()], p2=[_lastmove_victim()],
                turns=[("move fakeout", "move splash"), ("move encore", "move splash")],
                measured=None, setup_step=0,
                setup_landed=lambda L: _has(L, "|cant|p2a: Smeargle|flinch"),
                facts=lambda L: {"encore_applied": _has(L, "|-start|p2a: Smeargle|Encore")},
                expect={"encore_applied": False},
                landmark=lambda L: _has(L, "|-fail|p1a: Misdreavus"),
                landmark_desc="Encore failed against a target with no last move")
        control = name == "lastmoveexecutedcontrol"
        # Turn 1 must actually RECORD Splash, or the scenario is measuring the
        # wrong thing: Thunder Wave and Confuse Ray both resolve before the
        # victim acts, so the victim can already be immobilized on turn 1 and
        # end up with NO last move -- in which case Encore correctly fails and
        # the "which move got encored" question never arises. Gating both steps
        # keeps only the lines where turn 1 recorded and turn 2 did not.
        used_splash = "|move|p2a: Smeargle|Splash"
        if control:
            setup = ("move splash", "move splash")
            gate = lambda L: _has(L, used_splash)
        elif name == "lastmoveparaencore":
            setup = ("move splash", "move tackle")
            gate = lambda L: _has(L, used_splash) and _has(L, "|cant|p2a: Smeargle|par")
        else:
            setup = ("move splash", "move tackle")
            gate = lambda L: _has(L, used_splash) and _has(L, "[from] confusion")
        opener = ("move thunderwave" if name == "lastmoveparaencore"
                  else "move confuseray" if name == "lastmoveconfusionencore"
                  else "move splash")
        return dict(
            p1=[_lastmove_encorer()], p2=[_lastmove_victim()],
            turns=[(opener, "move splash"), ("move splash", setup[1]),
                   ("move encore", setup[1]), ("move splash", "move splash")],
            measured=None, setup_step=(0, 1), setup_landed=gate,
            tolerate_desync=True,
            # Both halves of the gate are ~50/50 (the victim must record on
            # turn 1 AND be immobilized on turn 2), so a 4-seed band is too thin
            # to guarantee a landed line; this widens it rather than hand-picking.
            seeds=tuple(range(1000, 1024)),
            facts=lambda L: {"encored_move_is_splash": _has(L, "|-start|p2a: Smeargle|Encore")},
            expect={"encored_move_is_splash": True},
            landmark=lambda L: _has(L, "|-start|p2a: Smeargle|Encore"),
            landmark_desc="Encore applied")
    # --- phazing (Whirlwind / Roar) -----------------------------------------
    if name in ("whirlwindprotect", "roarprotect", "whirlwinddrag"):
        # gen3 inherits gen4's flag override, which ADDS protect and drops
        # reflectable, so a phaze is stopped by Protect but still goes through a
        # Substitute. `whirlwinddrag` is the no-regression control: the same turn
        # without Protect must still drag.
        blocked = name.endswith("protect")
        phaze = "roar" if name.startswith("roar") else "whirlwind"
        answer = "move protect" if blocked else "move splash"
        return dict(
            p1=[_phazer()], p2=[_phaze_target(), _blissey(), _snorlax_hazard_victim()],
            turns=[(f"move {phaze}", answer)],
            measured=0, setup_step=None, setup_landed=None,
            facts=lambda L: {"dragged": _has(L, "|drag|p2a:"),
                             "protect_activated": _has(L, "|-activate|p2a: Snorlax|Protect")},
            expect={"dragged": not blocked, "protect_activated": blocked},
            landmark=lambda L: _has(L, "|move|p1a: Skarmory"),
            landmark_desc="the phaze was used")
    if name == "whirlwindsub":
        # bypasssub: a Substitute does NOT stop the drag.
        return dict(
            p1=[_phazer()], p2=[_phaze_target(), _blissey(), _snorlax_hazard_victim()],
            turns=[("move splash", "move substitute"), ("move whirlwind", "move splash")],
            measured=1, setup_step=0,
            setup_landed=lambda L: _has(L, "|-start|p2a: Snorlax|Substitute"),
            facts=lambda L: {"dragged": _has(L, "|drag|p2a:")},
            expect={"dragged": True},
            landmark=lambda L: _has(L, "|move|p1a: Skarmory|Whirlwind"),
            landmark_desc="Whirlwind was used into the Substitute")
    # --- variable base power (Flail / Reversal) ------------------------------
    if name in ("flailladder", "reversalladder", "flailladdercontrol"):
        # gen3 declares its OWN ladder for both moves (data/mods/gen3/moves.ts:273
        # and :496): ratio = floor(hp * 48 / maxhp) clamped to >= 1, then
        # 200/150/100/80/40/20. Sandstorm chips the attacker 16 HP a turn, so a
        # fixed Flail every turn walks the ladder upward and the damage must
        # climb with it. The control swaps Flail for a FIXED-power move on the
        # same line: its damage must NOT climb, which is what separates "the
        # ladder works" from "the attacker is just getting chip-damage variance".
        control = name.endswith("control")
        move = "Body Slam" if control else ("Reversal" if name.startswith("reversal") else "Flail")
        move_id = move.replace(" ", "").lower()
        return dict(
            p1=[_flailer(move)], p2=[_sand_anvil(move)],
            turns=[(f"move {move_id}", "move splash")] * 12,
            measured=None, setup_step=None, setup_landed=None,
            facts=lambda L: {"climbs": _damage_climbs(L, "p2a", factor=3.0)},
            expect={"climbs": not control},
            landmark=lambda L: _has(L, "|-weather|Sandstorm|[upkeep]"),
            landmark_desc="the sandstorm chipped the attacker")
    if name in ("ppimmobilizedfree", "ppimmobilizedcontrol"):
        # Does an immobilized turn cost PP? Showdown deducts in runMove only
        # after the BeforeMove gate, so it must not -- and the cleanest way to
        # SEE that is to count uses to Struggle.
        #
        # The victim's only move is Mind Reader (5 base PP -> 8 with PP Ups), so
        # it is forced onto Struggle the moment the slot empties. Fake Out gives a
        # 100% deterministic flinch on turn 1, with no seed dependence at all: if
        # that flinched turn had cost a PP the victim would get only SEVEN uses
        # and Struggle a turn early, which `mindreader_uses` reads off directly.
        #
        # The control is the same line with the flinch removed, and pins the 8 on
        # its own so the number is not taken on faith.
        flinch = name == "ppimmobilizedfree"
        opener = "move fakeout" if flinch else "move splash"
        # 8 grinding turns either way; the flinch line needs one extra boundary
        # because turn 1 is consumed by the flinch and spends nothing.
        turns = [(opener, "move mindreader")]
        turns += [("move splash", "move mindreader")] * (8 if flinch else 7)
        turns += [("move splash", "move struggle")]
        return dict(
            p1=[_pp_faker()], p2=[_pp_grinder()], turns=turns,
            measured=None, setup_step=None, setup_landed=None,
            facts=lambda L: {
                "mindreader_uses": _count(L, "|move|p2a: Smeargle|Mind Reader"),
                "struggled": _has(L, "|move|p2a: Smeargle|Struggle"),
            },
            # Exactly 8, flinch or no flinch: the immobilized turn is free.
            expect={"mindreader_uses": 8, "struggled": True},
            landmark=lambda L: (_has(L, "|cant|p2a: Smeargle|flinch") if flinch
                               else _has(L, "|move|p2a: Smeargle|Mind Reader")),
            landmark_desc="Fake Out flinched the victim" if flinch else "victim moved")
    if name in ("lockedmoveppdrain", "lockedmoveppcontrol"):
        # Does a LOCKED continuation turn cost PP? Showdown guards the deduction
        # on getLockedMove(), and `twoturnmove` is one of its onLockMove
        # providers -- which is why it tags the execute turn `[from] lockedmove`.
        # So a two-turn move costs ONE PP for the pair.
        #
        # Both arms give the victim a sole move with 8 PP (5 base, x8/5 from PP
        # Ups) and drive it to Struggle, which makes the PP count observable in
        # the protocol without reading requests. Sky Attack spends those 8 PP over
        # SIXTEEN boundaries; Mind Reader spends them over eight. Had the engine's
        # per-turn charging been right, Sky Attack would manage only four uses.
        #
        # This also answers the PP-exhaustion question empirically rather than by
        # reasoning: the eighth Sky Attack STARTS on the last PP and still
        # completes its lock, because the execute turn never consults PP again.
        drain = name == "lockedmoveppdrain"
        move = "skyattack" if drain else "mindreader"
        turns = [("move splash", f"move {move}")] * (16 if drain else 8)
        turns += [("move splash", "move struggle")]
        marker = "[from] lockedmove" if drain else "|move|p2a: Smeargle|Mind Reader"
        return dict(
            p1=[_lock_tank()], p2=[_lock_charger() if drain else _pp_grinder()],
            turns=turns, measured=None, setup_step=None, setup_landed=None,
            facts=lambda L: {
                "completed_uses": sum(1 for line in L if marker in line),
                "struggled": _has(L, "|move|p2a: Smeargle|Struggle"),
                "fainted": _has(L, "|faint|"),
            },
            expect={"completed_uses": 8, "struggled": True, "fainted": False},
            landmark=lambda L: _has(L, "|-prepare|p2a: Smeargle") if drain else True,
            landmark_desc="Sky Attack charged" if drain else "")
    if name in ("solarbeamclear", "solarbeamsand", "solarbeamsun"):
        # Solar Beam's weather interaction, the root of the two-turn release
        # damage gap. Showdown weakens it in rain / sand / hail ONLY
        # (data/moves.ts solarbeam.onBasePower, inherited unchanged by gen3);
        # clear weather is full power, and sun is a different mechanism entirely
        # -- onTryMove skips the charge turn without touching power.
        #
        # The engine asked `weather_is_active(&state.weather.weather_type)`, which
        # is a self-comparison and therefore true in CLEAR weather too, so every
        # release did half damage. Blissey's 651 HP keeps the number readable and
        # survives a full-power beam.
        if name == "solarbeamsand":
            turns = [("move sandstorm", "move splash"), ("move solarbeam", "move splash"),
                     ("move solarbeam", "move splash")]
            # Halved: 60 BP. Observed 71 at seed 1000; the band covers the roll.
            band = (55, 90)
        elif name == "solarbeamsun":
            turns = [("move sunnyday", "move splash"), ("move solarbeam", "move splash")]
            band = (110, 160)
        else:
            turns = [("move solarbeam", "move splash"), ("move solarbeam", "move splash")]
            # Full power: 120 BP. Observed 136 at seed 1000. Disjoint from the
            # weakened band above, so the two cannot be confused by any roll.
            band = (110, 160)
        # A crit doubles the hit, and a HALVED crit lands squarely inside the
        # full-power non-crit band -- so the two bands stop being disjoint the
        # moment crits are allowed in. Skip those seeds rather than widen the
        # band into uselessness; the crit ratio itself is pinned natively in
        # rust/pokezero-search/tests/gen3_solarbeam_weather.rs.
        release_step = 2 if name == "solarbeamsand" else 1
        return dict(
            p1=[_solarbeam_user()], p2=[_solarbeam_wall()], turns=turns,
            measured=None, setup_step=release_step,
            setup_landed=lambda L: not _has(L, "|-crit|"),
            facts=lambda L: {
                "release_damage_in_band": (
                    band[0] <= _solarbeam_release_damage(L) <= band[1]),
                # Sun releases on the SAME turn it is selected; the other two
                # spend a turn charging first.
                "charged_first": _has(L, "|-prepare|p1a: Exeggutor"),
            },
            expect={"release_damage_in_band": True, "charged_first": True},
            landmark=lambda L: _has(L, "[from] lockedmove") or name == "solarbeamsun",
            landmark_desc="Solar Beam released")
    # --- fixed damage vs Substitute (choice_special_effect audit) -----------
    if name in ("seismictosssub", "seismictosssubbreak"):
        # substitute.onTryPrimaryHit caps the hit at the sub's HP and never
        # overflows onto the Pokemon. Snorlax's 115 HP sub survives a 100 HP
        # Seismic Toss (`-activate ... Substitute [damage]`); Dodrio's 65 HP sub
        # breaks (`-end ... Substitute`). In BOTH cases the Pokemon behind takes
        # nothing — which is the bug: the engine wrote the hit straight to it.
        breaks = name.endswith("break")
        target = "Dodrio" if breaks else "Snorlax"
        return dict(
            p1=[_tosser()], p2=[_sub_user(target)],
            turns=[("move splash", "move substitute"), ("move seismictoss", "move splash")],
            measured=1, setup_step=0,
            setup_landed=lambda L: _has(L, "|-start|p2a") and _has(L, "Substitute"),
            facts=lambda L: {
                "sub_absorbed": _has(L, "|-activate|p2a") or _has(L, "|-end|p2a"),
                "sub_broke": _has(L, f"|-end|p2a: {target}|Substitute"),
                "pokemon_hit": any(l.startswith(f"|-damage|p2a: {target}") for l in L),
            },
            expect={"sub_absorbed": True, "sub_broke": breaks, "pokemon_hit": False},
            landmark=lambda L: _has(L, "|move|p1a: Blissey|Seismic Toss"),
            landmark_desc="Seismic Toss was used into the Substitute")
    if name == "seismictosscontrol":
        # No Substitute: the same hit must land on the Pokemon for exactly the
        # attacker's level (461 - 100 = 361).
        return dict(
            p1=[_tosser()], p2=[_sub_user("Snorlax")],
            turns=[("move seismictoss", "move splash")],
            measured=0, setup_step=None, setup_landed=None,
            facts=lambda L: {"landed": _has(L, "|-damage|p2a: Snorlax|361/461")},
            expect={"landed": True},
            landmark=lambda L: _has(L, "|move|p1a: Blissey|Seismic Toss"),
            landmark_desc="Seismic Toss was used")
    if name == "seismictossghost":
        # Fighting-typed fixed damage is zero-effect vs Ghost.
        return dict(
            p1=[_tosser()], p2=[_ghost_wall()],
            turns=[("move seismictoss", "move splash")],
            measured=0, setup_step=None, setup_landed=None,
            facts=lambda L: {"immune": _has(L, "|-immune|p2a: Gengar"),
                             "damaged": any(l.startswith("|-damage|p2a: Gengar") for l in L)},
            expect={"immune": True, "damaged": False},
            landmark=lambda L: _has(L, "|move|p1a: Blissey|Seismic Toss"),
            landmark_desc="Seismic Toss was used into the Ghost")
    raise ValueError(name)


SCENARIOS = ("spinprotect", "spinconnect", "batonpass", "batonpasscontrol",
             "leechseed", "leechseedcontrol", "partialtrap", "partialtrapcontrol",
             "spikes1layer", "spikes2layers", "spikes3layers", "spikesminimum",
             "faintresiduals", "faintresidualsdeferred", "faintresidualscontrol",
             "confusionduration", "confusiondurationcontrol",
             "confusionbatonpass", "confusionbatonpasscontrol",
             "transform", "transformcontrol", "transformsub", "transformmirror",
             "transformrevert",
             "meanlook", "meanlookcontrol", "spiderweb", "meanlookghost",
             "meanlookprotect", "meanlooksub", "meanlooktrapperleaves",
             "meanlookbatonpass", "meanlooktrapperbatonpass",
             "meanlooktrapperbatonpassfreed",
             "perishladderfirsttick", "perishladder",
             "encoreduration", "encoreoutlivesshortest", "encoredurationslow",
             "encoredurationcontrol",
             "encorefailstruggle", "encorefailnolastmove", "encorefailmirrormove",
             "encoreappliescontrol",
             "toxicladder", "toxicladdercontrol", "sandminimum",
             "lastmoveparaencore", "lastmoveconfusionencore",
             "lastmoveflinchencore", "lastmoveexecutedcontrol",
             "solarbeamrelease", "solarbeamcontrol",
             "whirlwindprotect", "roarprotect", "whirlwinddrag",
             "whirlwindsub",
             "flailladder", "reversalladder", "flailladdercontrol",
             "ppimmobilizedfree", "ppimmobilizedcontrol",
             "lockedmoveppdrain", "lockedmoveppcontrol",
             "solarbeamclear", "solarbeamsand", "solarbeamsun",
             "seismictosssub", "seismictosssubbreak", "seismictosscontrol",
             "seismictossghost")


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
            # A list of indices concatenates those steps, for setups that only
            # count as landed across more than one boundary — e.g. "the victim
            # really did record a move on turn 1, AND was immobilized on turn 2",
            # where a self-hit on turn 1 would leave it with no last move at all
            # and quietly change which case the scenario is measuring.
            if isinstance(spec["setup_step"], (list, tuple)):
                setup_lines = [line for index in spec["setup_step"]
                               for line in steps[index].protocol_lines]
            else:
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
        # Some ground truth is only visible in the REQUEST, not the protocol: a
        # trap is announced once with `-activate ... trapped` and then never
        # mentioned again, so "is this seat still trapped" can only be read off
        # the switch options Showdown offers at the boundary. `request_facts`
        # sees the requests standing AFTER the measured step, and its results are
        # merged into `facts` so `expect` stays one flat mapping.
        request_facts = spec.get("request_facts")
        if request_facts is not None:
            if spec["measured"] is None:
                raise ValueError(f"{name}: request_facts needs a measured step")
            facts = {**facts, **request_facts(steps[spec["measured"]].requests)}
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
