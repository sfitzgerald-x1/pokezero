"""Live-simulator verification for the turn-merged transition-token shapes.

Each test drives the vendored gen3 Showdown (BattleStream bridge, gen3customgame with
curated teams) through the exact intra-turn oddity it verifies, then runs the
turn-merged extraction over the REAL emitted protocol and asserts both the engine
disposition and the merged representation. These are the engine facts the merge design
rests on (probed 2026-07-05):

1. SELFDESTRUCT/EXPLOSION double-faint: both faints emit before any replacement; both
   replacements are requested in ONE forceSwitch cycle and emitted back-to-back before
   ``|upkeep`` -> ONE cold replacement-pair token.
2. HAZARD SACK: a switch-in that faints to Spikes pauses the turn (forceSwitch/wait);
   the opponent's declared move FIZZLES ENTIRELY — no ``|move|`` line, no redirect to
   the replacement — even for non-targeted moves. -> NEGATED sub-block + a single
   mid-turn replacement token.
3. PURSUIT KO of a switching target: the engine hint says previously chosen switches
   continue in Gen 2-4; the declared switch completes with NO forceSwitch cycle ->
   folded in as the target's declared switch sub-block, not a replacement phase.
4. BATON PASS: click -> mid-turn forceSwitch pause -> completion tagged
   ``[from] Baton Pass`` -> the slower opponent's move executes against the NEW mon ->
   completion collapses into the passer's sub-block.

Skip gate matches the other live-sim suites (built Showdown checkout + node).
"""

import os
import shutil
import unittest
from pathlib import Path

from pokezero.showdown import parse_showdown_replay
from pokezero.transitions import (
    TOKEN_KIND_SWITCH,
    extract_transition_tokens,
)
from pokezero.turn_merged import (
    PHASE_LEAD,
    PHASE_REPLACEMENT,
    PHASE_TURN,
    SUB_BLOCK_ABSENT,
    SUB_BLOCK_ACTION,
    SUB_BLOCK_NEGATED,
    extract_turn_merged_tokens,
    flatten_turn_merged_tokens,
)


def _integration_config():
    from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT, LocalShowdownConfig

    root = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)
    if not (root / "dist" / "sim" / "index.js").exists():
        return None
    if shutil.which("node") is None:
        return None
    return LocalShowdownConfig(showdown_root=root, read_timeout_seconds=20.0)


_EVS = {"hp": 85, "atk": 85, "def": 85, "spa": 85, "spd": 85, "spe": 85}


def _mon(species, moves, ability, item=None, level=100):
    from pokezero.showdown_fixture import FixturePokemon

    return FixturePokemon(
        species=species, moves=moves, ability=ability, item=item, level=level, evs=_EVS
    )


def _run_turns(config, p1_team, p2_team, turn_choices, seed=42):
    """Multi-step driver over one bridge session; ``None`` skips a waiting seat."""
    from pokezero.showdown_fixture import _BridgeFixtureSession, pack_team

    session = _BridgeFixtureSession(config)
    try:
        session.start(
            format_id="gen3customgame",
            seed=seed,
            p1_team=pack_team(p1_team),
            p2_team=pack_team(p2_team),
        )
        session.read_until_boundary()
        for choices in turn_choices:
            if session.terminal:
                break
            session.send_choices(
                {slot: choice for slot, choice in choices.items() if choice is not None}
            )
            session.read_until_boundary()
        return tuple(session.protocol_lines)
    finally:
        session.close()


def _extract_both(lines):
    replay = parse_showdown_replay(lines, battle_id="turn-merged-engine")
    merged = extract_turn_merged_tokens(replay, perspective_slot="p1")
    per_action = extract_transition_tokens(replay, perspective_slot="p1")
    return merged, per_action


@unittest.skipUnless(_integration_config() is not None, "requires built Showdown checkout and node")
class TurnMergedEngineVerificationTest(unittest.TestCase):
    def test_explosion_double_faint_is_one_cold_replacement_pair(self) -> None:
        lines = _run_turns(
            _integration_config(),
            [_mon("Golem", ("Explosion", "Rock Throw"), "Rock Head"), _mon("Sandslash", ("Slash",), "Sand Veil")],
            [_mon("Abra", ("Tackle",), "Synchronize", level=5), _mon("Starmie", ("Surf",), "Natural Cure")],
            [
                {"p1": "move explosion", "p2": "move tackle"},
                {"p1": "switch 2", "p2": "switch 2"},
            ],
        )
        # Engine disposition: the slower victim's declared Tackle left no trace.
        self.assertFalse(any("|move|p2a: Abra|" in line for line in lines))
        merged, per_action = _extract_both(list(lines))
        self.assertEqual(
            [token.phase for token in merged], [PHASE_LEAD, PHASE_TURN, PHASE_REPLACEMENT]
        )
        explosion_turn = merged[1]
        self.assertEqual(explosion_turn.first.action, "explosion")
        self.assertTrue(explosion_turn.first.ko)
        self.assertEqual(explosion_turn.second.status, SUB_BLOCK_NEGATED)
        self.assertEqual(explosion_turn.second.actor_species, "Abra")
        pair = merged[2]
        self.assertEqual(pair.first.status, SUB_BLOCK_ACTION)
        self.assertEqual(pair.second.status, SUB_BLOCK_ACTION)
        self.assertEqual(
            {pair.first.actor_slot, pair.second.actor_slot}, {"p1", "p2"}
        )
        self.assertEqual(pair.turn, 1)
        self.assertEqual(flatten_turn_merged_tokens(merged), per_action)

    def test_hazard_sack_fizzles_the_declared_move_into_a_negated_sub_block(self) -> None:
        # Shedinja (1 HP) eats the Spikes; Skarmory's declared Drill Peck must vanish.
        lines = _run_turns(
            _integration_config(),
            [
                _mon("Magikarp", ("Splash",), "Swift Swim"),
                _mon("Shedinja", ("Shadow Ball",), "Wonder Guard"),
                _mon("Sandslash", ("Slash",), "Sand Veil"),
            ],
            [_mon("Skarmory", ("Spikes", "Drill Peck"), "Keen Eye")],
            [
                {"p1": "move splash", "p2": "move spikes"},
                {"p1": "switch 2", "p2": "move drillpeck"},
                {"p1": "switch 3", "p2": None},  # mid-turn forceSwitch; p2 waits
            ],
        )
        sack_region = "\n".join(lines[lines.index("|turn|2") :])
        self.assertIn("|switch|p1a: Shedinja|", sack_region)
        self.assertIn("|faint|p1a: Shedinja", sack_region)
        # THE disposition fact: no Drill Peck line anywhere — full fizzle, no redirect.
        self.assertFalse(any("Drill Peck" in line for line in lines))
        merged, per_action = _extract_both(list(lines))
        self.assertEqual(
            [token.phase for token in merged],
            [PHASE_LEAD, PHASE_TURN, PHASE_TURN, PHASE_REPLACEMENT],
        )
        sack_turn = merged[2]
        self.assertEqual(sack_turn.first.action, "Shedinja")
        self.assertEqual(sack_turn.second.status, SUB_BLOCK_NEGATED)
        self.assertEqual(sack_turn.second.actor_species, "Skarmory")
        replacement = merged[3]
        self.assertEqual(replacement.first.action, "Sandslash")
        self.assertEqual(replacement.second.status, SUB_BLOCK_ABSENT)
        self.assertEqual(flatten_turn_merged_tokens(merged), per_action)

    def test_pursuit_ko_completes_the_declared_switch_without_a_replacement_phase(self) -> None:
        lines = _run_turns(
            _integration_config(),
            [_mon("Tyranitar", ("Pursuit",), "Sand Stream")],
            [
                _mon("Abra", ("Tackle",), "Synchronize", level=5),
                _mon("Starmie", ("Surf",), "Natural Cure"),
            ],
            [{"p1": "move pursuit", "p2": "switch 2"}],
        )
        self.assertTrue(any(line.startswith("|-activate|p2a: Abra|move: Pursuit") for line in lines))
        # Engine's own statement of the continuation semantics.
        self.assertTrue(
            any("Previously chosen switches continue" in line for line in lines)
        )
        merged, per_action = _extract_both(list(lines))
        self.assertEqual([token.phase for token in merged], [PHASE_LEAD, PHASE_TURN])
        turn = merged[1]
        self.assertTrue(turn.first.pursuit_intercept)
        self.assertTrue(turn.first.ko)
        self.assertEqual(turn.second.status, SUB_BLOCK_ACTION)
        self.assertEqual(turn.second.kind, TOKEN_KIND_SWITCH)
        self.assertEqual(turn.second.action, "Starmie")
        self.assertEqual(flatten_turn_merged_tokens(merged), per_action)

    def test_baton_pass_completion_folds_into_the_passers_sub_block(self) -> None:
        lines = _run_turns(
            _integration_config(),
            [
                _mon("Jolteon", ("Baton Pass", "Agility"), "Volt Absorb"),
                _mon("Snorlax", ("Body Slam",), "Immunity"),
            ],
            [_mon("Skarmory", ("Drill Peck", "Spikes"), "Keen Eye")],
            [
                {"p1": "move batonpass", "p2": "move drillpeck"},
                {"p1": "switch 2", "p2": None},  # mid-turn completion; p2 waits
            ],
        )
        completion_index = next(
            index for index, line in enumerate(lines) if "[from] Baton Pass" in line
        )
        drill_peck_index = next(
            index for index, line in enumerate(lines) if line.startswith("|move|p2a: Skarmory|Drill Peck|")
        )
        # The slower opponent acts AFTER the completion, against the NEW mon.
        self.assertLess(completion_index, drill_peck_index)
        self.assertIn("p1a: Snorlax", lines[drill_peck_index])
        merged, per_action = _extract_both(list(lines))
        self.assertEqual([token.phase for token in merged], [PHASE_LEAD, PHASE_TURN])
        turn = merged[1]
        self.assertEqual(turn.first.action, "batonpass")
        self.assertEqual(turn.first.baton_pass_species, "Snorlax")
        self.assertEqual(turn.second.action, "drillpeck")
        self.assertEqual(flatten_turn_merged_tokens(merged), per_action)


if __name__ == "__main__":
    unittest.main()
