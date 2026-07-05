"""Corpus-level equivalence gate for turn-merged transition tokens.

Replays the committed 5-capture corpus (controlled foul-play games) through both
extraction modes, from BOTH perspectives, at EVERY decision boundary (incremental
prefixes, matching how the live env consumes the parser), and asserts:

- BIJECTION: ``flatten_turn_merged_tokens(merge(...))`` reproduces the per-action
  stream field-for-field. The ONLY tolerated difference is the documented trio merge —
  a token rebuilt from a *second* sub-block inherits the first mover's context trio,
  which may differ from truth only when the first mover's own action changed the trio
  (side_effect ∈ {hazard-set, hazard-clear, weather-set}). Non-trio fields must match
  exactly on every token, always.
- No PHASE_EXTRA safety-valve tokens anywhere in the corpus (every real action
  sequence is a recognized shape).
- Structural invariants: first sub-blocks are always real actions; one PHASE_TURN
  token per numbered turn per prefix at most; phases are chronologically ordered.
- The compression that motivates the change: the full-game merged stream is measured
  and must cut token count by at least a third (observed ≈45% on this corpus).

Requires a built Gen 3 Showdown checkout only for path consistency with the corpus
gate (the capture logs themselves are committed); no live simulator is used.
"""

import unittest
from dataclasses import replace
from pathlib import Path

from pokezero.showdown import parse_showdown_replay
from pokezero.transitions import (
    SIDE_EFFECT_HAZARD_CLEAR,
    SIDE_EFFECT_HAZARD_SET,
    SIDE_EFFECT_WEATHER_SET,
    extract_transition_tokens,
)
from pokezero.turn_merged import (
    PHASE_EXTRA,
    PHASE_TURN,
    SUB_BLOCK_ACTION,
    SUB_BLOCK_NEGATED,
    SUB_BLOCK_PENDING,
    extract_turn_merged_tokens,
    flatten_turn_merged_tokens,
)

CAPTURE_ROOT = Path(__file__).parent / "fixtures" / "showdown" / "capture"

# side_effect values of a FIRST sub-block that can legitimately shift the context trio
# between the two declarations of one turn (the documented lossy merge).
_TRIO_CHANGERS = {SIDE_EFFECT_HAZARD_SET, SIDE_EFFECT_HAZARD_CLEAR, SIDE_EFFECT_WEATHER_SET}


def _second_sub_block_turns(merged):
    """(turn, first.side_effect) for merged tokens whose second half is a real action."""
    return {
        id(token): token
        for token in merged
        if token.second.status == SUB_BLOCK_ACTION
    }


class TurnMergedCorpusGateTest(unittest.TestCase):
    def test_corpus_bijection_and_compression(self) -> None:
        capture_paths = sorted(CAPTURE_ROOT.glob("lines-*.log"))
        self.assertEqual(len(capture_paths), 5, capture_paths)

        prefixes_checked = 0
        trio_allowances = 0
        total_per_action = 0
        total_merged = 0

        for path in capture_paths:
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
            boundaries = [
                index
                for index, line in enumerate(lines)
                if line.startswith("|turn|") or line.startswith("|win|")
            ]
            for boundary in boundaries:
                prefix = lines[: boundary + 1]
                replay = parse_showdown_replay(prefix, battle_id=path.stem)
                for slot in ("p1", "p2"):
                    merged = extract_turn_merged_tokens(replay, perspective_slot=slot)
                    per_action = extract_transition_tokens(replay, perspective_slot=slot)
                    trio_allowances += self._assert_equivalent(merged, per_action, path, slot)
                    self._assert_structure(merged, path, slot)
                    prefixes_checked += 1
                    if boundary == boundaries[-1] and slot == "p1":
                        total_per_action += len(per_action)
                        total_merged += len(merged)

        self.assertGreater(prefixes_checked, 200)
        # The lossy-trio shape must stay rare (hazard/weather setting as first mover
        # with a second mover behind it). Bound it loosely so corpus refreshes don't
        # flake; the per-field assert above already guarantees everything else.
        self.assertLess(trio_allowances, prefixes_checked)
        # Compression: full-game streams must shrink by at least a third (measured
        # ≈45% on this corpus; the K budget flag counts exactly these tokens).
        self.assertLess(total_merged, total_per_action * 2 / 3)
        self.assertGreater(total_merged, 0)

    def _assert_equivalent(self, merged, per_action, path, slot) -> int:
        """Field-exact bijection modulo the documented trio merge; returns allowances."""
        flattened = flatten_turn_merged_tokens(merged)
        self.assertEqual(
            len(flattened), len(per_action), f"{path.name}/{slot}: token count diverged"
        )
        # Map each merged token's second-sub-block expansion positions: rebuild a
        # parallel flatten that tags which output tokens came from a SECOND sub-block
        # under a trio-changing first mover.
        allowance_keys = set()
        cursor = 0
        for token in merged:
            for position, sub in enumerate((token.first, token.second)):
                if sub.status != SUB_BLOCK_ACTION:
                    continue
                expansion = 1 + (1 if sub.cant_reason else 0) + (1 if sub.cant_reason and sub.called else 0)
                expansion += 1 if sub.baton_pass_species else 0
                if position == 1 and token.first.side_effect in _TRIO_CHANGERS:
                    allowance_keys.update(range(cursor, cursor + expansion))
                cursor += expansion
        self.assertEqual(cursor, len(flattened))

        allowances = 0
        for index, (rebuilt, original) in enumerate(zip(flattened, per_action)):
            if rebuilt == original:
                continue
            self.assertIn(
                index,
                allowance_keys,
                f"{path.name}/{slot}: non-allowed mismatch at token {index}:\n"
                f"  rebuilt:  {rebuilt}\n  original: {original}",
            )
            # Only the trio may differ, and only under a trio-changing first mover.
            self.assertEqual(
                replace(
                    rebuilt,
                    own_spikes_layers=original.own_spikes_layers,
                    opp_spikes_layers=original.opp_spikes_layers,
                    weather=original.weather,
                ),
                original,
                f"{path.name}/{slot}: non-trio field diverged at token {index}",
            )
            allowances += 1
        return allowances

    def test_mid_turn_prefixes_never_fabricate_negation(self) -> None:
        """Review MED-1's corpus-scale gate: the |turn|/|win| boundaries above never see
        a mid-turn cut, so replay every prefix ending at an action/faint line too (the
        forceSwitch-boundary class). NEGATED may appear in an OPEN turn only with a
        faint as consumption proof; otherwise the missing half is PENDING, and PENDING
        never appears in a closed turn. Bijection (with the trio allowance) holds at
        every cut."""
        capture_paths = sorted(CAPTURE_ROOT.glob("lines-*.log"))
        prefixes_checked = 0
        pending_seen = 0
        for path in capture_paths:
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
            cut_points = [
                index
                for index, line in enumerate(lines)
                if line.startswith(("|move|", "|switch|", "|cant|", "|faint|"))
            ]
            for cut in cut_points:
                prefix = lines[: cut + 1]
                turn_lines = [line for line in prefix if line.startswith("|turn|")]
                current_turn = int(turn_lines[-1].split("|")[2]) if turn_lines else 0
                open_turn_has_faint = any(
                    line.startswith("|faint|")
                    for line in prefix[
                        prefix.index(turn_lines[-1]) if turn_lines else 0 :
                    ]
                ) and not any(
                    line.startswith(("|upkeep", "|win|"))
                    for line in prefix[
                        prefix.index(turn_lines[-1]) if turn_lines else 0 :
                    ]
                )
                replay = parse_showdown_replay(prefix, battle_id=path.stem)
                for slot in ("p1", "p2"):
                    merged = extract_turn_merged_tokens(replay, perspective_slot=slot)
                    per_action = extract_transition_tokens(replay, perspective_slot=slot)
                    self._assert_equivalent(merged, per_action, path, slot)
                    for token in merged:
                        for sub in (token.first, token.second):
                            if sub.status == SUB_BLOCK_PENDING:
                                pending_seen += 1
                                # Only ever in the current, still-open turn.
                                self.assertEqual(token.turn, current_turn, path.name)
                            if sub.status == SUB_BLOCK_NEGATED and token.turn == current_turn:
                                # An open-turn negation requires the faint proof.
                                self.assertTrue(open_turn_has_faint, f"{path.name}@{cut}")
                    prefixes_checked += 1
        self.assertGreater(prefixes_checked, 500)
        self.assertGreater(pending_seen, 50)

    def _assert_structure(self, merged, path, slot) -> None:
        turn_phase_turns = []
        last_key = None
        for token in merged:
            self.assertNotEqual(
                token.phase, PHASE_EXTRA, f"{path.name}/{slot}: EXTRA safety valve fired"
            )
            self.assertEqual(
                token.first.status,
                SUB_BLOCK_ACTION,
                f"{path.name}/{slot}: first sub-block must be a real action",
            )
            if token.phase == PHASE_TURN:
                turn_phase_turns.append(token.turn)
            key = token.turn
            if last_key is not None:
                self.assertGreaterEqual(key, last_key, f"{path.name}/{slot}: phase order")
            last_key = key
        self.assertEqual(
            len(turn_phase_turns),
            len(set(turn_phase_turns)),
            f"{path.name}/{slot}: multiple PHASE_TURN tokens for one turn",
        )


if __name__ == "__main__":
    unittest.main()
