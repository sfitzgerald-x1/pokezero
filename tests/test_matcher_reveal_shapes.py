"""V5 — matcher reveal-shape sweep: every pool move's protocol form must keep its own variant.

The matcher needed leniency for exactly one display-name/id mismatch we know of (bare
``Hidden Power`` vs the typed pool ids), and that was found late, by accident, after a wrong
premise. The class question this file answers exhaustively: for EVERY pool move, does the protocol
display form match its own variant through parse -> belief -> matcher?

The emitted form is not always the pool id. Two shapes exist in gen3:

- most moves emit their dex display name, which normalizes back to the pool id;
- Hidden Power emits **bare** ``Hidden Power``, with no type
  (``tests/test_hidden_power_typed_reveal.py`` pins that against committed real-server captures),
  while the pool ids are typed (``hiddenpowergrass``). So the reveal is strictly less specific than
  the id, and the matcher must keep EVERY Hidden Power variant rather than resolve to one.

A reveal that drops its own variant is the maximally harmful failure: the set still narrows, just
to the wrong thing, and ``UNCERTAINTY`` reports confidence while the truth has been excluded.
"""

from __future__ import annotations

import unittest

from pokezero.belief import PublicBattleBeliefEngine, variant_identity
from pokezero.dex import load_showdown_dex_cached
from pokezero.randbat import load_gen3_randbat_source_cached
from pokezero.showdown import _normalize_identifier, parse_showdown_replay

from _showdown_root import requires_showdown, showdown_root


def _emitted_move_name(dex, move_id: str) -> str:
    """The name the gen3 protocol puts on a ``|move|`` line for this pool move id.

    Hidden Power is emitted WITHOUT its type in gen3, so the wire form is less specific than the
    pool id. Every other pool move emits its dex display name.
    """
    if move_id.startswith("hiddenpower") and move_id != "hiddenpower":
        return "Hidden Power"
    info = dex.move_info(move_id)
    return info.name if info is not None else move_id


@requires_showdown()
class MatcherRevealShapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = showdown_root()
        cls.dex = load_showdown_dex_cached(cls.root)
        cls.source = load_gen3_randbat_source_cached(cls.root)

    def _reveal(self, species: str, level: int, emitted: str):
        """Feed one opponent reveal of ``emitted`` and return the resulting belief."""
        lines = [
            "|player|p1|PokeZeroBot|1",
            "|player|p2|Rival|2",
            "|switch|p1a: Snorlax|Snorlax, L80|500/500",
            f"|switch|p2a: {species}|{species}, L{level}|300/300",
            "|turn|1",
            f"|move|p2a: {species}|{emitted}|p1a: Snorlax",
            "|upkeep",
        ]
        replay = parse_showdown_replay(lines, battle_id="b")
        engine = PublicBattleBeliefEngine(
            format_id="gen3randombattle", set_source=self.source
        )
        for event in replay.public_events:
            engine.ingest_event(event)
        for belief in engine.snapshot().sides["p2"]:
            if _normalize_identifier(belief.species) == _normalize_identifier(species):
                return belief
        raise AssertionError(f"no belief for {species}")

    def test_every_pool_move_reveal_keeps_its_own_variant(self) -> None:
        """The exhaustive class check: (variant, move) -> the variant survives its own reveal."""
        checked = 0
        hidden_power_cases = 0
        failures: list[str] = []
        for universe in self.source.universes.values():
            display = universe.species
            for variant in universe.variants:
                for move_id in variant.moves:
                    emitted = _emitted_move_name(self.dex, move_id)
                    if emitted == "Hidden Power":
                        hidden_power_cases += 1
                    belief = self._reveal(display, variant.level, emitted)
                    survivors = {
                        str(entry.get("variant_id"))
                        for entry in (belief.candidate_variants or ())
                    }
                    checked += 1
                    if variant.variant_id not in survivors:
                        failures.append(
                            f"{display} {variant.variant_id}: reveal {emitted!r} "
                            f"(pool id {move_id!r}) dropped its own variant; "
                            f"{len(survivors)} survivor(s)"
                        )
        # Reachability: the sweep must have reached the pool AND the lenient shape, or the
        # assertion is about nothing.
        self.assertGreater(checked, 5000, "did not reach the pool's (variant, move) pairs")
        self.assertGreater(
            hidden_power_cases, 400, "no Hidden Power reveals reached; the lenient shape is untested"
        )
        self.assertEqual(failures[:20], [], f"{len(failures)} reveal(s) dropped their own variant")

    def test_a_bare_hidden_power_reveal_never_resolves_to_a_single_type(self) -> None:
        """Bare ``Hidden Power`` is LESS specific than the pool id, so it must not pin a type.

        If the matcher resolved a bare reveal to one typed variant it would be inventing
        information the wire never carried -- and on a species with two Hidden Power types it would
        be wrong half the time while reporting certainty.
        """
        checked = 0
        for universe in self.source.universes.values():
            hp_variants = [
                variant
                for variant in universe.variants
                if any(m.startswith("hiddenpower") for m in variant.moves)
            ]
            hp_types = {
                m[len("hiddenpower"):]
                for variant in hp_variants
                for m in variant.moves
                if m.startswith("hiddenpower") and m != "hiddenpower"
            }
            if len(hp_types) < 2:
                continue
            belief = self._reveal(universe.species, universe.variants[0].level, "Hidden Power")
            survivors = belief.candidate_variants or ()
            surviving_types = {
                m[len("hiddenpower"):]
                for entry in survivors
                for m in (entry.get("moves") or ())
                if str(m).startswith("hiddenpower") and str(m) != "hiddenpower"
            }
            checked += 1
            with self.subTest(species=universe.species):
                self.assertEqual(
                    surviving_types,
                    hp_types,
                    f"{universe.species}: a bare Hidden Power reveal narrowed the surviving "
                    f"types to {sorted(surviving_types)} of {sorted(hp_types)}",
                )
        self.assertGreater(
            checked, 0, "no species with two Hidden Power types reached; assertion was vacuous"
        )

    def test_the_audit_plans_unverified_move_naming_shapes_are_settled(self) -> None:
        """Closes the audit plan's four named UNVERIFIED items -- three of them by REACHABILITY.

        The audit plan (§4.1) listed: "Disable, Encore and Spite all name a move in their protocol
        lines, `|cant|...|Disable|MOVE` names one". Read from the engine and screened against the
        pool, that premise is wrong in both directions:

        - **Disable is absent from the gen3 randbats pool**, so its `|-start|POKEMON|Disable|MOVE`
          line (`data/moves.ts:3686`) is unreachable -- and so is the `|cant|POKEMON|Disable|MOVE`
          shape (`data/moves.ts:3697`), which only Disable produces.
        - **Spite is absent too**, so its `|-activate|TARGET|move: Spite|MOVE_ID|ROLL`
          (`data/mods/gen3/moves.ts:554`) is unreachable. Worth noting what that line WOULD have
          cost if reachable: it deducts a random 2-6 PP, which the `move_uses` ledger has no term
          for -- so the reachability screen is the only reason the PP ledger is safe here.
        - **Encore IS reachable** (8+ pool species) but emits `|-start|TARGET|Encore`
          (`data/moves.ts:4746`) with **no move name at all**, so there is nothing to capture.

        Per the audit's own reachability screen, auditing an absent mechanic is wasted work. This
        test pins the screen itself, so a pool update that ADDS Disable or Spite fails here rather
        than silently reopening a gap nobody is watching.
        """
        pool_moves = {
            move
            for universe in self.source.universes.values()
            for variant in universe.variants
            for move in variant.moves
        }
        for absent in ("disable", "spite", "mimic", "sketch", "metronome", "assist", "imprison"):
            with self.subTest(move=absent):
                self.assertNotIn(
                    absent,
                    pool_moves,
                    f"{absent} is now pool-reachable; its protocol shape needs auditing and this "
                    "test's conclusion no longer holds",
                )
        self.assertIn("encore", pool_moves, "Encore left the pool; re-check the reasoning above")
        # And the reachable one is genuinely nameless on the wire: a bare Encore reveal must not
        # narrow anything, because it carries no move information.
        for universe in self.source.universes.values():
            if not any("encore" in v.moves for v in universe.variants):
                continue
            before = len(universe.variants)
            belief = self._reveal(universe.species, universe.variants[0].level, "Encore")
            # Encore itself IS a revealed move, so the set may narrow on Encore -- but never below
            # the number of variants that actually carry Encore.
            carriers = sum(1 for v in universe.variants if "encore" in v.moves)
            self.assertGreaterEqual(len(belief.candidate_variants or ()), carriers)
            self.assertLessEqual(len(belief.candidate_variants or ()), before)
            break

    def test_variant_identity_is_stable_across_the_matcher(self) -> None:
        """Survivors must carry the same ``variant_identity`` the source handed out.

        The pin/intersect machinery keys on that tuple, so an identity that changes as it passes
        through the matcher would silently drop pins.
        """
        for universe in list(self.source.universes.values())[:40]:
            expected = {variant_identity(v.to_summary()) for v in universe.variants}
            belief = self._reveal(universe.species, universe.variants[0].level, "Protect")
            for entry in belief.candidate_variants or ():
                with self.subTest(species=universe.species):
                    self.assertIn(variant_identity(entry), expected)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
