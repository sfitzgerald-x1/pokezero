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


def _carries(variant, move_id: str) -> bool:
    """Whether ``variant`` carries ``move_id`` AS THE WIRE WOULD SEE IT.

    A bare ``Hidden Power`` reveal cannot distinguish types, so for Hidden Power the carrier set is
    every variant with ANY Hidden Power -- which is exactly the leniency under test.
    """
    if move_id.startswith("hiddenpower"):
        return any(str(m).startswith("hiddenpower") for m in variant.moves)
    return move_id in variant.moves


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

    @staticmethod
    def _details(species: str, level: int) -> str:
        """The wire form of a ``details`` string.

        The engine OMITS the level token at L100 (`sim/pokemon.ts:541`), and 9 pool species are
        L100 -- the real capture reads `|switch|p2a: Luvdisc|Luvdisc, M|248/248` (this helper omits
        the gender token, which belief RECORDS -- `belief.py` populates
        `RevealedPokemonBelief.gender` from the switch line -- but which no ENCODED column reads;
        it is one of the audit plan's §1b held-but-never-encoded fields). That is the one
        details shape the plan's own table blames for the L100 zeroing defect ("the details-shape no
        fixture carried"), so this sweep emits it rather than a synthetic `, L100`.
        """
        return species if level == 100 else f"{species}, L{level}"

    def _reveal(self, species: str, level: int, emitted: str):
        """Feed one opponent reveal of ``emitted`` and return the resulting belief."""
        lines = [
            "|player|p1|PokeZeroBot|1",
            "|player|p2|Rival|2",
            "|switch|p1a: Snorlax|Snorlax, L80|500/500",
            f"|switch|p2a: {species}|{self._details(species, level)}|300/300",
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
        off_script = 0
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
                        continue

                    # CONTAINMENT ALONE IS NOT ENOUGH -- and this is the whole reason the first
                    # version of this file proved nothing. `randbat.summarize` degrades a reveal
                    # that matches NOTHING to the full species pool (the documented off-script
                    # fallback), so "the true variant survived" is satisfied identically by correct
                    # filtering and by TOTAL matcher failure. Deleting the Hidden Power leniency, or
                    # making the belief engine never record a revealed move at all, both left the
                    # old assertion green.
                    #
                    # So assert the reveal actually DISCRIMINATED: it must not have gone off-script,
                    # and the survivors must be exactly the variants that carry the move.
                    summary = self.source.summarize(
                        format_id="gen3randombattle",
                        species=universe.species,
                        revealed_moves=tuple(belief.revealed_moves),
                    )
                    self.assertIsNotNone(summary, f"no candidate summary for {display}")
                    # `expected` models the MOVE filter only, while the summary also applies
                    # revealed ability/item and the ruled-out sets. Pin that those are empty here
                    # rather than weakening the comparison: if this scenario ever grows a second
                    # reveal or an upkeep non-proc, this precondition fails loudly instead of
                    # `survivors == expected` false-failing.
                    self.assertFalse(belief.ruled_out_items or belief.ruled_out_abilities)
                    self.assertEqual(len(belief.revealed_moves), 1)
                    if summary.inconsistent:
                        off_script += 1
                        failures.append(
                            f"{display}: reveal {emitted!r} (pool id {move_id!r}) matched NO set "
                            "-- off-script fallback, so the set was re-widened to the whole pool"
                        )
                        continue
                    expected = {
                        other.variant_id
                        for other in universe.variants
                        if _carries(other, move_id)
                    }
                    if survivors != expected:
                        failures.append(
                            f"{display}: reveal {emitted!r} (pool id {move_id!r}) survivors "
                            f"{len(survivors)} != carriers {len(expected)}"
                        )
        # These floors bind against the POOL shrinking, and nothing more -- they count loop
        # iterations, so they stayed satisfied under a mutation that blinded the matcher entirely.
        # The real protection is the off-script and survivors==carriers assertions above.
        self.assertEqual(off_script, 0, "reveals that matched no set at all")
        self.assertGreater(checked, 5000, "did not reach the pool's (variant, move) pairs")
        self.assertGreater(
            hidden_power_cases, 400, "no Hidden Power reveals reached; the lenient shape is untested"
        )
        self.assertEqual(
            failures[:20],
            [],
            f"{len(failures)} reveal(s) failed containment, went off-script, or did not narrow "
            "to exactly the move's carriers",
        )

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
        """Settles the audit plan's four UNVERIFIED items: closes three, NARROWS the fourth.

        The audit plan (§4.1) listed: "Disable, Encore and Spite all name a move in their protocol
        lines, `|cant|...|Disable|MOVE` names one". Read from the engine and screened against the
        pool, that premise is wrong in both directions:

        - **Disable is absent from the gen3 randbats pool**, so its `|-start|POKEMON|Disable|MOVE`
          line is unreachable. The gen3-EFFECTIVE emitter is `data/mods/gen4/moves.ts:319`, not the
          shared `data/moves.ts:3686`: gen3's `disable.condition` inherits gen4's override, which
          replaces `onStart` wholesale. Same shape, but citing the shared handler here would be the
          mod-chain mistake this plan keeps paying for. The `|cant|POKEMON|Disable|MOVE` shape
          (`data/moves.ts:3697`) IS gen3-effective -- `onBeforeMove` is not overridden -- and is
          equally unreachable without Disable.
        - **Spite is absent too**, so its `|-activate|TARGET|move: Spite|MOVE_ID|ROLL`
          (`data/mods/gen3/moves.ts:554`) is unreachable. Worth noting what that line WOULD have
          cost if reachable: `this.random(2, 6)` returns an integer in [2, 6)
          (`sim/prng.ts:88`), so it deducts **2-5** PP, which the `move_uses` ledger has no term
          for. And gen3 emits `roll` -- the REQUESTED deduction -- rather than `deductPP`'s return,
          unlike the shared implementation (`data/moves.ts:17656` deducts a fixed 4, and `:17658` emits the actual
          `ppDeducted`), so the wire number can EXCEED the PP actually lost. The reachability
          screen is the only reason V3's ledger is safe here.
        - **Encore IS reachable** (8+ pool species) but emits `|-start|TARGET|Encore`
          (`data/moves.ts:4746`) with **no move name at all**, so there is nothing to capture.

        One `|cant|` shape that names a move IS reachable, and it is not Disable: gen3 Sleep Talk
        emits `this.add('cant', pokemon, 'nopp', randomMove.move)`
        (`data/mods/gen3/moves.ts:543`), carried by 40 pool species, and it additionally implies
        that named move is at 0 PP. `belief.py` only decomposes `|cant|...|slp`, so that move name
        and the PP fact are both dropped today. Out of scope for this file -- recorded so the
        `|cant|` item is not reported as closed when one live shape remains.

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
        encore_species = 0
        for universe in self.source.universes.values():
            carriers = {v.variant_id for v in universe.variants if "encore" in v.moves}
            if not carriers:
                continue
            encore_species += 1
            belief = self._reveal(universe.species, universe.variants[0].level, "Encore")
            survivors = {str(e.get("variant_id")) for e in (belief.candidate_variants or ())}
            with self.subTest(species=universe.species):
                # Encore itself IS a revealed move, so the set narrows to exactly its carriers.
                # Asserting equality rather than a band: the band [carriers, all] was satisfied by
                # no narrowing at all, so it could not tell filtering from doing nothing.
                self.assertEqual(survivors, carriers)
        # ALL Encore species, not the first one. Breaking after one covered 1 of 16.
        self.assertGreaterEqual(encore_species, 10, "too few Encore species reached")

    def test_variant_identity_is_stable_across_the_matcher(self) -> None:
        """Survivors must carry the same ``variant_identity`` the source handed out.

        The pin/intersect machinery keys on that tuple, so an identity that changes as it passes
        through the matcher would silently drop pins.
        """
        checked = 0
        for universe in self.source.universes.values():
            # Reveal a move the species ACTUALLY has. The first draft revealed "Protect" over the
            # first 40 universes, and 34 of them carry it in no variant -- so the reveal went
            # off-script, the set was re-widened to the whole pool, and the identity check ran over
            # unfiltered data. Picking a real move keeps this on the filtered path.
            # Pick a variant with a non-HP MOVE, not a variant with NO Hidden Power anywhere.
            # The stricter form silently skipped 52 of 220 species -- Salamence, Rayquaza, Raikou,
            # Gyarados, Dragonite, Aerodactyl, Jolteon, Crobat, Forretress, Unown among them --
            # every one of which has plenty of non-HP moves and simply carries an HP in every
            # variant. Unown stays skipped BY CONSTRUCTION -- it is the one pool species whose
            # every variant is a single Hidden Power move -- so this arm reaches 219 of 220.
            carrier = move_id = None
            for candidate in universe.variants:
                pick = next(
                    (m for m in candidate.moves if not str(m).startswith("hiddenpower")), None
                )
                if pick is not None:
                    carrier, move_id = candidate, pick
                    break
            if carrier is None:
                continue
            expected = {variant_identity(v.to_summary()) for v in universe.variants}
            belief = self._reveal(
                universe.species, carrier.level, _emitted_move_name(self.dex, move_id)
            )
            survivors = belief.candidate_variants or ()
            with self.subTest(species=universe.species):
                self.assertTrue(survivors, "no survivors; identity check would be vacuous")
                for entry in survivors:
                    self.assertIn(variant_identity(entry), expected)
            checked += 1
        self.assertGreater(checked, 200, "identity arm reached too few species")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
