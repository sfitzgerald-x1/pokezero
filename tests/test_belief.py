from pathlib import Path
import unittest

from pokezero.belief import (
    CandidateSetSummary,
    PlayerBeliefView,
    PublicBattleBeliefEngine,
    RevealedPokemonBelief,
    sample_opponent_determinizations,
)
from pokezero.showdown import parse_showdown_replay


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "showdown"


def fixture_lines(name: str) -> list[str]:
    return (FIXTURE_ROOT / name).read_text(encoding="utf-8").splitlines()


class FakeSetSource:
    def summarize(self, *, format_id, species, revealed_moves):
        candidate_count = max(1, 4 - len(revealed_moves))
        uncertainty = candidate_count / 4.0
        return CandidateSetSummary(
            species=species,
            candidate_count=candidate_count,
            uncertainty=uncertainty,
        )


class PublicBattleBeliefEngineTest(unittest.TestCase):
    def test_tracks_public_reveals_moves_and_conditions(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        snapshot = PublicBattleBeliefEngine.from_events(replay.public_events).snapshot()
        p1_species = [pokemon.species for pokemon in snapshot.side("p1")]
        xatu = next(pokemon for pokemon in snapshot.side("p1") if pokemon.species == "Xatu")

        self.assertEqual(p1_species, ["Arcanine", "Xatu"])
        self.assertFalse(snapshot.side("p1")[0].active)
        self.assertTrue(xatu.active)
        self.assertEqual(xatu.condition, "70/100")
        self.assertEqual(xatu.revealed_moves, ("Psychic",))
        self.assertNotIn("Thunder Wave", xatu.revealed_moves)

    def test_tracks_item_reveals_from_passive_tags_and_enditem(self) -> None:
        # The explicit `-item` event (Frisk/Trick/Trace) is the rare case. The common items surface
        # as an inline `[from] item:` tag on -heal/-damage (Leftovers, Life Orb) or via `-enditem`
        # (berries, Knock Off). Neither was tracked before, so Leftovers etc. never registered.
        lines = [
            "|start",
            "|switch|p1a: Blissey|Blissey, F|352/352",
            "|switch|p2a: Skarmory|Skarmory, M|271/271",
            "|turn|1",
            "|-damage|p1a: Blissey|300/352",
            "|-heal|p1a: Blissey|322/352|[from] item: Leftovers",
            "|-enditem|p2a: Skarmory|Salac Berry|[eat]",
            "|turn|2",
        ]
        replay = parse_showdown_replay(lines, battle_id="battle-gen3randombattle-1")

        snapshot = PublicBattleBeliefEngine.from_events(replay.public_events).snapshot()
        blissey = next(pokemon for pokemon in snapshot.side("p1") if pokemon.species == "Blissey")
        skarmory = next(pokemon for pokemon in snapshot.side("p2") if pokemon.species == "Skarmory")

        self.assertEqual(blissey.revealed_item, "Leftovers")
        self.assertEqual(skarmory.revealed_item, "Salac Berry")

    def test_transform_records_target_and_suppresses_copied_moves(self) -> None:
        # Ditto transforms into Blissey and then "uses" Blissey's moves. Those copied moves are not
        # part of Ditto's own set and must not be recorded (they would collapse candidate inference).
        lines = [
            "|start",
            "|switch|p1a: Ditto|Ditto, L78|100/100",
            "|switch|p2a: Blissey|Blissey, F|352/352",
            "|turn|1",
            "|move|p1a: Ditto|Transform|p2a: Blissey",
            "|-transform|p1a: Ditto|p2a: Blissey",
            "|turn|2",
            "|move|p1a: Ditto|Ice Beam|p2a: Blissey",
            "|turn|3",
        ]
        replay = parse_showdown_replay(lines, battle_id="battle-gen3randombattle-1")
        snapshot = PublicBattleBeliefEngine.from_events(replay.public_events).snapshot()
        ditto = next(pokemon for pokemon in snapshot.side("p1") if pokemon.species == "Ditto")

        self.assertTrue(ditto.transformed)
        self.assertEqual(ditto.transform_species, "Blissey")
        self.assertIn("Transform", ditto.revealed_moves)  # its own move, used directly
        self.assertNotIn("Ice Beam", ditto.revealed_moves)  # copied — not Ditto's set

    def test_transform_flag_resets_when_the_mon_leaves_the_field(self) -> None:
        # Transform ends the moment the mon switches out (it reverts to itself on the bench), so the
        # flag must clear on switch-out — not only when it returns.
        lines = [
            "|start",
            "|switch|p1a: Ditto|Ditto, L78|100/100",
            "|switch|p2a: Blissey|Blissey, F|352/352",
            "|turn|1",
            "|move|p1a: Ditto|Transform|p2a: Blissey",
            "|-transform|p1a: Ditto|p2a: Blissey",
            "|turn|2",
            "|switch|p1a: Starmie|Starmie, L78|100/100",  # Ditto leaves the field -> reverts
            "|turn|3",
        ]
        replay = parse_showdown_replay(lines, battle_id="battle-gen3randombattle-1")
        snapshot = PublicBattleBeliefEngine.from_events(replay.public_events).snapshot()
        ditto = next(pokemon for pokemon in snapshot.side("p1") if pokemon.species == "Ditto")

        self.assertFalse(ditto.active)
        self.assertFalse(ditto.transformed)
        self.assertIsNone(ditto.transform_species)

    def test_called_moves_are_not_recorded_as_revealed(self) -> None:
        # Metronome / Sleep Talk invoke another move; the invoked move is not part of the caller's
        # set. Both the "[from]move: X" and bare "[from] X" protocol forms must be guarded.
        lines = [
            "|start",
            "|switch|p1a: Clefable|Clefable, F|100/100",
            "|switch|p2a: Blissey|Blissey, F|352/352",
            "|turn|1",
            "|move|p1a: Clefable|Metronome|p1a: Clefable",
            "|move|p1a: Clefable|Fissure|p2a: Blissey|[from]move: Metronome",
            "|turn|2",
            "|move|p1a: Clefable|Sleep Talk|p1a: Clefable",
            "|move|p1a: Clefable|Ice Beam|p2a: Blissey|[from] Sleep Talk",
            "|turn|3",
        ]
        replay = parse_showdown_replay(lines, battle_id="battle-gen3randombattle-1")
        snapshot = PublicBattleBeliefEngine.from_events(replay.public_events).snapshot()
        clefable = next(pokemon for pokemon in snapshot.side("p1") if pokemon.species == "Clefable")

        self.assertIn("Metronome", clefable.revealed_moves)
        self.assertIn("Sleep Talk", clefable.revealed_moves)
        self.assertNotIn("Fissure", clefable.revealed_moves)
        self.assertNotIn("Ice Beam", clefable.revealed_moves)

    def test_locked_move_continuation_is_still_recorded(self) -> None:
        # "[from]lockedmove" (Thrash/Outrage/Petal Dance) IS the mon's own move continuing — it must
        # NOT be treated as a called move.
        lines = [
            "|start",
            "|switch|p1a: Gyarados|Gyarados, M|100/100",
            "|switch|p2a: Blissey|Blissey, F|352/352",
            "|turn|1",
            "|move|p1a: Gyarados|Thrash|p2a: Blissey",
            "|turn|2",
            "|move|p1a: Gyarados|Thrash|p2a: Blissey|[from]lockedmove",
            "|turn|3",
        ]
        replay = parse_showdown_replay(lines, battle_id="battle-gen3randombattle-1")
        snapshot = PublicBattleBeliefEngine.from_events(replay.public_events).snapshot()
        gyarados = next(pokemon for pokemon in snapshot.side("p1") if pokemon.species == "Gyarados")

        self.assertIn("Thrash", gyarados.revealed_moves)

    def test_player_view_is_overlay_ready_and_player_relative(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        view = PublicBattleBeliefEngine.from_events(replay.public_events).snapshot().for_player("p2")
        payload = view.to_overlay_payload()

        self.assertEqual(payload["self_slot"], "p2")
        self.assertEqual(payload["opponent_slot"], "p1")
        self.assertEqual(
            [pokemon["species"] for pokemon in payload["opponent_pokemon"]],
            ["Arcanine", "Xatu"],
        )
        self.assertEqual(payload["opponent_pokemon"][1]["revealed_moves"], ["Psychic"])

    def test_set_source_can_attach_candidate_summary_without_engine_coupling(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        snapshot = PublicBattleBeliefEngine.from_events(
            replay.public_events,
            format_id="gen3randombattle",
            set_source=FakeSetSource(),
        ).snapshot()
        xatu = next(pokemon for pokemon in snapshot.side("p1") if pokemon.species == "Xatu")

        self.assertEqual(xatu.candidate_set_count, 3)
        self.assertEqual(xatu.uncertainty, 0.75)


class OpponentBeliefDeterminizationTest(unittest.TestCase):
    def test_samples_player_relative_opponent_variants_in_source_order(self) -> None:
        view = PlayerBeliefView(
            self_slot="p2",
            opponent_slot="p1",
            self_pokemon=(_belief("p2", "Charizard", variants=("self-charizard",)),),
            opponent_pokemon=(
                _belief("p1", "Arcanine", variants=("arcanine-a", "arcanine-b"), active=True),
                _belief("p1", "Xatu", variants=("xatu-a", "xatu-b")),
            ),
        )

        samples = sample_opponent_determinizations(view, sample_count=4)

        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0].combination_count, 4)
        self.assertEqual(
            [
                tuple(pokemon.variant_id for pokemon in sample.opponent_pokemon)
                for sample in samples
            ],
            [
                ("arcanine-a", "xatu-a"),
                ("arcanine-b", "xatu-a"),
                ("arcanine-a", "xatu-b"),
                ("arcanine-b", "xatu-b"),
            ],
        )
        self.assertEqual(samples[0].self_slot, "p2")
        self.assertEqual(samples[0].opponent_slot, "p1")
        self.assertTrue(samples[0].opponent_pokemon[0].resolved)
        self.assertEqual(samples[0].to_payload()["unresolved_count"], 0)
        self.assertEqual([pokemon.species for pokemon in samples[0].opponent_pokemon], ["Arcanine", "Xatu"])
        self.assertNotIn("Charizard", [pokemon.species for pokemon in samples[0].opponent_pokemon])

    def test_determinization_cap_does_not_repeat_source_order_combinations(self) -> None:
        view = PlayerBeliefView(
            self_slot="p1",
            opponent_slot="p2",
            self_pokemon=(),
            opponent_pokemon=(_belief("p2", "Xatu", variants=("xatu-a", "xatu-b")),),
        )

        samples = sample_opponent_determinizations(view, sample_count=5)

        self.assertEqual(len(samples), 2)
        self.assertEqual([sample.opponent_pokemon[0].variant_id for sample in samples], ["xatu-a", "xatu-b"])

    def test_unsourced_opponent_stays_unresolved_instead_of_inventing_hidden_facts(self) -> None:
        view = PlayerBeliefView(
            self_slot="p1",
            opponent_slot="p2",
            self_pokemon=(),
            opponent_pokemon=(
                RevealedPokemonBelief(
                    showdown_slot="p2",
                    species="Tauros",
                    active=True,
                    revealed_moves=("Hidden Power",),
                    candidate_set_count=0,
                    possible_moves=("hiddenpowerghost",),
                ),
            ),
        )

        samples = sample_opponent_determinizations(view, sample_count=3)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].combination_count, 1)
        self.assertEqual(samples[0].unresolved_count, 1)
        tauros = samples[0].opponent_pokemon[0]
        self.assertFalse(tauros.resolved)
        self.assertIsNone(tauros.variant_id)
        self.assertEqual(tauros.revealed_moves, ("Hidden Power",))
        self.assertEqual(tauros.possible_moves, ("hiddenpowerghost",))
        self.assertEqual(tauros.moves, ())

    def test_rejects_non_positive_sample_count(self) -> None:
        view = PlayerBeliefView(self_slot="p1", opponent_slot="p2", self_pokemon=(), opponent_pokemon=())

        with self.assertRaisesRegex(ValueError, "sample_count"):
            sample_opponent_determinizations(view, sample_count=0)


def _belief(
    slot: str,
    species: str,
    *,
    variants: tuple[str, ...],
    active: bool = False,
) -> RevealedPokemonBelief:
    return RevealedPokemonBelief(
        showdown_slot=slot,
        species=species,
        active=active,
        candidate_set_count=len(variants),
        uncertainty=1.0,
        possible_abilities=tuple(f"ability-{variant}" for variant in variants),
        possible_items=tuple(f"item-{variant}" for variant in variants),
        possible_moves=tuple(f"move-{variant}" for variant in variants),
        candidate_variants=tuple(
            {
                "variant_id": variant,
                "source_set_id": f"{variant}-source",
                "role": "fixture",
                "level": 80,
                "moves": [f"move-{variant}"],
                "ability": f"ability-{variant}",
                "item": f"item-{variant}",
            }
            for variant in variants
        ),
    )


if __name__ == "__main__":
    unittest.main()


class ExactStateLedgerTest(unittest.TestCase):
    """Exact-state belief layer (observation_compression_design.md): PP ledger, non-proc
    pruning, sleep/clause bookkeeping, turns-in-battle, Natural Cure / Early Bird / Shield
    Dust identification, Trick mutation freeze."""

    @staticmethod
    def engine_from(lines: list[str]) -> PublicBattleBeliefEngine:
        replay = parse_showdown_replay(["|player|p1|PokeZeroBot|1", "|player|p2|Rival|2", *lines], battle_id="b")
        engine = PublicBattleBeliefEngine()
        for event in replay.public_events:
            engine.ingest_event(event)
        return engine

    @staticmethod
    def opponent(engine: PublicBattleBeliefEngine, species: str) -> RevealedPokemonBelief:
        for belief in engine.snapshot().sides["p2"]:
            if belief.species == species:
                return belief
        raise AssertionError(f"no belief for {species}")

    def test_pp_ledger_charges_uses_pressure_and_called_moves(self) -> None:
        engine = self.engine_from([
            "|switch|p1a: Entei|Entei, L78|307/307",
            "|-ability|p1a: Entei|Pressure",
            "|switch|p2a: Blissey|Blissey, L80|600/600",
            "|turn|1",
            "|move|p2a: Blissey|Ice Beam|p1a: Entei",
            "|turn|2",
            "|move|p2a: Blissey|Ice Beam|p1a: Entei",
            "|turn|3",
            "|move|p2a: Blissey|Rest|p2a: Blissey",
            "|-status|p2a: Blissey|slp|[from] move: Rest",
            "|turn|4",
            "|cant|p2a: Blissey|slp",
            "|move|p2a: Blissey|Sleep Talk|p2a: Blissey",
            "|move|p2a: Blissey|Ice Beam|p1a: Entei|[from]Sleep Talk",
        ])
        blissey = self.opponent(engine, "Blissey")
        uses = dict(blissey.move_uses)
        # Pressure doubles every charge; the called Ice Beam charges nothing extra —
        # only Sleep Talk's own line pays (x2 under Pressure).
        self.assertEqual(uses.get("icebeam"), 4)
        # self-targeted moves are never pressured in gen3
        self.assertEqual(uses.get("rest"), 1)
        self.assertEqual(uses.get("sleeptalk"), 1)

    def test_sleep_talk_move_line_charges_caller_only(self) -> None:
        engine = self.engine_from([
            "|switch|p1a: Snorlax|Snorlax, L80|500/500",
            "|switch|p2a: Arcanine|Arcanine, L80|400/400",
            "|turn|1",
            "|move|p2a: Arcanine|Rest|p2a: Arcanine",
            "|-status|p2a: Arcanine|slp|[from] move: Rest",
            "|turn|2",
            "|move|p2a: Arcanine|Sleep Talk|p2a: Arcanine",
            "|move|p2a: Arcanine|Flamethrower|p1a: Snorlax|[from]Sleep Talk",
        ])
        arcanine = self.opponent(engine, "Arcanine")
        uses = dict(arcanine.move_uses)
        self.assertEqual(uses.get("sleeptalk"), 1)
        self.assertEqual(uses.get("rest"), 1)
        self.assertNotIn("flamethrower", uses)
        # revealed moves keep existing caller suppression: Flamethrower is not set evidence
        self.assertNotIn("Flamethrower", arcanine.revealed_moves)

    def test_sleep_counters_rest_flag_and_early_bird(self) -> None:
        engine = self.engine_from([
            "|switch|p1a: Snorlax|Snorlax, L80|500/500",
            "|switch|p2a: Dodrio|Dodrio, L80|300/300",
            "|turn|1",
            "|move|p2a: Dodrio|Rest|p2a: Dodrio",
            "|-status|p2a: Dodrio|slp|[from] move: Rest",
            "|turn|2",
            "|cant|p2a: Dodrio|slp",
            "|turn|3",
            "|-curestatus|p2a: Dodrio|slp|[msg]",
        ])
        dodrio = self.opponent(engine, "Dodrio")
        self.assertIsNone(dodrio.status)
        self.assertEqual(dodrio.revealed_ability, "Early Bird")

    def test_sleep_clause_holder_is_live(self) -> None:
        engine = self.engine_from([
            "|switch|p1a: Breloom|Breloom, L80|300/300",
            "|switch|p2a: Blissey|Blissey, L80|600/600",
            "|turn|1",
            "|move|p1a: Breloom|Spore|p2a: Blissey",
            "|-status|p2a: Blissey|slp",
        ])
        from pokezero.belief import belief_key as _bk
        self.assertEqual(engine.sleep_clause_holders["p1"], _bk("p2", "Blissey"))
        engine2 = self.engine_from([
            "|switch|p1a: Breloom|Breloom, L80|300/300",
            "|switch|p2a: Blissey|Blissey, L80|600/600",
            "|turn|1",
            "|move|p1a: Breloom|Spore|p2a: Blissey",
            "|-status|p2a: Blissey|slp",
            "|turn|2",
            "|-curestatus|p2a: Blissey|slp|[msg]",
        ])
        self.assertIsNone(engine2.sleep_clause_holders["p1"])

    def test_non_proc_pruning_family(self) -> None:
        engine = self.engine_from([
            "|switch|p1a: Tyranitar|Tyranitar, L76|350/350",
            "|switch|p2a: Salamence|Salamence, L76|320/320",
            "|turn|1",
            "|move|p1a: Tyranitar|Rock Slide|p2a: Salamence",
            "|-damage|p2a: Salamence|60/320",
            "|move|p2a: Salamence|Toxic|p1a: Tyranitar",
            "|-status|p1a: Tyranitar|tox",
            "|upkeep",
        ])
        salamence = self.opponent(engine, "Salamence")
        # damaged end of turn, no Leftovers heal, ended at <=25% with no berry, healthy status
        self.assertIn("leftovers", salamence.ruled_out_items)
        self.assertIn("salacberry", salamence.ruled_out_items)
        self.assertNotIn("lumberry", salamence.ruled_out_items)
        # our own statused Tyranitar (p1 side) also gets Lum ruled out
        tyranitar = [b for b in engine.snapshot().sides["p1"] if b.species == "Tyranitar"][0]
        self.assertIn("lumberry", tyranitar.ruled_out_items)

    def test_leftovers_heal_blocks_pruning_and_reveal_registers(self) -> None:
        engine = self.engine_from([
            "|switch|p1a: Tyranitar|Tyranitar, L76|350/350",
            "|switch|p2a: Blissey|Blissey, L80|600/600",
            "|turn|1",
            "|move|p1a: Tyranitar|Rock Slide|p2a: Blissey",
            "|-damage|p2a: Blissey|400/600",
            "|-heal|p2a: Blissey|437/600|[from] item: Leftovers",
            "|upkeep",
        ])
        blissey = self.opponent(engine, "Blissey")
        self.assertNotIn("leftovers", blissey.ruled_out_items)
        self.assertEqual(blissey.revealed_item, "Leftovers")

    def test_turns_active_and_switch_reset(self) -> None:
        engine = self.engine_from([
            "|switch|p1a: Snorlax|Snorlax, L80|500/500",
            "|switch|p2a: Skarmory|Skarmory, L80|300/300",
            "|turn|1",
            "|turn|2",
            "|switch|p2a: Blissey|Blissey, L80|600/600",
            "|turn|3",
        ])
        skarmory = self.opponent(engine, "Skarmory")
        blissey = self.opponent(engine, "Blissey")
        self.assertEqual(skarmory.turns_active, 2)
        self.assertEqual(blissey.turns_active, 1)

    def test_natural_cure_detected_on_clean_reentry(self) -> None:
        engine = self.engine_from([
            "|switch|p1a: Breloom|Breloom, L80|300/300",
            "|switch|p2a: Starmie|Starmie, L80|280/280",
            "|turn|1",
            "|move|p1a: Breloom|Spore|p2a: Starmie",
            "|-status|p2a: Starmie|slp",
            "|turn|2",
            "|switch|p2a: Blissey|Blissey, L80|600/600",
            "|turn|3",
            "|switch|p2a: Starmie|Starmie, L80|280/280",
        ])
        starmie = self.opponent(engine, "Starmie")
        self.assertEqual(starmie.revealed_ability, "Natural Cure")
        self.assertIsNone(starmie.status)

    def test_natural_cure_not_claimed_after_heal_bell(self) -> None:
        engine = self.engine_from([
            "|switch|p1a: Breloom|Breloom, L80|300/300",
            "|switch|p2a: Starmie|Starmie, L80|280/280",
            "|turn|1",
            "|move|p1a: Breloom|Spore|p2a: Starmie",
            "|-status|p2a: Starmie|slp",
            "|turn|2",
            "|switch|p2a: Miltank|Miltank, L80|400/400",
            "|move|p2a: Miltank|Heal Bell|p2a: Miltank",
            "|turn|3",
            "|switch|p2a: Starmie|Starmie, L80|280/280",
        ])
        starmie = self.opponent(engine, "Starmie")
        self.assertIsNone(starmie.revealed_ability)

    def test_trick_mutation_freezes_non_proc_pruning(self) -> None:
        engine = self.engine_from([
            "|switch|p1a: Kecleon|Kecleon, L80|300/300",
            "|switch|p2a: Blissey|Blissey, L80|600/600",
            "|turn|1",
            "|move|p1a: Kecleon|Trick|p2a: Blissey",
            "|-activate|p1a: Kecleon|move: Trick|[of] p2a: Blissey",
            "|-item|p2a: Blissey|Choice Band|[from] move: Trick",
            "|move|p1a: Kecleon|Shadow Ball|p2a: Blissey",
            "|-damage|p2a: Blissey|400/600",
            "|upkeep",
        ])
        blissey = self.opponent(engine, "Blissey")
        self.assertTrue(blissey.item_mutated)
        self.assertEqual(blissey.revealed_item, "Choice Band")
        # pruning frozen post-mutation: no Leftovers rule-out despite a damaged, heal-free turn
        self.assertNotIn("leftovers", blissey.ruled_out_items)
        # A swap, not a removal: the mon HOLDS an item that is not the sampled
        # assignment. The -item line names the current item, so worlds can
        # substitute it (the Trick-swap current-item override).
        self.assertFalse(blissey.item_removed)
        self.assertFalse(blissey.to_overlay_payload()["item_removed"])
        self.assertEqual(blissey.current_public_item, "Choice Band")
        self.assertEqual(blissey.to_overlay_payload()["current_public_item"], "Choice Band")

    def test_knock_off_marks_removal_not_just_mutation(self) -> None:
        engine = self.engine_from([
            "|switch|p1a: Tyranitar|Tyranitar, L74|340/340",
            "|switch|p2a: Blissey|Blissey, L80|600/600",
            "|turn|1",
            "|move|p1a: Tyranitar|Knock Off|p2a: Blissey",
            "|-damage|p2a: Blissey|580/600",
            "|-enditem|p2a: Blissey|Leftovers|[from] move: Knock Off|[of] p1a: Tyranitar",
            "|upkeep",
        ])
        blissey = self.opponent(engine, "Blissey")
        self.assertTrue(blissey.item_mutated)
        # The removal distinction: current public item state is "holds nothing",
        # which a determinized world CAN express by clearing the sampled item.
        self.assertTrue(blissey.item_removed)
        # The -enditem line names the removed item: the original assignment stays known.
        self.assertEqual(blissey.revealed_item, "Leftovers")
        self.assertTrue(blissey.to_overlay_payload()["item_removed"])

    def test_trick_that_takes_without_giving_is_a_removal(self) -> None:
        # Trick against an itemless partner: the victim's item is taken and
        # nothing replaces it — same representable end state as Knock Off.
        engine = self.engine_from([
            "|switch|p1a: Kecleon|Kecleon, L80|300/300",
            "|switch|p2a: Blissey|Blissey, L80|600/600",
            "|turn|1",
            "|move|p1a: Kecleon|Trick|p2a: Blissey",
            "|-activate|p1a: Kecleon|move: Trick|[of] p2a: Blissey",
            "|-enditem|p2a: Blissey|Leftovers|[from] move: Trick",
            "|upkeep",
        ])
        blissey = self.opponent(engine, "Blissey")
        self.assertTrue(blissey.item_mutated)
        self.assertTrue(blissey.item_removed)
        self.assertIsNone(blissey.current_public_item)

    def test_trick_after_knock_off_clears_the_removal(self) -> None:
        # Once a later Trick hands the knocked-off mon an item again, it holds
        # something that is not the sampled assignment — with the current item
        # protocol-named (override-eligible). Defensive only: the real gen3
        # sim REFUSES to Trick with a knocked-off mon involved (gen<=4
        # itemKnockedOff gate; probed: |move|...|Trick||[still] + |-fail|).
        engine = self.engine_from([
            "|switch|p1a: Tyranitar|Tyranitar, L74|340/340",
            "|switch|p2a: Blissey|Blissey, L80|600/600",
            "|turn|1",
            "|move|p1a: Tyranitar|Knock Off|p2a: Blissey",
            "|-damage|p2a: Blissey|580/600",
            "|-enditem|p2a: Blissey|Leftovers|[from] move: Knock Off|[of] p1a: Tyranitar",
            "|upkeep",
            "|turn|2",
            "|switch|p1a: Kecleon|Kecleon, L80|300/300",
            "|turn|3",
            "|move|p1a: Kecleon|Trick|p2a: Blissey",
            "|-activate|p1a: Kecleon|move: Trick|[of] p2a: Blissey",
            "|-item|p2a: Blissey|Choice Band|[from] move: Trick",
            "|upkeep",
        ])
        blissey = self.opponent(engine, "Blissey")
        self.assertTrue(blissey.item_mutated)
        self.assertFalse(blissey.item_removed)
        self.assertEqual(blissey.current_public_item, "Choice Band")

    @staticmethod
    def self_side(engine: PublicBattleBeliefEngine, species: str) -> RevealedPokemonBelief:
        for belief in engine.snapshot().sides["p1"]:
            if belief.species == species:
                return belief
        raise AssertionError(f"no belief for {species}")

    def test_trick_full_swap_confirms_current_item_on_both_mons(self) -> None:
        # Verbatim gen3 protocol (live probe 2026-07-19): a both-items Trick
        # emits ONE -item line per mon, each naming the mon's CURRENT item —
        # the exchange is fully public on both halves.
        engine = self.engine_from([
            "|switch|p1a: Alakazam|Alakazam, L80|219/219",
            "|switch|p2a: Furret|Furret, L80|267/267",
            "|turn|1",
            "|move|p1a: Alakazam|Trick|p2a: Furret",
            "|-activate|p1a: Alakazam|move: Trick|[of] p2a: Furret",
            "|-item|p2a: Furret|Choice Band|[from] move: Trick",
            "|-item|p1a: Alakazam|Petaya Berry|[from] move: Trick",
            "|upkeep",
        ])
        furret = self.opponent(engine, "Furret")
        self.assertTrue(furret.item_mutated)
        self.assertFalse(furret.item_removed)
        self.assertEqual(furret.current_public_item, "Choice Band")
        alakazam = self.self_side(engine, "Alakazam")
        self.assertTrue(alakazam.item_mutated)
        self.assertFalse(alakazam.item_removed)
        self.assertEqual(alakazam.current_public_item, "Petaya Berry")

    def test_trick_give_half_silent_enditem_is_a_removal(self) -> None:
        # Verbatim: Trick into an itemless target — the giver's half is a
        # [silent] -enditem naming the item it handed away; the giver is now
        # publicly itemless.
        engine = self.engine_from([
            "|switch|p1a: Alakazam|Alakazam, L80|219/219",
            "|switch|p2a: Furret|Furret, L80|267/267",
            "|turn|1",
            "|move|p1a: Alakazam|Trick|p2a: Furret",
            "|-activate|p1a: Alakazam|move: Trick|[of] p2a: Furret",
            "|-item|p2a: Furret|Choice Band|[from] move: Trick",
            "|-enditem|p1a: Alakazam|Choice Band|[silent]|[from] move: Trick",
            "|upkeep",
        ])
        alakazam = self.self_side(engine, "Alakazam")
        self.assertTrue(alakazam.item_mutated)
        self.assertTrue(alakazam.item_removed)
        self.assertIsNone(alakazam.current_public_item)
        furret = self.opponent(engine, "Furret")
        self.assertEqual(furret.current_public_item, "Choice Band")

    def test_trick_take_half_silent_enditem_is_a_removal(self) -> None:
        # Verbatim: an item-taking Trick (itemless user) — the victim's half
        # is a [silent] -enditem; the victim is publicly itemless and the
        # taker's -item names its new current item.
        engine = self.engine_from([
            "|switch|p1a: Alakazam|Alakazam, L80|219/219",
            "|switch|p2a: Furret|Furret, L80|267/267",
            "|turn|1",
            "|move|p1a: Alakazam|Trick|p2a: Furret",
            "|-activate|p1a: Alakazam|move: Trick|[of] p2a: Furret",
            "|-enditem|p2a: Furret|Leftovers|[silent]|[from] move: Trick",
            "|-item|p1a: Alakazam|Leftovers|[from] move: Trick",
            "|upkeep",
        ])
        furret = self.opponent(engine, "Furret")
        self.assertTrue(furret.item_mutated)
        self.assertTrue(furret.item_removed)
        self.assertIsNone(furret.current_public_item)
        alakazam = self.self_side(engine, "Alakazam")
        self.assertEqual(alakazam.current_public_item, "Leftovers")

    def test_berry_eat_marks_removed_without_mutation(self) -> None:
        # Verbatim pinch-berry eat: the item is publicly GONE (worlds must not
        # hand it back), but it was the original assignment — no mutation, and
        # revealed_item keeps pinning variant matching.
        engine = self.engine_from([
            "|switch|p1a: Blissey|Blissey, L80|539/539",
            "|switch|p2a: Furret|Furret, L80|267/267",
            "|turn|1",
            "|move|p1a: Blissey|Seismic Toss|p2a: Furret",
            "|-damage|p2a: Furret|27/267",
            "|-enditem|p2a: Furret|Petaya Berry|[eat]",
            "|-boost|p2a: Furret|spa|1|[from] item: Petaya Berry",
            "|upkeep",
        ])
        furret = self.opponent(engine, "Furret")
        self.assertFalse(furret.item_mutated)
        self.assertTrue(furret.item_removed)
        self.assertIsNone(furret.current_public_item)
        self.assertEqual(furret.revealed_item, "Petaya Berry")
        payload = furret.to_overlay_payload()
        self.assertTrue(payload["item_removed"])
        self.assertIsNone(payload["current_public_item"])

    def test_chesto_rest_eat_marks_removed(self) -> None:
        # Verbatim Chesto-Rest consumption on the SELF side.
        engine = self.engine_from([
            "|switch|p1a: Snorlax|Snorlax, L80|387/387",
            "|switch|p2a: Blissey|Blissey, L80|539/539",
            "|turn|1",
            "|move|p2a: Blissey|Seismic Toss|p1a: Snorlax",
            "|-damage|p1a: Snorlax|307/387",
            "|move|p1a: Snorlax|Rest|p1a: Snorlax",
            "|-status|p1a: Snorlax|slp|[from] move: Rest",
            "|-heal|p1a: Snorlax|387/387 slp|[silent]",
            "|-enditem|p1a: Snorlax|Chesto Berry|[eat]",
            "|-curestatus|p1a: Snorlax|slp|[msg]",
            "|upkeep",
        ])
        snorlax = self.self_side(engine, "Snorlax")
        self.assertFalse(snorlax.item_mutated)
        self.assertTrue(snorlax.item_removed)
        self.assertEqual(snorlax.revealed_item, "Chesto Berry")

    def test_tricked_berry_eaten_becomes_a_removal(self) -> None:
        # The seed-7013 composition: Trick puts a Petaya on the mon (override
        # state), the mon later eats it at pinch — final public state is
        # itemless (removal wins; the override must not linger).
        engine = self.engine_from([
            "|switch|p1a: Alakazam|Alakazam, L80|219/219",
            "|switch|p2a: Furret|Furret, L80|267/267",
            "|turn|1",
            "|move|p1a: Alakazam|Trick|p2a: Furret",
            "|-activate|p1a: Alakazam|move: Trick|[of] p2a: Furret",
            "|-item|p2a: Furret|Petaya Berry|[from] move: Trick",
            "|-item|p1a: Alakazam|Leftovers|[from] move: Trick",
            "|upkeep",
            "|turn|2",
            "|move|p1a: Alakazam|Seismic Toss|p2a: Furret",
            "|-damage|p2a: Furret|27/267",
            "|-enditem|p2a: Furret|Petaya Berry|[eat]",
            "|-boost|p2a: Furret|spa|1|[from] item: Petaya Berry",
            "|upkeep",
        ])
        furret = self.opponent(engine, "Furret")
        self.assertTrue(furret.item_mutated)  # Trick history stands
        self.assertTrue(furret.item_removed)  # ... but it now holds nothing
        self.assertIsNone(furret.current_public_item)

    def test_trick_then_knock_off_removal_wins(self) -> None:
        # Verbatim composition: the Tricked item is knocked off — removal ends
        # the override.
        engine = self.engine_from([
            "|switch|p1a: Alakazam|Alakazam, L80|219/219",
            "|switch|p2a: Furret|Furret, L80|267/267",
            "|turn|1",
            "|move|p1a: Alakazam|Trick|p2a: Furret",
            "|-activate|p1a: Alakazam|move: Trick|[of] p2a: Furret",
            "|-item|p2a: Furret|Choice Band|[from] move: Trick",
            "|-item|p1a: Alakazam|Petaya Berry|[from] move: Trick",
            "|upkeep",
            "|turn|2",
            "|move|p1a: Alakazam|Knock Off|p2a: Furret",
            "|-damage|p2a: Furret|243/267",
            "|-enditem|p2a: Furret|Choice Band|[from] move: Knock Off|[of] p1a: Alakazam",
            "|upkeep",
        ])
        furret = self.opponent(engine, "Furret")
        self.assertTrue(furret.item_mutated)
        self.assertTrue(furret.item_removed)
        self.assertIsNone(furret.current_public_item)
        # The self half of the original exchange is untouched by the Knock Off.
        alakazam = self.self_side(engine, "Alakazam")
        self.assertEqual(alakazam.current_public_item, "Petaya Berry")

    def test_unexpected_enditem_move_source_fails_closed(self) -> None:
        # Hardening (PR #741 review): a pool change to Thief/Covet must not be
        # silently treated as a plain reveal (worlds would hand the stolen
        # item back). Unaudited -enditem move sources mark the mutation with
        # no removal and no confirmed current item -> construction blocks.
        engine = self.engine_from([
            "|switch|p1a: Sneasel|Sneasel, L80|250/250",
            "|switch|p2a: Blissey|Blissey, L80|539/539",
            "|turn|1",
            "|move|p1a: Sneasel|Covet|p2a: Blissey",
            "|-damage|p2a: Blissey|500/539",
            "|-enditem|p2a: Blissey|Leftovers|[from] move: Covet|[of] p1a: Sneasel",
            "|upkeep",
        ])
        blissey = self.opponent(engine, "Blissey")
        self.assertTrue(blissey.item_mutated)
        self.assertFalse(blissey.item_removed)
        self.assertIsNone(blissey.current_public_item)

    def test_unexpected_item_move_source_fails_closed(self) -> None:
        # The receiving half of an unaudited item-moving move (the Covet/Thief
        # stealer's -item line): mutation with NO confirmed current item.
        engine = self.engine_from([
            "|switch|p1a: Sneasel|Sneasel, L80|250/250",
            "|switch|p2a: Blissey|Blissey, L80|539/539",
            "|turn|1",
            "|move|p2a: Blissey|Thief|p1a: Sneasel",
            "|-damage|p1a: Sneasel|220/250",
            "|-item|p2a: Blissey|Quick Claw|[from] move: Thief|[of] p1a: Sneasel",
            "|upkeep",
        ])
        blissey = self.opponent(engine, "Blissey")
        self.assertTrue(blissey.item_mutated)
        self.assertFalse(blissey.item_removed)
        self.assertIsNone(blissey.current_public_item)

    def test_mudshot_shield_dust_identification_requires_clean_hit(self) -> None:
        class DustSource(FakeSetSource):
            def summarize(self, *, format_id, species, revealed_moves, **kwargs):
                summary = super().summarize(format_id=format_id, species=species, revealed_moves=revealed_moves)
                return CandidateSetSummary(
                    species=species,
                    candidate_count=summary.candidate_count,
                    uncertainty=summary.uncertainty,
                    possible_abilities=("Shield Dust", "Swarm") if species == "Dustox" else ("Natural Cure",),
                )

        lines = [
            "|switch|p1a: Swampert|Swampert, L78|340/340",
            "|switch|p2a: Dustox|Dustox, L84|280/280",
            "|turn|1",
            "|move|p1a: Swampert|Mud Shot|p2a: Dustox",
            "|-damage|p2a: Dustox|200/280",
            "|upkeep",
        ]
        replay = parse_showdown_replay(["|player|p1|PokeZeroBot|1", "|player|p2|Rival|2", *lines], battle_id="b")
        engine = PublicBattleBeliefEngine(format_id="gen3randombattle", set_source=DustSource())
        for event in replay.public_events:
            engine.ingest_event(event)
        dustox = self.opponent(engine, "Dustox")
        self.assertEqual(dustox.revealed_ability, "Shield Dust")
        # and the drop firing cancels the inference
        lines_dropped = lines[:-1] + ["|-unboost|p2a: Dustox|spe|1", "|upkeep"]
        replay2 = parse_showdown_replay(["|player|p1|PokeZeroBot|1", "|player|p2|Rival|2", *lines_dropped], battle_id="b")
        engine2 = PublicBattleBeliefEngine(format_id="gen3randombattle", set_source=DustSource())
        for event in replay2.public_events:
            engine2.ingest_event(event)
        self.assertIsNone(self.opponent(engine2, "Dustox").revealed_ability)


    def test_residual_chip_does_not_manufacture_leftovers_evidence(self) -> None:
        # Review finding: gen3 runs the Leftovers slot before status chip. A full-HP mon that
        # gets toxic'd and only chips during residuals gave Leftovers no chance to fire.
        engine = self.engine_from([
            "|switch|p1a: Blissey|Blissey, L80|600/600",
            "|switch|p2a: Lugia|Lugia, L70|400/400",
            "|turn|1",
            "|move|p1a: Blissey|Toxic|p2a: Lugia",
            "|-status|p2a: Lugia|tox",
            "|-damage|p2a: Lugia|375/400 tox|[from] psn",
            "|upkeep",
        ])
        lugia = self.opponent(engine, "Lugia")
        self.assertNotIn("leftovers", lugia.ruled_out_items)
        # but Lum is still correctly ruled out (status stuck)
        self.assertIn("lumberry", lugia.ruled_out_items)

    def test_spikes_chip_counts_as_action_phase_for_leftovers(self) -> None:
        engine = self.engine_from([
            "|switch|p1a: Skarmory|Skarmory, L80|300/300",
            "|switch|p2a: Blissey|Blissey, L80|600/600",
            "|turn|1",
            "|move|p1a: Skarmory|Spikes|p2a: Blissey",
            "|-sidestart|p2: Rival|Spikes",
            "|turn|2",
            "|switch|p2a: Starmie|Starmie, L80|280/280",
            "|-damage|p2a: Starmie|245/280|[from] Spikes",
            "|upkeep",
        ])
        starmie = self.opponent(engine, "Starmie")
        self.assertIn("leftovers", starmie.ruled_out_items)

    def test_solar_beam_release_charges_once(self) -> None:
        engine = self.engine_from([
            "|switch|p1a: Snorlax|Snorlax, L80|500/500",
            "|switch|p2a: Venusaur|Venusaur, L80|360/360",
            "|turn|1",
            "|move|p2a: Venusaur|Solar Beam||[still]",
            "|-prepare|p2a: Venusaur|Solar Beam",
            "|turn|2",
            "|move|p2a: Venusaur|Solar Beam|p1a: Snorlax|[from] lockedmove",
            "|-damage|p1a: Snorlax|300/500",
        ])
        venusaur = self.opponent(engine, "Venusaur")
        self.assertEqual(dict(venusaur.move_uses).get("solarbeam"), 1)

    def test_sethp_updates_condition_for_pinch_sweep(self) -> None:
        engine = self.engine_from([
            "|switch|p1a: Misdreavus|Misdreavus, L80|280/280",
            "|switch|p2a: Snorlax|Snorlax, L80|500/500",
            "|turn|1",
            "|move|p1a: Misdreavus|Pain Split|p2a: Snorlax",
            "|-sethp|p2a: Snorlax|180/500|[from] move: Pain Split|[silent]",
            "|-sethp|p1a: Misdreavus|230/280|[from] move: Pain Split",
            "|upkeep",
        ])
        snorlax = self.opponent(engine, "Snorlax")
        self.assertEqual(snorlax.condition, "180/500")

    def test_mudshot_ko_never_claims_shield_dust(self) -> None:
        lines = [
            "|switch|p1a: Swampert|Swampert, L78|340/340",
            "|switch|p2a: Dustox|Dustox, L84|20/280",
            "|turn|1",
            "|move|p1a: Swampert|Mud Shot|p2a: Dustox",
            "|-damage|p2a: Dustox|0 fnt",
            "|faint|p2a: Dustox",
            "|upkeep",
        ]
        replay = parse_showdown_replay(["|player|p1|PokeZeroBot|1", "|player|p2|Rival|2", *lines], battle_id="b")
        engine = PublicBattleBeliefEngine()
        for event in replay.public_events:
            engine.ingest_event(event)
        dustox = self.opponent(engine, "Dustox")
        self.assertIsNone(dustox.revealed_ability)


class AbsorbAbilityAttributionTest(unittest.TestCase):
    """Ability-evidence attribution for the absorb-class protocol shapes.

    Every protocol line here (except the synthetic conflict case) is VERBATIM
    from the live gen3customgame captures of the absorb audit (2026-07-19,
    probe3_showdown_capture.out): Showdown's heal convention makes ``[of]``
    the MOVE SOURCE (sim/battle.ts:2311), so the pre-fix ``[of]``-is-holder
    read pinned the absorb ability on the ATTACKER and destroyed its
    previously confirmed ability.
    """

    @staticmethod
    def engine_from(lines: list[str], set_source=None) -> PublicBattleBeliefEngine:
        replay = parse_showdown_replay(
            ["|player|p1|PokeZeroBot|1", "|player|p2|Rival|2", *lines], battle_id="b"
        )
        engine = PublicBattleBeliefEngine(
            format_id="gen3randombattle" if set_source is not None else None,
            set_source=set_source,
        )
        for event in replay.public_events:
            engine.ingest_event(event)
        return engine

    @staticmethod
    def belief(engine: PublicBattleBeliefEngine, slot: str, species: str):
        for belief in engine.snapshot().sides[slot]:
            if belief.species == species:
                return belief
        raise AssertionError(f"no belief for {slot} {species}")

    _VOLTABSORB_HEAL_LINES = [
        "|switch|p1a: Zapdos|Zapdos, L80|275/275",
        "|switch|p2a: Lanturn|Lanturn, L80, F|331/331",
        "|-ability|p1a: Zapdos|Pressure|[silent]",
        "|turn|1",
        "|move|p1a: Zapdos|Thunderbolt|p2a: Lanturn",
        "|-heal|p2a: Lanturn|331/331|[from] ability: Volt Absorb|[of] p1a: Zapdos",
        "|turn|2",
    ]

    def test_absorb_heal_pins_the_healed_mon_and_preserves_the_attacker(self) -> None:
        engine = self.engine_from(self._VOLTABSORB_HEAL_LINES)
        lanturn = self.belief(engine, "p2", "Lanturn")
        zapdos = self.belief(engine, "p1", "Zapdos")
        # The healed mon is the ability holder — never the ``[of]`` attacker.
        self.assertEqual(lanturn.revealed_ability, "Volt Absorb")
        # Zapdos's protocol-confirmed Pressure survives (the live-captured bug
        # overwrote it with Volt Absorb).
        self.assertEqual(zapdos.revealed_ability, "Pressure")
        self.assertFalse(
            [e for e in zapdos.evidence if "Volt Absorb" in (e.detail or "")],
            "no Volt Absorb evidence may attach to the attacker",
        )

    def test_immune_pins_the_holder(self) -> None:
        engine = self.engine_from([
            "|switch|p1a: Zapdos|Zapdos, L80|275/275",
            "|switch|p2a: Lanturn|Lanturn, L80, F|331/331",
            "|-ability|p1a: Zapdos|Pressure|[silent]",
            "|turn|1",
            "|move|p1a: Zapdos|Thunderbolt|p2a: Lanturn",
            "|-immune|p2a: Lanturn|[from] ability: Volt Absorb",
            "|turn|2",
        ])
        self.assertEqual(self.belief(engine, "p2", "Lanturn").revealed_ability, "Volt Absorb")
        self.assertEqual(self.belief(engine, "p1", "Zapdos").revealed_ability, "Pressure")

    def test_flashfire_start_pins_the_holder(self) -> None:
        engine = self.engine_from([
            "|switch|p1a: Charizard|Charizard, L80, F|256/256",
            "|switch|p2a: Houndoom|Houndoom, L80, M|251/251",
            "|turn|1",
            "|move|p1a: Charizard|Flamethrower|p2a: Houndoom",
            "|-start|p2a: Houndoom|ability: Flash Fire",
            "|turn|2",
        ])
        self.assertEqual(self.belief(engine, "p2", "Houndoom").revealed_ability, "Flash Fire")
        self.assertIsNone(self.belief(engine, "p1", "Charizard").revealed_ability)

    def test_waterabsorb_heal_shape(self) -> None:
        engine = self.engine_from([
            "|switch|p1a: Suicune|Suicune, L80|291/291",
            "|switch|p2a: Quagsire|Quagsire, L80, F|283/283",
            "|-ability|p1a: Suicune|Pressure|[silent]",
            "|turn|1",
            "|move|p1a: Suicune|Surf|p2a: Quagsire",
            "|-heal|p2a: Quagsire|283/283|[from] ability: Water Absorb|[of] p1a: Suicune",
            "|turn|2",
        ])
        self.assertEqual(self.belief(engine, "p2", "Quagsire").revealed_ability, "Water Absorb")
        self.assertEqual(self.belief(engine, "p1", "Suicune").revealed_ability, "Pressure")

    def test_conflicting_ability_claim_keeps_earlier_confirmation_and_flags(self) -> None:
        # Synthetic conflict shape (the attribution fix removes the captured
        # route to it): a later raw-line claim of a DIFFERENT ability for a mon
        # with a protocol-confirmed one must not overwrite — keep the earlier
        # confirmation, append a conflict flag.
        engine = self.engine_from([
            "|switch|p1a: Zapdos|Zapdos, L80|275/275",
            "|switch|p2a: Lanturn|Lanturn, L80, F|331/331",
            "|-ability|p1a: Zapdos|Pressure|[silent]",
            "|turn|1",
            "|-heal|p1a: Zapdos|275/275|[from] ability: Volt Absorb",
            "|turn|2",
        ])
        zapdos = self.belief(engine, "p1", "Zapdos")
        self.assertEqual(zapdos.revealed_ability, "Pressure")
        conflicts = [e for e in zapdos.evidence if e.kind == "conflicting-ability-evidence"]
        self.assertEqual(len(conflicts), 1)
        self.assertIn("Volt Absorb", conflicts[0].detail)

    def test_pins_flow_into_candidate_summaries_without_off_script_degradation(self) -> None:
        # Regression for the live bug's blast radius: the mis-pinned attacker
        # went off-script (zero surviving variants -> full pool, uncertainty
        # 1.0). With the real randbats universe, the heal must leave BOTH mons
        # on-script: Lanturn pinned to its absorb set, Zapdos still Pressure.
        import os
        from pathlib import Path
        root = Path(
            os.environ.get("POKEZERO_SHOWDOWN_ROOT")
            or "/Users/scott/workspace/pokerena/vendor/pokemon-showdown"
        )
        if not (root / "data" / "random-battles" / "gen3" / "sets.json").exists():
            self.skipTest("requires a local Showdown checkout with gen3 randbats sets")
        from pokezero.randbat import Gen3RandbatSource

        engine = self.engine_from(
            self._VOLTABSORB_HEAL_LINES, set_source=Gen3RandbatSource.from_showdown_root(root)
        )
        lanturn = self.belief(engine, "p2", "Lanturn")
        zapdos = self.belief(engine, "p1", "Zapdos")
        self.assertEqual(lanturn.possible_abilities, ("Volt Absorb",))
        self.assertEqual(zapdos.possible_abilities, ("Pressure",))
        self.assertGreater(lanturn.candidate_set_count, 0)
        self.assertTrue(
            all(v.get("ability") == "Volt Absorb" for v in lanturn.candidate_variants)
        )
        # Pre-fix, Zapdos's revealed ability became Volt Absorb -> zero
        # surviving variants -> off-script fallback, which FORCES uncertainty
        # to exactly 1.0. On-script Pressure filtering stays below it.
        self.assertGreater(zapdos.candidate_set_count, 0)
        self.assertLess(zapdos.uncertainty, 1.0)
