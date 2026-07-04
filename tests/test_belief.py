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
