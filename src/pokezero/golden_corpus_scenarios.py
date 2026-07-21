"""Curated edge-case scenario games for the golden corpus (track B + fallback audit).

Random-seed games only exercise edge cases by luck. This module scripts
`gen3customgame` games (via ``BattleStartOverride``) that deterministically
reach the positions the belief/mask edge-case matrix cares about — Truant
loafing phases, Transform, Encore locks, Hyper Beam recharge, Baton Pass
boundaries, Wish, sand + Shedinja, RestTalk, screens, toxic stalls, Ghost
Curse, Trick exchanges, berry consumption — and
captures them through the exact same corpus machinery as the random games
(same schema; scenario identity carried in ``battle_id``, no schema change).

Two consumers:
- the encoder bit-exactness gate (scenario rows join the corpus surface);
- the fallback-detection sweep (`run_scenario_fallback_sweep`), which drives
  the engine-search policy over every scenario game and asserts each decision
  either searched or fell back with a KNOWN taxonomy reason — no crashes, no
  unmapped choices.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .env import BattleStartOverride
from .dex import normalize_id
from .golden_corpus import (
    GoldenGame,
    GoldenGameRecord,
    _CapturingPolicy,
    _decision_row_from_context,
    _true_teams_from_bridge_snapshot,
    write_golden_corpus,
)
from .golden_corpus_fold import FoldSurfaceRecorder, build_fold_rows
from .local_showdown import LocalShowdownConfig, LocalShowdownEnv
from .showdown import OBSERVATION_SCHEMA_VERSION_V2_2
from .observation import TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS
from .policy import PolicyContext, PolicyDecision, legal_action_indices
from .rollout import RolloutConfig, continue_rollout_from_current_state
from .showdown_fixture import FixturePokemon, pack_team

_EVS = {s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")}


def _mon(species: str, moves: Sequence[str], *, ability: str, item: str = "Leftovers",
         level: int = 80, ivs: Mapping[str, int] | None = None,
         gender: str | None = None) -> FixturePokemon:
    return FixturePokemon(species=species, moves=tuple(moves), ability=ability,
                          item=item, level=level, evs=dict(_EVS), ivs=ivs, gender=gender)


@dataclass(frozen=True)
class ScriptedPreferencePolicy:
    """Deterministic policy: per decision, pick the first legal candidate whose
    move id matches the preference list for that turn (cycled); otherwise the
    first legal move, otherwise the first legal action. Consumes no RNG."""

    preferences: Sequence[Sequence[str]]
    policy_id: str = "scripted-preference"
    _turn: list = field(default_factory=lambda: [0])

    def reset(self) -> None:
        self._turn[0] = 0

    def select_action(self, observation, *, rng: random.Random) -> PolicyDecision:
        index = self._pick(observation)
        return PolicyDecision(action_index=index, policy_id=self.policy_id)

    def select_action_with_context(self, context: PolicyContext, *, rng: random.Random) -> PolicyDecision:
        return self.select_action(context.observation, rng=rng)

    def _pick(self, observation) -> int:
        candidates = observation.metadata.get("action_candidates") or ()
        legal = [c for c in candidates if isinstance(c, Mapping) and c.get("legal")]
        wanted = self.preferences[self._turn[0] % len(self.preferences)] if self.preferences else ()
        self._turn[0] += 1
        for want in wanted:
            for candidate in legal:
                if candidate.get("kind") == "move" and normalize_id(str(want)) == normalize_id(
                    str(candidate.get("move_id") or "")
                ):
                    return int(candidate["action_index"])
                if candidate.get("kind") == "switch" and str(want).startswith("switch:"):
                    pokemon = candidate.get("pokemon") or {}
                    if normalize_id(want.split(":", 1)[1]) == normalize_id(str(pokemon.get("species") or "")):
                        return int(candidate["action_index"])
        moves = [c for c in legal if c.get("kind") == "move"]
        pick = moves[0] if moves else (legal[0] if legal else None)
        if pick is None:
            mask = observation.legal_action_mask
            return legal_action_indices(mask)[0]
        return int(pick["action_index"])


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    p1_team: Sequence[FixturePokemon]
    p2_team: Sequence[FixturePokemon]
    p1_prefs: Sequence[Sequence[str]]
    p2_prefs: Sequence[Sequence[str]]
    seed: int = 91000
    max_decision_rounds: int = 24


def scenario_specs() -> tuple[ScenarioSpec, ...]:
    swampert = _mon("Swampert", ("Earthquake", "Ice Beam", "Protect", "Toxic"), ability="Torrent", level=84)
    blissey = _mon("Blissey", ("Seismic Toss", "Soft-Boiled", "Toxic", "Ice Beam"), ability="Natural Cure")
    celebi = _mon("Celebi", ("Calm Mind", "Baton Pass", "Psychic", "Recover"), ability="Natural Cure")
    # Probe-verbatim bench partner for the absorb scenarios (absorb audit,
    # 2026-07-19): L80, Surf over Toxic — keep it byte-identical to the
    # captured games so the corpus rows reproduce the probe protocol exactly.
    absorb_swampert = _mon("Swampert", ("Earthquake", "Ice Beam", "Surf", "Protect"), ability="Torrent")

    return (
        ScenarioSpec(
            "truant_slaking",
            (_mon("Slaking", ("Double-Edge", "Earthquake", "Shadow Ball", "Return"),
                  ability="Truant", item="Choice Band", level=78), swampert),
            (blissey, swampert),
            p1_prefs=(("doubleedge",),),
            p2_prefs=(("softboiled", "seismictoss"),),
        ),
        ScenarioSpec(
            "ditto_transform",
            (_mon("Ditto", ("Transform",), ability="Limber", item="Quick Claw", level=100), swampert),
            (_mon("Snorlax", ("Body Slam", "Curse", "Rest", "Shadow Ball"), ability="Immunity"), swampert),
            p1_prefs=(("transform",), ("bodyslam",)),
            p2_prefs=(("curse",), ("bodyslam",)),
        ),
        ScenarioSpec(
            "encore_wobbuffet",
            (_mon("Wobbuffet", ("Encore", "Counter", "Mirror Coat", "Safeguard"), ability="Shadow Tag"), swampert),
            (celebi, swampert),
            p1_prefs=(("safeguard",), ("encore",), ("counter",)),
            p2_prefs=(("calmmind",), ("calmmind",), ("psychic",)),
        ),
        ScenarioSpec(
            "hyperbeam_recharge",
            (_mon("Slaking", ("Hyper Beam", "Earthquake", "Return", "Shadow Ball"),
                  ability="Truant", item="Choice Band", level=78), swampert),
            (blissey, swampert),
            p1_prefs=(("hyperbeam",),),
            p2_prefs=(("softboiled", "seismictoss"),),
        ),
        ScenarioSpec(
            "baton_pass_boundary",
            (celebi, _mon("Snorlax", ("Body Slam", "Curse", "Rest", "Shadow Ball"), ability="Immunity"), swampert),
            (blissey, swampert),
            p1_prefs=(("calmmind",), ("batonpass",), ("bodyslam",)),
            p2_prefs=(("seismictoss",),),
        ),
        ScenarioSpec(
            "wish_boundary",
            (_mon("Vaporeon", ("Wish", "Surf", "Protect", "Ice Beam"), ability="Water Absorb"), swampert),
            (blissey, swampert),
            p1_prefs=(("wish",), ("protect",), ("surf",)),
            p2_prefs=(("seismictoss",),),
        ),
        ScenarioSpec(
            "sand_shedinja",
            (_mon("Tyranitar", ("Rock Slide", "Earthquake", "Crunch", "Pursuit"), ability="Sand Stream"), swampert),
            (_mon("Shedinja", ("Shadow Ball", "Silver Wind", "Agility", "Baton Pass"),
                  ability="Wonder Guard", item="Lum Berry", level=100), blissey),
            p1_prefs=(("crunch",),),
            p2_prefs=(("agility",), ("shadowball",)),
        ),
        ScenarioSpec(
            "resttalk_snorlax",
            (_mon("Snorlax", ("Rest", "Sleep Talk", "Body Slam", "Curse"), ability="Immunity"), swampert),
            (blissey, swampert),
            p1_prefs=(("curse",), ("rest",), ("sleeptalk",), ("sleeptalk",)),
            p2_prefs=(("seismictoss",),),
        ),
        ScenarioSpec(
            "screens_jirachi",
            (_mon("Jirachi", ("Reflect", "Light Screen", "Psychic", "Wish"), ability="Serene Grace"), swampert),
            (_mon("Snorlax", ("Body Slam", "Curse", "Rest", "Shadow Ball"), ability="Immunity"), swampert),
            p1_prefs=(("reflect",), ("lightscreen",), ("psychic",)),
            p2_prefs=(("bodyslam",),),
        ),
        ScenarioSpec(
            "toxic_stall",
            (swampert, blissey),
            (_mon("Starmie", ("Surf", "Recover", "Psychic", "Rapid Spin"), ability="Natural Cure"), blissey),
            p1_prefs=(("toxic",), ("protect",), ("earthquake",)),
            p2_prefs=(("recover", "surf"),),
        ),
        # Ghost-typed Curse (live-protocol shape: |move| with explicit target,
        # |-start|target|Curse|[of] user, bare self |-damage| HP cut; a
        # repeat lands the already-cursed fail form). Keeps the renderer's
        # Ghost/non-Ghost Curse split and the leaf CURSE volatile placement
        # exercised; Seismic Toss into Gengar also pins the Normal-immunity
        # render.
        ScenarioSpec(
            "ghost_curse",
            (_mon("Gengar", ("Curse", "Shadow Ball", "Thunderbolt", "Protect"), ability="Levitate"), swampert),
            (blissey, swampert),
            p1_prefs=(("curse",), ("curse",), ("shadowball",)),
            p2_prefs=(("seismictoss", "softboiled"),),
        ),
        # --- Absorb-class abilities (Flash Fire / Volt Absorb / Water Absorb),
        # per the probe-captured protocol (absorb audit, 2026-07-19): activation
        # ``-start ability:``, repeat ``-immune [from] ability:``, absorb heal
        # ``-heal <absorber> ... [from] ability: X|[of] <attacker>``, the gen3
        # Thunder Wave NOT-absorbed gate, and the switch-clear of the flashfire
        # volatile.
        ScenarioSpec(
            "flashfire_houndoom",
            (_mon("Houndoom", ("Crunch", "Flamethrower", "Toxic", "Taunt"), ability="Flash Fire"),
             _mon("Rapidash", ("Double-Edge", "Fire Blast", "Megahorn", "Toxic"), ability="Flash Fire")),
            (_mon("Charizard", ("Flamethrower", "Dragon Claw", "Toxic", "Roar"), ability="Blaze"),
             absorb_swampert),
            # The Flash Fire holders sit on p1 with Charizard scripted to spam
            # Flamethrower: the fallback sweep drives p1 with the engine-search
            # policy, so this is the only orientation that guarantees the
            # flashfire volatile on the search seat's payload from round 1's
            # resolution onward regardless of what the search picks (pre-fix
            # this walled every post-activation decision with
            # ``volatile_unsupported: flashfire``). Scripted corpus run:
            # r1 -start on Houndoom, r2 -immune, r3 switch -> ``-end``+clear
            # plus a fresh -start on Rapidash, r4 -immune on Rapidash.
            p1_prefs=(("crunch",), ("crunch",), ("switch:rapidash",), ("doubleedge",)),
            p2_prefs=(("flamethrower",),),
            seed=91007,
            max_decision_rounds=6,
        ),
        ScenarioSpec(
            "voltabsorb_lanturn",
            (_mon("Zapdos", ("Thunderbolt", "Drill Peck", "Thunder Wave", "Roar"), ability="Pressure"),
             blissey),
            (_mon("Lanturn", ("Surf", "Ice Beam", "Confuse Ray", "Recover"), ability="Volt Absorb"),
             absorb_swampert),
            # r1 Thunderbolt at FULL hp (-immune, no heal); r2 Drill Peck
            # (chip); r3 Thunderbolt (-heal ... [of] Zapdos — the live-captured
            # shape behind the belief attribution fix); r4 Thunder Wave (gen3
            # quirk: NOT absorbed, paralysis lands).
            p1_prefs=(("thunderbolt",), ("drillpeck",), ("thunderbolt",), ("thunderwave",)),
            p2_prefs=(("surf",),),
            seed=91007,
            max_decision_rounds=5,
        ),
        ScenarioSpec(
            "waterabsorb_quagsire",
            (_mon("Suicune", ("Surf", "Ice Beam", "Calm Mind", "Rest"), ability="Pressure"),
             blissey),
            (_mon("Quagsire", ("Earthquake", "Toxic", "Rest", "Sleep Talk"), ability="Water Absorb"),
             absorb_swampert),
            # r1 Surf at FULL hp (-immune); r2 Ice Beam (chip); r3 Surf
            # (-heal ... [of] Suicune); r4 Surf (-immune again at full HP).
            p1_prefs=(("surf",), ("icebeam",), ("surf",), ("surf",)),
            p2_prefs=(("earthquake",),),
            seed=91007,
            max_decision_rounds=5,
        ),
        # --- item-state scenarios (Trick-swap current-item override + berry
        # consumption; protocol shapes probed verbatim 2026-07-19). Post-fix
        # these must SEARCH: the exchange's two -item lines confirm both mons'
        # CURRENT items, and an eaten berry clears the sampled item. ---
        ScenarioSpec(
            "trick_swap_exchange",
            # p1 direction: our Alakazam Tricks its Choice Band onto the
            # opponent's Furret and receives the Petaya Berry (full swap: one
            # |-item|...|[from] move: Trick per mon).
            (_mon("Alakazam", ("Trick", "Psychic", "Thunder Punch", "Fire Punch"),
                  ability="Synchronize", item="Choice Band"), swampert),
            (_mon("Furret", ("Return", "Shadow Ball", "Brick Break", "Quick Attack"),
                  ability="Run Away", item="Petaya Berry"), blissey),
            p1_prefs=(("trick",), ("psychic",)),
            p2_prefs=(("quickattack",),),
        ),
        ScenarioSpec(
            "trick_berry_pinch",
            # p2 direction + the seed-7013 shape: the scripted opponent Tricks
            # a Petaya Berry onto OUR mon, then Seismic-Tosses it to pinch —
            # the Tricked berry is eaten (override -> removal transition on
            # the self seat; the tricker holds our Leftovers on the opponent
            # seat). Scripted as p2 so the engine-search sweep exercises it
            # deterministically.
            (_mon("Furret", ("Return", "Shadow Ball", "Brick Break", "Quick Attack"),
                  ability="Run Away"), swampert),
            (_mon("Blissey", ("Trick", "Seismic Toss", "Soft-Boiled", "Ice Beam"),
                  ability="Natural Cure", item="Petaya Berry"), swampert),
            p1_prefs=(("quickattack",),),
            p2_prefs=(("trick",),) + (("seismictoss",),) * 9,
        ),
        ScenarioSpec(
            "berry_eat_chesto",
            # Public consumption without any mutation: the opponent's Snorlax
            # Rest-eats its Chesto Berry — the item is publicly GONE and the
            # sampled world must stop handing it back (repeated rest prefs so
            # the eat lands whenever the first damage arrives).
            (swampert, blissey),
            (_mon("Snorlax", ("Body Slam", "Rest", "Curse", "Shadow Ball"),
                  ability="Immunity", item="Chesto Berry"), blissey),
            p1_prefs=(("earthquake",),),
            p2_prefs=(("curse",), ("rest",), ("bodyslam",), ("rest",), ("bodyslam",), ("rest",)),
        ),
        # --- Attract (infatuation) 50%-per-turn move immobilization
        # (third_party/poke-engine-gen3-attract.patch; walls audit 2026-07-19).
        # The fallback sweep drives p1 with the engine-search policy, so the
        # ATTRACT-carrying mon must sit on p1 for its volatile to reach the
        # search seat's world construction (pre-fix this walled every attracted
        # decision with ``volatile_unsupported: attract``). Blissey is
        # female-only and bulky, so the infatuation is gender-valid and the
        # source survives to keep the volatile live. Genders are pinned (Custom
        # Game does not enforce randbats gender ratios) so Attract never fails
        # the ``onTryImmunity`` opposite-gender gate. One game exercises BOTH
        # branches the search must price: r1 p2 casts Attract -> r2 p1 decision
        # carries a FREE attract; r2 p2 casts Thunder Wave -> r3+ p1 decisions
        # carry the PARALYSIS + attract composition (net move prob 0.75*0.5,
        # commutative with Showdown's attract-before-par onBeforeMove order).
        ScenarioSpec(
            "attract_snorlax",
            (_mon("Snorlax", ("Body Slam", "Curse", "Rest", "Shadow Ball"),
                  ability="Immunity", gender="M"), swampert),
            (_mon("Blissey", ("Attract", "Thunder Wave", "Soft-Boiled", "Seismic Toss"),
                  ability="Natural Cure", gender="F"), blissey),
            p1_prefs=(("curse",), ("bodyslam",)),
            p2_prefs=(("attract",), ("thunderwave",), ("softboiled",)),
            seed=91011,
            max_decision_rounds=5,
        ),
    )


def interaction_registry_specs() -> tuple[ScenarioSpec, ...]:
    """Validated-interactions registry scenarios (docs/validated_interactions.md).

    A SEPARATE list from ``scenario_specs()`` on purpose: these drive the
    protocol-census interactions for the regression suite
    (``tests/test_interaction_registry.py``) WITHOUT joining the default corpus /
    fallback-sweep surface, so the existing bit-exactness and sweep gates are
    byte-for-byte unperturbed. Two of these (``castform_forecast_formechange`` and
    ``colorchange_kecleon``) reproduce the in-battle-retype ENCODER BUGS flagged in
    the registry — the test asserts the CORRECT behavior under ``expectedFailure``.
    """

    swampert = _mon("Swampert", ("Earthquake", "Ice Beam", "Surf", "Protect"), ability="Torrent", level=84)
    blissey = _mon("Blissey", ("Soft-Boiled", "Seismic Toss", "Ice Beam", "Toxic"), ability="Natural Cure", level=100)
    snorlax = _mon("Snorlax", ("Body Slam", "Curse", "Rest", "Shadow Ball"), ability="Immunity", level=100)

    return (
        # --- BUG: Castform Forecast in-battle -formechange retype (flagged case).
        # p1 Charizard sets sun -> p2 Castform Forecast-changes to Castform-Sunny
        # (Fire). The encoder must retype it; today it stays base Normal.
        ScenarioSpec(
            "castform_forecast_formechange",
            (_mon("Charizard", ("Sunny Day", "Flamethrower", "Dragon Claw", "Toxic"), ability="Blaze", level=90), blissey),
            (_mon("Castform", ("Fire Blast", "Ice Beam", "Thunderbolt", "Return"), ability="Forecast", level=90), swampert),
            p1_prefs=(("sunnyday",), ("flamethrower",)),
            p2_prefs=(("return",),),
            seed=91020, max_decision_rounds=3,
        ),
        # --- BUG: Color Change typechange retype (same root cause as Castform).
        # p2 Alakazam hits p1 Kecleon with Psychic -> Kecleon retypes to Psychic.
        ScenarioSpec(
            "colorchange_kecleon",
            (_mon("Kecleon", ("Return", "Shadow Ball", "Brick Break", "Thunder Wave"), ability="Color Change", level=92), swampert),
            (_mon("Alakazam", ("Psychic", "Fire Punch", "Ice Punch", "Thunder Punch"), ability="Synchronize"), blissey),
            p1_prefs=(("return",),),
            p2_prefs=(("psychic",),),
            seed=91021, max_decision_rounds=3,
        ),
        # --- Deoxys formes: real dex entries (distinct stats), NOT -formechange.
        ScenarioSpec(
            "deoxys_forme_swap",
            (_mon("Deoxys-Attack", ("Psycho Boost", "Superpower", "Shadow Ball", "Extreme Speed"), ability="Pressure", level=100),
             _mon("Deoxys-Defense", ("Recover", "Toxic", "Seismic Toss", "Spikes"), ability="Pressure", level=100)),
            (snorlax, swampert),
            p1_prefs=(("shadowball",), ("switch:deoxysdefense",), ("recover",)),
            p2_prefs=(("curse",),),
            seed=91022, max_decision_rounds=4,
        ),
        # --- Intimidate on switch-in: p2 Salamence drops p1 lead's atk at t1.
        ScenarioSpec(
            "intimidate_switchin",
            (snorlax, _mon("Porygon2", ("Recover", "Ice Beam", "Thunderbolt", "Toxic"), ability="Trace")),
            (_mon("Salamence", ("Dragon Dance", "Earthquake", "Rock Slide", "Fire Blast"), ability="Intimidate"), swampert),
            p1_prefs=(("bodyslam",), ("switch:porygon2",)),
            p2_prefs=(("dragondance",),),
            seed=91023, max_decision_rounds=3,
        ),
        # --- Natural Cure: Gen 3 singles emits its public -curestatus ability
        # line before a statused holder leaves the field. Re-entry must remain
        # status-free in both seats' public views. (The showCure=false branch
        # exists only for multi-active ambiguity, outside randbats singles.)
        ScenarioSpec(
            "natural_cure_switch",
            (_mon("Starmie", ("Recover", "Surf", "Psychic", "Rapid Spin"), ability="Natural Cure"), swampert),
            (blissey, snorlax),
            p1_prefs=(("recover",), ("switch:swampert",), ("switch:starmie",)),
            p2_prefs=(("toxic",), ("seismictoss",), ("softboiled",)),
            seed=91032, max_decision_rounds=3,
        ),
        # --- Belly Drum: -setboost atk 6 on the user.
        ScenarioSpec(
            "bellydrum_snorlax",
            (_mon("Snorlax", ("Belly Drum", "Body Slam", "Curse", "Rest"), ability="Immunity", level=100), swampert),
            (blissey, swampert),
            p1_prefs=(("bellydrum",), ("bodyslam",)),
            p2_prefs=(("softboiled",),),
            seed=91024, max_decision_rounds=3,
        ),
        # --- Spikes stacking to 3 layers on p1's side (switch churn to eat chip).
        ScenarioSpec(
            "spikes_stack",
            (blissey, swampert, _mon("Gengar", ("Shadow Ball", "Thunderbolt", "Ice Punch", "Explosion"), ability="Levitate")),
            (_mon("Skarmory", ("Spikes", "Drill Peck", "Roar", "Toxic"), ability="Keen Eye"), snorlax),
            p1_prefs=(("softboiled",), ("switch:swampert",), ("switch:gengar",), ("switch:blissey",), ("softboiled",)),
            p2_prefs=(("spikes",),),
            seed=91025, max_decision_rounds=6,
        ),
        # --- Substitute: volatile:substitute on the user.
        ScenarioSpec(
            "substitute_focuspunch",
            (_mon("Breloom", ("Substitute", "Mach Punch", "Spore", "Seismic Toss"), ability="Effect Spore"), swampert),
            (blissey, swampert),
            p1_prefs=(("substitute",), ("machpunch",), ("machpunch",)),
            p2_prefs=(("softboiled",),),
            seed=91026, max_decision_rounds=4,
        ),
        # --- Weather permanence: Sand Stream (ability) is permanent in gen3.
        ScenarioSpec(
            "sand_stream_permanence",
            (_mon("Tyranitar", ("Rock Slide", "Crunch", "Earthquake", "Pursuit"), ability="Sand Stream"), swampert),
            (_mon("Charizard", ("Sunny Day", "Flamethrower", "Dragon Claw", "Toxic"), ability="Blaze", level=90), blissey),
            p1_prefs=(("rockslide",),),
            p2_prefs=(("flamethrower",),),
            seed=91027, max_decision_rounds=3,
        ),
        # --- Roar drag resets the dragged-in mon's boosts (drag != Baton Pass).
        ScenarioSpec(
            "roar_drag_reset",
            (_mon("Suicune", ("Roar", "Surf", "Calm Mind", "Rest"), ability="Pressure"), swampert),
            (_mon("Raikou", ("Calm Mind", "Thunderbolt", "Crunch", "Rest"), ability="Pressure"), snorlax),
            p1_prefs=(("calmmind",), ("calmmind",), ("roar",)),
            p2_prefs=(("calmmind",),),
            seed=91028, max_decision_rounds=4,
        ),
        # --- Future Sight: pending strike scheduled on the OPPONENT's side.
        ScenarioSpec(
            "future_sight_pending",
            (_mon("Alakazam", ("Future Sight", "Psychic", "Recover", "Thunder Punch"), ability="Synchronize"), swampert),
            (snorlax, swampert),
            p1_prefs=(("futuresight",), ("recover",), ("recover",)),
            p2_prefs=(("curse",),),
            seed=91029, max_decision_rounds=4,
        ),
        # --- Perish Song: per-mon perishN volatiles counting down.
        ScenarioSpec(
            "perish_song",
            (_mon("Celebi", ("Perish Song", "Recover", "Psychic", "Baton Pass"), ability="Natural Cure"), blissey),
            (snorlax, swampert),
            p1_prefs=(("perishsong",), ("recover",), ("recover",)),
            p2_prefs=(("curse",),),
            seed=91030, max_decision_rounds=4,
        ),
        # --- Counter / Mirror Coat: fixed-damage; Mirror Coat (Psychic) vs Dark = immune.
        ScenarioSpec(
            "counter_mirrorcoat",
            (_mon("Wobbuffet", ("Counter", "Mirror Coat", "Encore", "Safeguard"), ability="Shadow Tag", level=100), swampert),
            (_mon("Tyranitar", ("Rock Slide", "Crunch", "Earthquake", "Ice Beam"), ability="Sand Stream"), blissey),
            p1_prefs=(("counter",), ("mirrorcoat",)),
            p2_prefs=(("crunch",), ("icebeam",)),
            seed=91031, max_decision_rounds=3,
        ),
    )


def _scenario_override(spec: ScenarioSpec) -> BattleStartOverride:
    return BattleStartOverride(player_teams={
        "p1": pack_team(tuple(spec.p1_team)),
        "p2": pack_team(tuple(spec.p2_team)),
    })


def play_scenario_games(
    *,
    showdown_root: Path | str | None = None,
    specs: Sequence[ScenarioSpec] | None = None,
    belief_set_source: bool | None = None,
) -> tuple[list[GoldenGame], str | None]:
    """Play every scenario and return corpus games (same capture machinery)."""

    config = LocalShowdownConfig(showdown_root=showdown_root, set_belief_source=belief_set_source)
    env = LocalShowdownEnv(config)
    turn_merged_active = config.observation_spec.schema_version in TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS
    games: list[GoldenGame] = []
    try:
        belief_hash = env.belief_set_source_hash
        for spec in specs if specs is not None else scenario_specs():
            override = _scenario_override(spec)
            battle_id = f"golden-scenario-{spec.name}-{spec.seed}"
            env.reset_with_start_override(seed=spec.seed, start_override=override)
            true_teams = _true_teams_from_bridge_snapshot(env.snapshot().bridge_snapshot)
            captures: list[tuple[PolicyContext, PolicyDecision]] = []
            recorder = FoldSurfaceRecorder(env)

            def _sink(context: PolicyContext, decision: PolicyDecision) -> None:
                captures.append((context, decision))
                recorder.record(context.player_id)

            policies = {
                "p1": _CapturingPolicy(ScriptedPreferencePolicy(spec.p1_prefs), _sink),
                "p2": _CapturingPolicy(ScriptedPreferencePolicy(spec.p2_prefs), _sink),
            }
            result = continue_rollout_from_current_state(
                env=env,
                policies=policies,
                config=RolloutConfig(
                    max_decision_rounds=spec.max_decision_rounds,
                    format_id=override.format_id,
                    hide_opponent_legal_action_masks=True,
                ),
                seed=spec.seed,
                battle_id=battle_id,
                reset_policies=True,
            )
            rows = tuple(
                _decision_row_from_context(context, decision, battle_seed=spec.seed)
                for context, decision in captures
            )
            fold_rows = build_fold_rows(
                replays=[context.public_materialization_state.replay for context, _ in captures],
                surfaces=recorder.surfaces,
                turn_merged_active=turn_merged_active,
            )
            record = GoldenGameRecord(
                battle_seed=spec.seed,
                battle_id=battle_id,
                format_id=override.format_id,
                policy_ids={"p1": policies["p1"].policy_id, "p2": policies["p2"].policy_id},
                true_teams=true_teams,
                terminal={
                    "winner": result.terminal.winner,
                    "turn_count": result.terminal.turn_count,
                    "capped": result.terminal.capped,
                },
            )
            games.append(GoldenGame(record=record, rows=rows, fold_rows=fold_rows))
    finally:
        env.close()
    return games, belief_hash


KNOWN_FALLBACK_REASONS = frozenset({
    "no_public_state", "no_worlds_constructed", "choices_unmapped",
})


def run_scenario_fallback_sweep(
    *,
    showdown_root: Path | str | None = None,
    specs: Sequence[ScenarioSpec] | None = None,
    worlds: int = 2,
    search_time_ms: int = 25,
) -> dict[str, Any]:
    """Drive the engine-search policy over every scenario game.

    Asserts nothing itself; returns per-scenario stats for callers/tests to
    assert on: every decision searched or fell back with a KNOWN reason,
    zero unmapped choices, world-failure taxonomy observed per scenario.
    """

    from .dex import load_showdown_dex
    from .engine_search import EngineMctsConfig, EngineMctsPolicy
    from .local_showdown import DEFAULT_SHOWDOWN_ROOT
    from .randbat import Gen3RandbatSource
    from .rollout import RolloutDriver

    root = showdown_root or DEFAULT_SHOWDOWN_ROOT
    dex = load_showdown_dex(root)
    set_source = Gen3RandbatSource.from_showdown_root(root)
    report: dict[str, Any] = {}
    env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=root))
    try:
        for spec in specs if specs is not None else scenario_specs():
            override = _scenario_override(spec)
            policy = EngineMctsPolicy(
                dex=dex, set_source=set_source,
                config=EngineMctsConfig(worlds=worlds, search_time_ms=search_time_ms),
                fixed_override=override,
            )
            env.reset_with_start_override(seed=spec.seed, start_override=override)
            continue_rollout_from_current_state(
                env=env,
                policies={"p1": policy, "p2": ScriptedPreferencePolicy(spec.p2_prefs)},
                config=RolloutConfig(
                    max_decision_rounds=spec.max_decision_rounds,
                    format_id=override.format_id,
                    hide_opponent_legal_action_masks=True,
                ),
                seed=spec.seed,
                battle_id=f"sweep-{spec.name}",
                reset_policies=True,
            )
            report[spec.name] = policy.stats.to_dict()
    finally:
        env.close()
    return report


def generate_scenario_corpus(
    *,
    out_dir: Path,
    showdown_root: Path | str | None = None,
    belief_set_source: bool | None = None,
) -> dict[str, Any]:
    """Write the scenario games as a standalone corpus directory."""

    games, belief_hash = play_scenario_games(
        showdown_root=showdown_root, belief_set_source=belief_set_source
    )
    config = LocalShowdownConfig(showdown_root=showdown_root)
    spec = config.observation_spec
    masks = config.feature_masks
    from .actions import ACTION_COUNT

    header = {
        "generator": {
            "scenario_suite": [s.name for s in scenario_specs()],
            "hide_opponent_legal_action_masks": True,
        },
        "observation": {
            "schema_version": spec.schema_version,
            "token_count": spec.token_count,
            "categorical_feature_count": spec.categorical_feature_count,
            "numeric_feature_count": spec.numeric_feature_count,
            "action_count": ACTION_COUNT,
            "feature_masks": {
                "stats_block": masks.opponent_tendency_stats_block,
                "exact_state": masks.exact_state,
                "transition_token_budget": masks.transition_token_budget,
                "tier2_residuals": masks.tier2_residuals,
                "tier2_investment": masks.tier2_investment,
            },
        },
        "belief_set_source_hash": belief_hash,
    }
    return write_golden_corpus(out_dir, header=header, games=games)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate the edge-case scenario corpus")
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--belief-set-source",
        choices=("env", "on", "off"),
        default="env",
        help="Candidate-set belief source: pin on/off, or defer to POKEZERO_BELIEF_SET_SOURCE (default).",
    )
    args = parser.parse_args(argv)
    belief_set_source = {"env": None, "on": True, "off": False}[args.belief_set_source]
    manifest = generate_scenario_corpus(
        out_dir=Path(args.out),
        showdown_root=args.showdown_root,
        belief_set_source=belief_set_source,
    )
    counts = manifest.get("counts", {})
    print(
        f"scenario corpus written: {counts.get('decisions', '?')} rows "
        f"({counts.get('fold_rows', 0)} fold rows) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
