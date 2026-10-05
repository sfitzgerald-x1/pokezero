"""Public-root-only fresh-world factory for the opt-in trajectory reference.

Never accepts a live environment, simulator snapshot, opponent request, or
historical numeric opponent action. The actor's opening request and public
belief ledger are the only team inputs. Every call samples and materializes a
new hypothetical world and checks exact actor/public observation identity.
Pending Baton Pass commitments currently refuse explicitly: choosing argmax
from the materializer's deferred-prior shortcut is NOT paper policy sampling.
"""

from __future__ import annotations

from dataclasses import asdict
import random
import re
from typing import Any

from .paper_reference import ReferenceRefusal
from .paper_reference_sampling import KnownSetTraits, PaperHiddenTeamSampler
from .paper_reference_showdown import ChampionEvaluator, ShowdownTrajectoryWorld, decision_state
from ..determinization import _self_team_from_metadata_result, player_belief_view_from_payload
from ..env import BattleStartOverride
from ..local_showdown import LocalShowdownEnv, PublicBattleMaterializationState
from ..public_decision_corpus import _public_belief_view
from ..randbat import Gen3RandbatSource, canonical_gen3_randbat_species_id
from ..showdown import _pokemon_metadata, _self_team_from_request
from ..showdown_fixture import pack_team


def _level(details: Any) -> int | None:
    if not isinstance(details, str):
        return None
    for part in details.split(",")[1:]:
        if re.fullmatch(r"L\d+", part.strip()):
            return int(part.strip()[1:])
    # In Showdown public details, an omitted level is level 100.
    return 100


def _gender(details: Any) -> str | None:
    if isinstance(details, str):
        return next((part.strip() for part in details.split(",")[1:]
            if part.strip() in ("M", "F")), None)
    return None


def _maximum_hp(condition: str | None) -> int | None:
    if isinstance(condition, str):
        match = re.match(r"\d+/(\d+)(?:\s|$)", condition)
        if match and int(match[1]) != 100:
            # The retained corpus sometimes carries exact HP, a controlled
            # information deviation from PS percentage-only observations.
            # Condition the draw on it; never silently change that root.
            return int(match[1])
    return None


class PublicRootWorldFactory:
    def __init__(self, *, env: LocalShowdownEnv, state: PublicBattleMaterializationState,
                 observation: Any, evaluator: ChampionEvaluator, set_source: Gen3RandbatSource) -> None:
        if state.replay.requests:
            raise ReferenceRefusal("public root must strip replay request payloads")
        if state.deferred_opponent_action_player is not None:
            raise ReferenceRefusal("pending committed opponent action needs a sampled-policy certificate")
        if state.observation_format_id != "gen3randombattle":
            raise ReferenceRefusal("reference factory supports only public Gen 3 random-battle roots")
        if env.belief_set_source_hash != set_source.metadata.source_hash:
            raise ReferenceRefusal("reference environment and public encoder source binding disagree")
        self.root = decision_state(observation, player=state.player_id)
        # Strict actor-only opening request reconstruction avoids mutated live
        # abilities/items and the opponent's real party. No private truth builder.
        rows = [_pokemon_metadata(mon) for mon in
            _self_team_from_request(state.self_initial_request, state.player_id)]
        own_team, failure = _self_team_from_metadata_result(rows, team_size=6, set_source=set_source)
        if own_team is None:
            raise ReferenceRefusal(f"actor-known opening party refuses: {failure}")
        payload = _public_belief_view(observation.metadata)
        view = player_belief_view_from_payload(payload)
        if (view is None or view.self_slot != state.player_id
                or not isinstance(payload.get("opponent_pokemon"), list)
                or len(view.opponent_pokemon) != len(payload["opponent_pokemon"])):
            raise ReferenceRefusal("public opponent belief has invalid or silently dropped members")
        if state.belief_engine.resolved_player_view(state.player_id).to_overlay_payload() != payload:
            raise ReferenceRefusal("public root observation and materialization belief ledger disagree")
        details = {str(row.get("species", "")).casefold(): row.get("details")
            for row in observation.metadata.get("opponent_team", [])}
        known = []
        for mon in view.opponent_pokemon:
            # A mutated held item is CURRENT state, not the generator's original
            # assignment. The public materializer applies current mutations later.
            item = mon.original_public_item
            if item is None and not mon.item_mutated:
                item = mon.revealed_item
            ability = mon.revealed_ability
            universe = set_source.universes.get(canonical_gen3_randbat_species_id(mon.species))
            # Validation only, NOT a draw from the catalog. The compact Dex
            # metadata used by the encoder intentionally omits ability tables.
            original_abilities = {re.sub(r"[^a-z0-9]", "", str(a).lower())
                for a in (v.ability for v in universe.variants)} if universe is not None else set()
            if ability and re.sub(r"[^a-z0-9]", "", ability.lower()) not in original_abilities:
                # A copied Transform/Trace ability is not an original-set trait.
                # Refuse unexpected ledger corruption rather than forcing it.
                if not mon.transformed:
                    raise ReferenceRefusal("public original-set ability is incompatible with species")
                ability = None
            known.append(KnownSetTraits(mon.species, mon.revealed_moves, ability, item,
                _level(details.get(mon.species.casefold())),
                mon.gender or _gender(details.get(mon.species.casefold())), _maximum_hp(mon.condition),
                mon.ruled_out_abilities, mon.ruled_out_items))
        self.known = tuple(known)
        self.env, self.state, self.evaluator = env, state, evaluator
        self.own_team = own_team
        self.sampler = PaperHiddenTeamSampler(env, set_source=set_source)
        self.active = False
        self.receipts: list[dict[str, Any]] = []

    def __call__(self, hidden_rng: random.Random) -> ShowdownTrajectoryWorld:
        if self.active:
            raise ReferenceRefusal("warm reference world is still owned by another trajectory")
        self.active = True
        evidence: dict[str, Any] = {"ordinal": len(self.receipts), "status": "STARTED"}
        self.receipts.append(evidence)
        try:
            draw = self.sampler.draw(self.known, hidden_rng)
            evidence.update(packed_team_sha256=draw.packed_team_sha256,
                known_draws=[asdict(row) for row in draw.known], forced_sets=draw.forced_sets,
                unknown_party_seeds=list(draw.unknown_party_seeds))
            opponent = "p2" if self.state.player_id == "p1" else "p1"
            override = BattleStartOverride(player_teams={self.state.player_id: pack_team(self.own_team),
                opponent: pack_team(draw.team)}, observation_format_id="gen3randombattle")
            seed = hidden_rng.getrandbits(32)
            evidence["materialization_seed"] = seed
            self.env.materialize_public_world(state=self.state, start_override=override, seed=seed,
                reference_rest_sleep=True)
            if decision_state(self.env.observe(self.state.player_id), player=self.state.player_id) != self.root:
                raise ReferenceRefusal("fresh sampled world does not preserve exact player-known root")
            evidence["status"] = "ROOT_VALIDATED"
            return ShowdownTrajectoryWorld(self.env, subject=self.state.player_id, evaluator=self.evaluator,
                release=lambda: self._release(evidence))
        except Exception as exc:
            self.active = False
            evidence.update(status="REFUSED", error=f"{type(exc).__name__}: {exc}")
            raise

    def _release(self, evidence: dict[str, Any]) -> None:
        self.active = False
        # Release is ownership accounting, not a completed-trajectory assertion.
        evidence["released"] = True
