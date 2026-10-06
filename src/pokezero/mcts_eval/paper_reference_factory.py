"""Public-root-only fresh-world factory for the opt-in trajectory reference.

Never accepts a live environment, simulator snapshot, opponent request, or
historical numeric opponent action. The actor's opening request and public
belief ledger are the only team inputs. Every call samples and materializes a
new hypothetical world and checks exact actor/public observation identity.
Pending Baton Pass commitments require a public-only previous-root certificate
and joint champion-policy/chance conditioning, never the deferred-prior argmax.
"""

from __future__ import annotations

from dataclasses import asdict
import random
import re
from typing import Any

from .paper_reference import ReferenceRefusal
from .paper_reference_sampling import KnownSetTraits, PaperHiddenTeamSampler
from .paper_reference_sleep import induced_sleep_certificates, induced_sleep_support
from .paper_reference_showdown import ChampionEvaluator, ShowdownTrajectoryWorld, decision_state
from ..determinization import _self_team_from_metadata_result, player_belief_view_from_payload
from ..env import BattleStartOverride
from ..local_showdown import LocalShowdownEnv, PublicBattleMaterializationState, _public_reference_encore
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
                 observation: Any, evaluator: ChampionEvaluator, set_source: Gen3RandbatSource,
                 allow_earlier_compatible_template: bool = False,
                 max_known_set_draws: int = 10, pending_transition: Any = None) -> None:
        if state.replay.requests:
            raise ReferenceRefusal("public root must strip replay request payloads")
        from .paper_reference_pending import requires_faint_encore_replay, validate_transition
        if state.deferred_opponent_action_player is not None or requires_faint_encore_replay(state):
            validate_transition(pending_transition, state, observation, set_source.metadata.source_hash)
        elif pending_transition is not None:
            raise ReferenceRefusal('nonpending public root cannot carry a pending certificate')
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
        self.set_source = set_source
        self.pending_transition = pending_transition
        self.allow_earlier_compatible_template = allow_earlier_compatible_template
        self.max_known_set_draws = max_known_set_draws
        self.own_team = own_team
        self.sampler = PaperHiddenTeamSampler(env, set_source=set_source,
            allow_earlier_compatible_template=allow_earlier_compatible_template,
            max_known_set_draws=max_known_set_draws)
        self.active = False
        self.receipts: list[dict[str, Any]] = []

    def __call__(self, hidden_rng: random.Random) -> ShowdownTrajectoryWorld:
        if self.active:
            raise ReferenceRefusal("warm reference world is still owned by another trajectory")
        self.active = True
        evidence: dict[str, Any] = {"ordinal": len(self.receipts), "status": "STARTED"}
        self.receipts.append(evidence)
        try:
            if self.pending_transition is not None:
                from .paper_reference_pending import condition_pending_world
                return condition_pending_world(self, hidden_rng, evidence)
            draw = self.sampler.draw(self.known, hidden_rng)
            evidence.update(packed_team_sha256=draw.packed_team_sha256,
                known_draws=[asdict(row) for row in draw.known], forced_sets=draw.forced_sets,
                unknown_party_seeds=list(draw.unknown_party_seeds))
            opponent = "p2" if self.state.player_id == "p1" else "p1"
            override = BattleStartOverride(player_teams={self.state.player_id: pack_team(self.own_team),
                opponent: pack_team(draw.team)}, observation_format_id="gen3randombattle")
            seed = hidden_rng.getrandbits(32)
            evidence["materialization_seed"] = seed
            durations, conditioning = {}, {}
            for side in ('p1', 'p2'):
                certificate = _public_reference_encore(self.state, side)
                if certificate is not None:
                    durations[side] = hidden_rng.choice(certificate['remaining_candidates'])
                    conditioning[side] = {**certificate, 'sampled_remaining': durations[side]}
            evidence['encore_conditioning'] = conditioning
            sleep_draws, sleep_conditioning = {}, {}
            teams = {self.state.player_id: self.own_team, opponent: draw.team}
            for key, certificate in induced_sleep_certificates(self.state).items():
                side, species = key.split(':', 1)
                candidates = [mon for mon in teams[side] if re.sub('[^a-z0-9]', '', mon.species.lower()) == species]
                if len(candidates) != 1:
                    raise ReferenceRefusal('induced sleep cannot match public victim to sampled party')
                support = induced_sleep_support(certificate, candidates[0].ability)
                if not support:
                    raise ReferenceRefusal('induced sleep sampled ability contradicts public survival')
                sleep_draws[key] = hidden_rng.choice(support)
                sleep_conditioning[key] = dict(certificate=certificate, support=support, sampled=sleep_draws[key])
            evidence['induced_sleep_conditioning'] = sleep_conditioning
            self.env.materialize_public_world(state=self.state, start_override=override, seed=seed,
                reference_rest_sleep=True, reference_consumed_items=True,
                reference_encore_durations=durations, reference_induced_sleep=sleep_draws,
                reference_turn_clocks=True)
            if decision_state(self.env.observe(self.state.player_id), player=self.state.player_id) != self.root:
                raise ReferenceRefusal("fresh sampled world does not preserve exact player-known root")
            evidence["status"] = "ROOT_VALIDATED"
            return ShowdownTrajectoryWorld(self.env, subject=self.state.player_id, evaluator=self.evaluator,
                release=lambda: self._release(evidence))
        except Exception as exc:
            self.active = False
            evidence.update(status="REFUSED", error=f"{type(exc).__name__}: {exc}")
            if getattr(exc, "sampling_diagnostic", None) is not None:
                evidence["sampling_diagnostic"] = exc.sampling_diagnostic
            raise

    def _release(self, evidence: dict[str, Any]) -> None:
        self.active = False
        # Release is ownership accounting, not a completed-trajectory assertion.
        evidence["released"] = True
