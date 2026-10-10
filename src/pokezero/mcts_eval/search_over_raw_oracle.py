"""Team-only oracle diagnostics; never a deployable public search input.

The trusted controller extracts original sets from opening requests. Workers
receive those sets plus a canonical public root, not a live source snapshot,
opponent observation or committed action. Public reconstruction stays intact.
"""
from dataclasses import dataclass
import hashlib

from .paper_reference import ReferenceRefusal
from .paper_reference_runtime import PublicRootRequest, ShowdownWorkerFactory
from .paper_reference_sampling import HiddenTeamDraw, KnownDrawReceipt, _matches
from .paper_reference_showdown import decision_state
from .search_over_raw import digest, require
from .search_over_raw_adapters import PublicModelSearchAdapter
from ..showdown_fixture import pack_team


def team_sha256(team):
    return hashlib.sha256(pack_team(team).encode()).hexdigest()


def opening_team(request, subject, source):
    # Same packer as TruthWorldBuilder, without retaining its source env.
    from ..determinization import _self_team_from_metadata_result
    from ..showdown import _pokemon_metadata, _self_team_from_request
    rows = [_pokemon_metadata(mon) for mon in _self_team_from_request(request, subject)]
    team, failure = _self_team_from_metadata_result(rows, team_size=6, set_source=source)
    require(team is not None and not failure and len(team) == 6, "oracle opening team cannot be reconstructed")
    return tuple(team)


def public_root_binding(request):
    from ..public_decision_corpus import _public_belief_view
    return digest(dict(subject=request.state.player_id, set_source_hash=request.set_source_hash,
        information_key=decision_state(request.observation, player=request.state.player_id).key.hex(),
        public_history=[e.raw_line for e in request.state.replay.public_events],
        own_request=request.state.self_request, own_opening=request.state.self_initial_request,
        belief=_public_belief_view(request.observation.metadata)))


@dataclass(frozen=True)
class TeamOracle:
    subject: str
    set_source_hash: str
    root_binding: str
    own_team_sha256: str
    opponent_team: tuple
    opponent_team_sha256: str

    @classmethod
    def capture(cls, snapshot, request, source):
        """Trusted-controller API only; snapshot is discarded after extraction."""
        from ..local_showdown import LocalShowdownSnapshot
        require(isinstance(snapshot, LocalShowdownSnapshot) and isinstance(request, PublicRootRequest)
            and snapshot.terminal is None and request.state.replay.requests == {}
            and request.set_source_hash == source.metadata.source_hash,
            "oracle requires a bound nonterminal source snapshot and public root")
        subject = request.state.player_id
        require(subject in {"p1", "p2"} and snapshot.latest_requests.get(subject) == request.state.self_request
            and snapshot.first_requests.get(subject) == request.state.self_initial_request
            and snapshot.replay.public_events == request.state.replay.public_events
            and snapshot.observation_format_id == request.state.observation_format_id == "gen3randombattle",
            "oracle source does not match actor/public root")
        own = opening_team(request.state.self_initial_request, subject, source)
        opponent = "p2" if subject == "p1" else "p1"
        team = opening_team(snapshot.first_requests.get(opponent, {}), opponent, source)
        return cls(subject, request.set_source_hash, public_root_binding(request),
            team_sha256(own), team, team_sha256(team))

    def validate(self, request, source):
        require(isinstance(request, PublicRootRequest) and self.subject == request.state.player_id
            and self.set_source_hash == request.set_source_hash == source.metadata.source_hash
            and self.root_binding == public_root_binding(request)
            and self.own_team_sha256 == team_sha256(opening_team(
                request.state.self_initial_request, self.subject, source))
            and self.opponent_team_sha256 == team_sha256(self.opponent_team),
            "oracle root, team or source binding changed")

    def receipt(self):
        return dict(schema="pokezero.search-over-raw.team-oracle.v1",
            information_scope="original_opponent_team_only", root_binding=self.root_binding,
            own_team_sha256=self.own_team_sha256, opponent_team_sha256=self.opponent_team_sha256,
            set_source_hash=self.set_source_hash, deployable=False,
            source_snapshot_transported=False, committed_opponent_action_transported=False)


class OracleTeamSampler:
    def __init__(self, team, source):
        from ..randbat import canonical_gen3_randbat_species_id
        require(type(team) is tuple and len(team) == 6
            and len({canonical_gen3_randbat_species_id(p.species) for p in team}) == 6,
            "oracle requires one complete original party")
        self.team, self.source = team, source
        self.sha256 = team_sha256(team)

    def draw(self, known, hidden_rng):
        require(team_sha256(self.team) == self.sha256, "oracle team mutated after binding")
        rows = []
        for traits in known:
            matches = [mon for mon in self.team if _matches(mon, traits, self.source)]
            if len(matches) != 1:
                raise ReferenceRefusal("true original team contradicts public set evidence")
            rows.append(KnownDrawReceipt(traits.species, (), False, team_sha256(matches)))
        # No opponent-team RNG draw: chance and historical action conditioning
        # still use their original domains and public evidence.
        return HiddenTeamDraw(self.team, tuple(rows), (), self.sha256)


@dataclass(frozen=True)
class OracleRootRequest:
    public: PublicRootRequest
    oracle: TeamOracle


@dataclass(frozen=True)
class OracleWorkerFactory:
    base: ShowdownWorkerFactory

    def __post_init__(self):
        require(isinstance(self.base, ShowdownWorkerFactory), "oracle requires the unchanged reference runtime")

    def __call__(self, index):
        return _OracleRuntime(self.base(index))


class _OracleRuntime:
    def __init__(self, base):
        self.base = base

    def prepare(self, request):
        from .paper_reference_factory import PublicRootWorldFactory
        from .paper_reference_parallel import PreparedDecision
        require(isinstance(request, OracleRootRequest), "oracle worker requires explicit diagnostic transport")
        request.oracle.validate(request.public, self.base.source)
        prepared = self.base.prepare(request.public)
        factory = prepared.sample_world
        require(isinstance(factory, PublicRootWorldFactory), "oracle requires genuine public reconstruction")
        factory.install_diagnostic_oracle(request.oracle.opponent_team)

        def evidence():
            result = prepared.evidence()
            result["team_oracle"] = request.oracle.receipt()
            return result

        return PreparedDecision(prepared.root, prepared.evaluate_root, factory, evidence,
            prepared.set_sampling_deadline)

    def close(self):
        self.base.close()


def validate_oracle_work(measured, oracle):
    require(bool(measured.worker_receipts), "oracle requires actual worker evidence")
    for receipt in measured.worker_receipts:
        evidence = receipt["evidence"]
        require(evidence.get("team_oracle") == oracle.receipt(), "oracle worker binding drift")
        for draw in evidence["draws"]:
            require(draw.get("diagnostic_oracle_team_sha256") == oracle.opponent_team_sha256
                and draw.get("information_scope") == "original_opponent_team_only",
                "oracle draw lost truth; sampled fallback is forbidden")


class DiagnosticTeamOracleSearchAdapter(PublicModelSearchAdapter):
    """Explicit nondeployable path; public adapters continue to reject oracles."""

    def __init__(self, configuration, *, oracle, checkpoint_contract, showdown_root, **kwargs):
        from ..randbat import load_gen3_randbat_source_cached
        require(isinstance(oracle, TeamOracle), "explicit team-only oracle required")
        self.oracle = oracle
        self._oracle_source = load_gen3_randbat_source_cached(showdown_root)
        require(oracle.set_source_hash == checkpoint_contract.showdown_source_sha256
            == self._oracle_source.metadata.source_hash, "oracle runtime source differs")
        super().__init__(configuration, checkpoint_contract=checkpoint_contract,
            showdown_root=showdown_root, **kwargs)

    def _check_configuration(self, configuration):
        require(configuration.belief == "oracle" and configuration.arm in {"incumbent", "reference"}
            and (configuration.arm == "reference" or configuration.leaf in {"model", "hp_fraction", "raw_rollout"}),
            "diagnostic oracle requires an explicit supported nondeployable configuration")

    def _reference_worker_factory(self, factory):
        return super()._reference_worker_factory(OracleWorkerFactory(factory))

    def _diagnostic_request(self, request):
        self.oracle.validate(request, self._oracle_source)
        return OracleRootRequest(request, self.oracle)

    def _prepare_incumbent(self, request):
        from ..env import BattleStartOverride
        self.oracle.validate(request, self._oracle_source)
        require(hasattr(self._native, "_fixed_override"), "incumbent lacks explicit oracle hook")
        # Reconstruct the actor's own opening party from its public transport;
        # do not copy a current private opponent request into the engine.
        own = opening_team(request.state.self_initial_request, self.oracle.subject, self._oracle_source)
        opponent = "p2" if self.oracle.subject == "p1" else "p1"
        self._oracle_override = BattleStartOverride(player_teams={self.oracle.subject: pack_team(own),
            opponent: pack_team(self.oracle.opponent_team)}, observation_format_id="gen3randombattle")
        self._native._fixed_override = self._oracle_override

    def _validate_diagnostic_work(self, measured):
        validate_oracle_work(measured, self.oracle)

    def _selection_evidence(self, evidence, request):
        self.oracle.validate(request, self._oracle_source)
        if self.configuration.arm == "incumbent":
            require(self._native._fixed_override == self._oracle_override,
                "incumbent lost oracle override during selection")
            require(not evidence["fallbacks"] and not evidence["prior_fallbacks"],
                "oracle incumbent fallback is forbidden")
        return {**evidence, "team_oracle": self.oracle.receipt()}
