"""Opt-in observation of ACTUAL reference worlds; truth joins only afterward.

This measures original-team agreement, not dynamic hidden-state fidelity or
posterior calibration. Recording adds work to the decision clock; instrumented
selections cannot qualify the uninstrumented runtime. No truth is sent to public
workers, no extra worlds are drawn, and no values/actions are changed.
"""
from dataclasses import dataclass

from .search_over_raw import digest, require

SCHEMA = "pokezero.search-over-raw.sampled-original-team.v1"
FIELDS = ("species", "moves", "ability", "item", "level")


def team_fingerprints(team):
    from ..randbat import canonical_gen3_randbat_species_id
    from ..tier2 import canonical_move_id
    import re

    require(type(team) is tuple and len(team) == 6, "diagnostic requires a complete original party")
    rows = []
    for mon in team:
        species = canonical_gen3_randbat_species_id(mon.species)
        normalize = lambda value: re.sub(r"[^a-z0-9]", "", (value or "").lower())
        moves = sorted(canonical_move_id(m) for m in mon.moves)
        require(species and 1 <= len(moves) <= 4 and len(set(moves)) == len(moves)
            and type(mon.level) is int and 1 <= mon.level <= 100, "malformed diagnostic set")
        rows.append(dict(species=species, moves=digest(moves),
            ability=digest(normalize(mon.ability)), item=digest(normalize(mon.item)),
            level=digest(mon.level)))
    require(len({r["species"] for r in rows}) == 6, "duplicate diagnostic species")
    return sorted(rows, key=lambda row: row["species"])


def truth_record(oracle, *, request, source):
    """Trusted controller only; never attach this result to public transport."""
    from .search_over_raw_oracle import TeamOracle, team_sha256
    require(isinstance(oracle, TeamOracle)
        and team_sha256(oracle.opponent_team) == oracle.opponent_team_sha256,
        "truth requires a bound immutable team oracle")
    oracle.validate(request, source)
    return dict(schema=SCHEMA, role="CONTROLLER_TRUTH_ONLY", root_binding=oracle.root_binding,
        subject=oracle.subject, set_source_hash=oracle.set_source_hash,
        members=team_fingerprints(oracle.opponent_team),
        measurement_scope="original_species_moves_ability_item_level",
        exact_nature_gender_spread_identity_established=False)


def sampled_team_origin(draw):
    """Copy the actual generator party, not a reconstructed/current request."""
    from .search_over_raw_oracle import team_sha256
    require(team_sha256(draw.team) == draw.packed_team_sha256, "sampled origin packed identity drift")
    members = team_fingerprints(draw.team)
    return dict(schema=SCHEMA, original_team_sha256=draw.packed_team_sha256,
        members=members, members_sha256=digest(members))


def selected_team_origin(evidence, depth=0):
    """Follow only the selected empirical particle's actual anchor ancestry."""
    require(type(evidence) is dict and depth <= 16, "invalid selected-team ancestry")
    if "substitute_particle_conditioning" in evidence:
        return selected_team_origin(evidence["substitute_particle_conditioning"]["anchor"], depth+1)
    origin = evidence.get("sampled_team_origin")
    require(type(origin) is dict and origin.get("schema") == SCHEMA
        and type(origin.get("original_team_sha256")) is str
        and len(origin["original_team_sha256"]) == 64
        and origin["original_team_sha256"] == evidence.get("packed_team_sha256")
        and origin.get("members_sha256") == digest(origin.get("members")),
        "selected world lacks its actual sampled-team origin")
    validate_members(origin["members"])
    return origin


def capture_sampled_team(factory, world, evidence):
    """Observe an owned materialized hypothetical world, never the source env."""
    from .paper_reference_showdown import ShowdownTrajectoryWorld, decision_state
    require(isinstance(world, ShowdownTrajectoryWorld) and world.env is factory.env
        and world.env._search_snapshot_permitted is True and not world.closed
        and evidence["status"] == "ROOT_VALIDATED"
        and decision_state(world.env.observe(world.subject), player=world.subject) == factory.root,
        "diagnostic must observe the actual owned root world")
    origin = selected_team_origin(evidence)
    evidence["sampled_original_team"] = dict(schema=SCHEMA, role="HYPOTHETICAL_DRAW_ONLY",
        root_binding=factory.team_diagnostic_root_binding, subject=world.subject,
        information_key=factory.root.key.hex(), set_source_hash=factory.set_source.metadata.source_hash,
        members=origin["members"], members_sha256=origin["members_sha256"],
        original_team_sha256=origin["original_team_sha256"],
        measurement_scope="original_species_moves_ability_item_level",
        instrumentation_can_change_deadlines=True, qualifies_uninstrumented_runtime=False)


@dataclass(frozen=True)
class BeliefDiagnosticWorkerFactory:
    """Separate prospective factory, leaving old factory schemas/defaults intact."""
    base: object

    def __post_init__(self):
        from .paper_reference_runtime import ShowdownWorkerFactory
        from .search_over_raw_oracle import OracleWorkerFactory
        from .search_over_raw_leaves import ReferenceLeafWorkerFactory
        require(isinstance(self.base, (ShowdownWorkerFactory, OracleWorkerFactory,
            ReferenceLeafWorkerFactory)), "diagnostic requires the genuine reference runtime")

    def __call__(self, index):
        return _BeliefDiagnosticRuntime(self.base(index))


class _BeliefDiagnosticRuntime:
    def __init__(self, base):
        self.base = base

    def prepare(self, request):
        from .search_over_raw_oracle import OracleRootRequest, public_root_binding
        prepared = self.base.prepare(request)
        runtime = self.base
        while not hasattr(runtime, "prepared_factory") and hasattr(runtime, "base"):
            runtime = runtime.base
        require(hasattr(runtime, "prepared_factory") and runtime.prepared_factory is not None,
            "diagnostic has no owned public factory")
        public = request.public if isinstance(request, OracleRootRequest) else request
        runtime.prepared_factory.enable_team_diagnostics(public_root_binding(public))
        return prepared

    def close(self):
        self.base.close()


def compare_sampled_teams(draws, *, truth, information_key):
    """Controller-side interval readout on the FULL attempted draw denominator.

    Call per worker/root, preserving that worker's full attempted draw ledger.
    Refused/cancelled/unvalidated worlds contribute [0,1] per agreement. Never
    interpret attempted worlds as independent statistical samples: source seeds
    are the inferential clusters, and this readout makes no significance claim.
    """
    require(type(draws) is list and type(truth) is dict and truth.get("schema") == SCHEMA
        and truth.get("role") == "CONTROLLER_TRUTH_ONLY", "bound controller truth required")
    validate_members(truth["members"])
    require(type(information_key) is str and bool(information_key), "root information key required")
    ordinals = [r.get("ordinal") for r in draws]
    require(all(type(i) is int and i >= 0 for i in ordinals)
        and len(set(ordinals)) == len(ordinals), "duplicate or invalid draw ordinal")
    totals = {field: [0, 0] for field in FIELDS}
    validated = 0
    for row in draws:
        require(row.get("status") in {"STARTED", "REFUSED", "DEADLINE_CANCELLED", "ROOT_VALIDATED"},
            "unknown world status")
        diagnostic = row.get("sampled_original_team")
        if diagnostic is None:
            require(row["status"] != "ROOT_VALIDATED", "validated world lacks required diagnostic")
            for field in FIELDS:
                totals[field][1] += 6
            continue
        require(row["status"] == "ROOT_VALIDATED" and diagnostic.get("schema") == SCHEMA
            and diagnostic.get("role") == "HYPOTHETICAL_DRAW_ONLY"
            and diagnostic.get("instrumentation_can_change_deadlines") is True
            and diagnostic.get("qualifies_uninstrumented_runtime") is False
            and diagnostic.get("information_key") == information_key
            and all(diagnostic.get(k) == truth[k] for k in ("root_binding", "subject", "set_source_hash")),
            "diagnostic root/source/information boundary drift")
        validate_members(diagnostic["members"])
        origin = selected_team_origin(row)
        require(diagnostic["members"] == origin["members"]
            and diagnostic.get("members_sha256") == origin["members_sha256"]
            and diagnostic.get("original_team_sha256") == origin["original_team_sha256"],
            "measurement differs from selected original team")
        sampled = {r["species"]: r for r in diagnostic["members"]}
        validated += 1
        for member in truth["members"]:
            match = sampled.get(member["species"])
            for field in FIELDS:
                agreed = int(match is not None and match[field] == member[field])
                totals[field][0] += agreed
                totals[field][1] += agreed
    denominator = 6 * len(draws)
    return dict(schema="pokezero.search-over-raw.team-agreement.v1", root_binding=truth["root_binding"],
        attempted_worlds=len(draws), validated_worlds=validated,
        unresolved_worlds=len(draws)-validated, member_denominator=denominator,
        agreement_intervals={field: [v / denominator for v in interval] if denominator else [0., 1.]
            for field, interval in totals.items()},
        estimand="original opponent species/moves/ability/item/level in actual attempted reference worlds",
        exact_nature_gender_spread_identity_established=False,
        root_member_weighting="six truth members per attempted world; absent species is disagreement",
        dynamic_hidden_state_fidelity=False, posterior_calibration_established=False,
        significance_claim=False, scientific_strength_evidence=False)


def validate_members(rows):
    require(type(rows) is list and len(rows) == 6
        and all(type(row) is dict and set(row) == set(FIELDS) for row in rows)
        and all(type(row["species"]) is str and bool(row["species"]) for row in rows)
        and len({row["species"] for row in rows}) == 6
        and all(type(row[field]) is str and len(row[field]) == 64
            and all(c in "0123456789abcdef" for c in row[field])
            for row in rows for field in FIELDS[1:]), "malformed team fingerprint ledger")


class _BeliefDiagnosticAdapterMixin:
    def __init__(self, configuration, **kwargs):
        require(configuration.arm == "reference", "incumbent belief instrumentation remains required")
        super().__init__(configuration, **kwargs)
        self.runtime_configuration["belief_diagnostics"] = dict(schema=SCHEMA,
            scope="actual_materialized_original_teams", truth_sent_to_public_worker=False,
            instrumentation_can_change_deadlines=True, qualifies_uninstrumented_runtime=False)
        self.runtime_sha256 = digest(self.runtime_configuration)

    def _reference_worker_factory(self, factory):
        return BeliefDiagnosticWorkerFactory(super()._reference_worker_factory(factory))

    def _selection_evidence(self, evidence, request):
        from .search_over_raw_oracle import public_root_binding
        binding = public_root_binding(request)
        for receipt in evidence["worker_receipts"]:
            for row in receipt["evidence"]["draws"]:
                diagnostic = row.get("sampled_original_team")
                if row["status"] == "ROOT_VALIDATED":
                    require(type(diagnostic) is dict and diagnostic.get("schema") == SCHEMA
                        and diagnostic.get("role") == "HYPOTHETICAL_DRAW_ONLY"
                        and diagnostic.get("root_binding") == binding
                        and diagnostic.get("subject") == request.state.player_id
                        and diagnostic.get("set_source_hash") == request.set_source_hash,
                        "selected reference world lacks bound belief evidence")
                    validate_members(diagnostic["members"])
                else:
                    require(diagnostic is None, "unvalidated world carries an agreement measurement")
        return super()._selection_evidence(evidence, request)


from .search_over_raw_adapters import PublicModelSearchAdapter
from .search_over_raw_oracle import DiagnosticTeamOracleSearchAdapter


class PublicBeliefDiagnosticSearchAdapter(_BeliefDiagnosticAdapterMixin, PublicModelSearchAdapter):
    """Public actor: sampled-team observations only, never source truth."""


class OracleBeliefDiagnosticSearchAdapter(_BeliefDiagnosticAdapterMixin, DiagnosticTeamOracleSearchAdapter):
    """Diagnostic oracle actor, retaining its explicit nondeployable scope."""
