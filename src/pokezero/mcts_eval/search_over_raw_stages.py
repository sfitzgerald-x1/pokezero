"""Exploration A2 ledger and sealed A3 choice; never runtime authorization.

Same retained roots and continuation randomness permit descriptive performance
ranking. Equal sampler seeds do NOT establish identical realized world counts
under asynchronous clocks. Actual whole-population matching is reported apart
from performance; unmatched tails are never filtered from either denominator.
This does not implement direct value labels, fidelity, holdout or launch gates.
"""
from copy import deepcopy
from dataclasses import asdict
import math
import statistics

from .search_over_raw import SearchConfiguration, digest, require, root_contrast
from .search_over_raw_belief_diagnostics import selected_team_origin

LEAVES = ("model", "hp_fraction", "raw_rollout")
GROUPS = tuple((arm, belief) for arm in ("incumbent", "reference")
    for belief in ("public", "oracle"))


def a2_configurations():
    return (SearchConfiguration("raw", seconds=1.), *(SearchConfiguration(
        arm, belief, leaf, 1., 20 if arm == "reference" else 1)
        for arm, belief in GROUPS for leaf in LEAVES))


def key(cfg):
    return "raw" if cfg.arm == "raw" else cfg.identity


def _world_population(selected, arm):
    """Keep every attempted world keyed by worker/ordinal, never a common prefix."""
    evidence = selected["evidence"]
    ledgers = [(0, evidence["belief_draws"])] if arm == "incumbent" else [
        (receipt["worker"], receipt["evidence"]["draws"])
        for receipt in evidence["worker_receipts"]]
    result = {}
    for worker, draws in ledgers:
        require(type(worker) is int and 0 <= worker < (1 if arm == "incumbent" else 20)
            and type(draws) is list, "invalid actual world population")
        for draw in draws:
            ordinal = draw.get("ordinal")
            require(type(ordinal) is int and ordinal >= 0 and (worker, ordinal) not in result,
                "duplicate or invalid actual world ordinal")
            status = draw.get("status")
            require(status in {"STARTED", "REFUSED", "DEADLINE_CANCELLED", "ROOT_VALIDATED"},
                "unknown actual world status")
            result[worker, ordinal] = (status, selected_team_origin(draw)["original_team_sha256"]
                if status == "ROOT_VALIDATED" else None)
    return result


def compare_leaf_worlds(selections, configs, *, root_id, selection_seed, legal_choices):
    require(type(legal_choices) is int and legal_choices > 0, "bound legal choice count required")
    populations = {}
    for cfg in configs:
        selected = selections[cfg.identity]
        require(selected.get("statistics_root_id") == root_id
            and type(selected.get("selection_seed")) is int
            and selected["selection_seed"] == selection_seed,
            "leaf sampler stream/root namespace differs")
        populations[cfg.leaf] = _world_population(selected, cfg.arm)
    require(set(populations) == set(LEAVES), "all three leaf populations required")
    complete = all(population and all(status == "ROOT_VALIDATED" for status, _ in population.values())
        for population in populations.values())
    matched = complete and all(population == populations["model"] for population in populations.values())
    forced = legal_choices == 1
    return dict(schema="pokezero.search-over-raw.a2-world-comparison.v1",
        same_sampler_stream_binding=True, original_team_population_match=bool(matched),
        strict_whole_population_match=False, full_world_identity_established=False,
        forced_action_not_evaluator_evidence=forced,
        attempted_world_counts={leaf: len(pop) for leaf, pop in populations.items()},
        unresolved_world_counts={leaf: sum(status != "ROOT_VALIDATED" for status, _ in pop.values())
            for leaf, pop in populations.items()},
        population_sha256={leaf: digest([[worker, ordinal, *value]
            for (worker, ordinal), value in sorted(pop.items())]) for leaf, pop in populations.items()},
        common_prefix_filtering=False, equal_tree_paths_established=False,
        visited_value_calibration_established=False, causal_evaluator_claim=False)


class ExplorationStages:
    """A2 full-roster accounting and irreversible exploration-only A3 freeze.

    The bank's sealed public manifest binds which new roots exist. All original
    two-source A2 slots and source-validated short-game slots remain uncertain.
    No historical A1 measurement is promoted to an A2 evaluator measurement.
    """
    def __init__(self, plan, bank_manifest, *, original_deadline_at=None, value_diagnostics=None):
        require(original_deadline_at is None or (type(original_deadline_at) in (int, float)
            and math.isfinite(original_deadline_at)), "finite A2 execution deadline required")
        if value_diagnostics is not None:
            from .search_over_raw_value_diagnostics import ValueDiagnosticContract
            require(isinstance(value_diagnostics, ValueDiagnosticContract)
                and value_diagnostics.original_deadline_at == original_deadline_at,
                "A2 live diagnostic contract must bind original deadline")
        require(digest(plan["phase_a_cohort"]) == plan["phase_a_cohort_sha256"], "original cohort digest drift")
        panel = plan["phase_a_cohort"]["panels"]["exploration"]
        slots, seeds = panel["root_slots"], panel["seeds"]
        require(len(slots) == 200 and len({s["root_id"] for s in slots}) == 200
            and len(seeds) == len(set(seeds)) == 32
            and set(s["source_seed"] for s in slots) == set(seeds), "full original exploration roster required")
        expected = {s["root_id"] for s in plan["comparable_root_contract"]["root_slots"]}
        captured = set(bank_manifest["root_public_bindings"])
        missing_rows = bank_manifest["source_validated_missing_root_ids"]
        missing = set(missing_rows)
        require(bank_manifest["state"] == "SEALED" and bank_manifest["capacity"] == len(expected) == 186
            and bank_manifest["captured_roots"] == len(captured) and len(missing_rows) == len(missing)
            and not captured & missing and captured | missing == expected
            and expected <= {s["root_id"] for s in slots}, "sealed exact186 comparable roots required")
        cohort_configs = {row["identity"] for row in plan["phase_a_cohort"]["configurations"]}
        require(all(cfg.identity in cohort_configs for cfg in a2_configurations()), "A2 changed original cohort")
        self._panel, self._manifest = deepcopy(panel), deepcopy(bank_manifest)
        self._slots = {row["root_id"]: row for row in slots}
        self._captured = captured
        self._records = {}
        self._frozen = None
        self._plan_sha256 = digest(plan)
        self._a3_started = False
        self._a3_records = {}
        self._original_deadline_at = original_deadline_at
        self._value_diagnostic_contract = asdict(value_diagnostics) if value_diagnostics is not None else None

    def record_root(self, audit, selections, *, public_record_sha256, selection_seed, legal_choices):
        require(self._frozen is None, "A2 frozen; no later evidence or reselection")
        root_id = audit["root_id"]
        require(root_id in self._captured and root_id not in self._records
            and public_record_sha256 == self._manifest["root_public_bindings"][root_id]
            and type(selection_seed) is int and selection_seed == self._slots[root_id]["source_seed"],
            "unregistered, duplicate or unbound A2 root")
        configs = a2_configurations()
        require(set(selections) == set(audit["actions"]) == {key(cfg) for cfg in configs},
            "exact raw and twelve A2 selections required")
        for cfg in configs:
            selected = selections[key(cfg)]
            require(selected["status"] == "SELECTED" and selected["configuration_sha256"] == cfg.identity
                and selected["root_id"] == root_id + ":" + key(cfg)
                and type(selected["action"]) is int and selected["action"] >= 0
                and selected["action"] == audit["actions"][key(cfg)], "A2 action/configuration binding drift")
        require(legal_choices != 1 or len(set(audit["actions"].values())) == 1,
            "forced-action selectors disagree")
        intervals = {cfg.identity: list(root_contrast(audit, cfg.identity))
            for cfg in configs if cfg.arm != "raw"}
        require(all(row["status"] in {"COMPLETE", "CAPPED"} for row in audit["outcomes"]),
            "operational refusal stops A2; no frozen choice")
        worlds = {arm + ":" + belief: compare_leaf_worlds(selections,
            [cfg for cfg in configs if (cfg.arm, cfg.belief) == (arm, belief)],
            root_id=root_id, selection_seed=selection_seed, legal_choices=legal_choices)
            for arm, belief in GROUPS}
        require(audit["status"] == ("COMPLETE" if all(row["status"] == "COMPLETE"
            for row in audit["outcomes"]) else "UNCERTAIN"), "A2 audit status drift")
        self._records[root_id] = deepcopy(dict(intervals=intervals, worlds=worlds,
            audit_sha256=digest(audit), selections_sha256=digest(selections),
            selected_sha256={name: digest(row) for name, row in selections.items()},
            outcome_sha256={f"{row['action']}:{row['replicate']}": digest(row) for row in audit["outcomes"]}))

    def _summary(self, cfg, records=None):
        records = self._records if records is None else records
        by_seed = {seed: [] for seed in self._panel["seeds"]}
        uncertain = 0
        for root_id, slot in self._slots.items():
            interval = records.get(root_id, {}).get("intervals", {}).get(cfg.identity, [-1., 1.])
            require(len(interval) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in interval)
                and -1 <= interval[0] <= interval[1] <= 1, "invalid full-roster A2 bounds")
            uncertain += interval[0] != interval[1]
            by_seed[slot["source_seed"]].append(interval)
        means = [[statistics.mean(row[i] for row in by_seed[seed]) for i in (0, 1)]
            for seed in self._panel["seeds"]]
        return dict(configuration=asdict(cfg), configuration_sha256=cfg.identity,
            root_slots=200, source_seeds=32, uncertain_roots=uncertain,
            identification_interval=[statistics.mean(row[i] for row in means) for i in (0, 1)],
            seed_intervals=[dict(source_seed=seed, interval=row) for seed, row in zip(self._panel["seeds"], means)])

    def freeze_for_a3(self):
        require(self._frozen is None and set(self._records) == self._captured,
            "A3 requires one complete fixed A2 accounting pass; no early freeze or reselection")
        choices = []
        for arm, belief in GROUPS:
            rows = [self._summary(cfg) for cfg in a2_configurations() if (cfg.arm, cfg.belief) == (arm, belief)]
            best = max(row["identification_interval"][0] for row in rows)
            tied = [row for row in rows if row["identification_interval"][0] == best]
            chosen = tied[0]  # LEAVES order is the prospective tie rule.
            worlds = [record["worlds"][arm + ":" + belief] for record in self._records.values()]
            informative = [row for row in worlds if not row["forced_action_not_evaluator_evidence"]]
            choices.append(dict(arm=arm, belief=belief, leaf=chosen["configuration"]["leaf"],
                selected_configuration_sha256=chosen["configuration_sha256"], candidate_scores=rows,
                choice_basis="LOWER_BOUND_PERFORMANCE_RANK" if len(tied) == 1 else "PREDECLARED_TIE_DEFAULT_NOT_A_WINNER",
                tied_leaves=[row["configuration"]["leaf"] for row in tied], deployable=belief == "public",
                original_team_matched_roots=sum(row["original_team_population_match"] for row in informative),
                strict_same_world_roots=0,
                substantive_roots=len(informative), full_panel_same_world_qualification=False,
                causal_evaluator_claim=False, positive_gain_claim=False))
        self._frozen = dict(schema="pokezero.search-over-raw.a3-exploration-freeze.v1", choices=choices,
            original_deadline_at=self._original_deadline_at,
            visited_value_diagnostics=deepcopy(self._value_diagnostic_contract),
            full_root_denominator=200, full_seed_denominator=32, accounted_new_roots=len(self._records),
            unavailable_a2_roots=200-len(self._records),
            a2_evidence_sha256=digest(self._records), bank_manifest_sha256=digest(self._manifest),
            selection_rule="equal-source-seed mean of pointwise lower bounds; ties model then hp_fraction then raw_rollout",
            diagnostic_scope="performance ranking only; actual full populations reported without filtering; direct value/fidelity checks still required",
            holdout_authorized=False, runtime_authorized=False, scientific_strength_evidence=False)
        return deepcopy(self._frozen)

    def a3_configurations(self):
        require(self._frozen is not None, "A3 configuration roster requires prior sealed exploration freeze")
        return (SearchConfiguration("raw", seconds=1.), *(SearchConfiguration(
            row["arm"], row["belief"], row["leaf"], seconds, 20 if row["arm"] == "reference" else 1)
            for row in self._frozen["choices"] for seconds in (1., 3., 10.)))

    def begin_a3(self, plan, bank_manifest, frozen, *, deadline_at):
        """Claim this in-process frozen stage once; no restore/retry entrypoint."""
        require(not self._a3_started and self._frozen is not None,
            "A3 requires unconsumed sealed exploration freeze; no retry")
        require(self._original_deadline_at is not None and deadline_at == self._original_deadline_at,
            "A3 cannot extend bound original A2 execution deadline")
        def identity(manifest):
            value = deepcopy(manifest)
            value.pop("parent_high_water_bytes", None)  # RSS may rise without changing bank contents.
            return digest(value)
        require(digest(plan) == self._plan_sha256 and digest(frozen) == digest(self._frozen)
            and identity(bank_manifest) == identity(self._manifest), "A3 plan/bank/freeze binding drift")
        require(self._value_diagnostic_contract is None,
            "instrumented A2 cannot feed an uninstrumented A3 curve; matched capture/reuse pending")
        self._a3_started = True
        return deepcopy(self._frozen)

    def a3_reuse(self, root_id, audit, selections):
        """Bind all original A2 evidence, then project only frozen 1s selectors."""
        require(self._a3_started and root_id in self._records, "A3 has no captured A2 root")
        record = self._records[root_id]
        require(digest(audit) == record["audit_sha256"]
            and digest(selections) == record["selections_sha256"], "prior A2 evidence hash drift")
        contract = audit["continuation_contract"]
        require(contract["root_id"] == root_id
            and contract["root_binding"]["public_record_sha256"] == self._manifest["root_public_bindings"][root_id]
            and contract["selection_seed"] == self._slots[root_id]["source_seed"], "prior A2 public root drift")
        reused = {key(cfg): deepcopy(selections[key(cfg)]) for cfg in self.a3_configurations()
            if cfg.arm == "raw" or cfg.seconds == 1.}
        return dict(audit=deepcopy(audit), audit_sha256=record["audit_sha256"], selections=reused)

    def record_a3_root(self, audit, selections):
        root_id = audit["root_id"]
        require(self._a3_started and root_id in self._captured and root_id not in self._a3_records,
            "unstarted, uncaptured or duplicate A3 root")
        configs = self.a3_configurations()
        record = self._records[root_id]
        require(set(selections) == set(audit["actions"]) == {key(cfg) for cfg in configs}
            and audit["prior_a2_audit_sha256"] == record["audit_sha256"], "A3 frozen roster/parent drift")
        for cfg in configs:
            selected = selections[key(cfg)]
            require(selected["status"] == "SELECTED" and selected["configuration_sha256"] == cfg.identity
                and selected["root_id"] == root_id + ":" + key(cfg)
                and type(selected["action"]) is int and selected["action"] == audit["actions"][key(cfg)],
                "A3 selection binding drift")
            if cfg.arm == "raw" or cfg.seconds == 1.:
                require(digest(selected) == record["selected_sha256"][key(cfg)], "A3 reran original one-second selector")
            else:
                require(selected.get("statistics_root_id") == root_id
                    and selected.get("selection_seed") == self._slots[root_id]["source_seed"],
                    "A3 sampler root/seed drift")
        expected_reused = {f"{action}:{rep}" for action in set(audit["actions"].values()) for rep in range(8)
            if f"{action}:{rep}" in record["outcome_sha256"]}
        keys = audit["reused_continuation_keys"]
        require(len(keys) == len(expected_reused)
            and {f"{a}:{r}" for a, r in keys} == expected_reused, "A3 shared-action evidence must be reused exactly")
        for row in audit["outcomes"]:
            name = f"{row['action']}:{row['replicate']}"
            if name in expected_reused:
                require(digest(row) == record["outcome_sha256"][name], "A3 changed or retried prior continuation")
        intervals = {cfg.identity: list(root_contrast(audit, cfg.identity)) for cfg in configs if cfg.arm != "raw"}
        require(all(row["status"] in {"COMPLETE", "CAPPED"} for row in audit["outcomes"])
            and audit["status"] == ("COMPLETE" if all(row["status"] == "COMPLETE"
                for row in audit["outcomes"]) else "UNCERTAIN"), "A3 operational refusal/status drift")
        self._a3_records[root_id] = dict(intervals=intervals, audit_sha256=digest(audit),
            selections_sha256=digest(selections))

    def a3_readout(self):
        require(self._a3_started and set(self._a3_records) == self._captured, "A3 full fixed accounting pass required")
        curves = []
        for arm, belief in GROUPS:
            rows = [self._summary(cfg, self._a3_records) for cfg in self.a3_configurations()
                if (cfg.arm, cfg.belief) == (arm, belief)]
            curves.append(dict(arm=arm, belief=belief, leaf=rows[0]["configuration"]["leaf"],
                deployable=belief == "public", budget_points=rows))
        observed = sorted({self._slots[root_id]["source_seed"] for root_id in self._a3_records})
        return dict(schema="pokezero.search-over-raw.a3-exploratory-readout.v1", curves=curves,
            full_root_denominator=200, full_seed_denominator=32, measured_roots=len(self._a3_records),
            observed_source_seeds=observed, observed_source_seed_count=len(observed),
            at_least_32_observed_sources=len(observed) >= 32,
            original_phase_a_measurement_requirement_satisfied=False,
            unavailable_roots=200-len(self._a3_records), frozen_choice_sha256=digest(self._frozen),
            a3_evidence_sha256=digest(self._a3_records),
            one_second_points_reused_not_independent=True, causal_mechanism_claim=False,
            visited_value_calibration_established=False, searched_line_fidelity_established=False,
            monotone_gain_claim=False, scientific_strength_evidence=False,
            holdout_authorized=False, phase_b_authorized=False, runtime_authorized=False)
