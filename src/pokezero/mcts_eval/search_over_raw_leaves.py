"""Reference leaf ablations with unchanged tree, worlds and champion priors.

HP values use the sum-HP/sum-max-HP difference on the OWNED hypothetical world.
Raw rollouts use masked argmax for both seats to an uncapped terminal, with a
separate chance domain. Neither mode accepts true source environments, changes
the champion, substitutes uniform rollouts or prices a cap with an HP fallback.
"""
from dataclasses import dataclass
import math
import time

from .paper_reference import Evaluation, LeafDeadlineExceeded, ReferenceRefusal
from .paper_reference_parallel import PreparedDecision
from .paper_reference_runtime import ShowdownWorkerFactory
from .paper_reference_showdown import ShowdownTrajectoryWorld, decision_state
from .search_over_raw import _raw_action, require, rng_seed

ROLLOUT_CAP = 250


def hp_value(snapshot, subject):
    """Signed equivalent of the native HP-fraction leaf, not visible-HP mean."""
    require(subject in ("p1", "p2"), "invalid HP leaf perspective")
    sides = snapshot.bridge_snapshot["battle"]["sides"]
    require(len(sides) == 2, "HP leaf requires both hypothetical parties")
    ratios = []
    for side in sides:
        rows = side["pokemon"]
        require(rows and all(type(row["hp"]) in (int, float) and type(row["maxhp"]) in (int, float)
            and math.isfinite(row["hp"]) and math.isfinite(row["maxhp"])
            and 0 <= row["hp"] <= row["maxhp"] and row["maxhp"] > 0 for row in rows),
            "invalid hypothetical HP ledger")
        ratios.append(math.fsum(row["hp"] for row in rows)/math.fsum(row["maxhp"] for row in rows))
    return (ratios[0] - ratios[1]) * (1 if subject == "p1" else -1)


class ReferenceLeafWorld:
    def __init__(self, world, *, leaf, deadline, receipts, cap=ROLLOUT_CAP, rollout_seed=0):
        require(isinstance(world, ShowdownTrajectoryWorld)
            and world.env._search_snapshot_permitted is True, "leaf requires an owned hypothetical reference world")
        require(leaf in {"hp_fraction", "raw_rollout"} and type(cap) is int and cap > 0,
            "invalid leaf mode or rollout cap")
        self.world, self.leaf, self.deadline, self.receipts, self.cap = world, leaf, deadline, receipts, cap
        self.rollout_seed = rollout_seed

    def frame(self):
        return self.world.frame()

    def advance(self, *args):
        return self.world.advance(*args)

    def close(self):
        self.world.close()

    def evaluate(self, state):
        model = self.world.evaluate(state)
        began = time.perf_counter()
        row = dict(schema="pokezero.search-over-raw.leaf.v1", leaf=self.leaf,
            information_key=state.key.hex(), model_signed_value=model.value,
            champion_priors_unchanged=True, status="ATTEMPTED", boundaries=0,
            alternative_signed_value=None, scientific_strength_evidence=False)
        try:
            if self.leaf == "hp_fraction":
                value = hp_value(self.world.env.snapshot(), self.world.subject)
            else:
                value = self._rollout(state, row)
            row.update(status="COMPLETE", alternative_signed_value=value)
            return Evaluation(model.priors, value)
        except LeafDeadlineExceeded:
            row.update(status="DEADLINE_CANCELLED", backed_up=False)
            raise
        except BaseException:
            row.update(status="REFUSED", backed_up=False)
            raise
        finally:
            row["elapsed_seconds"] = time.perf_counter()-began
            self.receipts.append(row)

    def _rollout(self, state, row):
        env, subject = self.world.env, self.world.subject
        snapshot = env.snapshot_for_search()
        row["continuation_policy"] = "raw_argmax_both_seats"
        row["maximum_boundaries"] = self.cap
        try:
            for step in range(self.cap + 1):
                checked = time.perf_counter()
                deadline = self.deadline()
                if deadline is not None and checked >= deadline:
                    row.update(checked_at=checked, deadline_at=deadline)
                    raise LeafDeadlineExceeded("raw leaf exhausted the decision clock")
                terminal = env.terminal()
                if terminal is not None:
                    if terminal.capped or terminal.winner not in {None, "p1", "p2"}:
                        raise ReferenceRefusal("raw leaf cap or invalid winner is not a terminal value")
                    return 0. if terminal.winner is None else 1. if terminal.winner == subject else -1.
                if step == self.cap:
                    raise ReferenceRefusal("raw leaf rollout cap reached; no fallback value")
                requested = env.requested_players()
                require(requested and set(requested) <= {"p1", "p2"}, "raw leaf has invalid request boundary")
                actions = {}
                for seat in requested:
                    legal, evaluation = self.world.evaluator(env.observe(seat))
                    actions[seat] = _raw_action(legal, evaluation.priors)
                env.reseed_simulator_rng(rng_seed("reference-raw-leaf.v1", self.rollout_seed,
                    state.key.hex(), step, "chance"))
                env.step(actions)
                row["boundaries"] = step + 1
        finally:
            try:
                env.restore_search_snapshot(snapshot)
                # Restoration must leave exactly the leaf that the unchanged
                # tree owns, not accidentally advance its search parent.
                require(env.terminal() is None and subject in env.requested_players()
                    and decision_state(env.observe(subject), player=subject) == state,
                    "raw leaf restoration changed the owned information state")
            finally:
                require(env.release_search_snapshot(snapshot) is True, "raw leaf snapshot release failed")


@dataclass(frozen=True)
class ReferenceLeafWorkerFactory:
    base: ShowdownWorkerFactory
    leaf: str
    rollout_cap: int = ROLLOUT_CAP

    def __post_init__(self):
        from .search_over_raw_oracle import OracleWorkerFactory
        require(isinstance(self.base, (ShowdownWorkerFactory, OracleWorkerFactory)) and self.leaf in {"hp_fraction", "raw_rollout"}
            and self.rollout_cap == ROLLOUT_CAP, "unregistered reference leaf factory")

    def __call__(self, index):
        return _ReferenceLeafRuntime(self.base(index), self.leaf, self.rollout_cap)


class _ReferenceLeafRuntime:
    def __init__(self, base, leaf, cap):
        self.base, self.leaf, self.cap = base, leaf, cap
        self._deadline = None

    def prepare(self, request):
        prepared = self.base.prepare(request)
        receipts = []
        consumed = 0

        def sample_world(rng):
            world = prepared.sample_world(rng)
            # Derive a disjoint continuation stream WITHOUT consuming another
            # hidden draw. The original sampler's RNG schedule is unchanged.
            return ReferenceLeafWorld(world, leaf=self.leaf,
                deadline=lambda: self._deadline, receipts=receipts, cap=self.cap,
                rollout_seed=rng_seed("reference-raw-leaf-domain.v1", rng.getstate()))

        def bind_deadline(deadline):
            prepared.set_sampling_deadline(deadline)
            self._deadline = deadline

        def evidence():
            nonlocal consumed
            result = prepared.evidence()
            result["leaf_ablation"] = dict(schema="pokezero.search-over-raw.leaf-batch.v1",
                leaf=self.leaf, rollout_cap=self.cap, root_priors="unchanged_champion",
                root_value_used_for_backup=False, evaluations=receipts[consumed:])
            consumed = len(receipts)
            return result

        return PreparedDecision(prepared.root, prepared.evaluate_root, sample_world, evidence, bind_deadline)

    def close(self):
        self.base.close()


def validate_leaf_work(measured, leaf):
    require(leaf in {"hp_fraction", "raw_rollout"} and bool(measured.worker_receipts),
        "alternative leaf requires owned worker receipts")
    for receipt in measured.worker_receipts:
        batch = receipt["evidence"].get("leaf_ablation", {})
        require(batch.get("schema") == "pokezero.search-over-raw.leaf-batch.v1"
            and batch.get("leaf") == leaf and batch.get("rollout_cap") == ROLLOUT_CAP
            and batch.get("root_priors") == "unchanged_champion"
            and batch.get("root_value_used_for_backup") is False,
            "reference leaf mode lacks authoritative worker evidence")
        require(type(batch.get("evaluations")) is list, "missing leaf evaluation ledger")
        for row in batch["evaluations"]:
            require(row.get("schema") == "pokezero.search-over-raw.leaf.v1"
                and row["leaf"] == leaf and row["champion_priors_unchanged"] is True
                and row.get("scientific_strength_evidence") is False
                and type(row["model_signed_value"]) in (int, float) and math.isfinite(row["model_signed_value"])
                and -1 <= row["model_signed_value"] <= 1, "invalid leaf/model binding")
            require(type(row.get("boundaries")) is int and 0 <= row["boundaries"] <= ROLLOUT_CAP
                and type(row.get("elapsed_seconds")) in (int, float)
                and math.isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] >= 0,
                "invalid leaf work/timing ledger")
            if row["status"] == "COMPLETE":
                value = row["alternative_signed_value"]
                require(type(value) in (int, float) and math.isfinite(value) and -1 <= value <= 1,
                    "invalid alternative leaf value")
                if leaf == "raw_rollout":
                    require(value in (-1, 0, 1) and row["continuation_policy"] == "raw_argmax_both_seats"
                        and row["maximum_boundaries"] == ROLLOUT_CAP, "raw leaf is not an uncapped policy terminal")
            else:
                checked, deadline = row.get("checked_at"), row.get("deadline_at")
                require(row["status"] == "DEADLINE_CANCELLED" and row["backed_up"] is False
                    and row["alternative_signed_value"] is None
                    and type(checked) in (int, float) and type(deadline) in (int, float)
                    and math.isfinite(checked) and math.isfinite(deadline) and checked >= deadline
                    and receipt["batch"].deadline_exhausted, "unjustified leaf cancellation or fallback")
