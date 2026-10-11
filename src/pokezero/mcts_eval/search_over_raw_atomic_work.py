"""Prospective reference commit observations and a separate arithmetic reader.

Records at the real kernel commit seam, not at an evaluator callback or counter.
This proves consistency of observed OWN backups, exported work and cleanup; it
does not independently replay simulator outcomes or qualify native work. Cost
of instrumentation is inside the decision clock. Historical runtimes opt out.
"""
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math

SCHEMA = "pokezero.search-over-raw.reference-atomic-work.v1"


def digest(value):
    """Canonical receipt hash, independent of the unlanded collection harness."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def state_record(state):
    return dict(key=state.key.hex(), actions=list(state.actions), faint_count=state.faint_count)


def statistics_record(rows):
    return [dict(state=state_record(row.state), visits=list(row.visits),
        totals=list(row.totals), count=row.count) for row in sorted(rows, key=lambda r: r.state.key)]


def update_record(update):
    return dict(battle_id=update.battle_id, worker_id=update.worker_id,
        sequence=update.sequence, rows=statistics_record(update.rows))


class ReferenceAtomicWork:
    def __init__(self, binding):
        self.binding = deepcopy(binding)
        self.rows = []
        self.current = None
        self.consumed = 0
        self.batch_before = self.batch_after = None
        self.root = None

    def begin_batch(self, search, root):
        require(self.current is None, "atomic batch overlaps an attempt")
        self.batch_before = statistics_record(search._local_statistics.values())
        self.batch_after = None
        self.root = state_record(root)

    def end_batch(self, search):
        require(self.current is None and self.batch_before is not None, "incomplete atomic batch")
        self.batch_after = statistics_record(search._local_statistics.values())

    def begin(self, search, root, ordinal):
        require(self.current is None and type(ordinal) is int and ordinal >= 0,
            "atomic trajectory attempt overlaps")
        self.current = dict(ordinal=ordinal, root=state_record(root),
            before=statistics_record(search._local_statistics.values()), after=None,
            status="ATTEMPTED", path=[], discovered_leaf=None, signed_value=None,
            transitions=0, sampled=False, cleanup_completed=False)
        self.rows.append(self.current)

    def commit(self, search, path, leaf, value, transitions):
        require(self.current is not None and self.current["status"] == "ATTEMPTED",
            "duplicate or unbound atomic commit")
        self.current.update(status="COMMITTED", sampled=True,
            path=[dict(state=state_record(node.state), action_index=index) for node, index in path],
            discovered_leaf=None if leaf is None else state_record(leaf.state),
            signed_value=value, transitions=transitions,
            after=statistics_record(search._local_statistics.values()))

    def cancel(self, search, transitions, *, sampled):
        require(self.current is not None and self.current["status"] == "ATTEMPTED",
            "atomic cancellation after commit")
        self.current.update(status="CANCELLED", sampled=sampled, transitions=transitions,
            after=statistics_record(search._local_statistics.values()))
        if not sampled:
            self.current = None  # No owned world was returned to the caller.

    def cleanup(self):
        row = self.current if self.current is not None else self.rows[-1] if self.rows else None
        require(row is not None and row["status"] in {"COMMITTED", "CANCELLED", "REFUSED"},
            "cleanup before atomic trajectory completed")
        row["cleanup_completed"] = True
        self.current = None

    def refuse(self, search):
        if self.current is not None:
            self.current.update(status="REFUSED",
                after=statistics_record(search._local_statistics.values()))
            self.current = None

    def evidence(self, draws):
        rows = deepcopy(self.rows[self.consumed:])
        by_ordinal = {row["ordinal"]: row for row in draws}
        for row in rows:
            draw = by_ordinal.get(row["ordinal"], {})
            row["input_sha256"] = draw.get("consumed_world_input", {}).get("input_sha256")
        self.consumed = len(self.rows)
        return dict(schema=SCHEMA, binding=deepcopy(self.binding), rows=rows,
            batch_before=deepcopy(self.batch_before), batch_after=deepcopy(self.batch_after), root=deepcopy(self.root),
            instrumentation_can_change_deadlines=True, runtime_qualified=False,
            independent_terminal_labels=False)


def _state(row):
    require(type(row) is dict and set(row) == {"key", "actions", "faint_count"}
        and type(row["key"]) is str and row["key"] and len(row["key"]) % 2 == 0
        and all(c in "0123456789abcdef" for c in row["key"])
        and type(row["actions"]) is list and row["actions"]
        and all(type(a) is str and a for a in row["actions"])
        and len(set(row["actions"])) == len(row["actions"])
        and type(row["faint_count"]) is int and 0 <= row["faint_count"] <= 12,
        "invalid atomic information-state identity")
    return row["key"]


def _statistics(rows):
    require(type(rows) is list, "atomic statistics roster required")
    result = {}
    for row in rows:
        require(type(row) is dict and set(row) == {"state", "visits", "totals", "count"},
            "atomic statistics fields drift")
        key = _state(row["state"])
        n = len(row["state"]["actions"])
        require(key not in result and type(row["visits"]) is list and type(row["totals"]) is list
            and len(row["visits"]) == len(row["totals"]) == n
            and all(type(v) is int and v >= 0 for v in row["visits"])
            and type(row["count"]) is int and row["count"] == sum(row["visits"])
            and all(type(q) in (int, float) and math.isfinite(q) and abs(q) <= v
                and (v != 0 or q == 0) for v, q in zip(row["visits"], row["totals"])),
            "invalid atomic OWN statistics")
        result[key] = deepcopy(row)
    require(list(result) == sorted(result), "atomic statistics ordering drift")
    return result


def validate_reference_batch(evidence, *, batch, update, root, battle_id, worker_id, sequence):
    """Reconcile exact observed commits with the actual outgoing OWN update.

    Every draw and exported node is checked, including cancelled work. Imported
    master statistics are deliberately not treated as new OWN work. Evidence is
    not self-certifying: durable source/runtime bindings and terminal replay are
    separate producer admission obligations.
    """
    saved = deepcopy(evidence)
    audit = saved.get("atomic_work", {})
    require(type(audit) is dict and set(audit) == {"schema", "binding", "rows",
        "batch_before", "batch_after", "root", "instrumentation_can_change_deadlines",
        "runtime_qualified", "independent_terminal_labels"}
        and audit.get("schema") == SCHEMA and audit.get("instrumentation_can_change_deadlines") is True
        and audit.get("runtime_qualified") is False and audit.get("independent_terminal_labels") is False,
        "atomic work scope drift")
    binding = audit["binding"]
    require(type(binding) is dict and set(binding) == {"root_binding", "subject", "set_source_hash", "information_key"}
        and binding["information_key"] == root.key.hex() and binding["subject"] in {"p1", "p2"}
        and all(type(binding[k]) is str and len(binding[k]) == 64
            and all(c in "0123456789abcdef" for c in binding[k])
            for k in ("root_binding", "set_source_hash")), "atomic root binding drift")
    rows, draws = audit.get("rows"), saved.get("draws")
    require(type(rows) is list and type(draws) is list and len(rows) == len(draws),
        "atomic attempt/draw roster drift")
    batch = asdict(batch) if is_dataclass(batch) else deepcopy(batch)
    if is_dataclass(update):
        own = statistics_record(update.rows)
        actual_update = update_record(update)
    else:
        require(type(update) is dict and set(update) == {"battle_id", "worker_id", "sequence", "rows"},
            "atomic exported update fields drift")
        own = update["rows"]
        actual_update = update
    require(type(battle_id) is str and battle_id and type(worker_id) is str and worker_id
        and type(sequence) is int and sequence >= 1 and actual_update["battle_id"] == battle_id
        and actual_update["worker_id"] == worker_id and type(actual_update["sequence"]) is int
        and actual_update["sequence"] == sequence, "atomic update identity drift")
    exported = _statistics(own)  # Empty work still cannot smuggle changed contributions.
    for name in ("world_draws", "trajectories", "transitions"):
        require(type(batch.get(name)) is int and batch[name] >= 0, "invalid atomic batch count")
    require(batch["world_draws"] == len(rows), "atomic attempted work count drift")
    expected_root = state_record(root)
    require(digest(audit["root"]) == digest(expected_root), "atomic batch root drift")
    previous = _statistics(audit["batch_before"])
    batch_after = _statistics(audit["batch_after"])
    completed, transitions, ordinals = 0, 0, []
    by_ordinal = {}
    for draw in draws:
        ordinal = draw.get("ordinal")
        require(type(ordinal) is int and ordinal >= 0 and ordinal not in by_ordinal,
            "duplicate atomic world ordinal")
        by_ordinal[ordinal] = draw
    for row in rows:
        require(type(row) is dict and set(row) == {"ordinal", "root", "before", "after", "status", "path",
            "discovered_leaf", "signed_value", "transitions", "sampled", "cleanup_completed", "input_sha256"},
            "atomic trajectory fields drift")
        ordinal = row["ordinal"]
        require(type(ordinal) is int and ordinal in by_ordinal and ordinal not in ordinals
            and digest(row["root"]) == digest(expected_root)
            and type(row["transitions"]) is int and row["transitions"] >= 0
            and type(row["sampled"]) is bool and type(row["cleanup_completed"]) is bool,
            "atomic trajectory root/ordinal/cost drift")
        ordinals.append(ordinal)
        draw = by_ordinal[ordinal]
        before, after = _statistics(row["before"]), _statistics(row["after"])
        require(digest(before) == digest(previous), "atomic batch chain drift")
        expected = deepcopy(before)
        transitions += row["transitions"]
        require(type(row["path"]) is list, "atomic path required")
        if row["status"] == "COMMITTED":
            require(row["sampled"] and row["cleanup_completed"]
                and draw.get("status") == "ROOT_VALIDATED" and draw.get("released") is True
                and row["input_sha256"] == draw.get("consumed_world_input", {}).get("input_sha256")
                and type(row["input_sha256"]) is str and len(row["input_sha256"]) == 64
                and all(c in "0123456789abcdef" for c in row["input_sha256"])
                and type(row["signed_value"]) in (float, int) and math.isfinite(row["signed_value"])
                and -1 <= row["signed_value"] <= 1 and row["path"]
                and digest(row["path"][0]["state"]) == digest(expected_root),
                "atomic commit lacks root backup, value, input or cleanup witness")
            leaf = row["discovered_leaf"]
            require(len(row["path"]) <= row["transitions"]
                and (leaf is not None or row["signed_value"] in (-1, 0, 1)),
                "atomic trajectory has invalid terminal or path cost")
            if leaf is not None:
                key = _state(leaf)
                if key not in expected:
                    expected[key] = dict(state=leaf, visits=[0]*len(leaf["actions"]),
                        totals=[0.0]*len(leaf["actions"]), count=0)
            for step in row["path"]:
                require(type(step) is dict and set(step) == {"state", "action_index"}, "atomic path fields drift")
                key = _state(step["state"])
                if key not in expected:
                    # Imported shared nodes acquire their first OWN contribution
                    # at this seam; imported visits are never new local work.
                    expected[key] = dict(state=step["state"], visits=[0]*len(step["state"]["actions"]),
                        totals=[0.0]*len(step["state"]["actions"]), count=0)
                require(digest(expected[key]["state"]) == digest(step["state"])
                    and type(step["action_index"]) is int
                    and 0 <= step["action_index"] < len(step["state"]["actions"]), "atomic backed action drift")
                index = step["action_index"]
                expected[key]["visits"][index] += 1
                expected[key]["totals"][index] += row["signed_value"]
                expected[key]["count"] += 1
            completed += 1
        else:
            require(row["status"] == "CANCELLED" and batch.get("deadline_exhausted") is True
                and not row["path"] and row["discovered_leaf"] is None and row["signed_value"] is None
                and ((row["sampled"] and row["cleanup_completed"]
                    and draw.get("status") == "ROOT_VALIDATED" and draw.get("released") is True)
                    or (not row["sampled"] and not row["cleanup_completed"]
                        and draw.get("status") == "DEADLINE_CANCELLED")),
                "refused or incomplete atomic trajectory cannot supply accepted work")
        require(digest(expected) == digest(after), "atomic OWN backup does not match actual full commit")
        previous = after
    require(ordinals == sorted(ordinals) and (not ordinals or ordinals == list(range(ordinals[0], ordinals[-1]+1)))
        and completed == batch["trajectories"] and transitions == batch["transitions"],
        "atomic trajectory/cost accounting drift")
    require(digest(previous) == digest(batch_after) == digest(exported), "atomic exported OWN update drift")
    require(digest(evidence) == digest(saved), "atomic evidence mutated during validation")
    return digest(audit)


def validate_reference_atomic_work(evidence, *, binding, battle_id):
    """Independent fresh-root reader, including all twenty workers and exchanges."""
    return _validate_reference_atomic_work(evidence, binding=binding, battle_id=battle_id,
                                         allow_zero_deadline=False)


def _validate_reference_atomic_work(evidence, *, binding, battle_id, allow_zero_deadline):
    from .paper_reference import DecisionState
    saved = deepcopy(evidence)
    receipts = saved.get("worker_receipts")
    require(type(receipts) in (tuple, list) and receipts, "actual reference atomic receipts required")
    previous, ordinals, sequences = {}, {}, {}
    completed = transitions = draws = 0
    root = None
    for receipt in receipts:
        worker, sequence = receipt.get("worker"), receipt.get("sequence")
        require(type(worker) is int and 0 <= worker < 20
            and type(sequence) is int and sequence == sequences.get(worker, 0)+1,
            "reference atomic worker/sequence drift")
        update = receipt.get("own_statistics")
        audit = receipt.get("evidence", {}).get("atomic_work", {})
        require(digest(audit.get("binding")) == digest(binding), "reference atomic consumed-input binding drift")
        state = audit.get("root")
        _state(state)
        if root is None:
            root = DecisionState(bytes.fromhex(state["key"]), tuple(state["actions"]), state["faint_count"])
        require(digest(state) == digest(state_record(root)), "reference workers disagree on atomic root")
        require(type(update) is dict, "actual outgoing reference OWN update required")
        require(type(battle_id) is str and battle_id, "reference atomic battle binding required")
        validate_reference_batch(receipt["evidence"], batch=receipt["batch"], update=update, root=root,
            battle_id=battle_id, worker_id=str(worker), sequence=sequence)
        before, after = _statistics(audit["batch_before"]), _statistics(audit["batch_after"])
        if worker in previous:
            require(digest(before) == digest(previous[worker]), "reference exchanged OWN history drift")
        else:
            require(all(row["count"] == 0 for row in before.values()), "historical OWN work is not fresh-root evidence")
        for row in audit["rows"]:
            require(row["ordinal"] == ordinals.get(worker, 0), "reference atomic world schedule drift")
            ordinals[worker] = row["ordinal"]+1
        previous[worker], sequences[worker] = after, sequence
        batch = receipt["batch"]
        batch = asdict(batch) if is_dataclass(batch) else batch
        completed += batch["trajectories"]; transitions += batch["transitions"]; draws += batch["world_draws"]
    require(set(previous) == set(range(20)), "reference atomic roster omits a worker")
    result = saved.get("result", {})
    result = asdict(result) if is_dataclass(result) else result
    require(all(type(result.get(k)) is int and result[k] == v for k,v in (
        ("trajectories",completed), ("transitions",transitions), ("world_draws",draws))),
        "reference aggregate accepted/attempted/transition count drift")
    size = len(root.actions)
    visits = [sum(rows.get(root.key.hex(), {}).get("visits", [0]*size)[i] for rows in previous.values())
        for i in range(size)]
    totals = [math.fsum(rows.get(root.key.hex(), {}).get("totals", [0.]*size)[i] for rows in previous.values())
        for i in range(size)]
    require(type(result.get("root_visits")) in (list, tuple) and len(result["root_visits"]) == size
        and all(type(v) is int and v == n for v,n in zip(result["root_visits"],visits))
        and type(result.get("root_means")) in (list, tuple) and len(result["root_means"]) == size,
        "reference root counts omit actual OWN backups")
    for q,n,mean in zip(totals,visits,result["root_means"]):
        require((n == 0 and mean is None) or (n > 0 and type(mean) in (int,float)
            and math.isfinite(mean) and math.isclose(mean,q/n,rel_tol=0,abs_tol=1e-12)),
            "reference root value differs from actual committed OWN work")
    if completed:
        require(result.get("action") == root.actions[max(range(size),key=lambda i:(visits[i],-i))],
            "reference action lacks max-visit selection")
    else:
        require(allow_zero_deadline and "action" in result and result["action"] is None
            and result.get("deadline_exhausted") is True and not any(visits)
            and all(total == 0 for total in totals)
            and all((asdict(receipt["batch"]) if is_dataclass(receipt["batch"]) else
                receipt["batch"]).get("deadline_exhausted") is True for receipt in receipts),
            "reference action lacks new atomic work or normal zero deadline")
    require(digest(evidence) == digest(saved), "reference atomic evidence mutated")
    return dict(schema=SCHEMA, binding=deepcopy(binding), battle_id=battle_id,
        accepted_trajectories=completed, attempted_draws=draws, transitions=transitions,
        witness_sha256=digest(receipts), runtime_qualified=False, independent_terminal_labels=False)
