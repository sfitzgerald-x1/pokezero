"""Independent reader of actual native float32 commits, not label truth.

Snapshots include collection reservations, chance pricing pseudocounts and
ordered real backups. Replay uses IEEE float32 operations, not a fitted tolerance.
The reader never opens a simulator, model, source or historical result.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import struct


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


SCHEMA = "native-atomic-backups-v1"


def _integer(value, maximum=2**32-1):
    require(type(value) is int and 0 <= value <= maximum, "invalid native integer")
    return value


def _float(bits):
    value = struct.unpack("!f", struct.pack("!I", _integer(bits)))[0]
    require(math.isfinite(value), "nonfinite native backup")
    return value


def _bits(value):
    return struct.unpack("!I", struct.pack("!f", value))[0]


def _f32(value):
    return _float(_bits(value))


def _add(bits, value):
    return _bits(_float(bits) + value)


def _keys(value, names):
    require(type(value) is dict and set(value) == set(names.split()), "native receipt schema drift")


def _snapshot(snapshot):
    _keys(snapshot, "decisions chances")
    nodes, chances = snapshot["decisions"], snapshot["chances"]
    require(type(nodes) is list and bool(nodes) and type(chances) is list, "empty native tree")
    owners = []
    for index, node in enumerate(nodes):
        _keys(node, "visits depth side_one side_two children")
        _integer(node["visits"]); _integer(node["depth"], 32)
        require((node["depth"] == 0) == (index == 0), "native root/depth drift")
        for seat in ("side_one", "side_two"):
            rows = node[seat]
            require(type(rows) is list and rows, "missing native options")
            for row in rows:
                _keys(row, "visits total_bits")
                _integer(row["visits"])
                value = _float(row["total_bits"])
                require(-1e-5 <= value <= row["visits"] + 1e-4, "invalid native total")
            require(sum(r["visits"] for r in rows) == node["visits"], "native side visit drift")
        children = node["children"]
        require(type(children) is list and children == sorted(children)
            and all(type(c) is list and len(c) == 3 for c in children), "native edge order drift")
        pairs = []
        for i, j, chance in children:
            _integer(i, len(node["side_one"])-1); _integer(j, len(node["side_two"])-1)
            _integer(chance, len(chances)-1)
            pairs.append((i, j)); owners.append(chance)
            for branch in chances[chance]:
                if branch.get("child") is not None:
                    _integer(branch["child"], len(nodes)-1)
                    require(nodes[branch["child"]]["depth"] == node["depth"] + 1,
                        "native child depth drift")
        require(len(set(pairs)) == len(pairs), "duplicate native edge")
    require(sorted(owners) == list(range(len(chances))), "native chance ownership drift")
    for chance in chances:
        require(type(chance) is list and bool(chance), "empty native chance node")
        probabilities = []
        for branch in chance:
            _keys(branch, "probability_bits value_sum_bits visits pending_row child terminal_bits no_expand")
            _integer(branch["visits"]); require(branch["visits"] > 0, "uninitialized native branch")
            probability, total = _float(branch["probability_bits"]), _float(branch["value_sum_bits"])
            require(0 <= probability <= 1 and -1e-5 <= total <= branch["visits"] + 1e-4,
                "invalid native branch value")
            probabilities.append(probability)
            if branch["pending_row"] is not None: _integer(branch["pending_row"])
            if branch["child"] is not None: _integer(branch["child"], len(nodes)-1)
            require(type(branch["no_expand"]) is bool, "invalid native pseudo-branch marker")
            if branch["terminal_bits"] is not None:
                require(_float(branch["terminal_bits"]) in (0.,1.) and branch["pending_row"] is None,
                    "invalid native terminal branch marker")
        require(abs(math.fsum(probabilities)-1) < 1e-4, "native probability mass drift")
    return snapshot


def _path(path, state):
    require(type(path) is list and path and len(path) <= 33, "missing native backup path")
    previous = None
    for step in path:
        _keys(step, "decision side_one side_two chance branch")
        node = state["decisions"][_integer(step["decision"], len(state["decisions"])-1)]
        i = _integer(step["side_one"], len(node["side_one"])-1)
        j = _integer(step["side_two"], len(node["side_two"])-1)
        chance = _integer(step["chance"], len(state["chances"])-1)
        require([i,j,chance] in node["children"], "native path is not an actual joint edge")
        if previous is None: require(step["decision"] == 0, "backup does not start at native root")
        else:
            require(previous["branch"] is not None and state["chances"][previous["chance"]]
                [previous["branch"]]["child"] == step["decision"], "native ancestry drift")
        if step["branch"] is not None:
            _integer(step["branch"], len(state["chances"][chance])-1)
        previous = step
    return path


def _collection(before, collected, paths):
    """Only visits, virtual loss, new topology and child links may change."""
    require(len(before["decisions"]) <= len(collected["decisions"])
        and len(before["chances"]) <= len(collected["chances"]), "collection deleted native topology")
    for path in paths: _path(path,collected)
    expansions = [p[-1]["chance"] for p in paths if p[-1]["branch"] is None]
    require(sorted(expansions) == list(range(len(before["chances"]),len(collected["chances"]))),
        "new native edges lack exactly one actual expansion")
    new_nodes = {s["decision"] for p in paths for s in p if s["decision"] >= len(before["decisions"])}
    require(new_nodes == set(range(len(before["decisions"]),len(collected["decisions"]))),
        "new native children lack actual descended paths")
    expected = deepcopy(collected)
    for index, node in enumerate(expected["decisions"]):
        old = before["decisions"][index] if index < len(before["decisions"]) else None
        if old:
            require(node["depth"] == old["depth"]
                and all(c in node["children"] for c in old["children"]), "collection rewrote native node")
        node["visits"] = old["visits"] if old else 0
        for seat in ("side_one", "side_two"):
            require(not old or len(node[seat]) == len(old[seat]), "collection rewrote native options")
            node[seat] = deepcopy(old[seat]) if old else [dict(visits=0,total_bits=0) for _ in node[seat]]
    for path in paths:
        for step in _path(path, collected):
            node = expected["decisions"][step["decision"]]
            node["visits"] += 1
            node["side_one"][step["side_one"]]["visits"] += 1
            row = node["side_two"][step["side_two"]]
            row["visits"] += 1; row["total_bits"] = _add(row["total_bits"], 1.0)
    require(expected["decisions"] == collected["decisions"], "reservations differ from actual paths")
    for index, old in enumerate(before["chances"]):
        new = collected["chances"][index]
        require(len(old) == len(new), "collection rewrote chance branches")
        for a, b in zip(old, new):
            require(all(a[k] == b[k] for k in a if k != "child")
                and (a["child"] is None or a["child"] == b["child"]), "collection changed completed samples")
    for chance in collected["chances"][len(before["chances"]):]:
        require(all(b["visits"] == 1 and (b["pending_row"] is None or b["value_sum_bits"] == 0)
            for b in chance), "new native chance pricing is not a single pseudocount")
        require(all(b["terminal_bits"] is None or b["value_sum_bits"] == b["terminal_bits"]
            for b in chance), "new native terminal price differs from declared terminal value")


def _commit(commit, state, values, collected):
    _keys(commit, "path end before after root_value_bits")
    require(commit["before"] == state, "native commit baseline drift")
    path = _path(commit["path"], state)
    end = commit["end"]
    require(type(end) is dict and end.get("kind") in {"ready", "row", "expanded"}, "unknown native leaf end")
    expected = deepcopy(state)
    if end["kind"] == "expanded":
        _keys(end, "kind"); require(path[-1]["branch"] is None, "expanded end lacks expansion")
        for branch in expected["chances"][path[-1]["chance"]]:
            row = branch["pending_row"]
            if row is not None:
                branch["value_sum_bits"] = _add(branch["value_sum_bits"], values[_integer(row,len(values)-1)])
                branch["pending_row"] = None
        value = None
    elif end["kind"] == "row":
        _keys(end, "kind row"); value = values[_integer(end["row"],len(values)-1)]
        require(path[-1]["branch"] is not None, "row end lacks sampled branch")
        origin = collected["chances"][path[-1]["chance"]][path[-1]["branch"]]
        require(origin["pending_row"] == end["row"] and origin["terminal_bits"] is None,
            "native leaf row is not the selected branch's deferred price")
    else:
        _keys(end, "kind value_bits"); value = _float(end["value_bits"])
        require(0 <= value <= 1 and path[-1]["branch"] is not None, "invalid ready leaf")
        origin = collected["chances"][path[-1]["chance"]][path[-1]["branch"]]
        expected_value = (_float(origin["terminal_bits"]) if origin["terminal_bits"] is not None
            else _f32(_float(origin["value_sum_bits"]) / _f32(origin["visits"])))
        require(origin["pending_row"] is None and end["value_bits"] == _bits(expected_value),
            "native ready leaf differs from its pre-backup estimate")
    for step in reversed(path):
        branches = expected["chances"][step["chance"]]
        if step["branch"] is not None:
            require(value is not None, "unresolved native backup")
            branch = branches[step["branch"]]
            branch["visits"] += 1
            branch["value_sum_bits"] = _add(branch["value_sum_bits"], value)
        else: require(step is path[-1], "expansion before final native step")
        value = 0.0
        for branch in branches:
            mean = _f32(_float(branch["value_sum_bits"]) / _f32(branch["visits"]))
            value = _f32(value + _f32(_float(branch["probability_bits"]) * mean))
        node = expected["decisions"][step["decision"]]
        one, two = node["side_one"][step["side_one"]], node["side_two"][step["side_two"]]
        one["total_bits"] = _add(one["total_bits"], value)
        two["total_bits"] = _add(two["total_bits"], _f32(value - 1.0))
    require(expected == _snapshot(commit["after"])
        and commit["root_value_bits"] == _bits(value), "native backup arithmetic drift")
    return expected


def validate_native_report(report, *, state_sha256, requested_iterations, batch_size, chance_seed):
    """Bind actual call parameters; validate every reserved/committed batch."""
    require(type(report) is dict and type(report.get("iterations")) is int
        and 0 <= report["iterations"] <= _integer(requested_iterations), "invalid native report work")
    work = report.get("atomic_work")
    _keys(work, "schema scope independent_terminal_labels runtime_qualified input_state_utf8_sha256 requested_iterations batch_size chance_seed rounds final joint_action_witness")
    require(work["schema"] == SCHEMA and work["scope"] == "observed_float32_backup_consistency"
        and work["independent_terminal_labels"] is False and work["runtime_qualified"] is False,
        "native atomic scope drift")
    require(type(state_sha256) is str and len(state_sha256) == 64
        and all(c in "0123456789abcdef" for c in state_sha256), "invalid native state binding")
    for key, value in dict(input_state_utf8_sha256=state_sha256, requested_iterations=requested_iterations,
            batch_size=batch_size, chance_seed=chance_seed).items():
        require(type(work[key]) is type(value) and work[key] == value, "native actual-call binding drift")
    _integer(batch_size); require(batch_size > 0, "empty native batch allocation")
    _integer(chance_seed,2**64-1)
    require(type(work["rounds"]) is list, "missing native rounds")
    previous = None; completed = 0; pairs = Counter(); node_pairs = Counter(); selections = 0
    for ordinal, round in enumerate(work["rounds"]):
        require(type(round) is dict and type(round.get("ordinal")) is int
            and round["ordinal"] == ordinal, "native round order drift")
        before = _snapshot(round["before_collection"])
        if previous is None:
            require(len(before["decisions"]) == 1 and not before["chances"]
                and before["decisions"][0]["visits"] == 0
                and all(r["total_bits"] == 0 for seat in ("side_one","side_two")
                    for r in before["decisions"][0][seat]), "native root has historical work")
        else: require(before == previous, "native batch chain drift")
        if round.get("status") == "CANCELLED":
            _keys(round, "ordinal status before_collection after_collection after accepted attempted_paths")
            require(type(round["accepted"]) is int and round["accepted"] == 0
                and round["after"] == before and type(round["attempted_paths"]) is list,
                "cancelled native work was accepted or not restored")
            require(len(round["attempted_paths"]) <= min(batch_size, requested_iterations-completed),
                "cancelled native reservation exceeds bound allocation")
            require(ordinal == len(work["rounds"])-1 and report.get("time_budget_exhausted") is True,
                "cancelled native round lacks terminal deadline evidence")
            _collection(before, _snapshot(round["after_collection"]), round["attempted_paths"])
            previous = before; continue
        _keys(round, "ordinal status before_collection after_collection row_value_bits commits after accepted")
        require(round["status"] == "COMMITTED" and type(round["commits"]) is list
            and type(round["accepted"]) is int and round["accepted"] == len(round["commits"])
            and 0 < round["accepted"] <= min(batch_size, requested_iterations-completed), "partial native batch")
        state = _snapshot(round["after_collection"])
        require(type(round["row_value_bits"]) is list, "missing native prices")
        values = [_float(bits) for bits in round["row_value_bits"]]
        require(all(0 <= value <= 1 for value in values), "invalid native row price")
        _collection(before, state, [c["path"] for c in round["commits"]])
        pending = [b["pending_row"] for chance in state["chances"] for b in chance if b["pending_row"] is not None]
        require(sorted(pending) == list(range(len(values))), "native evaluated rows lack exact branch coverage")
        for commit in round["commits"]:
            state = _commit(commit, state, values, round["after_collection"])
            step = commit["path"][0]
            pairs[(step["side_one"],step["side_two"])] += 1
            selections += len(commit["path"])
            node_pairs.update((s["decision"],s["side_one"],s["side_two"]) for s in commit["path"])
        require(state == round["after"], "native round lacks full commit boundary")
        require(all(b["pending_row"] is None for chance in state["chances"] for b in chance),
            "native committed boundary retains unresolved leaf work")
        completed += round["accepted"]; previous = state
    final = _snapshot(work["final"])
    require(all(b["pending_row"] is None for chance in final["chances"] for b in chance),
        "native final tree retains unresolved leaf work")
    if previous is None:
        require(len(final["decisions"]) == 1 and not final["chances"]
            and final["decisions"][0]["visits"] == 0
            and all(r["total_bits"] == 0 for seat in ("side_one","side_two")
                for r in final["decisions"][0][seat]), "unreported native work")
    else: require(final == previous, "unreported native tail")
    require(completed == report["iterations"] == final["decisions"][0]["visits"], "native finalized work drift")
    joint = work["joint_action_witness"]
    require(type(joint) is dict and joint == report.get("joint_action_witness")
        and joint.get("schema") == "completed-joint-actions-v1"
        and joint.get("basis") == "finalized_backups_not_reservations", "missing native finalized-path witness")
    require(type(joint.get("root_completed_traversals")) is int
        and joint["root_completed_traversals"] == completed
        and type(joint.get("tree_completed_selections")) is int
        and joint["tree_completed_selections"] == selections, "native finalized-path totals drift")
    for name,value in dict(root_distinct_pairs=len(pairs),tree_distinct_node_pairs=len(node_pairs)).items():
        require(type(joint.get(name)) is int and joint[name] == value, "native finalized pair cardinality drift")
    root = final["decisions"][0]
    for seat in ("one", "two"):
        visits = [r["visits"] for r in root["side_"+seat]]
        require(joint.get("root_side_"+seat+"_visits") == visits
            and all(type(v) is int for v in joint["root_side_"+seat+"_visits"]), "native root marginal drift")
        names = joint.get("root_side_"+seat+"_moves")
        order = sorted(range(len(visits)), key=lambda index:(-visits[index],index))
        require(type(names) is list and len(names) == len(visits) and all(type(n) is str and n for n in names)
            and joint.get("root_side_"+seat+"_report_order") == order
            and all(type(i) is int for i in joint["root_side_"+seat+"_report_order"]), "native report option identity drift")
        rows = report.get("side_"+seat)
        require(type(rows) is list and len(rows) == len(order), "missing native selected-action statistics")
        for row,index in zip(rows,order):
            stats = root["side_"+seat][index]
            mean = _f32(_float(stats["total_bits"]) / _f32(stats["visits"])) if stats["visits"] else .5
            require(type(row) is dict and row.get("move") == names[index]
                and type(row.get("visits")) is int and row["visits"] == stats["visits"]
                and type(row.get("q")) in (int,float) and row["q"] == float(f"{mean:.6f}"),
                "native selected-action statistics differ from actual backups")
    actual = joint.get("root_pairs")
    require(type(actual) is list and actual == [dict(side_one_option=i,side_two_option=j,completed_visits=n)
        for (i,j),n in sorted(pairs.items())], "native root pair coverage drift")
    require(all(type(row) is dict and all(type(v) is int for v in row.values()) for row in actual),
        "native joint-pair integer alias")
    total = 0.0
    for stats in root["side_one"]: total = _f32(total + _float(stats["total_bits"]))
    mean = _f32(total / _f32(completed)) if completed else .5
    require(type(report.get("root_value")) in (int,float)
        and report["root_value"] == float(f"{mean:.6f}"), "native reported root value drift")
    return dict(schema=SCHEMA, completed_iterations=completed, committed_rounds=sum(
        r["status"] == "COMMITTED" for r in work["rounds"]), witness_sha256=digest(work),
        independent_terminal_labels=False, runtime_qualified=False)


def state_hash(text):
    require(type(text) is str and text, "missing actual native state")
    return hashlib.sha256(text.encode()).hexdigest()


def validate_timed_native_report(report, *, state_sha256, requested_iterations,
                                batch_size, chance_seed, time_budget_ms):
    """Read a normal native deadline return, including a zero-commit prefix.

    This binds the actual ABI budget, not Python's nominal ceiling. It proves
    observed backup consistency only; neither clock provenance nor on-time
    deployability follows from an authored or unowned report.
    """
    _integer(time_budget_ms, 2**64-1)
    require(time_budget_ms > 0 and type(report) is dict
        and report.get("time_budget_enabled") is True
        and type(report.get("time_budget_ms")) is int
        and report["time_budget_ms"] == time_budget_ms, "native deadline actual-call drift")
    for key in ("requested_iterations", "remaining_iterations"):
        _integer(report.get(key))
    require(report["requested_iterations"] == requested_iterations
        and type(report.get("iterations")) is int
        and report["iterations"] + report["remaining_iterations"] == requested_iterations,
        "native deadline work partition drift")
    elapsed, overshoot = report.get("time_budget_elapsed_ms"), report.get("time_budget_batch_overshoot_ms")
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0
        for v in (elapsed, overshoot)) and type(report.get("time_budget_exhausted")) is bool,
        "invalid native deadline durations")
    exhausted = report["time_budget_exhausted"]
    require((not exhausted or elapsed + .001 >= time_budget_ms)
        and abs(overshoot - max(0., elapsed-time_budget_ms)) <= .001
        and (report["remaining_iterations"] == 0 or exhausted),
        "native deadline prefix/overshoot drift")
    validated = validate_native_report(report, state_sha256=state_sha256,
        requested_iterations=requested_iterations, batch_size=batch_size, chance_seed=chance_seed)
    completed = 0
    for round in report["atomic_work"]["rounds"]:
        if round["status"] == "COMMITTED":
            # Native collection ends when either traversal capacity OR deferred
            # model-row capacity is reached. One expansion can price several
            # chance branches, so a complete round need not have batch_size
            # traversals. validate_native_report already reconstructs every
            # pending row and backup; an invented row count cannot pass it.
            require(round["accepted"] == min(batch_size, requested_iterations-completed)
                or len(round["row_value_bits"]) >= batch_size,
                "timed native commit is not a whole batch")
            completed += round["accepted"]
    return dict(validated, schema="pokezero.search-over-raw.timed-native-prefix.v1",
        native_budget_ms=time_budget_ms, native_elapsed_ms=elapsed,
        native_overshoot_ms=overshoot, exhausted=exhausted,
        remaining_iterations=report["remaining_iterations"],
        cancelled_rounds=sum(r["status"] == "CANCELLED" for r in report["atomic_work"]["rounds"]),
        status="ZERO_COMMIT_DEADLINE_RETURN" if completed == 0 else "COMMITTED_PREFIX_RETURN",
        on_time_deployability_established=False, owning_process_validated=False)
