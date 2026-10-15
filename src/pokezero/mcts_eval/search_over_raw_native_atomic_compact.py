"""Prospective compact native backup consistency reader; not runtime admission.

One initial and final tree plus sparse collection edits replace repeated full
snapshots. Every actual commit must still supply before/after SHA256s of the
entire tree, computed by the producer, not inferred from its path. Replay keeps
private trees for the current batch only. No simulator/model/source is opened.
The v1 reader and all existing native call sites remain separately scoped.
"""
from collections import Counter
from copy import deepcopy

from .search_over_raw_native_atomic_work import (
    _backup_in_place, _collection, _deadline, _final_report, _float, _integer, _keys,
    _snapshot, _timed_prefix, digest, require,
)

SCHEMA = "native-atomic-backups-compact-v2"


def snapshot_patch(before, after):
    """Canonical sparse, append-only collection delta, not backup evidence.

    Ordered scalar edits and appended tails retain exact integer/float32 bits.
    Existing containers cannot disappear or change type. New tree rows are
    encoded once, including their full structure.
    """
    edits = []

    def walk(a, b, path):
        if a is None or b is None:
            # Nullable branch rows/children change type without replacing a
            # container. Snapshot validation constrains which fields are null.
            require(a is None or type(a) is int, "invalid nullable native field")
            require(b is None or type(b) is int, "invalid nullable native field")
            if type(a) is not type(b) or a != b:
                edits.append(dict(kind="set", path=path, before=a, after=b))
            return
        require(type(a) is type(b), "native collection container/type drift")
        if type(a) is dict:
            require(set(a) == set(b), "native collection field drift")
            for key in sorted(a):
                walk(a[key], b[key], path + [key])
        elif type(a) is list:
            require(len(b) >= len(a), "native collection deleted rows")
            for index, old in enumerate(a):
                walk(old, b[index], path + [index])
            if len(b) > len(a):
                edits.append(dict(kind="append", path=path, before_length=len(a),
                    rows=deepcopy(b[len(a):])))
        elif a != b:
            edits.append(dict(kind="set", path=path, before=a, after=b))

    walk(before, after, [])
    return edits


def _patched(before, edits):
    require(type(edits) is list, "missing native collection edits")
    state = deepcopy(before)
    for edit in edits:
        require(type(edit) is dict and edit.get("kind") in ("set", "append"),
            "unknown native collection edit")
        path = edit.get("path")
        require(type(path) is list and path and len(path) <= 8,
            "invalid native collection edit path")
        target = state
        for key in path[:-1]:
            if type(target) is dict:
                require(type(key) is str and key in target, "unknown native edit field")
            else:
                require(type(target) is list, "native edit descends into scalar")
                _integer(key, len(target)-1)
            target = target[key]
        key = path[-1]
        if type(target) is dict:
            require(type(key) is str and key in target, "unknown native edit field")
        else:
            require(type(target) is list, "native edit targets scalar")
            _integer(key, len(target)-1)
        old = target[key]
        if edit["kind"] == "set":
            _keys(edit, "kind path before after")
            require(type(old) is type(edit["before"]) and old == edit["before"]
                and type(old) not in (dict, list)
                and type(edit["after"]) not in (dict, list), "native edit baseline drift")
            target[key] = deepcopy(edit["after"])
        else:
            _keys(edit, "kind path before_length rows")
            require(type(old) is list and type(edit["before_length"]) is int
                and len(old) == edit["before_length"] and type(edit["rows"]) is list
                and edit["rows"], "native append baseline drift")
            old.extend(deepcopy(edit["rows"]))
    _snapshot(state)
    require(edits == snapshot_patch(before, state), "noncanonical native collection edits")
    return state


def _hash(actual, state):
    require(type(actual) is str and len(actual) == 64
        and all(c in "0123456789abcdef" for c in actual)
        and actual == digest(state), "native whole-tree fingerprint drift")


def validate_compact_native_report(report, *, state_sha256, requested_iterations,
                                   batch_size, chance_seed):
    """Validate v2 only. This does not admit it to any existing owning job."""
    require(type(report) is dict and type(report.get("iterations")) is int
        and 0 <= report["iterations"] <= _integer(requested_iterations), "invalid native report work")
    work = report.get("atomic_work")
    _keys(work, "schema scope independent_terminal_labels runtime_qualified input_state_utf8_sha256 requested_iterations batch_size chance_seed initial rounds final joint_action_witness")
    require(work["schema"] == SCHEMA and work["scope"] == "observed_float32_backup_consistency"
        and work["independent_terminal_labels"] is False and work["runtime_qualified"] is False,
        "compact native atomic scope drift")
    require(type(state_sha256) is str and len(state_sha256) == 64
        and all(c in "0123456789abcdef" for c in state_sha256), "invalid native state binding")
    for key, value in dict(input_state_utf8_sha256=state_sha256,
            requested_iterations=requested_iterations, batch_size=batch_size, chance_seed=chance_seed).items():
        require(type(work[key]) is type(value) and work[key] == value, "native actual-call binding drift")
    _integer(batch_size); require(batch_size > 0, "empty native batch allocation")
    _integer(chance_seed, 2**64-1)
    rounds = work["rounds"]
    require(type(rounds) is list, "missing native rounds")
    state = deepcopy(_snapshot(work["initial"]))
    require(len(state["decisions"]) == 1 and not state["chances"]
        and state["decisions"][0]["visits"] == 0
        and all(row["total_bits"] == 0 for seat in ("side_one", "side_two")
            for row in state["decisions"][0][seat]), "native root has historical work")
    completed = 0; committed_rounds = 0; selections = 0
    pairs = Counter(); node_pairs = Counter()
    for ordinal, round in enumerate(rounds):
        require(type(round) is dict and type(round.get("ordinal")) is int
            and round["ordinal"] == ordinal, "native round order drift")
        _hash(round.get("before_sha256"), state)
        collected = _patched(state, round.get("collection_edits"))
        _hash(round.get("collection_sha256"), collected)
        if round.get("status") == "CANCELLED":
            _keys(round, "ordinal status before_sha256 collection_edits collection_sha256 after_sha256 accepted attempted_paths")
            require(type(round["accepted"]) is int and round["accepted"] == 0
                and type(round["attempted_paths"]) is list, "cancelled native work was accepted")
            require(len(round["attempted_paths"]) <= min(batch_size, requested_iterations-completed),
                "cancelled native reservation exceeds bound allocation")
            require(ordinal == len(rounds)-1 and report.get("time_budget_exhausted") is True,
                "cancelled native round lacks terminal deadline evidence")
            _collection(state, collected, round["attempted_paths"])
            # The producer hashes its ACTUAL restored tree; no reverse edits can
            # conceal leftover reservations, topology or other rollback drift.
            _hash(round["after_sha256"], state)
            continue
        _keys(round, "ordinal status before_sha256 collection_edits collection_sha256 row_value_bits commits after_sha256 accepted")
        require(round["status"] == "COMMITTED" and type(round["commits"]) is list
            and type(round["accepted"]) is int and round["accepted"] == len(round["commits"])
            and 0 < round["accepted"] <= min(batch_size, requested_iterations-completed), "partial native batch")
        require(type(round["row_value_bits"]) is list, "missing native prices")
        values = [_float(bits) for bits in round["row_value_bits"]]
        require(all(0 <= value <= 1 for value in values), "invalid native row price")
        _collection(state, collected, [commit["path"] for commit in round["commits"]])
        pending = [branch["pending_row"] for chance in collected["chances"]
            for branch in chance if branch["pending_row"] is not None]
        require(sorted(pending) == list(range(len(values))), "native evaluated rows lack exact branch coverage")
        state = deepcopy(collected)
        for commit in round["commits"]:
            _keys(commit, "path end before_sha256 after_sha256 root_value_bits")
            _hash(commit["before_sha256"], state)
            _backup_in_place(commit, state, values, collected)
            # A matching hash does not make an observed intermediate tree
            # valid. Retain v1's range/topology checks at EVERY backup boundary,
            # even if a later backup would restore a valid final tree.
            _snapshot(state)
            _hash(commit["after_sha256"], state)
            first = commit["path"][0]
            pairs[(first["side_one"], first["side_two"])] += 1
            selections += len(commit["path"])
            node_pairs.update((step["decision"], step["side_one"], step["side_two"])
                for step in commit["path"])
        _hash(round["after_sha256"], state)
        require(all(branch["pending_row"] is None for chance in state["chances"] for branch in chance),
            "native committed boundary retains unresolved leaf work")
        completed += round["accepted"]; committed_rounds += 1
    final = _snapshot(work["final"])
    require(final == state and all(branch["pending_row"] is None
        for chance in final["chances"] for branch in chance), "unreported native tail")
    require(completed == report["iterations"] == final["decisions"][0]["visits"], "native finalized work drift")
    return _final_report(report, work, final, completed, pairs, node_pairs, selections,
        committed_rounds=committed_rounds, witness_sha256=digest(work))


def validate_timed_compact_native_report(report, *, state_sha256, requested_iterations,
                                        batch_size, chance_seed, time_budget_ms):
    """Prospective v2 deadline consistency, not owner or deployment admission."""
    elapsed, overshoot, exhausted = _deadline(report, requested_iterations, time_budget_ms)
    validated = validate_compact_native_report(report, state_sha256=state_sha256,
        requested_iterations=requested_iterations, batch_size=batch_size, chance_seed=chance_seed)
    result = _timed_prefix(report, validated, requested_iterations, batch_size,
        time_budget_ms, elapsed, overshoot, exhausted)
    result["schema"] = "pokezero.search-over-raw.timed-native-prefix.compact-v2"
    return result
