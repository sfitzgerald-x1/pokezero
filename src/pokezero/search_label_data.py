"""Optional search target columns and sparse collection provenance.

No arrays or metadata keys are emitted for legacy/unlabeled-off datasets.
An attempted but refused label retains its teacher identity and reason, with
both target masks false. A masked-out row is never a uniform search target.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from .actions import ACTION_COUNT
from .search_labels import validate_search_targets


SEARCH_COLUMNS = ("search_policy_targets", "search_policy_target_mask",
                  "search_value_targets", "search_value_target_mask")
SEARCH_PROVENANCE = "search_label_rows"


def validate_optional_targets(policy, value, legal_mask):
    if policy is not None:
        validate_search_targets(tuple(policy), 0.0, legal_mask)
    if value is not None and (not math.isfinite(float(value)) or not -1 <= float(value) <= 1):
        raise ValueError("search value target must be finite in acting-seat [-1, 1]")


def targets_from_metadata(metadata: Mapping[str, Any], legal_mask):
    label = metadata.get("search_label")
    if label is None:
        return {}
    if not isinstance(label, Mapping) or label.get("status") not in ("LABELED", "UNLABELED"):
        raise ValueError("invalid search label status")
    if label["status"] == "UNLABELED":
        if "policy_target" in label or "value_target" in label:
            raise ValueError("unlabeled search attempt cannot carry targets")
        return {}
    policy, value = validate_search_targets(label["policy_target"], label["value_target"], legal_mask)
    return {"search_policy_target": policy, "search_value_target": value}


def columns_from_examples(examples):
    if not any(e.search_policy_target is not None or e.search_value_target is not None
               or "search_label" in (e.step_metadata or {}) for e in examples):
        return {}
    return dict(zip(SEARCH_COLUMNS, (
        tuple(e.search_policy_target if e.search_policy_target is not None else (0.0,) * ACTION_COUNT for e in examples),
        tuple(e.search_policy_target is not None for e in examples),
        tuple(float(e.search_value_target) if e.search_value_target is not None else 0.0 for e in examples),
        tuple(e.search_value_target is not None for e in examples),
    )))


def validate_batch_columns(batch):
    columns = [getattr(batch, n) for n in SEARCH_COLUMNS]
    if all(c is None for c in columns):
        return
    if any(c is None for c in columns) or any(len(c) != batch.batch_size for c in columns):
        raise ValueError("search target columns must be present together and match batch size")
    _validate_column_shapes(dict(zip(SEARCH_COLUMNS, columns)), batch.batch_size)
    policy, policy_mask, value, value_mask = columns
    for i in range(batch.batch_size):
        if len(policy[i]) != ACTION_COUNT:
            raise ValueError("search policy target must contain nine probabilities")
        validate_optional_targets(policy[i] if policy_mask[i] else None,
                                  value[i] if value_mask[i] else None, batch.legal_action_mask[i])


def _validate_column_shapes(columns, count):
    for name in SEARCH_COLUMNS:
        values = columns[name]
        expected = (count, ACTION_COUNT) if name == "search_policy_targets" else (count,)
        if hasattr(values, "shape"):
            if tuple(values.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")
            if name.endswith("mask") and values.dtype.kind != "b":
                raise ValueError(f"{name} must be boolean")
        else:
            if len(values) != count:
                raise ValueError(f"{name} must have shape {expected}")
            if name == "search_policy_targets":
                if any(not isinstance(row, (list, tuple)) or len(row) != ACTION_COUNT for row in values):
                    raise ValueError(f"{name} must have shape {expected}")
            elif name.endswith("mask"):
                if any(type(value) is not bool for value in values):
                    raise ValueError(f"{name} must be boolean")
            elif any(not isinstance(value, (int, float)) for value in values):
                raise ValueError(f"{name} must contain scalars")


def validate_cache_columns(arrays, metadata):
    """A damaged declared treatment cache must never become a legacy cache."""
    present = set(arrays) & set(SEARCH_COLUMNS)
    validate_array_set(present)
    declared = set(metadata.get("array_dtypes", {})) & set(SEARCH_COLUMNS)
    provenance = metadata.get(SEARCH_PROVENANCE, {})
    if not isinstance(provenance, Mapping):
        raise ValueError("search label provenance must be a mapping")
    if (declared or provenance) and present != set(SEARCH_COLUMNS):
        raise ValueError("declared search label cache is missing target columns")
    if not present:
        return
    count = int(metadata["example_count"])
    _validate_column_shapes(arrays, count)
    for index, row in provenance.items():
        if (not isinstance(index, str) or not index.isdecimal() or str(int(index)) != index
                or not 0 <= int(index) < count or not isinstance(row, Mapping)):
            raise ValueError("invalid search label provenance row index")
        i = int(index)
        label = row.get("search_label")
        if not isinstance(label, Mapping) or not isinstance(label.get("teacher"), Mapping) or not label["teacher"]:
            raise ValueError("collected search label must retain its teacher identity")
        expected = targets_from_metadata(row, arrays["legal_action_mask"][i])
        masks = (arrays["search_policy_target_mask"][i], arrays["search_value_target_mask"][i])
        if label["status"] == "UNLABELED":
            if any(masks):
                raise ValueError("unlabeled provenance cannot have active target masks")
        elif not all(masks) or any(
            not math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-7)
            for a, b in zip(arrays["search_policy_targets"][i], expected["search_policy_target"], strict=True)
        ) or not math.isclose(float(arrays["search_value_targets"][i]), expected["search_value_target"], rel_tol=0, abs_tol=1e-7):
            raise ValueError("labeled provenance disagrees with stored search targets")


class SearchLabelRows:
    """Accumulate only labeled/attempted rows, with no dense off-path storage."""
    def __init__(self):
        self.count = 0
        self.rows = {}
        self.provenance = {}

    def append(self, example):
        label = (example.step_metadata or {}).get("search_label")
        if example.search_policy_target is not None or example.search_value_target is not None or label is not None:
            self.rows[self.count] = (example.search_policy_target, example.search_value_target)
        if label is not None:
            self.provenance[str(self.count)] = {"search_label": dict(label)}
        self.count += 1

    def arrays(self, numpy):
        if not self.rows:
            return {}
        arrays = empty_columns(numpy, self.count)
        for i, (policy, value) in self.rows.items():
            if policy is not None:
                arrays["search_policy_targets"][i] = policy
                arrays["search_policy_target_mask"][i] = True
            if value is not None:
                arrays["search_value_targets"][i] = value
                arrays["search_value_target_mask"][i] = True
        return arrays

    def metadata(self):
        return {SEARCH_PROVENANCE: self.provenance} if self.provenance else {}


def empty_columns(numpy, count):
    return {name: numpy.zeros((count, ACTION_COUNT) if name == "search_policy_targets" else count,
                             dtype=numpy.bool_ if name.endswith("mask") else numpy.float32)
            for name in SEARCH_COLUMNS}


def validate_array_set(names):
    present = set(names) & set(SEARCH_COLUMNS)
    if present and present != set(SEARCH_COLUMNS):
        raise ValueError("search target cache columns must be present together")
