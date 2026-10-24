"""Exact model-input identities for the bounded native engineering pilot.

These are *encoded* rows, never engine-state or history identities. Callers
must still construct the complete fold/history and encode every chance leaf.
Keys retain complete bytes; Python hash collisions cannot merge unequal rows.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct
import time
from typing import Mapping

from .shallow_exact import ExactSearchRefused


@dataclass(frozen=True)
class EncodedLeafRow:
    model_sha256: str
    # Native forward's five actual arrays, including its true history mask.
    # Each descriptor is (dtype, shape, complete little-endian byte string).
    arrays: tuple[tuple[str, tuple[int, ...], bytes], ...]
    # Legal actions do not enter this value forward. Retaining their full
    # encoded bytes is a conservative extra distinction, not a policy input.
    legal_action_mask: bytes

    def flat_inputs(self):
        return tuple([value[0] for value in struct.iter_unpack(code, data)]
                     for code, (_, _, data) in zip(("<q", "<f", "<q", "<?", "<?"), self.arrays))


def encoded_leaf_row(row: Mapping[str, bytes], *, model_sha256: str,
                     window: int, tokens: int, categorical_features: int,
                     numeric_features: int) -> EncodedLeafRow:
    """Validate canonical encoder bytes and bind the exact effective input.

    The Rust model converts i4→i8, f8→f4, i2→i8 before its forward. Numeric
    bytes here use that same finite f4 conversion, not decimal rounding or a
    lossy hash. Shapes and dtypes remain part of equality even for empty rows.
    """
    if (not isinstance(model_sha256, str) or len(model_sha256) != 64
            or any(c not in "0123456789abcdef" for c in model_sha256)
            or any(type(n) is not int or n <= 0
                   for n in (window, tokens, categorical_features, numeric_features))):
        raise ExactSearchRefused("fixed model SHA and positive observation dimensions required")
    shapes = ((window, tokens, categorical_features), (window, tokens, numeric_features),
              (window, tokens), (window, tokens), (window,))
    sources = (("categorical_ids", "<i", "<q"), ("numeric_features", "<d", "<f"),
               ("token_type_ids", "<h", "<q"), ("attention_mask", "<?", "<?"))
    arrays = []
    for (name, source_code, target_code), shape in zip(sources, shapes):
        data = row.get(name)
        if not isinstance(data, bytes) or len(data) != math.prod(shape) * struct.calcsize(source_code):
            raise ExactSearchRefused("encoded leaf array dtype/shape mismatch: " + name)
        if name == "attention_mask" and any(value not in (0, 1) for value in data):
            raise ExactSearchRefused("encoded leaf attention mask is not boolean")
        values = [value[0] for value in struct.iter_unpack(source_code, data)]
        try:
            converted = b"".join(struct.pack(target_code, value) for value in values)
        except (OverflowError, struct.error) as error:
            raise ExactSearchRefused("encoded leaf cannot enter native input dtype") from error
        if name == "numeric_features" and any(not math.isfinite(value[0])
                                               for value in struct.iter_unpack("<f", converted)):
            raise ExactSearchRefused("encoded leaf has nonfinite model input")
        arrays.append((target_code, shape, converted))
    arrays.append(("<?", shapes[4], bytes([1]) * window))
    legal = row.get("legal_action_mask")
    if not isinstance(legal, bytes) or not legal or any(value not in (0, 1) for value in legal):
        raise ExactSearchRefused("encoded leaf legal mask is not nonempty boolean bytes")
    return EncodedLeafRow(model_sha256, tuple(arrays), legal)


class DecisionLeafValues:
    """Memoize a fixed deterministic value head, reset for each decision.

    Encoding/fold construction lives upstream and is never skipped. This only
    avoids re-forwarding byte-identical effective input rows. Model binding is
    the attested artifact SHA plus the exact held model object; replacing the
    model, changing weights, or training it during a decision is unsupported.
    """

    def __init__(self, model, *, model_sha256: str):
        if (not isinstance(model_sha256, str) or len(model_sha256) != 64
                or any(c not in "0123456789abcdef" for c in model_sha256)):
            raise ExactSearchRefused("fixed model artifact SHA required for leaf memoization")
        self.model = model
        self.model_sha256 = model_sha256
        self._input_contract = None
        self.begin_decision(model)

    def begin_decision(self, model):
        if model is not self.model:
            raise ExactSearchRefused("memoized leaf model binding changed")
        self.values: dict[EncodedLeafRow, float] = {}
        self.row_requests = self.cache_hits = self.unique_evaluations = self.forward_batches = 0
        self.forward_seconds = 0.0

    def evaluate(self, rows: list[EncodedLeafRow], model) -> list[float]:
        if model is not self.model or any(row.model_sha256 != self.model_sha256 for row in rows):
            raise ExactSearchRefused("memoized leaf model binding changed")
        if rows:
            expected = self._input_contract or tuple((dtype, shape) for dtype, shape, _ in rows[0].arrays)
            if any(tuple((dtype, shape) for dtype, shape, _ in row.arrays) != expected for row in rows):
                raise ExactSearchRefused("mixed encoded leaf dtype/shape contracts")
            self._input_contract = expected
        missing = list(dict.fromkeys(row for row in rows if row not in self.values))
        self.row_requests += len(rows)
        self.cache_hits += len(rows) - len(missing)
        if missing:
            columns = [[], [], [], [], []]
            for row in missing:
                for output, data in zip(columns, row.flat_inputs()):
                    output.extend(data)
            self.forward_batches += 1
            started = time.perf_counter()
            try:
                values, _, _, _ = model.eval_obs_flat(len(missing), *columns)
            finally:
                self.forward_seconds += time.perf_counter() - started
            if len(values) != len(missing):
                raise ExactSearchRefused("memoized value head changed unique-row cardinality")
            values = [float(value) for value in values]
            if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in values):
                raise ExactSearchRefused("memoized value outside finite tanh scale")
            # Publish only after validating the entire returned batch.
            self.values.update(zip(missing, values))
            self.unique_evaluations += len(missing)
        return [self.values[row] for row in rows]
