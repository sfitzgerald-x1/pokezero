"""Fold-advance backends for the schema-v2 row-pair validation harness (track B).

Companion to ``scripts/validate_corpus_v2.py`` (mirroring how
``scripts/golden_encoder_backends.py`` serves ``validate_rust_encoder.py``).
The harness seam is ``pokezero.golden_corpus_fold.FoldBackend``: three methods
(``start`` / ``load`` / ``step``) over canonical payload dicts.

Backends:

- ``rust`` — the native crate's ``pokezero_search.FoldState`` (rust/pokezero-search
  ``src/fold.rs``), the Rust port of ``pokezero.transitions_fold.FoldState``. It
  returns NATIVE Python objects from ``to_payload()`` / ``products_payload()`` so
  the harness's own canonical ``json.dumps`` defines the compared bytes — float
  canonicalization stays entirely on the Python side.
- ``compare-backends`` — runs rust and python-reference side by side on every
  boundary and prints JSON-path locators for any divergence between the two
  (the debugging loop for the port), while returning the RUST outputs so the
  harness's corpus comparison remains the rust gate.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from pokezero.golden_corpus_fold import PythonReferenceFoldBackend


def rust_fold_available() -> bool:
    """True when the installed pokezero_search wheel carries the fold port."""

    try:
        import pokezero_search
    except (ImportError, OSError):  # pragma: no cover - environment guard
        return False
    return hasattr(pokezero_search, "FoldState")


class RustFoldBackend:
    """The native crate's ``FoldState`` behind the ``FoldBackend`` seam."""

    name = "rust"

    def __init__(self) -> None:
        import pokezero_search

        if not hasattr(pokezero_search, "FoldState"):
            raise RuntimeError(
                "the installed pokezero_search wheel has no FoldState; rebuild from "
                "rust/pokezero-search (scripts/build_search_crate_model.sh)."
            )
        self._fold_state = pokezero_search.FoldState

    def start(
        self, *, perspective_slot: str, merged_tail_limit: int, action_tail_limit: int
    ) -> Any:
        return self._fold_state.initial(
            perspective_slot, merged_tail_limit, action_tail_limit
        )

    def load(self, fold_state_payload: Mapping[str, Any]) -> Any:
        return self._fold_state.from_payload(fold_state_payload)

    def step(
        self,
        handle: Any,
        event_slice: Sequence[str],
        annotation_overlay: Mapping[str, Sequence[Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        handle.advance_in_place(list(event_slice))
        # Mirror the reference backend: an EMPTY overlay must not touch the state
        # (apply_annotations prunes; the reference only calls it when non-empty).
        if annotation_overlay:
            handle.apply_annotations_in_place(dict(annotation_overlay))
        return handle.to_payload(), handle.products_payload()


def _canonical_scalar(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def payload_diff_paths(
    got: Any, want: Any, *, prefix: str = "$", limit: int = 8
) -> list[str]:
    """JSON-path locators of the first differing payload regions (byte semantics:
    scalars compared on their canonical-JSON text, so 1 vs 1.0 vs True differ)."""

    out: list[str] = []
    _diff(got, want, prefix, out, limit)
    return out


def _diff(got: Any, want: Any, path: str, out: list[str], limit: int) -> None:
    if len(out) >= limit:
        return
    if isinstance(got, Mapping) and isinstance(want, Mapping):
        for key in sorted(set(got) | set(want)):
            if key not in got:
                out.append(f"{path}.{key}: missing in got (want {want[key]!r})")
            elif key not in want:
                out.append(f"{path}.{key}: extra in got ({got[key]!r})")
            else:
                _diff(got[key], want[key], f"{path}.{key}", out, limit)
            if len(out) >= limit:
                return
        return
    got_is_seq = isinstance(got, (list, tuple))
    want_is_seq = isinstance(want, (list, tuple))
    if got_is_seq and want_is_seq:
        if len(got) != len(want):
            out.append(f"{path}: length {len(got)} != {len(want)}")
            if len(out) >= limit:
                return
        for index, (got_item, want_item) in enumerate(zip(got, want)):
            _diff(got_item, want_item, f"{path}[{index}]", out, limit)
            if len(out) >= limit:
                return
        return
    try:
        if _canonical_scalar(got) != _canonical_scalar(want):
            out.append(f"{path}: got {got!r} want {want!r}")
    except (TypeError, ValueError):
        out.append(f"{path}: non-canonicalizable got {got!r} want {want!r}")


class CompareFoldBackend:
    """rust and python-reference side by side, divergences printed per boundary.

    Returns the RUST outputs, so the harness's comparison against the recorded
    corpus stays the rust gate; the printed paths are the per-row rust-vs-python
    diff for the debugging loop.
    """

    name = "compare-backends"

    def __init__(self) -> None:
        self._rust = RustFoldBackend()
        self._python = PythonReferenceFoldBackend()

    def start(
        self, *, perspective_slot: str, merged_tail_limit: int, action_tail_limit: int
    ) -> Any:
        kwargs = dict(
            perspective_slot=perspective_slot,
            merged_tail_limit=merged_tail_limit,
            action_tail_limit=action_tail_limit,
        )
        return (self._rust.start(**kwargs), self._python.start(**kwargs))

    def load(self, fold_state_payload: Mapping[str, Any]) -> Any:
        return (
            self._rust.load(fold_state_payload),
            self._python.load(fold_state_payload),
        )

    def step(
        self,
        handle: Any,
        event_slice: Sequence[str],
        annotation_overlay: Mapping[str, Sequence[Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        rust_handle, python_handle = handle
        rust_state, rust_products = self._rust.step(
            rust_handle, event_slice, annotation_overlay
        )
        python_state, python_products = self._python.step(
            python_handle, event_slice, annotation_overlay
        )
        for surface, got, want in (
            ("fold_state", rust_state, python_state),
            ("products", rust_products, python_products),
        ):
            paths = payload_diff_paths(got, want)
            if paths:
                print(f"[compare-backends] rust != python-reference on {surface}:")
                for path in paths:
                    print(f"  {path}")
        return rust_state, rust_products
