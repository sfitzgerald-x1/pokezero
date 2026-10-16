"""Exact collector-to-live correspondence, distinct from raw/native parity.

This reader checks declared inputs, boundary identity, complete action mapping
and production aggregates before Python display rounding. Upstream native
serializer quantization is retained; original model float32s are not recovered.
It is not a source-process or outcome validator.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
from importlib.machinery import EXTENSION_SUFFIXES
import json
import math
from pathlib import Path
import shutil
import sys

SCHEMA = "pokezero.collector-live-root-priors.v1"


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        if any(type(k) is not str for k in value):
            raise ValueError("string keys required")
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("finite JSON-compatible input required")


def fingerprint(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def boundary_fingerprint(context):
    """All observation arrays/metadata and the public materialization payload.

    Seed and original decision index are joined separately. A matching numeric
    row alone, or the public transport's reduced metadata, is insufficient.
    """
    from .local_showdown import _public_materialization_payload
    obs = context.observation
    return fingerprint(dict(format_id=context.format_id, player_id=context.player_id,
        observation={name: getattr(obs, name) for name in (
            "schema_version", "categorical_ids", "numeric_features", "token_type_ids",
            "attention_mask", "legal_action_mask", "metadata", "perspective")},
        public_materialization=_public_materialization_payload(context.public_materialization_state)))


def input_binding(config, *, checkpoint, model_ts, tables, selectors, world_rng):
    """Pin actual source/config/engine/model bytes; do not launch anything."""
    import poke_engine
    import pokezero_search
    package = Path(__file__).resolve().parent
    paths = [Path(checkpoint), Path(model_ts), Path(tables),
        native_extension_path(poke_engine), native_extension_path(pokezero_search)]
    # Bind bytes rather than a possibly dirty branch name. Relative source keys
    # let independently checked-out identical code compare without path aliases.
    sources = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob("*.py"))}
    scripts = package.parents[1] / "scripts"
    sources.update({"tools/"+p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(scripts.glob("*.py"))})
    bridge = config.resolved_bridge_path().resolve()
    dist = config.resolved_showdown_root().resolve() / "dist"
    node = shutil.which(config.node_binary)
    if node is None or not dist.is_dir():
        raise ValueError("available Node executable and built Showdown required")
    sources.update({"bridge/"+p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(bridge.parent.glob("*.mjs"))})
    sources.update({"showdown-dist/"+str(p.relative_to(dist)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(dist.rglob("*")) if p.is_file()})
    sources["node-executable"] = hashlib.sha256(Path(node).resolve().read_bytes()).hexdigest()
    for label, module in (("engine", poke_engine), ("search", pokezero_search)):
        sources["native-wrapper/"+label] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    return dict(source_configuration=_plain(config), selector_configurations=_plain(selectors),
        world_rng=world_rng,
        checkpoint_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(paths[1].read_bytes()).hexdigest(),
        tables_sha256=hashlib.sha256(paths[2].read_bytes()).hexdigest(),
        engine_sha256=hashlib.sha256(paths[3].read_bytes()).hexdigest(),
        search_sha256=hashlib.sha256(paths[4].read_bytes()).hexdigest(),
        source_files=sources)


def native_extension_path(module):
    """Resolve the loaded binary, not a package's Python re-export wrapper."""
    paths = {Path(loaded.__file__).resolve() for name, loaded in tuple(sys.modules.items())
        if (name == module.__name__ or name.startswith(module.__name__ + "."))
        and getattr(loaded, "__file__", None)
        and any(str(loaded.__file__).endswith(suffix) for suffix in EXTENSION_SUFFIXES)}
    if len(paths) != 1:
        raise ValueError("exactly one loaded native extension required for " + module.__name__)
    return next(iter(paths))


def _key(row, variant):
    seed, index = row.get("seed"), row.get("round")
    if type(seed) is not int or not 0 <= seed < 2**32 or type(index) is not int or index < 0 \
            or type(variant) is not str or not variant:
        raise ValueError("original seed, decision index and variant required")
    return seed, index, variant


def _vector(row, variant):
    legal = row.get("legal")
    if type(legal) is not list or not legal or any(type(a) is not int or a < 0 for a in legal) \
            or len(set(legal)) != len(legal):
        raise ValueError("unique legal action mapping required")
    witness = row.get("boundary_sha256")
    if type(witness) is not str or len(witness) != 64 or any(c not in "0123456789abcdef" for c in witness):
        raise ValueError("full original boundary identity required")
    item = row["variants"][variant]
    if "error" in item or item.get("unmapped_arms", 0) != 0:
        raise ValueError("unavailable or partially mapped native prior")
    prior = item.get("native_prior_pre_display")
    if type(prior) is not dict or set(prior) != {str(a) for a in legal}:
        raise ValueError("pre-display production aggregate over every legal action required")
    values = [prior[str(a)] for a in sorted(legal)]
    if any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in values) \
            or not math.isclose(math.fsum(values), 1., rel_tol=0., abs_tol=2e-5):
        raise ValueError("finite normalized probabilities required")
    # Do not renormalize here: even a tiny changed value must fail exactness.
    return witness, sorted(legal), values


def exact_collector_gate(collector, live):
    """Require complete, uniquely joined, unrounded zero-TV correspondence.

    Additional live boundaries may exist. Every declared collector root/variant
    must exist exactly once in both reports; invalid/fallback rows never pass.
    The declared roster is not independent proof of scientific panel coverage.
    """
    errors = []
    if collector.get("schema") != SCHEMA or live.get("schema") != SCHEMA:
        errors.append("pre_display_aggregate_report_schema_required")
    if collector.get("complete") is not True or live.get("complete") is not True:
        errors.append("both_reports_must_be_closed")
    bindings = collector.get("bindings")
    required = {"source_configuration", "selector_configurations", "world_rng", "checkpoint_sha256",
        "model_sha256", "tables_sha256", "engine_sha256", "search_sha256", "source_files"}
    if type(bindings) is not dict or set(bindings) != required or not bindings.get("source_files") \
            or bindings.get("world_rng") != "per-decision" or fingerprint(bindings) != fingerprint(live.get("bindings")):
        errors.append("declared_input_binding_mismatch")
    elif type(bindings["source_configuration"]) is not dict or not bindings["source_configuration"] \
            or type(bindings["selector_configurations"]) is not dict or not bindings["selector_configurations"] \
            or type(bindings["source_files"]) is not dict \
            or any(type(sha) is not str or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)
                for sha in [*(bindings[k] for k in required if k.endswith("_sha256")),
                    *bindings["source_files"].values()]):
        errors.append("invalid_declared_input_binding")
    expected = []
    try:
        expected = [_key(r, r["variant"]) for r in collector["expected"]]
        if not expected or len(set(expected)) != len(expected):
            raise ValueError("nonempty unique expected roster required")
        if type(bindings) is not dict or any(key[2] not in bindings.get("selector_configurations", {}) for key in expected):
            raise ValueError("expected variant lacks selector binding")
    except (KeyError, TypeError, ValueError):
        errors.append("invalid_expected_roster")
    indexed = []
    for label, report in (("collector", collector), ("live", live)):
        by_key = {}
        try:
            for row in report["rows"]:
                for variant in row["variants"]:
                    key = _key(row, variant)
                    if key in by_key:
                        raise ValueError("duplicate original boundary/variant")
                    by_key[key] = row
        except (KeyError, TypeError, ValueError):
            errors.append(label + "_invalid_or_duplicate_rows")
        indexed.append(by_key)
    compared, mismatches = 0, []
    for key in expected:
        try:
            left, right = (_vector(index[key], key[2]) for index in indexed)
            if left[:2] != right[:2]:
                raise ValueError("boundary identity or legal actions differ")
            tv = .5 * math.fsum(abs(a-b) for a, b in zip(left[2], right[2]))
            compared += 1
            if tv != 0.:
                mismatches.append(dict(seed=key[0], round=key[1], variant=key[2], tv=tv))
        except (KeyError, TypeError, ValueError) as error:
            errors.append(dict(seed=key[0], round=key[1], variant=key[2], reason=str(error)))
    return dict(comparison="collector_native_vs_independent_live_native", gate_tv=0.,
        prior_quantity="production aggregate before Python display rounding; upstream native serializer quantization retained",
        expected=len(expected), compared=compared, mismatches=mismatches, errors=errors,
        passed=bool(expected) and compared == len(expected) and not errors and not mismatches,
        scope="declared roster/input correspondence only; not source, work-count, outcome or panel-completeness qualification")
