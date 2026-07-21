"""Static and observed protocol inventory for the v3 silent-noop sweep.

The inventory deliberately separates three different kinds of evidence:

``E`` engine-emittable protocol tags found in the Gen 3 module or shared
simulator source, ``O`` canonical signatures observed in audited public play,
and ``C`` tags syntactically dispatched by the public observation/belief path.
Payload-sensitive direct handlers and justified redundant announcements are
recorded separately, so a handled tag cannot hide an unhandled subtype. It is
still a candidate generator, not proof that a handler preserves every argument
or subtype. Collision and harm-probe lanes adjudicate that later.
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .deep_line_audit import PROTOCOL_SIGNATURE_SCHEMA_VERSION


_EMITTER_RELATIVE_PATHS = (
    "sim",
    "data/moves.ts",
    "data/abilities.ts",
    "data/items.ts",
    "data/mods/gen3",
)
_CONSUMER_RELATIVE_PATHS = (
    "src/pokezero/showdown.py",
    "src/pokezero/transitions.py",
    "src/pokezero/belief.py",
    "src/pokezero/public_action_capture.py",
    "src/pokezero/turn_merged.py",
)
_ADD_CALL = re.compile(
    r"(?:\bthis(?:\.battle)?|\bbattle)\.add\(\s*(?:\n\s*)*(['\"])(?P<tag>[^'\"\n]+)\1",
    re.MULTILINE,
)
_PROTOCOL_TAG = re.compile(r"^-?[a-z][a-z0-9:-]*$")


@dataclass(frozen=True)
class ProtocolSourceLocation:
    """One static source location contributing an emission or consumer tag."""

    path: str
    line: int
    evidence: str

    def to_json_dict(self) -> dict[str, Any]:
        return {"path": self.path, "line": self.line, "evidence": self.evidence}


@dataclass(frozen=True)
class ProtocolSignatureCoverage:
    """One audited semantic treatment of a canonical protocol signature.

    ``direct`` means the listed protocol signature itself is consumed by a
    public observation, replay, or belief path. ``semantic-alias`` means the
    line is intentionally redundant because another public line in the same
    mechanic carries the model-visible fact. ``non-model`` is a public
    transport, setup, diagnostic, or terminal announcement that cannot alter
    a later model decision in this fixed-format audit. Both non-direct classes
    stay outside C so the inventory distinguishes a true handler from a
    documented non-omission.
    """

    signature: str
    coverage: str
    handler: str
    detail: str

    def matches(self, signature: str) -> bool:
        if self.signature.endswith(":*"):
            return signature.startswith(self.signature[:-1])
        return self.signature == signature

    def to_json_dict(self) -> dict[str, str]:
        return {
            "signature": self.signature,
            "coverage": self.coverage,
            "handler": self.handler,
            "detail": self.detail,
        }


# This registry complements AST tag discovery with the small payload-sensitive
# surface whose semantics are intentionally non-generic. Keep aliases out of
# C: they document why an announcement can be absent from model state without
# pretending that the announcement itself was parsed.
_SIGNATURE_COVERAGE = (
    ProtocolSignatureCoverage(
        signature="move:*",
        coverage="direct",
        handler="src/pokezero/transitions.py action-window fold",
        detail="A declared move and its identifier are retained in transition history.",
    ),
    ProtocolSignatureCoverage(
        signature="cant:*",
        coverage="direct",
        handler="src/pokezero/showdown.py turn-merged cant reason + src/pokezero/belief.py sleep bookkeeping",
        detail="The public lost-turn reason is retained in turn-merged history; sleep has additional belief bookkeeping.",
    ),
    ProtocolSignatureCoverage(
        signature="-activate:protect",
        coverage="direct",
        handler="src/pokezero/transitions.py damage-outcome fold",
        detail="Protect activation marks the opposing action as blocked.",
    ),
    ProtocolSignatureCoverage(
        signature="-activate:detect",
        coverage="direct",
        handler="src/pokezero/transitions.py damage-outcome fold",
        detail="Detect activation marks the opposing action as blocked.",
    ),
    ProtocolSignatureCoverage(
        signature="-activate:substitute",
        coverage="direct",
        handler="src/pokezero/transitions.py damage-outcome fold",
        detail="Substitute activation records the hit-sub damage outcome.",
    ),
    ProtocolSignatureCoverage(
        signature="-activate:endure",
        coverage="direct",
        handler="src/pokezero/transitions.py damage-outcome fold",
        detail="Endure activation records the endured damage outcome.",
    ),
    ProtocolSignatureCoverage(
        signature="-activate:shedskin",
        coverage="direct",
        handler="src/pokezero/belief.py Shed Skin public inference",
        detail="The ability activation is paired with the following public cure-status event.",
    ),
    ProtocolSignatureCoverage(
        signature="-start:futuresight",
        coverage="direct",
        handler="src/pokezero/showdown.py _update_future_sight",
        detail="The delayed attack landing turn is encoded on the field token.",
    ),
    ProtocolSignatureCoverage(
        signature="-end:futuresight",
        coverage="direct",
        handler="src/pokezero/showdown.py _update_future_sight",
        detail="Landing clears the delayed-attack field state.",
    ),
    ProtocolSignatureCoverage(
        signature="-start:doomdesire",
        coverage="direct",
        handler="src/pokezero/showdown.py _update_future_sight",
        detail="The delayed attack landing turn is encoded on the field token.",
    ),
    ProtocolSignatureCoverage(
        signature="-end:doomdesire",
        coverage="direct",
        handler="src/pokezero/showdown.py _update_future_sight",
        detail="Landing clears the delayed-attack field state.",
    ),
    *(
        ProtocolSignatureCoverage(
            signature=f"{event}:{counter}",
            coverage="direct",
            handler="src/pokezero/showdown.py _update_volatiles",
            detail="The active Perish Song counter is retained as a volatile token.",
        )
        for event in ("-start", "-end")
        for counter in ("perish0", "perish1", "perish2", "perish3", "perishsong")
    ),
    ProtocolSignatureCoverage(
        signature="-singleturn:protect",
        coverage="semantic-alias",
        handler="move:protect -> src/pokezero/transitions.py",
        detail="The declared Protect action remains in transition history even when it blocks no move.",
    ),
    ProtocolSignatureCoverage(
        signature="-singleturn:endure",
        coverage="semantic-alias",
        handler="move:endure -> src/pokezero/transitions.py",
        detail="The declared Endure action remains in transition history even when it prevents no KO.",
    ),
    ProtocolSignatureCoverage(
        signature="-fieldactivate:perishsong",
        coverage="semantic-alias",
        handler="-start/-end:perish0..perish3 -> src/pokezero/showdown.py volatile tracker",
        detail="Perish Song counters are the durable decision-state fact; field activation is an announcement.",
    ),
    ProtocolSignatureCoverage(
        signature="-mustrecharge",
        coverage="semantic-alias",
        handler="cant:recharge -> src/pokezero/transitions.py and src/pokezero/turn_merged.py",
        detail="The following public cant:recharge action is the model-visible forced-turn fact; this line announces it one turn early.",
    ),
    ProtocolSignatureCoverage(
        signature="-singleturn:focuspunch",
        coverage="semantic-alias",
        handler="move:focuspunch and cant:focuspunch -> src/pokezero/transitions.py",
        detail="Focus Punch's declared move and any public interruption are retained; the one-turn charge announcement has no later state of its own.",
    ),
    *(
        ProtocolSignatureCoverage(
            signature=signature,
            coverage="non-model",
            handler="fixed-format protocol boundary",
            detail=detail,
        )
        for signature, detail in (
            ("debug", "Diagnostic transport output is not battle state."),
            ("-hint", "Human-facing protocol guidance does not change battle state."),
            ("start", "Battle-start framing has no later decision-state payload."),
            ("gametype", "The fixed singles format is already selected by the audit runner."),
            ("gen", "The fixed Gen 3 format is already selected by the audit runner."),
            ("rule", "Format rules are fixed before the first model decision."),
            ("teamsize", "Gen 3 random battles use the fixed team-size contract."),
            ("tier", "The fixed Gen 3 random-battle format is already selected by the runner."),
            ("tie", "A terminal tie has no subsequent model decision."),
        )
    ),
)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _source_files(showdown_root: Path) -> tuple[Path, ...]:
    """Return the bounded source surface that can contribute Gen 3 protocol lines."""

    files: set[Path] = set()
    for relative in _EMITTER_RELATIVE_PATHS:
        path = showdown_root / relative
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(candidate for candidate in path.rglob("*.ts") if candidate.is_file())
    return tuple(sorted(files))


def _emitter_scope(relative_path: str) -> str:
    return "gen3-module" if relative_path.startswith("data/mods/gen3/") else "shared-simulator"


def discover_engine_emissions(showdown_root: Path | str) -> dict[str, list[ProtocolSourceLocation]]:
    """Find literal protocol tags emitted by relevant Showdown TypeScript sources.

    Dynamic tag expressions are deliberately not guessed. They remain visible
    through the dynamic census and are called out in the report limitations.
    """

    root = Path(showdown_root)
    emissions: dict[str, list[ProtocolSourceLocation]] = defaultdict(list)
    for path in _source_files(root):
        text = path.read_text(encoding="utf-8")
        relative = _relative(path, root)
        scope = _emitter_scope(relative)
        for match in _ADD_CALL.finditer(text):
            tag = match.group("tag").strip()
            if not _PROTOCOL_TAG.fullmatch(tag):
                continue
            line = text.count("\n", 0, match.start()) + 1
            # A whole-line comment can resemble an emitter. We keep block
            # comments conservative (reported as a potential literal source)
            # rather than attempting an unsafe TypeScript parser here.
            source_line = text.splitlines()[line - 1].lstrip()
            if source_line.startswith("//"):
                continue
            emissions[tag].append(
                ProtocolSourceLocation(path=relative, line=line, evidence=scope)
            )
    return {tag: sorted(locations, key=lambda item: (item.path, item.line)) for tag, locations in sorted(emissions.items())}


def _is_event_type_expression(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "event_type"
    if isinstance(node, ast.Attribute):
        return node.attr == "event_type"
    return False


def _literal_strings(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,)
    if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
        values: list[str] = []
        for element in node.elts:
            values.extend(_literal_strings(element))
        return tuple(values)
    return ()


class _ConsumerVisitor(ast.NodeVisitor):
    def __init__(self, *, relative_path: str) -> None:
        self.relative_path = relative_path
        self.tags: dict[str, list[ProtocolSourceLocation]] = defaultdict(list)

    def visit_Compare(self, node: ast.Compare) -> None:  # noqa: N802 - ast visitor API
        operands = (node.left, *node.comparators)
        for position, operand in enumerate(operands):
            if not _is_event_type_expression(operand):
                continue
            for other in (*operands[:position], *operands[position + 1 :]):
                for tag in _literal_strings(other):
                    if _PROTOCOL_TAG.fullmatch(tag):
                        self.tags[tag].append(
                            ProtocolSourceLocation(
                                path=self.relative_path,
                                line=node.lineno,
                                evidence="event_type-comparison",
                            )
                        )
        self.generic_visit(node)


def discover_consumer_dispatches(public_root: Path | str) -> dict[str, list[ProtocolSourceLocation]]:
    """Find explicit public-path protocol dispatch comparisons using Python AST.

    This is intentionally narrower than a raw string grep: only comparisons
    involving an ``event_type`` value count as a consumer. A positive result
    means the path names the tag, not that every argument is represented.
    """

    root = Path(public_root)
    consumers: dict[str, list[ProtocolSourceLocation]] = defaultdict(list)
    for relative in _CONSUMER_RELATIVE_PATHS:
        path = root / relative
        if not path.is_file():
            continue
        visitor = _ConsumerVisitor(relative_path=relative)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        for tag, locations in visitor.tags.items():
            consumers[tag].extend(locations)
    return {tag: sorted(locations, key=lambda item: (item.path, item.line)) for tag, locations in sorted(consumers.items())}


def load_observed_signatures(paths: Iterable[Path | str]) -> tuple[Counter[str], dict[str, list[str]]]:
    """Load canonical protocol-signature counts from deep-line audit reports."""

    counts: Counter[str] = Counter()
    provenance_paths: dict[str, list[str]] = defaultdict(list)
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        signatures = payload.get("protocol_signatures", {})
        if not isinstance(signatures, Mapping):
            raise ValueError(f"{path} protocol_signatures must be an object")
        for signature, raw_count in signatures.items():
            count = int(raw_count)
            if count < 0:
                raise ValueError(f"{path} has a negative count for {signature!r}")
            if count:
                counts[str(signature)] += count
                provenance_paths[str(signature)].append(str(path))
    return counts, {signature: sorted(paths) for signature, paths in sorted(provenance_paths.items())}


def load_observed_audit_provenance(paths: Iterable[Path | str]) -> list[dict[str, Any]]:
    """Require each dynamic census input to declare v3 and signature provenance."""

    entries: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance = payload.get("audit_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"{path} is missing audit_provenance")
        signature_schema = payload.get("protocol_signature_schema_version")
        if signature_schema != PROTOCOL_SIGNATURE_SCHEMA_VERSION:
            raise ValueError(
                f"{path} has protocol signature schema {signature_schema!r}; "
                f"expected {PROTOCOL_SIGNATURE_SCHEMA_VERSION!r}"
            )
        entries.append({"path": str(path), "audit_provenance": dict(provenance)})
    return entries


def _signature_tag(signature: str) -> str:
    return signature.split(":", 1)[0]


def _signature_coverage(signature: str) -> ProtocolSignatureCoverage | None:
    """Return direct consumption or a documented semantic alias for a signature."""

    return next((coverage for coverage in _SIGNATURE_COVERAGE if coverage.matches(signature)), None)


def build_protocol_inventory(
    *,
    showdown_root: Path | str,
    public_root: Path | str,
    observed_audits: Sequence[Path | str] = (),
) -> dict[str, Any]:
    """Build the E/O/C differential with source-level evidence and limitations."""

    engine = discover_engine_emissions(showdown_root)
    consumers = discover_consumer_dispatches(public_root)
    observed_paths = tuple(observed_audits)
    observed_signatures, observed_sources = load_observed_signatures(observed_paths)
    observed_provenance = load_observed_audit_provenance(observed_paths)
    observed_by_tag: Counter[str] = Counter()
    for signature, count in observed_signatures.items():
        observed_by_tag[_signature_tag(signature)] += count

    engine_tags = set(engine)
    consumer_tags = set(consumers)
    observed_tags = set(observed_by_tag)
    observed_signature_rows: list[dict[str, Any]] = []
    observed_without_direct_consumer: list[dict[str, Any]] = []
    observed_without_semantic_coverage: list[dict[str, Any]] = []
    observed_unconsumed_unclassified: Counter[str] = Counter()
    for signature, count in sorted(observed_signatures.items(), key=lambda item: (-item[1], item[0])):
        coverage = _signature_coverage(signature)
        row: dict[str, Any] = {
            "signature": signature,
            "tag": _signature_tag(signature),
            "count": count,
            "sources": observed_sources[signature],
            "coverage": coverage.coverage if coverage is not None else "unclassified",
        }
        if coverage is not None:
            row["handler"] = coverage.handler
            row["detail"] = coverage.detail
        observed_signature_rows.append(row)
        if coverage is None or coverage.coverage != "direct":
            observed_without_direct_consumer.append(row)
        if coverage is None:
            observed_without_semantic_coverage.append(row)
            if row["tag"] not in consumer_tags:
                observed_unconsumed_unclassified[row["tag"]] += count
    return {
        "schema_version": "pokezero.protocol-emission-inventory.v2",
        "engine_emittable": {
            "tags": [
                {
                    "tag": tag,
                    "source_locations": [location.to_json_dict() for location in locations],
                    "source_scopes": sorted({location.evidence for location in locations}),
                }
                for tag, locations in engine.items()
            ],
            "tag_count": len(engine_tags),
        },
        "consumer_dispatch": {
            "tags": [
                {"tag": tag, "source_locations": [location.to_json_dict() for location in locations]}
                for tag, locations in consumers.items()
            ],
            "tag_count": len(consumer_tags),
        },
        "consumer_signatures": {
            "entries": [coverage.to_json_dict() for coverage in _SIGNATURE_COVERAGE],
            "direct_signature_count": sum(
                coverage.coverage == "direct" for coverage in _SIGNATURE_COVERAGE
            ),
            "semantic_alias_count": sum(
                coverage.coverage == "semantic-alias" for coverage in _SIGNATURE_COVERAGE
            ),
            "non_model_signature_count": sum(
                coverage.coverage == "non-model" for coverage in _SIGNATURE_COVERAGE
            ),
        },
        "observed": {
            "signature_counts": [
                {
                    "signature": signature,
                    "tag": _signature_tag(signature),
                    "count": count,
                    "sources": observed_sources[signature],
                }
                for signature, count in sorted(observed_signatures.items(), key=lambda item: (-item[1], item[0]))
            ],
            "signature_count": len(observed_signatures),
            "tag_count": len(observed_tags),
            "audit_provenance": observed_provenance,
            "signature_coverage": observed_signature_rows,
        },
        "differential": {
            "observed_but_unconsumed": [
                {"tag": tag, "count": observed_by_tag[tag]}
                for tag in sorted(observed_tags - consumer_tags, key=lambda tag: (-observed_by_tag[tag], tag))
            ],
            "observed_but_unconsumed_unclassified": [
                {"tag": tag, "count": observed_unconsumed_unclassified[tag]}
                for tag in sorted(
                    observed_unconsumed_unclassified,
                    key=lambda tag: (-observed_unconsumed_unclassified[tag], tag),
                )
            ],
            "observed_signatures_without_direct_consumer": observed_without_direct_consumer,
            "observed_signatures_without_semantic_coverage": observed_without_semantic_coverage,
            "emittable_but_unobserved": sorted(engine_tags - observed_tags),
            "consumer_not_emittable": sorted(consumer_tags - engine_tags),
        },
        "limitations": [
            "Engine discovery records literal this.add/battle.add tags only; dynamic tag expressions require dynamic census evidence.",
            "Shared-simulator source locations are potential Gen 3 emissions, not standalone Gen 3 reachability proof.",
            "Consumer discovery records event_type comparisons only. It does not prove that every tag argument or subtype is encoded.",
            "Canonical signature coverage records direct handlers separately from semantic aliases; aliases are justified redundant announcements, not members of C.",
            "Observed signatures are public audit-census counts. They do not include private requests or raw protocol payloads.",
        ],
    }
