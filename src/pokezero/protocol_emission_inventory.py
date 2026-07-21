"""Static and observed protocol inventory for the v3 silent-noop sweep.

The inventory deliberately separates three different kinds of evidence:

``E`` engine-emittable canonical signatures found in the Gen 3 module or shared
simulator source, ``O`` canonical signatures observed in audited public play,
and ``C`` canonical signatures or patterns consumed by the public
observation/belief path. Payload-sensitive direct handlers and justified
redundant announcements are recorded separately, so a handled tag cannot hide
an unhandled subtype. Dynamic source expressions are reported as unresolved
rather than being promoted to canonical evidence. This remains a candidate
generator, not proof that a handler preserves every argument or subtype.
Collision and harm-probe lanes adjudicate that later.
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .coverage_enumeration_audit import AUDIT_PROVENANCE_MATCH_KEYS, require_matching_audit_provenance
from .deep_line_audit import PROTOCOL_SIGNATURE_SCHEMA_VERSION, canonical_protocol_signature


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
_ADD_CALL_START = re.compile(r"(?:\bthis(?:\.battle)?|\bbattle)\.add\s*\(")
_PROTOCOL_TAG = re.compile(r"^-?[a-z][a-z0-9:-]*$")
_TYPESCRIPT_STRING = re.compile(r"^\s*(['\"])(?P<value>(?:\\.|(?!\1).)*)\1\s*$", re.DOTALL)
_OBSERVED_CENSUS_KINDS = frozenset({"fixture", "fixed-opponent", "learned-selfplay"})


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


@dataclass(frozen=True)
class _EngineEmissionDiscovery:
    tags: dict[str, list[ProtocolSourceLocation]]
    signatures: dict[str, list[ProtocolSourceLocation]]
    unresolved: tuple[dict[str, Any], ...]


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


def _mask_typescript_comments(text: str) -> str:
    """Blank comments and strings while preserving offsets and newlines."""

    output = list(text)
    state = "code"
    quote = ""
    index = 0
    while index < len(text):
        character = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
            if character in {"'", '"', "`"}:
                output[index] = " "
                state = "string"
                quote = character
            elif character == "/" and following == "/":
                output[index] = output[index + 1] = " "
                state = "line-comment"
                index += 1
            elif character == "/" and following == "*":
                output[index] = output[index + 1] = " "
                state = "block-comment"
                index += 1
        elif state == "string":
            if character == "\\":
                output[index] = " "
                if index + 1 < len(text) and following != "\n":
                    output[index + 1] = " "
                index += 1
            elif character == quote:
                output[index] = " "
                state = "code"
            elif character != "\n":
                output[index] = " "
        elif state == "line-comment":
            if character == "\n":
                state = "code"
            else:
                output[index] = " "
        elif state == "block-comment":
            if character == "*" and following == "/":
                output[index] = output[index + 1] = " "
                state = "code"
                index += 1
            elif character != "\n":
                output[index] = " "
        index += 1
    return "".join(output)


def _split_typescript_call_arguments(text: str, open_parenthesis: int) -> tuple[str, ...] | None:
    """Split one call without retaining expressions in the public artifact."""

    arguments: list[str] = []
    start = open_parenthesis + 1
    stack = ["("]
    closing = {"(": ")", "[": "]", "{": "}"}
    state = "code"
    quote = ""
    index = start
    while index < len(text):
        character = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "string":
            if character == "\\":
                index += 1
            elif character == quote:
                state = "code"
            index += 1
            continue
        if state == "line-comment":
            if character == "\n":
                state = "code"
            index += 1
            continue
        if state == "block-comment":
            if character == "*" and following == "/":
                state = "code"
                index += 2
            else:
                index += 1
            continue
        if character in {"'", '"', "`"}:
            state = "string"
            quote = character
        elif character == "/" and following == "/":
            state = "line-comment"
            index += 1
        elif character == "/" and following == "*":
            state = "block-comment"
            index += 1
        elif character in closing:
            stack.append(character)
        elif character in closing.values():
            if not stack or closing[stack[-1]] != character:
                return None
            stack.pop()
            if not stack:
                final = text[start:index].strip()
                if final or arguments:
                    arguments.append(final)
                return tuple(arguments)
        elif character == "," and len(stack) == 1:
            arguments.append(text[start:index].strip())
            start = index + 1
        index += 1
    return None


def _typescript_string_literal(expression: str) -> str | None:
    match = _TYPESCRIPT_STRING.fullmatch(expression)
    if match is None:
        return None
    value = match.group("value")
    # Protocol identifiers are ASCII in the audited format. Decode the common
    # escaped quote/backslash forms without accepting executable expressions.
    quote = match.group(1)
    return value.replace(f"\\{quote}", quote).replace("\\\\", "\\")


def _semantic_payload_argument_index(tag: str) -> int | None:
    """Return the ``battle.add`` argument carrying the canonical subtype."""

    field_signature = canonical_protocol_signature(("", tag, "sentinel"))
    if field_signature != tag:
        return 1
    target_signature = canonical_protocol_signature(("", tag, "target", "sentinel"))
    return 2 if target_signature != tag else None


def _canonical_emission_signature(tag: str, arguments: Sequence[str]) -> tuple[str | None, str | None]:
    payload_index = _semantic_payload_argument_index(tag)
    if payload_index is None or payload_index >= len(arguments):
        return tag, None
    payload = _typescript_string_literal(arguments[payload_index])
    if payload is None:
        return None, "dynamic-payload"
    protocol_parts = ["", tag, "target", payload]
    if payload_index == 1:
        protocol_parts = ["", tag, payload]
    return canonical_protocol_signature(protocol_parts), None


def _discover_engine_emission_inventory(showdown_root: Path | str) -> _EngineEmissionDiscovery:
    root = Path(showdown_root)
    tags: dict[str, list[ProtocolSourceLocation]] = defaultdict(list)
    signatures: dict[str, list[ProtocolSourceLocation]] = defaultdict(list)
    unresolved: list[dict[str, Any]] = []
    for path in _source_files(root):
        text = path.read_text(encoding="utf-8")
        searchable = _mask_typescript_comments(text)
        relative = _relative(path, root)
        scope = _emitter_scope(relative)
        for match in _ADD_CALL_START.finditer(searchable):
            line = text.count("\n", 0, match.start()) + 1
            location = ProtocolSourceLocation(path=relative, line=line, evidence=scope)
            open_parenthesis = searchable.find("(", match.start(), match.end())
            arguments = _split_typescript_call_arguments(text, open_parenthesis)
            if arguments is None:
                unresolved.append({
                    "tag": None,
                    "reason": "unparseable-call",
                    "source_location": location.to_json_dict(),
                })
                continue
            tag = _typescript_string_literal(arguments[0]) if arguments else None
            if tag is None:
                unresolved.append({
                    "tag": None,
                    "reason": "dynamic-tag",
                    "source_location": location.to_json_dict(),
                })
                continue
            tag = tag.strip()
            if not _PROTOCOL_TAG.fullmatch(tag):
                unresolved.append({
                    "tag": None,
                    "reason": "noncanonical-literal-tag",
                    "source_location": location.to_json_dict(),
                })
                continue
            tags[tag].append(location)
            signature, reason = _canonical_emission_signature(tag, arguments)
            if reason is not None:
                unresolved.append({
                    "tag": tag,
                    "reason": reason,
                    "source_location": location.to_json_dict(),
                })
            elif signature:
                signatures[signature].append(location)
    return _EngineEmissionDiscovery(
        tags={
            tag: sorted(locations, key=lambda item: (item.path, item.line))
            for tag, locations in sorted(tags.items())
        },
        signatures={
            signature: sorted(locations, key=lambda item: (item.path, item.line))
            for signature, locations in sorted(signatures.items())
        },
        unresolved=tuple(
            sorted(
                unresolved,
                key=lambda row: (
                    row["source_location"]["path"],
                    row["source_location"]["line"],
                    row["reason"],
                ),
            )
        ),
    )


def discover_engine_emissions(showdown_root: Path | str) -> dict[str, list[ProtocolSourceLocation]]:
    """Return literal protocol tags for compatibility with tag-level callers."""

    return _discover_engine_emission_inventory(showdown_root).tags


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


def _signature_pattern_matches(pattern: str, signature: str) -> bool:
    return signature.startswith(pattern[:-1]) if pattern.endswith(":*") else signature == pattern


def _canonical_consumer_inventory(
    consumers: Mapping[str, Sequence[ProtocolSourceLocation]],
) -> tuple[dict[str, list[ProtocolSourceLocation]], tuple[dict[str, Any], ...]]:
    """Lift tag dispatches into canonical signatures or explicit patterns."""

    signatures: dict[str, list[ProtocolSourceLocation]] = defaultdict(list)
    unresolved: list[dict[str, Any]] = []
    direct_by_tag: dict[str, list[ProtocolSignatureCoverage]] = defaultdict(list)
    for coverage in _SIGNATURE_COVERAGE:
        if coverage.coverage == "direct":
            direct_by_tag[_signature_tag(coverage.signature)].append(coverage)
    for tag, locations in consumers.items():
        if _semantic_payload_argument_index(tag) is None:
            signatures[tag].extend(locations)
            continue
        direct = direct_by_tag.get(tag, ())
        if direct:
            for coverage in direct:
                signatures[coverage.signature].extend(locations)
            continue
        unresolved.append({
            "tag": tag,
            "reason": "tag-only-dispatch-without-canonical-signature",
            "source_locations": [location.to_json_dict() for location in locations],
        })
    return (
        {
            signature: sorted(set(locations), key=lambda item: (item.path, item.line))
            for signature, locations in sorted(signatures.items())
        },
        tuple(sorted(unresolved, key=lambda row: row["tag"])),
    )


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


def require_expected_observed_audit_provenance(
    entries: Iterable[Mapping[str, Any]], *, expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Require dynamic census inputs to match each other and this inventory run.

    A protocol differential is evidence for one encoded world. Accepting a
    source-compatible artifact made by another image would make a clean result
    look stronger than it is, so compare the immutable coverage identity only.
    """

    actual = require_matching_audit_provenance(entries)
    missing = [key for key in AUDIT_PROVENANCE_MATCH_KEYS if not expected.get(key)]
    if missing:
        raise ValueError(
            "inventory run provenance is missing required fields: " + ", ".join(missing)
        )
    expected_identity = {key: expected[key] for key in AUDIT_PROVENANCE_MATCH_KEYS}
    if actual != expected_identity:
        raise ValueError(
            "observed audit provenance differs from this inventory run: "
            f"expected {expected_identity!r}, got {actual!r}"
        )
    return actual


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
    observed_census_kinds: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the E/O/C differential with source-level evidence and limitations."""

    engine_discovery = _discover_engine_emission_inventory(showdown_root)
    engine = engine_discovery.tags
    engine_signatures = engine_discovery.signatures
    consumers = discover_consumer_dispatches(public_root)
    consumer_signatures, unresolved_consumers = _canonical_consumer_inventory(consumers)
    observed_paths = tuple(observed_audits)
    if observed_census_kinds and len(observed_census_kinds) != len(observed_paths):
        raise ValueError("observed census kinds must have one entry per observed audit")
    census_kinds = tuple(observed_census_kinds) if observed_census_kinds else ("fixture",) * len(observed_paths)
    invalid_census_kinds = sorted(set(census_kinds) - _OBSERVED_CENSUS_KINDS)
    if invalid_census_kinds:
        raise ValueError("unknown observed census kind: " + ", ".join(invalid_census_kinds))
    observed_signatures, observed_sources = load_observed_signatures(observed_paths)
    observed_provenance = load_observed_audit_provenance(observed_paths)
    observed_counts_by_kind: Counter[str] = Counter()
    for path, kind in zip(observed_paths, census_kinds, strict=True):
        counts, _sources = load_observed_signatures((path,))
        observed_counts_by_kind[kind] += sum(counts.values())
    learned_policy_status = (
        "present"
        if observed_counts_by_kind["learned-selfplay"] > 0
        else (
            "empty-learned-v3-census"
            if "learned-selfplay" in census_kinds
            else "unavailable-no-trained-v3-checkpoint"
        )
    )
    observed_by_tag: Counter[str] = Counter()
    for signature, count in observed_signatures.items():
        observed_by_tag[_signature_tag(signature)] += count

    engine_tags = set(engine)
    engine_signature_set = set(engine_signatures)
    consumer_tags = set(consumers)
    consumer_signature_set = set(consumer_signatures)
    consumer_exact_signatures = {
        signature for signature in consumer_signature_set if not signature.endswith(":*")
    }
    unresolved_engine_tags = {
        row["tag"] for row in engine_discovery.unresolved if row["tag"] is not None
    }
    has_unresolved_dynamic_tag = any(
        row["tag"] is None for row in engine_discovery.unresolved
    )

    def could_have_unresolved_engine_source(signature: str) -> bool:
        return has_unresolved_dynamic_tag or _signature_tag(signature) in unresolved_engine_tags

    def has_canonical_consumer(signature: str) -> bool:
        return any(
            _signature_pattern_matches(pattern, signature)
            for pattern in consumer_signature_set
        )

    stale_direct_coverage = sorted(
        signature
        for signature in observed_signatures
        if (coverage := _signature_coverage(signature)) is not None
        and coverage.coverage == "direct"
        and _signature_tag(signature) not in consumer_tags
    )
    if stale_direct_coverage:
        raise ValueError(
            "direct protocol-signature coverage has no discovered consumer dispatch: "
            + ", ".join(stale_direct_coverage)
        )
    observed_tags = set(observed_by_tag)
    observed_signature_rows: list[dict[str, Any]] = []
    observed_without_direct_consumer: list[dict[str, Any]] = []
    observed_without_semantic_coverage: list[dict[str, Any]] = []
    observed_unconsumed_unclassified: list[dict[str, Any]] = []
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
                observed_unconsumed_unclassified.append(row)
    return {
        "schema_version": "pokezero.protocol-emission-inventory.v3",
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
            "canonical_signatures": [
                {
                    "signature": signature,
                    "tag": _signature_tag(signature),
                    "source_locations": [location.to_json_dict() for location in locations],
                    "source_scopes": sorted({location.evidence for location in locations}),
                }
                for signature, locations in engine_signatures.items()
            ],
            "canonical_signature_count": len(engine_signature_set),
            "unresolved_emissions": list(engine_discovery.unresolved),
            "unresolved_emission_count": len(engine_discovery.unresolved),
            "canonical_complete": not engine_discovery.unresolved,
        },
        "consumer_dispatch": {
            "tags": [
                {"tag": tag, "source_locations": [location.to_json_dict() for location in locations]}
                for tag, locations in consumers.items()
            ],
            "tag_count": len(consumer_tags),
            "canonical_signatures": [
                {
                    "signature": signature,
                    "tag": _signature_tag(signature),
                    "kind": "pattern" if signature.endswith(":*") else "exact",
                    "source_locations": [location.to_json_dict() for location in locations],
                }
                for signature, locations in consumer_signatures.items()
            ],
            "canonical_signature_count": len(consumer_signature_set),
            "unresolved_dispatches": list(unresolved_consumers),
            "unresolved_dispatch_count": len(unresolved_consumers),
            "canonical_complete": not unresolved_consumers,
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
            "census_input_kinds": [
                {
                    "kind": kind,
                    "artifact_count": census_kinds.count(kind),
                    "observed_signature_count": observed_counts_by_kind[kind],
                }
                for kind in sorted(set(census_kinds))
            ],
            "learned_policy_census": {
                "status": learned_policy_status
            },
            "signature_coverage": observed_signature_rows,
        },
        "differential": {
            "observed_but_unconsumed": [
                {"tag": tag, "count": observed_by_tag[tag]}
                for tag in sorted(observed_tags - consumer_tags, key=lambda tag: (-observed_by_tag[tag], tag))
            ],
            "observed_but_unconsumed_unclassified": [
                row for row in observed_unconsumed_unclassified
            ],
            "observed_signatures_without_direct_consumer": observed_without_direct_consumer,
            "observed_signatures_without_semantic_coverage": observed_without_semantic_coverage,
            "emittable_but_unobserved": sorted(engine_tags - observed_tags),
            "consumer_not_emittable": sorted(consumer_tags - engine_tags),
            "emittable_signatures_without_consumer": sorted(
                signature
                for signature in engine_signature_set
                if not has_canonical_consumer(signature)
            ),
            "emittable_signatures_but_unobserved": sorted(
                engine_signature_set - set(observed_signatures)
            ),
            "observed_signatures_not_statically_emittable": sorted(
                signature
                for signature in observed_signatures
                if signature not in engine_signature_set
                and not could_have_unresolved_engine_source(signature)
            ),
            "observed_signatures_with_unresolved_engine_source": sorted(
                signature
                for signature in observed_signatures
                if signature not in engine_signature_set
                and could_have_unresolved_engine_source(signature)
            ),
            "consumer_exact_signatures_not_statically_emittable": sorted(
                signature
                for signature in consumer_exact_signatures
                if signature not in engine_signature_set
                and not could_have_unresolved_engine_source(signature)
            ),
            "consumer_exact_signatures_with_unresolved_engine_source": sorted(
                signature
                for signature in consumer_exact_signatures
                if signature not in engine_signature_set
                and could_have_unresolved_engine_source(signature)
            ),
        },
        "limitations": [
            "Engine discovery canonicalizes literal this.add/battle.add payloads; dynamic or unparseable expressions remain explicit unresolved emissions and require census evidence.",
            "Shared-simulator source locations are potential Gen 3 emissions, not standalone Gen 3 reachability proof.",
            "Consumer discovery combines event_type comparisons with the audited direct-signature registry; tag-only payload-sensitive dispatches remain explicit unresolved dispatches.",
            "Canonical signature coverage records direct handlers separately from semantic aliases; aliases are justified redundant announcements, not members of C.",
            "Observed signatures are public audit-census counts. They do not include private requests or raw protocol payloads.",
        ],
    }
