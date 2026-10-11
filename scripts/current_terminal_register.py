"""Create-only current metadata binding for the immutable historical C155 register.

No pool is measured and no status is discharged. Only source inventory, source
addresses, fingerprint and this binding's exact nonmeasurement carrier may move.
Stored facts are read independently of derivation; drift never repairs itself.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = "reports/artifacts/current_terminal_register_20261010.json"
HISTORICAL = "reports/c155_terminal_disposition_register.md"
HISTORICAL_SHA256 = "be65dff053da8cb41b9e7ece66cdd94afacc8dad6c1915a72372ce367893193c"
SOURCE_ONLY_KEYS = frozenset({
    "base.patch_stack", "base.expected_counter_artifacts", "t1.head_fingerprint",
    "t1.hashed_crate_sources", "t1.hashed_input_files", "t3.tie_refusal_line",
    "t4.leftovers_truncated_consumer_line", "t1.committed_json_carrying_head_fingerprint",
})
SCHEMA = "pokezero.current-terminal-register.v1"
SCOPE = "CURRENT_SOURCE_METADATA_ONLY_HISTORICAL_MEASUREMENTS_AND_STATUSES_UNCHANGED"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json(text):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs)


def validate_structure(document, *, historical_root=ROOT):
    if sha(historical_root / HISTORICAL) != HISTORICAL_SHA256:
        raise ValueError("historical C155 register changed")
    if (document.get("schema") != SCHEMA or document.get("scope") != SCOPE
            or document.get("historical_path") != HISTORICAL
            or document.get("historical_sha256") != HISTORICAL_SHA256
            or document.get("artifact_path") != ARTIFACT
            or document.get("pool_remeasured") is not False
            or document.get("scientific_admission") is not False
            or document.get("status_changes") != []
            or document.get("source_binding_carriers") != [ARTIFACT]
            or not isinstance(document.get("facts"), dict)
            or len(document["facts"]) != 75
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in document["facts"].items())):
        raise ValueError("current terminal-register binding malformed or widens scope")
    return document


def load_binding():
    return validate_structure(strict_json((ROOT / ARTIFACT).read_text(encoding="utf-8")))


def guard_counts():
    from test_unreachable_readjudication import EveryWorkflowTestCountGuardMatchesItsModuleTests as scan
    lines = (ROOT / ".github/workflows/engine-fidelity-gates.yml").read_text().splitlines()
    executable = sum(bool(scan.INVOCATION.search(line)) and not line.strip().startswith("#") for line in lines)
    sites, guards = scan._sites(), scan._guards()
    unresolved = sorted({target for _, targets, guard, _ in sites if guard is None for target in targets})
    if len(sites) != executable or executable != len(guards) + len(unresolved) or unresolved:
        raise ValueError("workflow guard inventory is incomplete")
    return dict(executable=executable, resolved=len(guards), unresolved=unresolved)


def source_carriers(corpus, root, fingerprint):
    carriers = [name for name in corpus if fingerprint in (root / name).read_text()]
    if set(carriers) - {ARTIFACT}:
        raise ValueError(f"nonbinding artifacts carry current fingerprint: {carriers}")
    return carriers


def build_binding():
    sys.path.insert(0, str(ROOT / "tests"))
    register = sys.modules.get("tests.test_terminal_disposition_register")
    if register is None:
        register = importlib.import_module("test_terminal_disposition_register")
    historical = register.historical_register_facts()
    facts = register.derive()
    corpus = register.committed_json()
    # The new binding itself carries the fingerprint, but no measurement does.
    # Before publication, include exactly this prospective path. After staging,
    # derive the raw count from actual indexed bytes; no broad exemption exists.
    carriers = source_carriers(corpus, ROOT, facts["t1.head_fingerprint"])
    if ARTIFACT not in corpus:
        facts["t1.committed_json_carrying_head_fingerprint"] = "1"
    elif carriers != [ARTIFACT]:
        raise ValueError("indexed binding lacks its actual current fingerprint")
    if facts["t1.committed_json_carrying_head_fingerprint"] != "1":
        raise ValueError("current source-binding carrier count is not one")
    if set(facts) != set(historical):
        raise ValueError("terminal register fact inventory changed")
    deltas = {key: dict(historical=historical[key], current=value)
        for key, value in facts.items() if historical[key] != value}
    if set(deltas) - SOURCE_ONLY_KEYS:
        raise ValueError("measurement or disposition change requires separate adjudication")
    source = (ROOT / "tests/test_terminal_disposition_register.py").read_text()
    tree = ast.parse(source)
    methods = sum(isinstance(n, ast.FunctionDef) and n.name.startswith("test")
        for cls in tree.body if isinstance(cls, ast.ClassDef) for n in cls.body)
    paths = set(register._tree_side_hashed_inputs()) | {
        ROOT / name for name in (HISTORICAL, register.LEDGER, register.WORKFLOW,
            "scripts/current_terminal_register.py", "scripts/engine_build_fingerprint.py",
            "tests/test_terminal_disposition_register.py", "tests/test_unreachable_readjudication.py",
            "tests/test_never_fired_counter_census.py", "tests/test_boundary_verdict_partition.py")}
    document = dict(schema=SCHEMA, scope=SCOPE, artifact_path=ARTIFACT,
        historical_path=HISTORICAL, historical_sha256=HISTORICAL_SHA256,
        scientific_admission=False, pool_remeasured=False, status_changes=[],
        source_binding_carriers=[ARTIFACT], facts=dict(sorted(facts.items())),
        deltas=dict(sorted(deltas.items())), items=register.register_items(),
        historical_self_description=dict(test_methods=53, executable_guards=40, resolved_guards=40),
        current_test_methods=methods, workflow_guard_counts=guard_counts(),
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)},
        corpus_sha256={name: sha(ROOT / name) for name in corpus if name != ARTIFACT},
        self_hash_excluded=True)
    return validate_structure(document)


def verify_binding(document):
    validate_structure(document)
    if document != build_binding():
        raise ValueError("current terminal-register binding is stale or altered")
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        document = build_binding()
        with (ROOT / ARTIFACT).open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
        print(f"created {ARTIFACT}; historical register and measurements unchanged")
    else:
        verify_binding(load_binding())
        print("current terminal-register binding verified; no scientific admission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
