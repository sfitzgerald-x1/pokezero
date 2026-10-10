"""Current-source citation supplement; never regenerate historical measurements.

C153/C154 remain immutable snapshots. This instrument re-resolves their citation
and call-graph evidence, using C154's recorded pool rather than running a new
Showdown census. It refuses any change beyond citation addresses: changed claims
need a separate adjudication, not an address refresh. The supplement is not
fidelity, playing-strength, or prospective scientific admission evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

import c153_wide_negative_census as c153
import c154_unreachable_readjudication as c154

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = "reports/artifacts/current_census_citations_20261010_public_prefix.json"
C153_ARTIFACT = "reports/artifacts/c153_wide_negative_census.json"
C154_ARTIFACT = "reports/artifacts/c154_unreachable_readjudication.json"
HISTORICAL = {
    C153_ARTIFACT:
        "b4cfee633a79af3fcd246906a898168aedd3e5ac254311856d67b70d4affd995",
    C154_ARTIFACT:
        "67b6711b007df553ed1e63be635fef2f048b021d1464b6168851fce2e3f2485c",
}
INPUTS = (
    "scripts/current_census_citations.py",
    "scripts/c153_wide_negative_census.py",
    "scripts/c154_unreachable_readjudication.py",
    "scripts/engine_transition_differential.py",
    "src/pokezero/engine_world.py",
    "src/pokezero/engine_search.py",
    "src/pokezero/local_showdown.py",
    "src/pokezero/showdown.py",
    "src/pokezero/transitions_fold.py",
    "src/pokezero/golden_corpus_scenarios.py",
    "src/pokezero/randbat_vocab.py",
    "src/pokezero/scenario_studio/domain.py",
    "src/pokezero/engine_fidelity.py",
    "rust/pokezero-search/src/events.rs",
    "third_party/poke-engine-gen3-encore-failencore.patch",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _historical_document(path: str) -> dict:
    data = (ROOT / path).read_bytes()
    if _sha(data) != HISTORICAL[path]:
        raise ValueError(f"historical evidence changed: {path}")
    return json.loads(data)


def _without_addresses(value, key: str = ""):
    """Only for the address-only admission check, NEVER for witness equality.

    Preserve edge identities, multiplicities, verdicts, counts, and prose. A
    complete supplement is still compared byte-for-value to fresh derivation.
    """
    if isinstance(value, str):
        return re.sub(r":\d+\b", ":LINE", value)
    if isinstance(value, dict):
        return {k: _without_addresses(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_without_addresses(v, key) for v in value]
    if key in {"line", "edges_out_of_the_chokepoint"}:
        return "LINE"
    return value


def _require_address_only(historical, current, label: str) -> None:
    if _without_addresses(historical) != _without_addresses(current):
        raise ValueError(f"semantic change needs separate adjudication: {label}")


def build_supplement() -> dict:
    old153 = _historical_document(C153_ARTIFACT)
    old154 = _historical_document(C154_ARTIFACT)
    cannot_reach = dict(c153.CENSUS_CANNOT_REACH)
    structural = dict(c153.STRUCTURAL_DIVERGENCE_CLASSES)
    old_cannot_reach = {name: record["census_cannot_reach"]
                        for name, record in old153["verdicts"].items()
                        if "census_cannot_reach" in record}
    old_structural = {name.removeprefix("divergence_class:"): record["structural_demonstration"]
                      for name, record in old153["verdicts"].items()
                      if "structural_demonstration" in record}
    if set(old_cannot_reach) != set(cannot_reach) or set(old_structural) != set(structural):
        raise ValueError("C153 claim inventory changed; needs separate adjudication")
    for name, text in cannot_reach.items():
        _require_address_only(old153["verdicts"][name]["census_cannot_reach"], text, name)
    for name, text in structural.items():
        _require_address_only(
            old153["verdicts"][f"divergence_class:{name}"]["structural_demonstration"],
            text, name,
        )
    graph = c154.rust_call_graph(c154.EV, "heal_subcase", c154.HEAL_SUBCASE_ROOT)
    _require_address_only(old154["pool"]["heal_subcase_call_graph"], graph, "heal_subcase graph")
    # The graph is source-derived metadata, not a pool measurement. The original
    # pool block and all of its measured counts remain unchanged on disk.
    pool = copy.deepcopy(old154["pool"])
    pool["heal_subcase_call_graph"] = graph
    verdicts = c154.build_verdicts(pool)
    _require_address_only(old154["verdicts"], verdicts, "C154 verdicts")
    return {
        "schema": "pokezero.current-census-citations.v1",
        "scope": "CURRENT_SOURCE_ADDRESSES_ONLY_HISTORICAL_MEASUREMENTS_UNCHANGED",
        "scientific_admission": False,
        "pool_remeasured": False,
        "historical_sha256": dict(HISTORICAL),
        "source_sha256": {path: _sha((ROOT / path).read_bytes()) for path in INPUTS},
        "c153_cannot_reach": cannot_reach,
        "c153_structural": structural,
        "c154_verdicts": verdicts,
        "heal_subcase_call_graph": graph,
    }


def verify_supplement(document: dict) -> dict:
    if document != build_supplement():
        raise ValueError("current-source citation supplement is stale or altered")
    return document


def load_verified_supplement() -> dict:
    return verify_supplement(json.loads((ROOT / ARTIFACT).read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="create the distinct supplement exclusively")
    args = parser.parse_args()
    if args.write:
        document = build_supplement()
        with (ROOT / ARTIFACT).open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
        print(f"created {ARTIFACT}; historical evidence not rewritten")
    else:
        load_verified_supplement()
        print("current-source citations verified; historical measurements unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
