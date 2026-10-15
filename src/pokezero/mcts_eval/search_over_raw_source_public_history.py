"""Public history from an original, transport-joined source boundary only.

Never use a materialized shell log or the old omniscient Python replay as a
prefix. This synchronous projection accesses only the original battle's log;
no original-source request, team, or Pokemon fields enter the returned value.
It is an input prerequisite, not leaf labels or engine qualification.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

from .search_over_raw import digest, require
from .search_over_raw_protocol_channels import FILTER_SHA256,MAXIMUM_BYTES,channel_zero_lines,channel_lines
from . import search_over_raw_protocol_channels as _channel_kernel

SCHEMA = "pokezero.search-over-raw.original-source-public-history.v1"


def _filter_binding(runtime):
    path = Path(runtime["argv"][3]) / "dist/sim/battle.js"
    state_path = path.with_name("state.js")
    paths = (path, state_path, Path(__file__).resolve(),Path(_channel_kernel.__file__).resolve())
    require(all(str(p) in runtime["files"] and hashlib.sha256(p.read_bytes()).hexdigest()
        == runtime["files"][str(p)] for p in paths), "original source/filter runtime bytes drifted")
    text = path.read_text()
    start = text.find("const splitRegex = ")
    stop = text.find("\nclass Battle {", start)
    require(start >= 0 and stop > start and hashlib.sha256(text[start:stop].encode()).hexdigest()
        == FILTER_SHA256 and "state.log = battle.log;" in state_path.read_text(),
        "unsupported pinned source-log serializer/channel filter")
    return dict(filter_function_sha256=FILTER_SHA256,
        files={str(p): runtime["files"][str(p)] for p in paths})


def archived_public_history(archive, root, join, runtime):
    """Only called inside an already validated TrustedOwnedSource handle.

    The independent source reader supplies ``join``. All private/source owner
    guards remain active in the caller before and after this projection.
    """
    from .search_over_raw_archive import SealedSourceArchive, retained_archive_root_identity
    require(type(archive) is SealedSourceArchive and type(root) is dict and type(join) is dict,
        "exact original source archive/root/snapshot join required")
    index = root["source_request_index"]
    require(type(index) is int and join["source_request_index"] == index
        and join["retained_archive_root_sha256"] == retained_archive_root_identity(archive, index),
        "original source root identity differs from received boundary join")
    bridge = archive._private[index].bridge_snapshot
    require(digest(bridge) == join["source_snapshot_sha256"] and type(bridge.get("battle")) is dict,
        "original source snapshot changed before public projection")
    log = bridge["battle"].get("log")
    binding = _filter_binding(runtime)
    public = channel_zero_lines(log)
    # This is a full reset-to-boundary prefix, not a truncated/attach-midstream
    # suffix. Without start/player evidence, parser coverage stays unavailable.
    require(any(line in {"|start", "|start|"} for line in public)
        and all(any(line.startswith("|player|" + seat + "|") for line in public) for seat in ("p1", "p2")),
        "LABEL_INPUT_UNAVAILABLE: source reset-to-boundary public prefix incomplete")
    require(not any(line.startswith("|win|") or line in {"|tie", "|tie|"} for line in public),
        "terminal log cannot masquerade as a selected pre-step source root")
    result = dict(schema=SCHEMA, root_id=root["root_id"], public_record_sha256=root["public_record_sha256"],
        source_join=deepcopy(join), runtime_sha256=digest(runtime), channel=0, filter_binding=binding,
        public_lines=public, public_lines_sha256=digest(public), complete_reset_prefix=True,
        old_omniscient_fold_used=False, original_source_requests_used=False,
        private_source_snapshot_included=False, labels_produced=False, runtime_qualified=False)
    require(digest(bridge) == join["source_snapshot_sha256"]
        and join["retained_archive_root_sha256"] == retained_archive_root_identity(archive, index)
        and _filter_binding(runtime) == binding, "source/filter changed during public projection")
    return result


def rebuild_public_fold(history, *, expected_history_sha256):
    """Rebuild public parser/belief history, deliberately WITHOUT own requests.

    A later actor must supply its actual constructed own-request history and
    annotation configuration. This fold alone cannot encode/evaluate an actor.
    Never copy any old omniscient replay/belief state or private p2 history.
    """
    from ..showdown import parse_showdown_replay
    from ..belief import PublicBattleBeliefEngine
    require(type(history) is dict and digest(history) == expected_history_sha256
        and history.get("schema") == SCHEMA and type(history.get("channel")) is int
        and history["channel"] == 0 and history.get("complete_reset_prefix") is True
        and all(history.get(k) is False for k in ("old_omniscient_fold_used", "original_source_requests_used",
            "private_source_snapshot_included", "labels_produced", "runtime_qualified"))
        and digest(history.get("public_lines")) == history.get("public_lines_sha256"),
        "externally bound original source public history required")
    lines = history["public_lines"]
    require(channel_zero_lines(lines) == lines and not any(line.startswith("|split") for line in lines),
        "public fold contains unfiltered/private protocol")
    replay = parse_showdown_replay(lines, battle_id=history["root_id"], complete_prefix=True,
        hp_visibility={"p1": "percentage", "p2": "percentage"})
    require(replay.requests == {}, "source public fold acquired a private request")
    belief = PublicBattleBeliefEngine.from_events(replay.public_events, format_id="gen3randombattle")
    require(digest(history) == expected_history_sha256, "public history mutated while folding")
    return replay, belief
