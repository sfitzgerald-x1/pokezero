"""Fresh paired-seat observations and outcome-only, frozen-trunk value probes.

Public observations use separate owning-wire parsers. The privileged arm swaps
the peer's six own Pokémon rows and adds its current true move identities and
PP, using only the initial sets and requests available at that boundary.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np

CHAMPION_SHA = "0fd095923b4ac7e05d6e2b3ccab9c1e6869dff4893c2dae456caff10dce690be"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def write_new(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def cohort(root: str, count: int = 256) -> list[dict[str, Any]]:
    """Freeze roster and exact 3/4,1/8,1/8 grouped split before collection."""
    if not root or type(count) is not int or count < 8 or count % 8:
        raise ValueError("nonempty root and positive multiple-of-eight roster required")
    rows = []
    for index in range(count):
        battle_id = f"{root}-battle-{index:04d}"
        seed = int(digest([root, "showdown-seed", index])[:8], 16) % (2**31)
        rows.append(dict(index=index, battle_id=battle_id, seed=seed,
                         split_key=digest([root, "battle-split", battle_id])))
    if len({r["seed"] for r in rows}) != count:
        raise ValueError("seed collision; derive a new unused root before collection")
    for rank, row in enumerate(sorted(rows, key=lambda r: r["split_key"])):
        row["split"] = "train" if rank < count * 3 // 4 else "validation" if rank < count * 7 // 8 else "test"
    return rows


def swap_own_view(actor, peer, vocab, *, move_facts=None):
    """Copy raw own-team rows only, preserving every other actor input."""
    from .showdown import CATEGORY_ROLE
    if actor.schema_version != "pokezero.observation.v4" or peer.schema_version != actor.schema_version:
        raise ValueError("paired v4 observations required")
    if actor.perspective is None or peer.perspective is None or {
            actor.perspective.player_id, peer.perspective.player_id} != {"p1", "p2"}:
        raise ValueError("opposite explicit seats required")
    for label in ("pokemon:self", "pokemon:opponent"):
        if not vocab.is_enumerated(label):
            raise ValueError("enumerated Pokémon roles required")
    categories = np.array(actor.categorical_ids, copy=True)
    numeric = np.array(actor.numeric_features, copy=True)
    attention = np.array(actor.attention_mask, copy=True)
    if categories.shape != (23, 41) or numeric.shape != (23, 132):
        raise ValueError("exact champion input shape required")
    categories[7:13] = np.asarray(peer.categorical_ids)[1:7]
    numeric[7:13] = np.asarray(peer.numeric_features)[1:7]
    attention[7:13] = np.asarray(peer.attention_mask)[1:7]
    for index in range(7, 13):
        if attention[index]:
            if categories[index, CATEGORY_ROLE] != vocab.encode("pokemon:self"):
                raise ValueError("peer own row role mismatch")
            categories[index, CATEGORY_ROLE] = vocab.encode("pokemon:opponent")
    if move_facts is not None:
        from .showdown import (CATEGORY_BELIEF_MOVE_OFFSET, BELIEF_MOVE_BUCKET_COUNT,
            NUMERIC_OPP_MOVE_PP_OFFSET, NUMERIC_REVEALED_MOVE_COUNT,
            NUMERIC_POSSIBLE_MOVE_COUNT)
        if len(move_facts) != 6:
            raise ValueError("six peer move records required")
        for index, facts in enumerate(move_facts, 7):
            categories[index, CATEGORY_BELIEF_MOVE_OFFSET:CATEGORY_BELIEF_MOVE_OFFSET + BELIEF_MOVE_BUCKET_COUNT] = 0
            numeric[index, NUMERIC_OPP_MOVE_PP_OFFSET:NUMERIC_OPP_MOVE_PP_OFFSET + BELIEF_MOVE_BUCKET_COUNT] = 0
            for bucket, (move, pp_fraction) in enumerate(sorted(facts.items())):
                label = "belief:possible_move:" + move
                if bucket >= BELIEF_MOVE_BUCKET_COUNT or not vocab.is_enumerated(label):
                    raise ValueError("true move must have an exact champion vocabulary entry: " + label)
                categories[index, CATEGORY_BELIEF_MOVE_OFFSET + bucket] = vocab.encode(label)
                numeric[index, NUMERIC_OPP_MOVE_PP_OFFSET + bucket] = pp_fraction
            numeric[index, NUMERIC_REVEALED_MOVE_COUNT] = len(facts)
            numeric[index, NUMERIC_POSSIBLE_MOVE_COUNT] = len(facts)
    return replace(actor, categorical_ids=tuple(tuple(map(int, r)) for r in categories),
                   numeric_features=tuple(tuple(map(float, r)) for r in numeric),
                   attention_mask=tuple(map(bool, attention)),
                   metadata={**actor.metadata, "privileged_own_view_swap": True})


def true_move_facts(state, initial_request, dex):
    """Exact current moves, with active PP from its request and base PP ledger.

    Transform copies are request-scoped (including the 5-PP denominator), never
    retained after switch-out. Benched base moves use the initial set and the
    public PP-charge ledger, which excludes called/copied moves and counts
    Pressure. No future request or future reveal enters these facts.
    """
    from .showdown import _normalize_identifier
    from .tier2 import canonical_move_id
    from .local_showdown import _base_move_id
    def key(value):
        return _normalize_identifier(canonical_move_id(str(value)))
    initial = {str(r["ident"]): r for r in initial_request["side"]["pokemon"]}
    current = state.request["side"]["pokemon"]
    if len(initial) != 6 or len(current) != 6 or len(state.self_team) != 6:
        raise ValueError("six unique initial and current own identities required")
    exact = {_normalize_identifier(b.species): b for b in state.belief_view.self_pokemon}
    active = state.request.get("active", [])
    active_moves = active[0].get("moves", []) if active else []
    result = []
    for candidate, row in zip(state.self_team, current):
        if candidate.ident != row["ident"] or row["ident"] not in initial:
            raise ValueError("peer token/request identity alignment")
        belief = exact.get(_normalize_identifier(candidate.species))
        transformed = bool(belief and belief.transformed and row.get("active"))
        source = row if transformed else initial[row["ident"]]
        moves = {key(m): m for m in source["moves"]}
        if not moves or len(moves) > 4:
            raise ValueError("one to four true moves required")
        uses = dict(belief.move_uses) if belief else {}
        facts = {}
        for move in sorted(moves):
            info = dex.move_info(move)
            if info is None or info.max_pp <= 0:
                raise ValueError("true move dex PP required: " + move)
            if transformed:
                # Every copied active slot must be confirmed by this request.
                pp = None
            else:
                charged = sum(n for m, n in uses.items() if _base_move_id(m) == _base_move_id(move))
                pp = max(0, info.max_pp - charged) / info.max_pp
            if row.get("active"):
                matches = [m for m in active_moves if "pp" in m and
                           _base_move_id(m.get("id", "")) == _base_move_id(move)]
                if len(matches) == 1:
                    slot = matches[0]
                    if slot.get("maxpp", 0) <= 0 or not 0 <= slot["pp"] <= slot["maxpp"]:
                        raise ValueError("invalid request PP")
                    pp = slot["pp"] / slot["maxpp"]
            if pp is None:
                raise ValueError("copied move PP absent from current peer request")
            facts[move] = float(pp)
        result.append(facts)
    return result


def make_owning_wire_env(config):
    """Collector-only driver retaining literal streams and separate public folds."""
    from .local_showdown import LocalShowdownEnv, _decode_request_line
    from .showdown import _ReplayParser
    class OwningWireEnv(LocalShowdownEnv):
        def __init__(self, config):
            super().__init__(config)
            self.owning_lines = {s: [] for s in ("p1", "p2")}
            self.public_observers = {s: LocalShowdownEnv(config) for s in ("p1", "p2")}
            for seat, observer in self.public_observers.items():
                observer._parser = _ReplayParser("mirror-owning-wire", complete_prefix=True,
                    hp_visibility={s: "exact" if s == seat else "percentage" for s in ("p1", "p2")})

        def _apply_event(self, event):
            # Keep every literal player line, including private side updates,
            # blanks and terminal lines. Never project an omniscient stream.
            seat = event.get("stream")
            if (event.get("type") == "stream" and seat in self.owning_lines and
                    event.get("battleId") == self._battle_token):
                observer = self.public_observers[seat]
                for raw in event["lines"]:
                    line = str(raw)
                    self.owning_lines[seat].append(line)
                    observer._lines.append(line)
                    if line.startswith("|request|"):
                        request = _decode_request_line(line)
                        if request.get("side", {}).get("id") != seat:
                            raise ValueError("literal player request ownership mismatch")
                        observer._latest_requests[seat] = request
                        observer._first_requests.setdefault(seat, request)
                        observer._request_history[seat].append(json.loads(json.dumps(request)))
                observer._battle_id = self._battle_id
            return super()._apply_event(event)

        def observe(self, player):
            return self.public_observers[player].observe(player)

        def peer_state(self, player):
            return self.public_observers[player]._state_for_player(player)

    return OwningWireEnv(config)


def champion_linear_head(model):
    """Copy the actual champion layer, preserving its masked mean and tanh."""
    from .neural_policy import require_torch
    torch = require_torch()
    if (not isinstance(model.value_head, torch.nn.Linear) or model.value_head.out_features != 1
            or model.config.value_activation != "tanh"):
        raise ValueError("linear tanh champion required")
    from .neural_policy import _masked_mean
    class ChampionHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = torch.nn.Linear(model.value_head.in_features, 1)

        def forward(self, tokens, valid):
            pooled = tokens if tokens.ndim == 2 else _masked_mean(tokens, valid)
            return torch.tanh(self.network(pooled).squeeze(-1)), None
    head = ChampionHead().float()
    head.network.load_state_dict(model.value_head.state_dict(), strict=True)
    return head


def terminal_labels(terminal: dict | None, seats) -> tuple[np.ndarray, np.ndarray]:
    """Real uncapped terminal draws are zero; caps/absent terminals stay NaN."""
    known = terminal is not None and not terminal["capped"]
    if known and terminal["winner"] not in (None, "p1", "p2"):
        raise ValueError("invalid terminal winner")
    y = np.full(len(seats), np.nan, np.float32)
    status = np.full(len(seats), 0 if known else 1, np.uint8)
    if known:
        y[:] = [0 if terminal["winner"] is None else 1 if seat == terminal["winner"] else -1 for seat in seats]
    return y, status


def model_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(canonical([name, array.shape, array.dtype.str])); h.update(array.tobytes())
    return h.hexdigest()


def load_runtime(checkpoint, showdown_root, node, threads=2):
    from .local_showdown import LocalShowdownConfig, env_config_from_checkpoint_provenance
    from .neural_policy import (require_torch, load_transformer_checkpoint,
        feature_masks_from_model_config, observation_spec_from_model_config, category_vocab_from_model_config)
    from .randbat import load_gen3_randbat_source_cached
    from .value_precision_probe import checkpoint_contract
    torch = require_torch()
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    contract = checkpoint_contract(checkpoint)
    if contract["checkpoint_sha256"] != CHAMPION_SHA or contract["value_calibration_transform"] is not None:
        raise ValueError("exact uncalibrated frozen champion required")
    model, result = load_transformer_checkpoint(checkpoint, map_location="cpu")
    model.float().eval()
    cfg = result.model_config
    vocab = category_vocab_from_model_config(cfg, showdown_root)
    env_config = env_config_from_checkpoint_provenance(
        LocalShowdownConfig(showdown_root=showdown_root, node_binary=node, set_belief_source=True),
        feature_masks_from_model_config(cfg), context="fresh mirror value probe",
        required_specs=observation_spec_from_model_config(cfg), required_vocabs=vocab)
    source = load_gen3_randbat_source_cached(showdown_root)
    if source.metadata.source_hash != contract["belief_set_source_hash"]:
        raise ValueError("Showdown belief source differs from champion")
    if file_sha(checkpoint) != CHAMPION_SHA:
        raise ValueError("checkpoint changed during loading")
    return model, result, vocab, env_config, contract


def collect_battle(row, output, runtime, *, max_rounds=300):
    """No branch/replay: once through a fresh battle; capture before selection."""
    from .dex import load_showdown_dex_cached
    from .neural_policy import TransformerSoftmaxPolicy, observation_windows_to_torch
    from .value_precision_probe import frozen_token_features
    model, result, vocab, env_config, contract = runtime
    dest = Path(output) / f"battle-{row['index']:04d}"
    dest.mkdir(exist_ok=False)
    before = model_digest(model)
    policy = TransformerSoftmaxPolicy(model, result, deterministic=False, sampling_temperature=1.,
                                      exploration_epsilon=0., device="cpu")
    rngs = {seat: random.Random(int(digest([row["battle_id"], seat, "action-rng"])[:16], 16)) for seat in ("p1", "p2")}
    env = make_owning_wire_env(env_config)
    dex = load_showdown_dex_cached(env_config.resolved_showdown_root())
    public, privileged, identities = [], [], []
    terminal = None
    try:
        env.reset(seed=row["seed"], format_id="gen3randombattle")
        initial_requests = json.loads(json.dumps(env._latest_requests))
        if set(initial_requests) != {"p1", "p2"} or any(
            len(initial_requests[s].get("side", {}).get("pokemon", [])) != 6 for s in initial_requests):
            raise ValueError("complete initial true-set requests for both seats required")
        protocol_count = 0
        owning_counts = {s: 0 for s in ("p1", "p2")}
        with gzip.open(dest / "boundaries.jsonl.gz", "xt") as log:
            for round_index in range(max_rounds):
                terminal = env.terminal()
                if terminal is not None:
                    break
                requested = tuple(env.requested_players())
                if not requested:
                    raise ValueError("no requested seats before terminal")
                observations = {seat: env.observe(seat) for seat in requested}
                lines = tuple(env.protocol_lines)
                boundary = dict(schema="pokezero.mirror_boundary.v1", battle_id=row["battle_id"],
                    seed=row["seed"], decision_round_index=round_index, requested_players=requested,
                    protocol_start=protocol_count, protocol_end=len(lines),
                    protocol_delta=lines[protocol_count:], public_prefix_sha256=digest(lines),
                    requests=json.loads(json.dumps(env._latest_requests)),
                    paired=set(requested) == {"p1", "p2"})
                boundary["root_id"] = digest([row["battle_id"], round_index, boundary["public_prefix_sha256"], boundary["requests"]])
                boundary["owning_wire"] = {s: dict(start=owning_counts[s], end=len(env.owning_lines[s]),
                    delta=env.owning_lines[s][owning_counts[s]:], prefix_sha256=digest(env.owning_lines[s])) for s in owning_counts}
                for seat, obs in observations.items():
                    if obs.perspective.player_id != seat:
                        raise ValueError("observation seat mismatch")
                    # All requested rows can support later source-encoder replay;
                    # A2 fits only simultaneous pairs, never a stale opposite view.
                    boundary.setdefault("observations", {})[seat] = {
                        name: np.asarray(getattr(obs, name)).tolist() for name in
                        ("categorical_ids", "numeric_features", "token_type_ids", "attention_mask", "legal_action_mask")}
                    boundary["observations"][seat]["turn_number"] = obs.metadata.get("turn_number")
                if boundary["paired"]:
                    turns = {o.metadata.get("turn_number") for o in observations.values()}
                    if len(turns) != 1:
                        raise ValueError("paired observations disagree on turn")
                    for seat in ("p1", "p2"):
                        actor, peer = observations[seat], observations["p2" if seat == "p1" else "p1"]
                        peer_seat = peer.perspective.player_id
                        facts = true_move_facts(env.peer_state(peer_seat), initial_requests[peer_seat], dex)
                        public.append(actor); privileged.append(swap_own_view(actor, peer, vocab, move_facts=facts))
                        identities.append(dict(root_id=boundary["root_id"], seat=seat,
                            turn_number=actor.metadata.get("turn_number"), decision_round_index=round_index))
                actions = {seat: policy.select_action(obs, rng=rngs[seat]).action_index for seat, obs in observations.items()}
                boundary["actions"] = actions
                log.write(json.dumps(boundary, sort_keys=True, separators=(",", ":")) + "\n")
                log.flush()
                protocol_count = len(lines)
                owning_counts = {s: len(env.owning_lines[s]) for s in owning_counts}
                env.step(actions)
            terminal = env.terminal()
            # Preserve last transition and terminal protocol, including a genuine tie.
            log.write(json.dumps(dict(event="end", protocol_start=protocol_count,
                protocol_end=len(env.protocol_lines), protocol_delta=env.protocol_lines[protocol_count:],
                actual_terminal=None if terminal is None else asdict(terminal),
                battle_id=row["battle_id"],
                owning_wire={s: dict(start=owning_counts[s], end=len(env.owning_lines[s]),
                    delta=env.owning_lines[s][owning_counts[s]:], prefix_sha256=digest(env.owning_lines[s])) for s in owning_counts})) + "\n")
    finally:
        env.close()
    if not public:
        raise ValueError("battle has no simultaneous paired decision")
    terminal_dict = None if terminal is None else asdict(terminal)
    y, status = terminal_labels(terminal_dict, [i["seat"] for i in identities])
    arrays = {"outcomes": y, "label_status": status}
    for arm, observations in (("public", public), ("privileged", privileged)):
        tokens, values = [], []
        for start in range(0, len(observations), 128):
            batch = observations[start:start + 128]
            tensors = observation_windows_to_torch([[o] for o in batch], window_size=1, device="cpu")
            encoded, prediction = frozen_token_features(model, tensors)
            tokens.append(encoded.cpu().numpy()); values.append(prediction.value.cpu().numpy())
        arrays[arm + "_tokens"] = np.concatenate(tokens)
        arrays[arm + "_values"] = np.concatenate(values)
        arrays[arm + "_mask"] = np.array([o.attention_mask for o in observations], bool)
        from .neural_policy import _masked_mean, require_torch
        torch = require_torch()
        arrays[arm + "_pooled"] = _masked_mean(torch.from_numpy(arrays[arm + "_tokens"]),
            torch.from_numpy(arrays[arm + "_mask"])).numpy()
    if model_digest(model) != before:
        raise ValueError("champion state changed")
    with (dest / "features.npz").open("xb") as stream:
        np.savez(stream, **arrays)
    receipt = dict(schema="pokezero.mirror_battle.v1", **row, contract=contract,
        paired_rows=len(identities), identities=identities, actual_terminal=terminal_dict,
        known_rows=int((status == 0).sum()), unknown_rows=int((status != 0).sum()),
        initial_true_requests=initial_requests, initial_true_requests_sha256=digest(initial_requests),
        public_view_source="literal owning player wire plus own request only; separate production parser/belief/annotation histories",
        privileged_move_source="current own request, initial true sets, boundary-local own PP charge ledger",
        frozen_model_sha256=before, observation_roots_are_siblings=False,
        files={p.name: file_sha(p) for p in sorted(dest.iterdir())})
    write_new(dest / "battle.json", receipt)
    return {k: v for k, v in receipt.items() if k not in ("identities", "initial_true_requests", "contract")}


def paired_battle_interval(public_losses, privileged_losses, *, seed=20261029, samples=10000):
    """Cluster at whole battle, averaging positions/seats within each battle."""
    difference = np.asarray(privileged_losses, np.float64) - np.asarray(public_losses, np.float64)
    if difference.ndim != 1 or len(difference) < 2 or not np.isfinite(difference).all():
        raise ValueError("at least two paired finite battle losses required")
    rng = np.random.default_rng(seed)
    estimates = difference[rng.integers(0, len(difference), (samples, len(difference)))].mean(1)
    return dict(privileged_minus_public_mse=float(difference.mean()), battles=len(difference),
        interval90=np.quantile(estimates, [.05, .95]).tolist(),
        public_equal_battle_mse=float(np.mean(public_losses)),
        privileged_equal_battle_mse=float(np.mean(privileged_losses)),
        method="percentile paired bootstrap over battles; equal battle weight")
