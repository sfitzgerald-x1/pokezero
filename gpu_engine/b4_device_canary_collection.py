"""Original-v4 canary collection from device public knowledge, never capture.

The original canary owns its 200k declaration and PPO schedule. This adapter
executes fixed chunks, spools literal PRE observations and actual decisions,
and publishes only natural terminal trajectories to the original v4 cache.
"""
from typing import NamedTuple
import hashlib
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as j
import numpy as np

from . import champion, transition, engine_device_trackers as trackers
from . import engine_device_observation as encoder, benchmark_device_selfplay as sampling
from . import source_provenance
from .engine_request_source import _gender
from pokezero.b4_fidelity_canary import require, file_sha, encoded, sha
from pokezero.observation import PokeZeroObservationV0, ObservationPerspective
from pokezero.trajectory import TrajectoryStep, BattleTrajectory
from pokezero.collection import RolloutRecord, rollout_record_to_dict, rollout_record_from_dict, write_rollout_record
from pokezero.env import TerminalState

RUNNING, COMPLETE, CAPPED, REFUSED = range(4)
STATUS = ('RUNNING', 'COMPLETE', 'CAPPED', 'REFUSED')


class Carry(NamedTuple):
    state: object
    public: object
    streams: object
    frames: object
    status: object
    winner: object


class Frames(NamedTuple):
    numeric: object
    categorical: object
    token_types: object
    attention: object
    legal: object
    turns: object
    requested: object
    actions: object
    probabilities: object
    values: object
    errors: object
    advanced: object


def make_collector(prepared, config, *, max_requests, chunk_steps=8):
    """One pure callable per current checkpoint/batch; all carry is current."""
    require(type(chunk_steps) is int and chunk_steps > 0 and type(max_requests) is int and max_requests > 0,
            'finite device chunk and request cap')
    require(config.tokens == 23 and config.numeric == 132 and config.categorical == 41,
            'untouched v4 public actor shape')
    table, observation_tables = prepared.engine_tables, prepared.observation_tables
    def advance(c, params, sampling_seed):
        state = c.state; b = len(c.streams); live = c.status == RUNNING
        state = state._replace(done=state.done | ~live)
        requested = sampling.request_seats(state) & live[:, None]
        masks = transition.action_mask(state, table)
        observations = [encoder.encode(state, trackers.readout(c.public, j.full(b, side, j.int32)),
            observation_tables, observer_sides=j.full(b, side, j.int32), legal_action_mask=masks[:, side])
            for side in (0, 1)]
        inputs = [encoder.champion_inputs(o) for o in observations]
        paired = {name: j.stack([x[name] for x in inputs], axis=1) for name in inputs[0]}
        out = champion.forward(params, **paired, c=config, compute_dtype='bfloat16')
        finite = j.all(j.isfinite(out.policy_logits), axis=-1) & j.isfinite(out.value) & j.all(j.isfinite(out.opponent_action_logits), axis=-1)
        legal = j.stack([o.legal_action_mask for o in observations], axis=1)
        errors = j.stack([o.error for o in observations], axis=1)
        logits = j.where(legal, j.nan_to_num(out.policy_logits, nan=0., posinf=0., neginf=0.), -j.inf)
        logits = j.where(j.any(legal, axis=-1)[..., None], logits, 0.)
        keys = sampling.sample_keys(c.streams, c.frames, sampling_seed=sampling_seed)
        actions = jax.vmap(lambda key, values: jax.random.categorical(key, values))(
            keys.reshape(-1, 2), logits.reshape(-1, 9)).reshape(b, 2).astype(j.int32)
        probabilities = j.take_along_axis(jax.nn.softmax(logits, axis=-1), actions[..., None], axis=-1)[..., 0]
        valid = finite & (errors == 0) & j.any(legal, axis=-1) & j.isfinite(probabilities) & (probabilities > 0)
        refused = live & j.any(requested & ~valid, axis=1)
        actual_actions = j.stack([encoder.map_actions(observations[side], state, j.full(b, side, j.int32), actions[:, side]) for side in (0, 1)], axis=1)
        stepped = transition.step_with_public(state._replace(done=state.done | refused), actual_actions,
            table, c.public, belief_tables=observation_tables.belief)
        after = stepped.step.state
        refused |= live & stepped.step.invalid_actions
        complete = live & ~refused & after.done
        frames = c.frames + live.astype(j.int32)
        capped = live & ~refused & ~complete & (frames >= max_requests)
        status = j.where(refused, REFUSED, j.where(complete, COMPLETE, j.where(capped, CAPPED, c.status)))
        new = Carry(after, stepped.trackers, c.streams, frames, status,
            j.where(complete, after.winner, c.winner))
        fields = Frames(j.stack([x['numeric'] for x in inputs], 1), j.stack([x['categorical'] for x in inputs], 1),
            j.stack([x['token_types'] for x in inputs], 1), j.stack([x['attention'] for x in inputs], 1), legal,
            c.public.turn, requested, actions, probabilities, out.value, errors, live & ~refused)
        return new, fields
    def chunk(carry, params, sampling_seed):
        return jax.lax.scan(lambda c, _: advance(c, params, sampling_seed), carry, None, length=chunk_steps)
    return jax.jit(chunk)


def _publish(path, blob):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(blob); f.flush(); os.fsync(f.fileno())
    path.chmod(0o444)
    with _directory(path.parent): pass


class _directory:
    def __init__(self, path): self.path = path
    def __enter__(self): self.fd = os.open(self.path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)); os.fsync(self.fd)
    def __exit__(self, *args): os.close(self.fd)


class Spool:
    """Only one chunk or one game's history is resident; arrays remain remote."""
    def __init__(self, root):
        self.root = Path(root); self.records = []
    def append(self, frames):
        folder = self.root / ('chunk-' + str(len(self.records))); folder.mkdir()
        record = {}
        for name, value in zip(Frames._fields, frames):
            path = folder / (name + '.npy')
            with path.open('xb') as f: np.save(f, np.swapaxes(value, 0, 1), allow_pickle=False)
            path.chmod(0o444)
            record[name] = dict(path=str(path), sha256=file_sha(path), shape=list(value.shape), dtype=value.dtype.str,
                identity=self._stat(path))
        _publish(folder / 'metadata.json', encoded(record)); self.records.append(record)
    def game(self, row):
        values = {name: [] for name in Frames._fields}
        for record in self.records:
            for name, description in record.items():
                path = Path(description['path'])
                # Owned read-only files were hashed at publication. Recheck
                # inode/size/mtime/ctime without rereading B rows for each game.
                require(self._stat(path) == description['identity'], 'literal device spool file changed')
                a = np.load(path, mmap_mode='r', allow_pickle=False)
                try:
                    require(a.shape == (description['shape'][1], description['shape'][0], *description['shape'][2:]) and a.dtype.str == description['dtype'], 'device spool geometry changed')
                    values[name].append(np.array(a[row], copy=True))
                finally: a._mmap.close()
                require(self._stat(path) == description['identity'], 'device spool changed during read')
        return Frames(*(np.concatenate(values[name], axis=0) for name in Frames._fields))
    @staticmethod
    def _stat(path):
        s = path.stat(); return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def complete_record(game, frames, *, winner, turn, duration, source_hash, checkpoint_sha, archive_sha):
    """Original v4 trajectory, preserving both actual seat-action labels."""
    require(winner in (0, 1, 2), 'literal natural terminal winner')
    steps = []
    for frame in range(len(frames.turns)):
        if not frames.advanced[frame]: continue
        for side in (0, 1):
            if not frames.requested[frame, side]: continue
            require(frames.errors[frame, side] == 0, 'bad view cannot enter completed PPO trajectory')
            def frozen(a):
                value = a.tolist()
                def fold(x): return tuple(fold(v) for v in x) if isinstance(x, list) else x
                return fold(value)
            observation = PokeZeroObservationV0(*(frozen(value[frame, side]) for value in
                (frames.categorical, frames.numeric, frames.token_types, frames.attention, frames.legal)),
                perspective=ObservationPerspective.from_showdown_slot('p' + str(side + 1), 'p' + str(side + 1)), schema_version='pokezero.observation.v4')
            other = 1 - side
            steps.append(TrajectoryStep('p' + str(side + 1), int(frames.turns[frame]), observation,
                tuple(observation.legal_action_mask), int(frames.actions[frame, side]),
                opponent_action_index=int(frames.actions[frame, other]) if frames.requested[frame, other] else None,
                action_probability=float(frames.probabilities[frame, side]), value_estimate=float(frames.values[frame, side]),
                metadata=dict(device_frame=frame, stream=game.stream_id, original_weights_archive_sha256=archive_sha)))
    from dataclasses import replace
    terminal = TerminalState(None if winner == 2 else 'p' + str(winner + 1), int(turn), False)
    for side in ('p1', 'p2'):
        indices = [i for i, s in enumerate(steps) if s.player_id == side]
        if indices:
            at = indices[-1]; reward = 0. if winner == 2 else 1. if side == terminal.winner else -1.
            steps[at] = replace(steps[at], reward=reward)
    trajectory = BattleTrajectory(game.battle_id, 'gen3randombattle', game.seed, steps, terminal,
        dict(source='device-public-v4-canary', checkpoint_sha256=checkpoint_sha))
    return RolloutRecord(game.battle_id, game.seed, 'gen3randombattle', {'p1': checkpoint_sha, 'p2': checkpoint_sha},
        max((s.turn_index for s in steps), default=0), duration, terminal, trajectory, source_hash)


class Backend:
    def __init__(self, prepared, pool, *, gender_policy, plan, model_config, chunk_steps=8):
        from .v5_device_backend import Prepared
        require(type(prepared) is Prepared and type(chunk_steps) is int and chunk_steps > 0, 'actual device preparation and bounded chunks')
        require(source_provenance._pool(pool, rehash=True)['pool_sha256'] == plan.pool_sha256 == prepared.pool_sha256, 'same declared pool')
        require(json.loads(prepared.source_binding_json)['source_commit'] == plan.source_commit and prepared.source_sha256 == plan.prepared_binding, 'same literal device source preparation')
        require(gender_policy is not None, 'same original gender policy required')
        self.prepared, self.pool, self.gender_policy, self.plan, self.model_config = prepared, pool, gender_policy, plan, model_config
        self.chunk_steps = chunk_steps; self.source_hash = prepared.catalog.source.source_hash
        self.collectors = {}
        self.initialize = jax.jit(lambda s: transition.initialize_with_public(s, prepared.engine_tables, belief_tables=prepared.observation_tables.belief))
    def _binding(self, game, checkpoint):
        return dict(stream=game.stream_id, battle_id=game.battle_id, seed=game.seed, team_indices=list(game.team_indices),
            pool_sha256=self.plan.pool_sha256, checkpoint_sha256=file_sha(checkpoint), source_commit=self.plan.source_commit,
            prepared_binding=self.plan.prepared_binding, source_hash=self.source_hash)
    def _cache(self, batch, checkpoint, path, dataset_config):
        from pokezero import dataset
        found = {}; records = path / ('complete-' + str(time.time_ns()) + '.jsonl')
        with records.open('x') as f:
            for game in batch.games:
                target = path / ('complete-' + str(game.stream_id) + '.json')
                if not target.exists(): continue
                x = json.loads(target.read_bytes())
                require(x['binding'] == self._binding(game, checkpoint) and x['ending']['status'] == 'COMPLETE' and sha(x['rollout']) == x['rollout_sha256'], 'durable original completion differs')
                record = rollout_record_from_dict(x['rollout'])
                require(not record.terminal.capped and record.seed == game.seed and record.battle_id == game.battle_id, 'original terminal trajectory identity')
                write_rollout_record(f, record); found[game.stream_id] = (x['ending'], sha(x))
        cache = None
        if found:
            cache = path / ('cache-' + str(time.time_ns()))
            dataset.write_training_cache_streaming(records, cache, config=dataset_config, collection_env='engine', belief_fidelity='full')
        return dict(games=len(batch.games), complete=len(found), unknown=len(batch.games) - len(found),
            cache=None if cache is None else str(cache), cache_metadata_sha256=None if cache is None else file_sha(cache / 'metadata.json'),
            completed_sources={str(k): v[1] for k, v in found.items()}), found
    def recover(self, batch, checkpoint, path, *, dataset_config, on_end):
        path = Path(path)
        if not path.exists():
            for game in batch.games:
                on_end(game.stream_id,'UNKNOWN',0.,sha(dict(reason='attempted device batch missing; no replay',binding=self._binding(game,checkpoint))))
            return dict(status='INTERRUPTED_UNKNOWN', games=len(batch.games), complete=0, unknown=len(batch.games), cache=None, completed_sources={})
        report, found = self._cache(batch, checkpoint, path, dataset_config)
        dispositions={}
        for game in batch.games:
            if game.stream_id in found:
                ending, digest = found[game.stream_id]
                on_end(game.stream_id, 'COMPLETE', ending['duration_seconds'], digest)
                status='COMPLETE'
            else:
                target=path/('ending-'+str(game.stream_id)+'.json')
                if target.exists():
                    ending=json.loads(target.read_bytes());status=ending['status']
                    require(ending['stream']==game.stream_id and status in ('CAPPED','REFUSED','FAILED'),'literal nontraining ending')
                    on_end(game.stream_id,status,ending['duration_seconds'],ending['source_sha256'])
                else:
                    status='UNKNOWN'
                    on_end(game.stream_id,status,0.,sha(dict(reason='attempted device cell missing; no replay',binding=self._binding(game,checkpoint))))
            dispositions[status]=dispositions.get(status,0)+1
        return dict(report, status='RECOVERED_STORED_ATTEMPT',dispositions=dispositions)
    def collect(self, batch, checkpoint, path, *, dataset_config, on_end, check_progress):
        path = Path(path); path.mkdir(parents=True, exist_ok=False)
        archive = path / 'actor.npz'; receipt = champion.export(checkpoint, archive)
        params, config, metadata = champion.load(archive)
        require(config == champion.config_from_checkpoint(self.model_config.to_dict()) and receipt['checkpoint_sha256'] == file_sha(checkpoint), 'original v4 checkpoint/shape')
        indices = np.asarray([g.team_indices for g in batch.games], np.int32)
        raw = transition.state_from_pool(self.pool.arrays, indices, seed=batch.constructor_seed)
        raw = _gender(raw, self.pool, indices, dict(policy=self.gender_policy, seed=batch.gender_seed, domain=batch.gender_domain, root_ids=batch.gender_root_ids))
        initial = self.initialize(raw); b = len(batch.games)
        carry = Carry(initial.state, initial.trackers, j.asarray([g.stream_id for g in batch.games], j.int32),
            j.zeros(b, j.int32), j.where(initial.state.done, REFUSED, RUNNING), j.full(b, -1, j.int32))
        # Model weights and the original batch seed are dynamic inputs. A new
        # current checkpoint does not recreate the same pure chunk closure.
        if config not in self.collectors:
            self.collectors[config] = make_collector(self.prepared, config, max_requests=self.plan.max_requests, chunk_steps=self.chunk_steps)
        execute = self.collectors[config]
        sampling_seed = j.asarray(batch.sampling_seed, j.uint32)
        spool = Spool(path); closed = set(); started = time.monotonic()
        def close(row, status, reason=None):
            if row in closed: return
            game = batch.games[row]; duration = time.monotonic() - started
            ending = dict(status=status, duration_seconds=duration, executed_steps=int(carry.frames[row]), reason=reason)
            digest = sha(ending)
            if status == 'COMPLETE':
                record = complete_record(game, spool.game(row), winner=int(carry.winner[row]), turn=int(carry.state.turn[row]), duration=duration,
                    source_hash=self.source_hash, checkpoint_sha=file_sha(checkpoint), archive_sha=metadata['weights_archive_sha256'])
                body = rollout_record_to_dict(record)
                document = dict(binding=self._binding(game, checkpoint), ending=ending, rollout=body, rollout_sha256=sha(body))
                _publish(path / ('complete-' + str(game.stream_id) + '.json'), encoded(document)); digest = sha(document)
            _publish(path / ('ending-' + str(game.stream_id) + '.json'), encoded(dict(stream=game.stream_id, **ending, source_sha256=digest)))
            on_end(game.stream_id, status, duration, digest); closed.add(row)
        try:
            for _ in range((self.plan.max_requests + self.chunk_steps - 1) // self.chunk_steps):
                check_progress(); carry, frames = jax.block_until_ready(execute(carry, params, sampling_seed))
                host = jax.device_get(frames); spool.append(host)
                statuses = np.asarray(carry.status)
                for row in np.flatnonzero(statuses != RUNNING): close(int(row), STATUS[int(statuses[row])])
                if len(closed) == b: break
        except BaseException:
            for row in range(b): close(row, 'FAILED', 'interrupted device collection; no automatic replay')
            raise
        for row in range(b): close(row, 'CAPPED', 'finite request cap; no observed outcome')
        report, _ = self._cache(batch, checkpoint, path, dataset_config)
        return dict(report, status='COLLECTED', source='DEVICE_PUBLIC_V4_NO_CAPTURE',
            dispositions={name:int(np.sum(np.asarray(carry.status)==index)) for index,name in enumerate(STATUS)})
