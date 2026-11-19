"""Bind the frozen root and input rows used by the Native prefix dispatcher.

This module reads the sealed constructor and action/chance inputs. Its command
schedule fixes the extent of the future replay. The dispatcher must still
derive the Native choices and phase from admitted causal and transport evidence.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from . import device_corpus_attestation as attestation
from .fresh200_native import CHANCE_FIELDS, initial_packet
from .fresh200_native_constructor import FROZEN_FIELDS, literal_teams
from .fresh200_slices import SealedInputs
from .fresh200_storage import INTEGRATION, SealedStorage
from .fresh200_witness_audit import document


PREFIX_STEPS = 8
FORMAT = 'gen3customgame@@@sleepclausemod'
CAPTURE_SHA = 'f6c606ffbce81b0b43351d88188c448d5b54a16334e6f75c20681aca2800b13a'
PACKER_SHA = '1c260f8f9d251f16c743ca1361041d3bbc6761869943decc9d5470cfe781c77d'
GAME_IDENTITY = ('original_ordinal', 'stream_id', 'battle_id', 'seed', 'team_indices')


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _wire(value):
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _manifest(root, reference, source, max_bytes=8 * 1024 ** 2):
    wire = attestation.verify_artifact(root, reference, max_bytes=max_bytes)
    value = document(wire.decode('utf8'))
    _need(type(value) is dict and set(value) == {'schema', 'kind', 'source', 'parents', 'payload'}
          and value['schema'] == attestation.SCHEMA and value['kind'] == 'manifest'
          and value['source'] == source, 'source-bound capture manifest required')
    return value


@dataclass(frozen=True)
class PrefixCommand:
    phase: str
    frame: int | None


def prefix_schedule():
    commands = [PrefixCommand('start', None), PrefixCommand('snapshot', 0)]
    for frame in range(PREFIX_STEPS):
        commands.extend((PrefixCommand('choices', frame), PrefixCommand('snapshot', frame + 1)))
    commands.extend((PrefixCommand('end', None), PrefixCommand('close', None)))
    return tuple(commands)


@dataclass(frozen=True)
class FrozenStep:
    step: int
    frame: int
    requested: tuple[bool, bool]
    actions: tuple[int, int]
    chance_wire: bytes

    def fresh_chance(self):
        return document(self.chance_wire.decode('utf8'))


@dataclass(frozen=True)
class FrozenPrefix:
    row: int
    root_index: int
    owner: int
    stream_id: int
    original_ordinal: int
    battle_id: str
    steps: tuple[FrozenStep, ...]
    start_wire: bytes
    initial_chance_wire: bytes
    team_wire: bytes
    binding_wire: bytes


def _game_matches(address, game):
    return all(key in address and address[key] == value for key, value in game.items())


def _selected_game(capture, construction, rows):
    selected = capture['selected']
    _need(len(selected) == 200, 'frozen 200-root roster required')
    address = selected[0]
    row = address['row']
    _need(type(row) is int and 0 <= row < rows, 'selected root row is out of bounds')
    _need(address['root_index'] == 0 and address['step'] == PREFIX_STEPS
          and address['frame'] == PREFIX_STEPS and address['live'] is True, 'selected root is not PRE8')
    owner = address['owner']
    _need(type(owner) is int and owner in (0, 1) and address['requested'][owner] is True,
          'frozen owner lacks its PRE8 request')
    game = construction['games'][row]
    _need(set(game) == set(GAME_IDENTITY) and _game_matches(address, game), 'selected root and construction differ')
    _need(type(game['stream_id']) is int and game['stream_id'] > 0
          and type(game['original_ordinal']) is int and game['original_ordinal'] >= 0
          and type(game['seed']) is int and type(game['battle_id']) in (str, int)
          and bool(str(game['battle_id'])), 'literal sealed game identity required')
    return address, game


def _read_steps(root, source, storage, capture, game, row, construction_reference, initial_link, selected):
    stream = game['stream_id']
    addresses = {}
    for address in capture['addresses']:
        if address['row'] == row and address['step'] <= PREFIX_STEPS:
            _need(address['step'] not in addresses, 'duplicate frozen prefix address')
            _need(address['live'] is True and _game_matches(address, game), 'frozen prefix address diverges from game')
            addresses[address['step']] = address
    _need(set(addresses) == set(range(PREFIX_STEPS + 1)), 'frozen prefix address is missing')
    _need(all(selected[key] == value for key, value in addresses[PREFIX_STEPS].items()),
          'PRE8 address differs from the selected root')
    steps = []
    source_inputs = []
    previous_link = construction_reference
    for chunk in capture['chunks']:
        start, count = chunk['start_step'], chunk['steps']
        _need(type(start) is int and start == len(steps) and type(count) is int and 1 <= count <= 4
              and start + count <= PREFIX_STEPS, 'action/chance chunk is outside the frozen prefix')
        link = initial_link if start == 0 else _manifest(root, chunk['link'], source)
        _need(link['parents'] == [previous_link, chunk['arrays'], chunk['replay_inputs']]
              and link['payload']['start_step'] == start and link['payload']['steps'] == count,
              'replay input link differs from the frozen capture')
        reader = SealedInputs(root, chunk['replay_inputs'], storage, source=source, kind='replay',
                              start_step=start, steps=count)
        source_inputs.append(reader.identity())
        for relative in range(count):
            record = reader.replay_inputs(relative, row)
            index = start + relative
            address = addresses[index]
            requested = tuple(record['requested'].tolist())
            actions = tuple(record['actions'].tolist())
            frame = int(record['frames'])
            _need(int(record['streams']) == stream and frame == index == address['frame']
                  and int(record['status']) == 0 and list(requested) == address['requested'],
                  'action/chance row has a different stream, frame, status or request mask')
            _need(len(requested) == len(actions) == 2
                  and all(type(flag) is bool for flag in requested) and any(requested)
                  and all(type(action) is int and (not flag or 0 <= action < 9)
                          for flag, action in zip(requested, actions)),
                  'frozen action row is outside the nine physical lanes')
            chance = record['chance']
            _need(set(chance) == set(CHANCE_FIELDS), 'frozen chance row has missing fields')
            chance_wire = _wire({name: chance[name].tolist() for name in CHANCE_FIELDS})
            steps.append(FrozenStep(index, frame, requested, actions, chance_wire))
        previous_link = chunk['link']
        if len(steps) == PREFIX_STEPS:
            return tuple(steps), source_inputs
    raise ValueError('frozen action/chance prefix is incomplete')


def _pack(teams, *, node, native_root):
    inputs = _wire(teams)
    packer = Path(__file__).with_name('fresh200_pack_teams.cjs')
    result = subprocess.run([str(node), str(packer), str(native_root)], input=inputs,
                            capture_output=True, check=True, timeout=30)
    packed = document(result.stdout.decode('utf8'))
    evidence = packed['evidence']
    _need(evidence['input_wire_sha256'] == _sha(inputs) and evidence['native_serializer_sha256'] == PACKER_SHA
          and evidence['node'] == 'v22.22.2' and evidence['arch'] == 'arm64'
          and evidence['literal_sets_unchanged'] is True and evidence['no_RNG'] is True,
          'pinned literal team serializer identity differs')
    values = packed['packed']
    _need(type(values) is list and len(values) == 2 and all(type(team) is str and team for team in values)
          and evidence['packed'] == [
              dict(bytes=len(team.encode('utf8')), sha256=_sha(team.encode('utf8'))) for team in values
          ], 'packed team bytes differ from their evidence')
    return inputs, values, evidence


def bind_frozen_prefix(*, capsule, reference, source, storage, pool_path, node, native_root):
    """Bind selected[0] and its eight input rows without opening expected state."""
    _need(type(storage) is SealedStorage and storage.rows == 256
          and reference.get('sha256') == CAPTURE_SHA and source['engine_commit'] == INTEGRATION,
          'the original frozen fresh200 capture and source layout are required')
    capture = _manifest(capsule, reference, source)['payload']
    _need(capture['schema'] == 'pokezero.fresh200.capture.v1' and capture['status'] == 'CAPTURED_ONLY'
          and capture['actual_horizon'] == 64 and capture['selection_error'] is None,
          'complete sealed capture required')
    initial_link = _manifest(capsule, capture['chunks'][0]['link'], source)
    construction_ref = initial_link['parents'][0]
    _need(construction_ref['path'] == 'construction.json', 'constructor manifest address differs')
    construction = _manifest(capsule, construction_ref, source)
    _need(capture['constructor_input'] in construction['parents'], 'constructor input is missing from sealed construction')
    selected, game = _selected_game(capture, construction['payload'], storage.rows)
    row = selected['row']
    steps, replay_inputs = _read_steps(capsule, source, storage, capture, game, row, construction_ref, initial_link, selected)

    reader = SealedInputs(capsule, capture['constructor_input'], storage, source=source, kind='constructor')
    constructor = reader.constructor(row)
    _need(int(constructor['streams']) == game['stream_id'], 'constructor stream differs from the selected game')
    raw = SimpleNamespace(**{key: np.expand_dims(value, 0) for key, value in constructor['raw'].items()})
    indices = game['team_indices']
    _need(type(indices) is list and len(indices) == 2 and all(type(index) is int and index >= 0 for index in indices),
          'frozen pool team addresses required')
    _need(Path(pool_path).resolve() == Path(storage.binding['pool_path']).resolve(), 'team pool path differs from admitted storage')
    with np.load(pool_path, allow_pickle=False) as archive:
        pool = {key: archive[key] for key in (*FROZEN_FIELDS, 'nature', 'ivs', 'evs')}
        metadata = document(archive['metadata_utf8'].tobytes().decode('utf8'))
        teams = literal_teams(pool, metadata, raw, row=0, team_indices=indices)

    team_wire, packed, packing = _pack(teams, node=node, native_root=native_root)
    initial = _wire(initial_packet(raw, np.expand_dims(constructor['initial_tie'], 0), row=0))
    words = constructor['raw']['keys']
    _need(words.shape == (2,) and words.dtype == np.dtype('uint32'), 'original two constructor key words required')
    seed = ','.join(str(part) for word in map(int, words) for part in (word >> 16, word & 65535))
    battle_id = str(game['battle_id'])
    start = _wire(dict(type='start', battleId=battle_id, formatid=FORMAT, seed=seed,
                       players={owner: dict(name=owner, team=team) for owner, team in zip(('p1', 'p2'), packed)}))
    binding = _wire(dict(
        schema='pokezero.fresh200.fixed_prefix_inputs.v1', qualification=False,
        capture_reference=reference, source=source, selected_root=selected,
        game=game, construction=construction_ref, constructor_input=reader.identity(),
        replay_inputs=replay_inputs, packing=packing, initial_sha256=_sha(initial), start_sha256=_sha(start),
        steps=[dict(step=x.step, frame=x.frame, requested=x.requested, actions=x.actions,
                    chance_sha256=_sha(x.chance_wire)) for x in steps],
        schedule=[vars(command) for command in prefix_schedule()],
    ))
    return FrozenPrefix(row, selected['root_index'], selected['owner'], game['stream_id'],
                        game['original_ordinal'], battle_id, steps, start, initial, team_wire, binding)
