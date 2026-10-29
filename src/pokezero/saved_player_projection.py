"""Source-derived player observation inputs from a flattened trusted transcript.

This is NOT captured player-wire reconstruction. The pinned collector omitted
own Struggle sideupdates. Their canonical omission has a separately guarded
production-consumer contract; no message or cross-stream ordering is invented.
Callers must attest the whole fixed source before using this module. The old
trusted transcript cursor and its stored observation references remain separate.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, is_dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

PINNED_SHOWDOWN_REVISION = 'f76228a1354b5d0f307ca2d16101294ad3a2308b'
PROJECTION_SCOPE = 'SOURCE_DERIVED_PLAYER_OBSERVATION_INPUT_UNMEASURED'
STRUGGLE_WARNING_POLICY = 'UNRECORDED_OWN_WARNING_NOT_FABRICATED_BOUNDARY_EQUIVALENCE_ONLY'
INTIMIDATE_HINT = 'In Gen 3, Intimidate does not activate if every target has a Substitute.'
SHARED_HINTS = frozenset({
    "In Gens 3-4, Knock Off only makes the target's item unusable; it cannot obtain a new item.",
    'Previously chosen switches continue in Gen 2-4 after a Pursuit target faints.',
    "Sleep Clause Mod prevents players from putting more than one of their opponent's Pokémon to sleep at a time",
})
RULES = frozenset({
    'Beat Up Nicknames Mod: Beat Up will not reveal any party members',
    'Endless Battle Clause: Forcing endless battles is banned',
    'Evasion Items Clause: Evasion items are banned',
    'Evasion Moves Clause: Evasion moves are banned',
    'HP Percentage Mod: HP is shown in percentages',
    'OHKO Clause: OHKO moves are banned',
    'Sleep Clause Mod: Limit one foe put to sleep',
    'Species Clause: Limit one of each Pokémon',
    'Switch Priority Clause Mod: Faster Pokémon switch first',
})
# Complete observed tag roster plus source-proved shared -item (base Trick
# emitter, data/moves.ts:19906/19912), needed by the consumer-clock control.
# This is not an assertion that every payload is consumed. Only source-proved
# split-sensitive families below are transformed.
SOURCE_KINDS = frozenset('''-ability -activate -anim -boost -clearallboost -clearnegativeboost
    -crit -curestatus -cureteam -damage -end -enditem -fail -fieldactivate -heal -hint
    -hitcount -immune -item -message -miss -mustrecharge -prepare -resisted -setboost -sethp
    -sideend -sidestart -singlemove -singleturn -start -status -supereffective -transform
    -unboost -weather cant drag faint gametype gen move player rule start switch t:
    teamsize tie tier turn upkeep'''.split()) | {''}
_IDENT = re.compile(r'^(p[12])a: [^|\r\n]+$')
_HEALTH = re.compile(r'^([1-9][0-9]*)/([1-9][0-9]*)(?: (brn|frz|par|psn|slp|tox))?$')
_CONDITION_FIELD = {'switch': 4, 'drag': 4, '-damage': 3, '-heal': 3, '-sethp': 3}


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _seat(ident: str) -> str:
    matched = _IDENT.fullmatch(ident)
    _require(matched is not None, 'canonical singles active ident required')
    return matched[1]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _public_condition(condition: str) -> str:
    if condition == '0 fnt':
        return condition
    match = _HEALTH.fullmatch(condition)
    _require(match is not None, 'canonical exact source condition required')
    hp, maximum = int(match[1]), int(match[2])
    # Far above the fixed level100 format's HP range, and small enough that
    # integer ceiling and pinned JS Math.ceil(100*hp/maximum) agree exactly.
    _require(0 < hp <= maximum <= 1_000_000, 'bounded exact source HP required')
    percentage = (100 * hp + maximum - 1) // maximum
    if percentage == 100 and hp < maximum:
        percentage = 99
    return f'{percentage}/100' + (f' {match[3]}' if match[3] else '')


@dataclass(frozen=True)
class PlayerSeatProjection:
    seat: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class PlayerProjectionBatch:
    battle_id: str
    p1: PlayerSeatProjection
    p2: PlayerSeatProjection
    source_nonrequest_start: int
    source_nonrequest_end: int
    source_nonrequest_prefix_sha256: str
    counts: Mapping[str, int]
    showdown_revision: str = PINNED_SHOWDOWN_REVISION
    scope: str = PROJECTION_SCOPE
    struggle_warning_policy: str = STRUGGLE_WARNING_POLICY
    captured_player_wire: bool = False
    owning_stream_bytes_reconstructed: bool = False
    public_wire_qualified: bool = False
    fullencoder: bool = False

    def for_seat(self, seat: str) -> PlayerSeatProjection:
        _require(seat in ('p1', 'p2'), 'owning seat required')
        return self.p1 if seat == 'p1' else self.p2


class SavedSourcePlayerProjector:
    """Project both observers atomically; never pass owning requests as public.

    The internal exact-HP parser is an ownership-proof substrate ONLY. It is
    never normalized, encoded, or returned as an actor observation. In singles,
    the private Intimidate hint proves the holder's foe has Substitute. A unique
    substituted side therefore proves ownership independently of switch order.
    Both/neither substituted is refused, never guessed from the last switch.
    """
    def __init__(self, *, battle_id: str, showdown_revision: str = PINNED_SHOWDOWN_REVISION):
        from .showdown import _ReplayParser
        _require(bool(battle_id) and isinstance(battle_id, str), 'battle identity required')
        _require(showdown_revision == PINNED_SHOWDOWN_REVISION, 'pinned emission source required')
        self.battle_id = battle_id
        self._parser = _ReplayParser(battle_id, complete_prefix=True,
                                     hp_visibility={'p1': 'exact', 'p2': 'exact'})
        self._setup: dict[str, str] = {}
        self._rules: set[str] = set()
        self._team_sizes: set[str] = set()
        self._started = False
        self._ready = False
        self._closed = False
        self._count = 0
        self._prefix = hashlib.sha256(b'[')

    def project_nonrequest_delta(self, lines: Sequence[str]) -> PlayerProjectionBatch:
        from .showdown import _ReplayParser
        _require(not self._closed, 'source projector already terminal')
        _require(isinstance(lines, (list, tuple)), 'ordered nonrequest delta required')
        # Work on copied proof state; a refused batch does not advance either seat.
        parser = _ReplayParser.from_snapshot(self._parser.snapshot())
        setup, rules, team_sizes = dict(self._setup), set(self._rules), set(self._team_sizes)
        started, ready, closed = self._started, self._ready, self._closed
        prefix, count, start = self._prefix.copy(), self._count, self._count
        output: dict[str, list[str]] = {'p1': [], 'p2': []}
        counters: Counter[str] = Counter()
        for line in lines:
            _require(not closed, 'source line after terminal')
            _require(isinstance(line, str) and line.startswith('|') and '\n' not in line and '\r' not in line,
                     'one raw protocol line required')
            parts = line.split('|')
            kind = parts[1]
            _require(kind in SOURCE_KINDS, 'unqualified source protocol family')
            if count:
                prefix.update(b',')
            prefix.update(_canonical(line))
            count += 1
            pair = {'p1': line, 'p2': line}
            if kind in {'gen', 'gametype', 'tier'}:
                _require(len(parts) == 3 and not ready and kind not in setup, 'unique setup field required')
                setup[kind] = parts[2]
            elif kind == 'rule':
                _require(len(parts) == 3 and not ready and parts[2] in RULES and parts[2] not in rules,
                         'fixed standard percentage rules required')
                rules.add(parts[2])
            elif kind == 'teamsize':
                _require(len(parts) == 4 and not ready and parts[2] in ('p1', 'p2') and parts[3] == '6'
                         and parts[2] not in team_sizes, 'fixed six-mon singles sides required')
                team_sizes.add(parts[2])
            elif kind == 'start':
                _require(len(parts) == 2 and not started and not ready, 'one battle start required')
                started = True
            if kind in _CONDITION_FIELD:
                if not ready:
                    _require(started and setup == {'gen': '3', 'gametype': 'singles', 'tier': '[Gen 3] Random Battle'}
                             and rules == RULES and team_sizes == {'p1', 'p2'}, 'complete fixed-format setup required')
                    ready = True
                field = _CONDITION_FIELD[kind]
                _require(len(parts) > field and all(part.startswith('[') for part in parts[field+1:]),
                         'one condition and canonical health tags required')
                owner = _seat(parts[2])
                condition = _public_condition(parts[field])
                peer = 'p2' if owner == 'p1' else 'p1'
                changed = list(parts)
                changed[field] = condition
                pair[peer] = '|'.join(changed)
                counters['health_messages'] += 1
            elif kind == '-ability':
                _require(ready and len(parts) >= 4, 'ability after qualified setup required')
                owner = _seat(parts[2])
                if parts[3] == 'Pressure':
                    if parts[4:] == ['[silent]']:
                        pair.pop('p2' if owner == 'p1' else 'p1')
                        counters['private_pressure_' + owner] += 1
                    else:
                        # Pokemon.setAbility emits these with ordinary add, NOT
                        # addSplit. Trace is one producer; sourceEffect.fullname
                        # and the old ability are common static arguments. The
                        # method gates its Start callback at gen>3.
                        _require(len(parts) == 7 and bool(parts[4]) and
                                 parts[5].startswith(('[from] ability: ', '[from] move: ')),
                                 'unproved nonsilent Pressure source shape')
                        _require(parts[6].startswith('[of] '), 'public setAbility source required')
                        _seat(parts[6][len('[of] '):])
                        counters['public_pressure_setability'] += 1
                else:
                    _require('[silent]' not in parts[4:], 'unqualified silent ability shape')
            elif kind == '-hint':
                _require(ready and len(parts) == 3, 'canonical qualified hint required')
                if parts[2] == INTIMIDATE_HINT:
                    substituted = [seat for seat in ('p1', 'p2') if 'substitute' in parser.volatiles[seat]]
                    _require(len(substituted) == 1, 'private Intimidate hint owner is ambiguous')
                    owner = 'p2' if substituted[0] == 'p1' else 'p1'
                    pair.pop(substituted[0])
                    counters['private_intimidate_hint_' + owner] += 1
                else:
                    _require(parts[2] in SHARED_HINTS, 'unproved hint source')
                    counters['shared_hints'] += 1
            elif kind == '-message':
                _require(ready and parts == ['', '-message', 'Sleep Clause Mod activated.'],
                         'unproved diagnostic message source')
                counters['shared_sleep_clause_messages'] += 1
            elif kind == '-activate' and len(parts) > 3 and parts[3] == 'move: Struggle':
                raise ValueError('private Struggle sideupdate is absent from this saved source contract')
            elif kind in {'win', 'tie'}:
                _require(ready, 'terminal after setup required')
                closed = True
            parser.feed((line,))
            for seat, projected in pair.items():
                output[seat].append(projected)
        self._parser, self._setup, self._rules, self._team_sizes = parser, setup, rules, team_sizes
        self._started, self._ready, self._closed = started, ready, closed
        self._prefix, self._count = prefix, count
        final = prefix.copy()
        final.update(b']')
        return PlayerProjectionBatch(self.battle_id,
            PlayerSeatProjection('p1', tuple(output['p1'])), PlayerSeatProjection('p2', tuple(output['p2'])),
            start, count, final.hexdigest(), dict(counters))


def _assert_annotation_histories(environments: Mapping[str, Any]) -> None:
    """Bind every contained producer to its owner and independent mutable state.

    Read-only dex/config/team metadata and immutable context/cache values may
    be shared. The mutable histories that the production tracker clone methods
    copy (folds, ledgers and accumulator/cache containers) may not cross owners.
    This is a bounded walk rooted only in those annotation-history fields, not
    a walk through the source/dex/model object graph.
    """
    from .investment import InvestmentLiveTracker
    from .tier2 import Tier2LiveTracker, _species_key, own_team_from_request
    owners: dict[int, str] = {}

    def claim(value: Any, owner: str) -> None:
        previous = owners.setdefault(id(value), owner)
        _require(previous == owner, 'contained annotation histories must not alias across owners')

    def mutable_tree(value: Any, owner: str, visited: set[int]) -> None:
        if id(value) in visited:
            return
        if isinstance(value, (dict, list, set)):
            claim(value, owner)
            visited.add(id(value))
            children = value.values() if isinstance(value, dict) else value
        elif is_dataclass(value) and not value.__dataclass_params__.frozen:
            claim(value, owner)
            visited.add(id(value))
            children = vars(value).values()
        elif isinstance(value, tuple):
            children = value
        else:
            return
        for child in children:
            mutable_tree(child, owner, visited)

    families = (
        ('_tier2_trackers', Tier2LiveTracker,
         ('_residuals', '_cb_turns', '_cb_non_ko', '_cb_bit_indices'), ('_stats_cache',)),
        ('_investment_trackers', InvestmentLiveTracker,
         ('_state', '_defender_levels'), ('_spread_cache',)),
    )
    for owner in ('p1', 'p2'):
        env = environments[owner]
        peer = 'p2' if owner == 'p1' else 'p1'
        for field, expected_type, history_fields, cache_fields in families:
            trackers = getattr(env, field)
            _require(isinstance(trackers, dict) and set(trackers) <= {owner},
                     'annotation map keys must bind only the owning seat')
            for tracker in trackers.values():
                claim(tracker, owner)
                _require(isinstance(tracker, expected_type) and tracker._perspective == owner and
                         tracker._opponent == peer, 'contained annotation tracker perspective must bind owner')
                request = env._first_requests.get(owner) or env._latest_requests.get(owner)
                _require(request is not None, 'annotation tracker requires its owning initial request')
                own_team = {_species_key(mon.species): mon for mon in own_team_from_request(request)}
                _require(bool(own_team) and tracker._own_by_species == own_team,
                         'annotation tracker own team must bind owning initial request')
                claim(tracker._fold, owner)
                visited: set[int] = set()
                mutable_tree(vars(tracker._fold), owner, visited)
                for name in history_fields:
                    mutable_tree(getattr(tracker, name), owner, visited)
                # Cache values are read-only after creation in the production
                # clone contract; the mutable cache container is still owned.
                for name in cache_fields:
                    claim(getattr(tracker, name), owner)


def new_player_reference_environments(config: Any, *, battle_id: str) -> dict[str, Any]:
    """Fresh offline production histories; no simulator is started or restored."""
    from .local_showdown import LocalShowdownEnv
    from .showdown import _ReplayParser
    _require(config.observation_spec.schema_version == 'pokezero.observation.v4', 'explicit v4 reference schema required')
    _require(config.category_vocab is not None, 'frozen reference vocabulary must be supplied')
    class OwningReferenceEnvironment(LocalShowdownEnv):
        def observe(self, player):
            _require(not self._player_projection_failed and self._terminal is None and
                     player == self._player_projection_seat and self.requested_players() == (player,),
                     'failed or foreign owning observation refused')
            _assert_annotation_histories(result)
            observation = super().observe(player)
            # Includes trackers lazily constructed by this observation.
            _assert_annotation_histories(result)
            return observation
    result = {}
    for seat in ('p1', 'p2'):
        env = OwningReferenceEnvironment(config)
        _require(env._process is None and env._battle_token is None and not env.protocol_lines,
                 'fresh offline reference environment required')
        env._battle_id = battle_id
        env._parser = _ReplayParser(battle_id, complete_prefix=True,
            hp_visibility={seat: 'exact', 'p2' if seat == 'p1' else 'p1': 'percentage'})
        env._player_projection_seat = seat
        env._player_projection_next = 0
        env._player_projection_failed = False
        result[seat] = env
    return result


def _only_struggle(request: Mapping[str, Any]) -> bool:
    active = request.get('active')
    if request.get('wait') or request.get('forceSwitch') or not isinstance(active, list) or len(active) != 1:
        return False
    moves = active[0].get('moves', [])
    return (len(moves) == 1 and moves[0].get('id') == 'struggle' and not moves[0].get('disabled'))


def assert_struggle_omission_boundary(env: Any, seat: str) -> None:
    """Prove the relevant generic consumers are quiescent at this own boundary.

    Struggle warnings are private chooseMove sideupdates before turn execution.
    They do not match an action, confusion, volatile, ability or item reveal.
    The generic pending-Trick/switch and faint latches are guarded here rather
    than assumed inert merely because one immediate observation happens to match.
    """
    _require(seat in ('p1', 'p2') and env._process is None and env._battle_token is None,
             'offline owning choice boundary required')
    env._sync_incremental_state()
    request = env._latest_requests.get(seat, {})
    _require(_only_struggle(request), 'own enabled Struggle-only request required')
    _require(env._belief_engine._pending_trick is None and not env._belief_engine._pending_switches,
             'Struggle omission cannot cross pending belief evidence')
    _require(env._parser.toxic_faint_replacement_pending[seat] is False,
             'Struggle omission cannot cross own pending faint replacement')


def feed_player_reference_boundary(environments: Mapping[str, Any], batch: PlayerProjectionBatch, *,
        own_request_deltas: Mapping[str, Sequence[str]], requested_seats: Sequence[str],
        request_tail_verified: bool = False) -> dict[str, Any]:
    """Advance BOTH production histories, including waits, before actor observe.

    Supply EVERY emitted private owning request from the validated source row,
    not just current requested views. Requests are appended via their separate
    private ingestion path. The upstream caller MUST inspect the original ordered
    mixed delta and prove every request is in its literal tail, with no later
    nonrequest. Separated arguments cannot themselves attest that chronology.
    Interleaving requires a separately reviewed ordered source-index interface;
    it is never reordered or filtered to make the fixed corpus pass.
    Original stored references are unused.
    Any production-ingestion failure poisons both histories; no partial observe.
    """
    from .local_showdown import requested_players_from_requests
    _require(request_tail_verified is True, 'literal original private-request tail must be verified upstream')
    _require(set(environments) == {'p1', 'p2'} and environments['p1'] is not environments['p2'],
             'two independent owning histories required')
    for field in ('_parser', '_belief_engine', '_tier2_trackers', '_investment_trackers'):
        _require(getattr(environments['p1'], field) is not getattr(environments['p2'], field),
                 'owning parser/belief/annotation histories must not alias')
    _assert_annotation_histories(environments)
    _require(set(own_request_deltas) == {'p1', 'p2'}, 'both owning request delta channels required')
    _require(batch.scope == PROJECTION_SCOPE and batch.showdown_revision == PINNED_SHOWDOWN_REVISION and
             batch.struggle_warning_policy == STRUGGLE_WARNING_POLICY and
             not batch.public_wire_qualified and not batch.captured_player_wire and
             not batch.owning_stream_bytes_reconstructed and not batch.fullencoder,
             'explicit source-derived observation projection required')
    _require(type(batch.source_nonrequest_start) is int and type(batch.source_nonrequest_end) is int and
             0 <= batch.source_nonrequest_start <= batch.source_nonrequest_end and
             isinstance(batch.source_nonrequest_prefix_sha256, str) and
             re.fullmatch('[0-9a-f]{64}', batch.source_nonrequest_prefix_sha256) is not None,
             'source prefix/count provenance required')
    encoded_requests, public_lines = {}, {}
    for seat in ('p1', 'p2'):
        env = environments[seat]
        _require(not env._player_projection_failed and env._player_projection_seat == seat and
                 env._battle_id == batch.battle_id and env._player_projection_next == batch.source_nonrequest_start
                 and env._process is None and env._battle_token is None, 'fresh chronological owning history required')
        _require(env._parser.hp_visibility == {seat: 'exact', 'p2' if seat == 'p1' else 'p1': 'percentage'},
                 'own exact and opposing percentage provenance required')
        projected = batch.for_seat(seat)
        _require(projected.seat == seat and isinstance(projected.lines, tuple), 'immutable owning public lines required')
        public_lines[seat] = tuple(projected.lines)
        for line in public_lines[seat]:
            _require(isinstance(line, str) and line.startswith('|') and '\n' not in line and '\r' not in line,
                     'one projected public line required')
            parts = line.split('|')
            _require(parts[1] in SOURCE_KINDS, 'request/private/unknown line in public projection')
            if parts[1] in _CONDITION_FIELD:
                field = _CONDITION_FIELD[parts[1]]
                _require(len(parts) > field, 'projected condition required')
                owner = _seat(parts[2])
                if owner != seat and parts[field] != '0 fnt':
                    health = _HEALTH.fullmatch(parts[field])
                    _require(health is not None and health[2] == '100' and int(health[1]) <= 100,
                             'peer exact HP refused by reference seam')
            if parts[1] == '-ability' and len(parts) > 4 and parts[3:] == ['Pressure', '[silent]']:
                _require(_seat(parts[2]) == seat, 'peer private Pressure refused by reference seam')
        encoded_requests[seat] = []
        for line in own_request_deltas[seat]:
            _require(isinstance(line, str) and line.startswith('|request|') and '\r' not in line and '\n' not in line,
                     'one raw owning request line required')
            request = json.loads(line[len('|request|'):])
            _require(isinstance(request, Mapping) and request.get('side', {}).get('id') == seat,
                     'foreign request refused before either history advances')
            encoded_requests[seat].append(line)
    _require(tuple(requested_seats) == tuple(seat for seat in ('p1', 'p2') if seat in requested_seats),
             'unique ordered requested seats required')
    try:
        for seat in ('p1', 'p2'):
            env = environments[seat]
            # Internal LocalShowdownEnv's nonrequest ingestion seam is named
            # omniscient. ONLY the already projected owning lines reach it here.
            env._apply_event({'type': 'stream', 'stream': 'omniscient', 'lines': list(public_lines[seat])})
            for line in encoded_requests[seat]:
                env._apply_event({'type': 'stream', 'stream': seat, 'lines': [line]})
            env._sync_incremental_state()
            _require(set(env._latest_requests) <= {seat} and set(env._parser.requests) <= {seat},
                     'peer request entered an owning history')
            env._player_projection_next = batch.source_nonrequest_end
        actual = tuple(seat for seat in ('p1', 'p2') if environments[seat]._terminal is None and
                       requested_players_from_requests(environments[seat]._latest_requests))
        _require(tuple(requested_seats) == actual, 'all current owning request roles must match')
        for seat in actual:
            if _only_struggle(environments[seat]._latest_requests[seat]):
                assert_struggle_omission_boundary(environments[seat], seat)
        observations = {}
        for seat in actual:
            _assert_annotation_histories(environments)
            observation = environments[seat].observe(seat)
            _assert_annotation_histories(environments)
            perspective = observation.perspective
            metadata = observation.metadata
            peer = 'p2' if seat == 'p1' else 'p1'
            _require(observation.schema_version == 'pokezero.observation.v4' and perspective is not None and
                     perspective.player_id == seat and perspective.showdown_slot == seat and
                     perspective.opponent_showdown_slot == peer and isinstance(metadata, Mapping) and
                     metadata.get('battle_id') == batch.battle_id and metadata.get('player_id') == seat and
                     metadata.get('showdown_slot') == seat and metadata.get('opponent_showdown_slot') == peer and
                     type(metadata.get('turn_number')) is int and
                     metadata['turn_number'] == environments[seat]._parser.turn_number,
                     'production reference identity/schema does not bind owning seat')
            observations[seat] = observation
        return observations
    except BaseException:
        for env in environments.values():
            env._player_projection_failed = True
        raise
