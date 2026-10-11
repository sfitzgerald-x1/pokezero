"""Gen 3 Yawn's fixed public residual clock and disclosed source ownership.

No simulator snapshot, sleep-duration draw, or opponent private request is read.
Source switching/fainting does not clear Yawn: the delayed sleep still belongs
to that Pokemon, including for Sleep Clause. Target switching does clear it.
"""


def _scan_yawn_history(state, player, *, error):
    active, identities = {}, {}
    lock, last_move, turn, last_kind = None, None, None, None
    pending_expiry, sleep_sources = None, {}
    normalize = lambda value: ''.join(c for c in value.lower() if c.isalnum())
    for index, event in enumerate(state.replay.public_events):
        parts = event.raw_line.split('|')
        kind = parts[1] if len(parts) > 1 else ''
        last_kind = kind
        # A native Yawn expiry emits precisely this next public event. Consume
        # once; a blocked status, intervening event or later unrelated sleep
        # can never borrow a stale source from the prior Yawn episode.
        if pending_expiry is not None:
            if (len(parts) == 4 and kind == '-status'
                    and parts[2] == pending_expiry['targetIdent'] and parts[3] == 'slp'):
                sleep_sources[index] = (pending_expiry['sourceSide'], pending_expiry['sourceIdent'])
            pending_expiry = None
        if kind == 'turn':
            last_move = None
            if len(parts) != 3 or not parts[2].isdigit() or int(parts[2]) < 1:
                raise error('Reference Yawn lacks a valid public turn ledger.')
            next_turn = int(parts[2])
            if turn is None and next_turn != 1:
                raise error('Reference Yawn requires a complete public opening turn.')
            if turn is not None and next_turn != turn + 1:
                raise error('Reference Yawn lacks a contiguous public turn ledger.')
            turn = next_turn
            continue
        if kind == 'upkeep':
            last_move = None
            if lock is not None:
                if turn is None or turn in lock['residual_turns']:
                    raise error('Reference Yawn lacks an unambiguous public residual ledger.')
                lock['residual_turns'].append(turn)
            continue
        if len(parts) < 3:
            continue
        actor, side = parts[2], parts[2][:2]
        if kind in {'switch', 'drag', 'replace'} and side in {'p1', 'p2'}:
            last_move = None
            if side == player:
                lock = None
            if len(parts) < 4 or not actor.startswith(side + 'a: '):
                raise error('Reference Yawn lacks public Pokemon identity.')
            species = parts[3].split(',', 1)[0]
            if not species:
                raise error('Reference Yawn lacks public Pokemon species.')
            old = identities.get(actor)
            if old is not None and old != species:
                raise error('Reference Yawn has an ambiguous public source identity.')
            identities[actor] = active[side] = species
            active[side + ':ident'] = actor
        elif kind == 'move':
            last_move = parts if len(parts) >= 5 and '[miss]' not in parts[5:] else None
        elif kind in {'-miss', '-fail', '-immune'}:
            if last_move is not None and actor in {last_move[2], last_move[4]}:
                last_move = None
        elif kind == 'faint' and side in {'p1', 'p2'}:
            if side == player:
                lock = None
            active.pop(side, None)
            active.pop(side + ':ident', None)
            if last_move is not None and actor in {last_move[2], last_move[4]}:
                last_move = None
        elif kind == '-end' and side == player and len(parts) >= 4 and normalize(parts[3]) == 'moveyawn':
            if (parts[4:] == ['[silent]'] and lock is not None
                    and actor == lock['targetIdent'] and active.get(player + ':ident') == actor
                    and lock['residual_turns'] == [lock['start_turn']]
                    and turn == lock['start_turn'] + 1):
                pending_expiry = dict(lock)
            lock = None
        elif kind == '-start' and side == player and len(parts) >= 4 and normalize(parts[3]) == 'moveyawn':
            if lock is not None:
                raise error('Reference Yawn has duplicate public starts.')
            tags = [part.strip() for part in parts[4:] if part.strip()]
            sources = [tag[5:] for tag in tags if tag.startswith('[of] ')]
            source = sources[0] if len(tags) == len(sources) == 1 else None
            source_side = source[:2] if source else ''
            if (turn is None or source_side not in {'p1', 'p2'} or source_side == player
                    or active.get(player + ':ident') != actor
                    or active.get(source_side + ':ident') != source
                    or last_move is None or normalize(last_move[3]) != 'yawn'
                    or last_move[2] != source or last_move[4] != actor):
                raise error('Reference Yawn lacks an unambiguous public start/source.')
            lock = dict(sourceSide=source_side, sourceIdent=source, targetIdent=actor,
                sourceSpecies=identities[source], targetSpecies=identities[actor],
                start_turn=turn, residual_turns=[])
    return lock, turn, last_kind, sleep_sources


def public_yawn_certificate(state, player, *, error):
    if 'yawn' not in state.replay.volatiles.get(player, ()):
        return None
    if state.observation_format_id not in {'gen3randombattle', 'gen3customgame'}:
        raise error('Reference Yawn requires Gen 3.')
    if (state.self_request.get('forceSwitch') or state.self_request.get('wait')
            or not state.self_request.get('active')
            or state.deferred_opponent_action_player is not None):
        raise error('Reference Yawn needs an ordinary post-upkeep move boundary.')
    lock, turn, last_kind, _ = _scan_yawn_history(state, player, error=error)
    if (lock is None or lock['residual_turns'] != [lock['start_turn']]
            or turn != lock['start_turn'] + 1
            or last_kind != 'turn'):
        raise error('Reference Yawn lacks exactly one paid public residual at an ordinary boundary.')
    return {**lock, 'duration': 1}


def yawn_expiry_sleep_sources(state, *, error):
    """Event-indexed opposing sources; no current volatile or private clock."""
    if not any('|-start|' in event.raw_line and '|move: Yawn|' in event.raw_line
               for event in state.replay.public_events):
        return {}
    if getattr(state, 'observation_format_id', None) not in {'gen3randombattle', 'gen3customgame'}:
        raise error('Reference Yawn expiry requires Gen 3 public provenance.')
    sources = {}
    for player in ('p1', 'p2'):
        _, _, _, expired = _scan_yawn_history(state, player, error=error)
        if sources.keys() & expired.keys():
            raise error('Reference Yawn expiry has ambiguous victim ownership.')
        sources.update(expired)
    return sources
