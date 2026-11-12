"""The order-10 residual list's order keys, and the positions each event kind can take there (jax-free).

transition.step_with_chance runs Showdown's order-10 residual handlers (fieldEvent('Residual'), sorted by comparePriority:
Speed, then subOrder; sim/battle.ts:404-411) as a fixed list of EVENTS events, one per side per kind. Event e is side
EVENT_SIDES[e]'s handler of kind EVENT_KIND[e] at subOrder EVENT_SUBORDER[e]. Per lane the step builds every event's key

    event_key(end_speed[EVENT_SIDES[e]], EVENT_SUBORDER[e], tie[e]) = -speed*SPEED_MULTIPLIER + subOrder*SUBORDER_STRIDE + tie

where the two events of a pair (TIE_PAIR, one per side, same subOrder) share one tie draw and take complementary bits
(draw ^ side; the inert Yawn pair's draw is the fixed 0). It argsorts the keys and runs one handler iteration per position.

Two invariants make the kinds a position can hold a static fact (tests/test_gpu_engine_residual_order.py checks both on
transition.py's source):
(a) a position is a rank among all EVENTS keys whether or not its event is active: an inactive event keeps its real key
    (no sentinel key) and the loop visits every position (no compaction);
(b) Speed and the tie are fixed before the loop (end_speed, tie, event_keys and event_order are built once, ahead of it),
    and nothing inside the loop reorders.
Then the event at position p depends only on the two actives' Speed relation and the tie draws. POSITIONS[kind] is every
position a kind's events take, over every Speed gap that can give its own order and every tie draw: an exhaustive
enumeration of the step's own keys (the fixed Yawn draw is enumerated too, which can only widen its set). At any other
position the kind's event mask is all false. Fidelity M2's R1 emits the Wrap and Yawn handler blocks only at
POSITIONS['wrap'] and POSITIONS['yawn'], which cuts the traced step's size; every other handler runs at every position.
"""
import itertools
import types

KINDS = ('ability', 'heal', 'seed', 'status', 'wrap', 'encore', 'yawn')
# Speed Boost (subOrder 3), Leftovers and the pinch berries (4), Leech Seed (5), brn/psn/tox (6), partiallytrapped (9),
# Encore (14) and Yawn (19), side 0's event then side 1's.
EVENT_KIND = ('ability', 'ability', 'heal', 'heal', 'seed', 'seed', 'status', 'status', 'wrap', 'wrap', 'encore', 'encore',
              'yawn', 'yawn')
EVENT_SIDES = (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
EVENT_SUBORDER = (3, 3, 4, 4, 5, 5, 6, 6, 9, 9, 14, 14, 19, 19)
# The tie draw each event reads: pair k's two events take draw_k ^ EVENT_SIDES[e], complementary bits.
TIE_PAIR = (0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6)
SPEED_MULTIPLIER = 64
SUBORDER_STRIDE = 2
EVENTS = len(EVENT_KIND)
if not len(EVENT_SIDES) == len(EVENT_SUBORDER) == len(TIE_PAIR) == EVENTS or not set(EVENT_KIND) <= set(KINDS):
    raise ValueError('residual_order: the event tables disagree')


def event_key(speed, suborder, tie):
    """An event's order key: Speed first (faster first), then subOrder, then the tie bit. transition applies it to its
    jnp arrays (M2's ops: negate, two products, two sums); orders() applies it to ints."""
    return -speed*SPEED_MULTIPLIER + suborder*SUBORDER_STRIDE + tie


def kind_events(kind):
    """(first, stop): the contiguous event indices of `kind`."""
    events = [e for e, k in enumerate(EVENT_KIND) if k == kind]
    if not events or events != list(range(events[0], events[-1] + 1)):
        raise ValueError(f'residual_order: the {kind} events {events} are not contiguous')
    return events[0], events[-1] + 1


def is_kind(event, kind):
    """The step's event-kind mask for an event index (an int or an integer array). One-sided at the list's ends, so the
    comparisons are M2's own: event < 2, (event >= 2) & (event < 4), ..., event >= 12."""
    first, stop = kind_events(kind)
    if first == 0:return event < stop
    if stop == EVENTS:return event >= first
    return (event >= first) & (event < stop)


def speed_gaps():
    """Every Speed gap (side 0's end_speed minus side 1's, an integer) that can give its own order. A cross-side comparison
    is decided by Speed alone once gap*SPEED_MULTIPLIER exceeds the span of subOrder*SUBORDER_STRIDE + tie, so any larger
    gap repeats the bound's order; same-side comparisons never read Speed."""
    parts = [event_key(0, suborder, tie) for suborder in EVENT_SUBORDER for tie in (0, 1)]
    bound = (max(parts) - min(parts)) // SPEED_MULTIPLIER + 1
    return range(-bound, bound + 1)


def orders():
    """Every order the step can give: (gap, draws, keys, order) for each Speed gap and each tie draw per pair. order is the
    stable argsort of the keys (jnp.argsort's): position p holds event order[p]."""
    for gap in speed_gaps():
        speed = (max(gap, 0), max(-gap, 0))
        for draws in itertools.product((0, 1), repeat=max(TIE_PAIR) + 1):
            keys = [event_key(speed[EVENT_SIDES[e]], EVENT_SUBORDER[e], draws[TIE_PAIR[e]] ^ EVENT_SIDES[e]) for e in range(EVENTS)]
            yield gap, draws, keys, sorted(range(EVENTS), key=lambda e: (keys[e], e))


def reachable_positions():
    """{kind: frozenset of the positions its events take over orders()}."""
    found = {kind: set() for kind in KINDS}
    for *_, order in orders():
        for position, event in enumerate(order):found[EVENT_KIND[event]].add(position)
    return {kind: frozenset(found[kind]) for kind in KINDS}


# R1's gating sets, derived from the constants above (wrap: 4, 8, 9, 11; yawn: 6, 12, 13). Read-only (a mapping proxy of
# frozensets), so nothing can change the gating sets at runtime.
POSITIONS = types.MappingProxyType(reachable_positions())
