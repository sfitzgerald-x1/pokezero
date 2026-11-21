"""Set posterior: one narrowing source for a revealed opponent's randbats set (host side, pure Python, no JAX).

Per revealed opponent species, over the source's existing variant list (source order, at most 64 variants):

    w(v) = 1[hard facts](v) * prod over admitted strikes of (how many of v's 16 damage rolls fit the strike)

- Hard facts are today's reveals exactly as the worlds path filters them: determinization._canonical_revealed_variant_payloads
  over the root's belief view (revealed moves, ability, item unless mutated, ruled-out abilities and items).
- An observation is an incoming hit on our active or a full-health outgoing hit on that species. Our HP is exact (our own lines). The 16 rolls per variant come
  from tier2's shared damage core, with the attacker built as the world constructor builds it (_fixture_from_variant_payload,
  _computed_stats) and our mon exact from our first request. A KO is censored (rolls >= our HP before the hit); a crit is a
  known condition.
- A strike the core cannot reproduce exactly is refused (L = 1, the reason recorded). The classes are the r5a census's
  (narrowing-source-scope-20261115 §1a: 0 false exclusions in 58,802 admitted strikes), plus every class r5a never
  exercised: Reflect / Light Screen, conditional power, the Sports, Charge and a semi-invulnerable defender.
- An effectiveness line is not a separate exclusion rule (the census validated damage only): each variant's rolls already
  carry its own effectiveness, and a strike whose line a variant with fitting rolls could not have printed (a KO censored
  above every roll, say) is refused rather than trusted either way.
- Off-model safety: a strike that would give every surviving variant 0 is dropped and recorded, so the support never empties.
  The worlds consumer has the same rule: if every one of the sampler's own survivors has weight 0 (restrict), it draws
  uniformly over them, today's call, and the world records the off-model draw (its PUBLIC_SET_POSTERIOR row).
- A hard support of one variant cannot narrow: its strikes are classified but their rolls are not computed (reason
  'single-variant').

Deliberately stricter than the r5a census (each can only refuse a strike the census admitted, never exclude a variant):
- only the first nonbreaking hit on our newly created Substitute has an exact capacity (floor(maxHP/4)); its likelihood
  counts rolls strictly below that capacity. Passed-in, broken, later, or mixed HP/Substitute events remain refused;
- the whole strike is refused when one hard-support variant has no rolls from the core ('core-unsupported:<move>'), where
  the census gave that variant 0, except type immunity: a variant whose move cannot touch our mon (an immune Hidden Power
  type) fits 0 rolls of a hit that did damage, as in the census;
- our hit on their Substitute counts as hitting them for Revenge ('conditional-power:revenge');
- a window still open at the end of the lines is refused ('window-open'); a prefix ending at a request is a closed boundary;
- a strike whose effectiveness line a variant with fitting rolls could not have printed is refused
  ('effectiveness-disagreement');
- Reflect / Light Screen on our side refuse the strike ('unvalidated:screen'), and the pinch / Flail ambiguity is checked
  on the hard support, not on the whole universe.

Weights are integers, reduced by their gcd (uniform weights are all 1) and never floats, so every consumer and process gets
the same bytes. draw() with all-equal weights makes exactly today's sampler call (rng.randrange over the survivors in source
order, determinization._sample_revealed_opponent_fixture), so worlds of roots without damage narrowing are byte-identical.
The prior is uniform (generator_weights must be None). Sleep Talk callees use their declared move. Our outgoing hits are
admitted only from public100/100 HP, fitting each candidate maximum to the after-percent bin; no exact foe HP is read.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import random
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

from . import determinization as _worlds
from . import tier2
from .dex import ShowdownDex, normalize_id
from .investment import FIXED_EXACT_DAMAGE_MOVES, UNMODELED_DAMAGE_MOVES
from .randbat import Gen3RandbatSource, Gen3RandbatVariant

SCHEMA = 'pokezero.set-posterior.v2'
UNIFORM = 'uniform'
ROLLS = 16
# Root-level refusal codes (EncounterPosterior.refusals, and the worlds that consume them).
B2_DISAGREEMENT = 'PUBLIC_SET_POSTERIOR_B2_DISAGREEMENT'
SAMPLER_DISAGREEMENT = 'PUBLIC_SET_POSTERIOR_SAMPLER_DISAGREEMENT'
SAMPLER_EMPTY = 'PUBLIC_SET_POSTERIOR_SAMPLER_EMPTY'
MISSING_SPECIES = 'PUBLIC_SET_POSTERIOR_MISSING_SPECIES'
DUPLICATE_SPECIES = 'PUBLIC_SET_POSTERIOR_DUPLICATE_SPECIES'
# The reason of a strike dropped by the off-model rule, and of one whose rolls a single-variant support makes moot.
OFF_MODEL = 'off-model-drop'
SINGLE_VARIANT = 'single-variant'

# Showdown protocol lines that carry no game state: wall clock (|t:| is the turn's unix time, from sim/battle.ts; |:| a server
# room log's; |inactive| / |inactiveoff| the timer) and room transport (chat, join, leave, rename, raw HTML). r4a3's owning
# wire carries |t:| alone (25,823 of its lines); the rest are listed defensively (a server-relayed stream could carry them).
# Filtering only feeds hashes (world seeds, provenance), never a parser.
NON_GAME_LINE_TAGS = frozenset(('t:', ':', 'c', 'c:', 'chat', 'j', 'J', 'join', 'l', 'L', 'leave', 'n', 'N', 'name', 'raw',
    'html', 'uhtml', 'uhtmlchange', 'inactive', 'inactiveoff'))


def game_lines_sha256(lines: Iterable[str]) -> str:
    """sha256 of the owner's protocol lines without the non-game ones (NON_GAME_LINE_TAGS), in the owner_lines_sha256 format:
    the same game state hashes the same in every run, whatever the wall clock said."""
    keep = (x for x in lines if not (x.startswith('|') and x.split('|', 2)[1] in NON_GAME_LINE_TAGS))
    return hashlib.sha256(''.join(x + '\n' for x in keep).encode()).hexdigest()


# --- Refused strike classes (the r5a census's, scope §1a; the first that applies names the strike). ---
# Damage that is not the 16-roll formula: fixed or unmodelled amounts (investment's lists) and Bide.
FIXED_DAMAGE_MOVES = frozenset(FIXED_EXACT_DAMAGE_MOVES) | frozenset(UNMODELED_DAMAGE_MOVES) | {'bide'}
OHKO_MOVES = frozenset({'fissure', 'horndrill', 'guillotine', 'sheercold'})
# Base power that public lines cannot condition (hidden or ordering state); Eruption / Water Spout below full HP and
# Revenge after our hit this turn are refused by their own rules.
CONDITIONAL_POWER_MOVES = frozenset({'weatherball', 'magnitude', 'lowkick', 'smellingsalt', 'spitup', 'present', 'beatup',
    'triplekick', 'rollout', 'iceball', 'furycutter', 'futuresight', 'doomdesire', 'frustration', 'stomp', 'rage'})
MULTI_HIT_MOVES = frozenset({'doublekick', 'bonemerang', 'twineedle', 'doubleslap', 'cometpunch', 'furyattack', 'furyswipes',
    'pinmissile', 'spikecannon', 'barrage', 'bulletseed', 'iciclespear', 'rockblast', 'armthrust', 'triplekick', 'beatup'})
SEMI_INVULNERABLE = frozenset({'dig', 'fly', 'dive', 'bounce'})
SPORTS = frozenset({'mudsport', 'watersport'})
# A strike's window (the lines its move printed) ends at the next action, turn or residual boundary, or at a delayed hit.
_WINDOW_END = frozenset({'', 'move', 'switch', 'drag', 'cant', 'turn', 'upkeep', 'replace', 'win', 'tie'})
_PINCH_BAND = tier2.Tier2Config().pinch_ambiguity_band


@dataclass(frozen=True)
class Strike:
    """One public damage event, admitted or refused."""
    line: int                       # index of the attacker's |move| line in the owning prefix
    attacker: str                   # seats 'p1' / 'p2'
    defender: str
    admitted: bool
    reason: str                     # the refusal's name when not admitted ('' when admitted)
    counts: tuple[int, ...] | None  # per variant of the source list: its rolls that fit (of 16); None when refused
    move: str = ''
    observed: int | None = None     # exact own HP lost; None for Substitute or public-percent observations
    ko: bool = False
    crit: bool = False
    observation_kind: str = 'exact'
    sub_hp: int | None = None
    after_percent: int | None = None


@dataclass(frozen=True)
class EncounterPosterior:
    seat: str                       # the opponent's seat
    species: str                    # species key (tier2._species_key)
    hard: tuple[bool, ...]          # the hard-fact support (today's reveals), per variant of the source list, in source order
    support: tuple[bool, ...]       # weights > 0: the hard support narrowed by the admitted strikes
    weights: tuple[int, ...]        # integer weights, 0 outside the support, gcd-reduced (uniform weights are all 1)
    strikes: tuple[Strike, ...]
    refusals: tuple[str, ...]       # root-level codes (e.g. PUBLIC_SET_POSTERIOR_B2_DISAGREEMENT)
    provenance: str                 # sha256(schema, source binding, 'uniform', game-lines hash)


def species_key(species: str) -> str:
    return tier2._species_key(species)


def posterior(lines: Sequence[str], own_request: Mapping[str, Any] | None, *, source: Gen3RandbatSource, owner: str,
              beliefs: Sequence[Any], format_id: str, dex: ShowdownDex, binding: str | None = None,
              generator_weights: Any = None) -> dict[str, EncounterPosterior]:
    """Keyed by species key, for every revealed opponent species (beliefs: the root's belief view's opponent_pokemon, whose
    reveals the worlds path filters with). Pure function of its arguments.

    lines are the owner's protocol lines (the owning prefix); our team is read from their first |request| (the original
    items), else from own_request. binding names the source (the bound public source's binding; default its hash)."""
    if generator_weights is not None:
        raise ValueError('set_posterior stage 1 has a uniform prior: generator_weights must be None')
    owner = _seat(owner)
    if owner is None:
        raise ValueError('owner must be p1 or p2')
    foe = 'p2' if owner == 'p1' else 'p1'
    lines = tuple(lines)
    provenance = hashlib.sha256(json.dumps((SCHEMA, binding or source.metadata.source_hash, UNIFORM, game_lines_sha256(lines)),
        separators=(',', ':')).encode()).hexdigest()
    raw = parse_strikes(lines, owner, include_outgoing=True)
    request = _first_request(lines) or own_request or {}
    own_team = tier2.own_team_from_request(request)
    out: dict[str, EncounterPosterior] = {}
    for belief in beliefs:
        key = species_key(belief.species)
        if key in out:
            raise ValueError(DUPLICATE_SPECIES)
        universe = source.universe_for(belief.species)
        variants = universe.variants if universe is not None else ()
        index = {variant.variant_id: i for i, variant in enumerate(variants)}
        hard = [False] * len(variants)
        for payload in _worlds._canonical_revealed_variant_payloads(belief, set_source=source, format_id=format_id):
            hard[index[payload['variant_id']]] = True
        rows = []
        single = sum(hard) == 1
        for strike in raw:
            if strike['posterior_key'] != key:
                continue
            reason, counts = strike['reason'], None
            if reason is None and single:
                reason = SINGLE_VARIANT
            elif reason is None:
                reason, counts = _counts(strike, belief.species, variants, hard, own_team, source=source, dex=dex)
            rows.append((strike, reason, counts))
        out[key] = _encounter(foe, owner, key, tuple(hard), rows, provenance)
    return out


def draw(post: EncounterPosterior, rng: random.Random) -> int:
    """A variant index of the source list. With all-equal weights this is exactly today's rng.randrange(len(surviving)) over
    the surviving variants in source order (byte identity); otherwise rng.randrange(sum(weights)) mapped by cumulative weight."""
    alive = [i for i, w in enumerate(post.weights) if w > 0]
    if not alive:
        raise ValueError(SAMPLER_EMPTY)
    first = post.weights[alive[0]]
    if all(post.weights[i] == first for i in alive):
        return alive[rng.randrange(len(alive))]
    pick = rng.randrange(sum(post.weights[i] for i in alive))
    for i in alive:
        pick -= post.weights[i]
        if pick < 0:
            return i
    raise AssertionError('unreachable')


def restrict(post: EncounterPosterior, indices: Sequence[int]) -> EncounterPosterior:
    """The posterior with zero weight outside `indices` (a sampler's own survivors, which must lie in the hard support: its
    replay filters only remove variants; else it refuses, named). Off-model safety at the consumer: when the survivors hold
    no weight, every survivor gets weight 1, so draw() makes today's uniform call over them (the world's PUBLIC_SET_POSTERIOR
    row records the zero-weight draw)."""
    keep = set(indices)
    if not keep or any(not 0 <= i < len(post.hard) or not post.hard[i] for i in keep):
        raise ValueError(SAMPLER_DISAGREEMENT)
    weights = tuple(w if i in keep else 0 for i, w in enumerate(post.weights))
    if not any(weights):
        weights = tuple(int(i in keep) for i in range(len(post.weights)))
    if weights == post.weights:
        return post
    return replace(post, weights=weights, support=tuple(w > 0 for w in weights))


def two_way(posts: Mapping[str, EncounterPosterior], b2: Mapping[str, Iterable[int]]) -> dict[str, EncounterPosterior]:
    """posts, with B2_DISAGREEMENT added to every species whose hard support is not exactly B2's public posterior (b2: species
    key -> the source indices B2 holds). Full variant identity (source index, so Hidden Power type too), both directions; a
    species only one side holds disagrees too (an empty encounter stands in for a species only B2 holds). Damage-excluded
    variants are a strict subset by design, so the hard support is compared, never the weighted one."""
    out = dict(posts)
    seat = next((post.seat for post in posts.values()), '')
    for key in sorted(set(posts) | set(b2)):
        post = posts.get(key)
        theirs = frozenset(b2[key]) if key in b2 else None
        if post is not None and theirs == frozenset(i for i, h in enumerate(post.hard) if h):
            continue
        if post is None:
            post = EncounterPosterior(seat, key, (), (), (), (), (), '')
        out[key] = replace(post, refusals=post.refusals + (B2_DISAGREEMENT,))
    return out


def summary(post: EncounterPosterior) -> dict[str, Any]:
    """One encounter's diagnostic record (JSON-ready): hard and weighted support sizes, strikes admitted, refused by reason
    (off-model drops and single-variant strikes among them), and the provenance."""
    refused: dict[str, int] = {}
    for strike in post.strikes:
        if not strike.admitted:
            refused[strike.reason] = refused.get(strike.reason, 0) + 1
    return dict(species=post.species, hard=sum(post.hard), weighted=sum(post.support), admitted=sum(s.admitted for s in post.strikes),
                refused=dict(sorted(refused.items())), off_model_drops=refused.get(OFF_MODEL, 0), provenance=post.provenance)


# --- Strikes from the owner's lines (the census's parse; a strike is classified when its window has closed). ---


def _seat(ident: Any) -> str | None:
    text = str(ident or '')
    return text[:2] if text[:2] in ('p1', 'p2') else None


def _hp(condition: str) -> int | None:
    head = condition.split()[0] if condition else ''
    if 'fnt' in condition and not head[:1].isdigit():
        return 0
    try:
        return int(head.split('/')[0])
    except ValueError:
        return None


def _tagged(parts: Sequence[str], start: int) -> bool:
    return any(x.startswith('[from]') for x in parts[start:])


def _substitute(q: Sequence[str], kind: str) -> bool:
    """A hit on a Substitute: |-activate|X|Substitute|[damage] (Gen 3 inherits gen4/moves.ts), |-activate|X|move: Substitute|
    [damage] (moves.ts) or the break, |-end|X|Substitute."""
    return kind in ('-activate', '-end') and len(q) > 3 and normalize_id(q[3].split(':', 1)[-1]) == 'substitute'


def _first_request(lines: Sequence[str]) -> Mapping[str, Any] | None:
    for line in lines:
        if line.startswith('|request|'):
            try:
                request = json.loads(line[len('|request|'):])
            except ValueError:
                continue
            if isinstance(request, Mapping) and request.get('side'):
                return request
    return None


def parse_strikes(lines: Sequence[str], owner: str, *, include_outgoing: bool = False) -> list[dict[str, Any]]:
    """Incoming observations; optionally merge our full-health outgoing hits in line order.

    Both scans use the original lines and the actual attacker-oriented contexts. Outgoing HP is public percent, never an
    exact opponent HP value. Existing census callers can keep requesting the incoming stream alone.
    """
    incoming = _parse_strikes(lines, owner)
    if not include_outgoing:
        return incoming
    foe = 'p2' if owner == 'p1' else 'p1'
    return sorted(incoming + _parse_strikes(lines, foe, public_target=True), key=lambda strike: strike['line'])


def _parse_strikes(lines: Sequence[str], owner: str, *, public_target: bool = False) -> list[dict[str, Any]]:
    """Every opponent strike that damaged (or was absorbed by) our active, in line order, each with the census's refusal
    reason (None: admitted to the damage likelihood) and its observation (our HP before, the loss, KO, crit, effectiveness)."""
    me = owner
    foe = 'p2' if me == 'p1' else 'p1'
    contexts = tier2._strike_contexts(lines)
    hp = maxhp = None
    fresh_sub = None
    pending_sub = False
    active: dict[str, str] = {}
    weather = ''
    effects: dict[str, set[str]] = {'p1': set(), 'p2': set()}   # -start volatiles and -prepare states, cleared on switch
    item_changed: set[tuple[str, str]] = set()                  # (side, species key) after a Thief / Covet / Trick / Knock Off
    hit_by = {'p1': False, 'p2': False}                         # side was hit by the other side's move this turn (Revenge)
    strikes = []
    count = len(lines)
    # A prefix that ends at a request is a decision boundary: the battle is waiting, so every window still open there (a KO
    # before a forced switch, say) has printed all its lines. Any other end of the lines leaves an open window unclassified.
    boundary = bool(lines) and lines[-1].startswith('|request|')
    for i, line in enumerate(lines):
        p = line.split('|')
        if len(p) < 2:
            continue
        kind = p[1]
        if kind == 'turn':
            hit_by = {'p1': False, 'p2': False}
            continue
        if kind == '-weather' and len(p) > 2:
            weather = '' if normalize_id(p[2]) == 'none' else normalize_id(p[2])
            continue
        if len(p) < 3:
            continue
        side = _seat(p[2])
        if kind in ('switch', 'drag', 'replace') and side and len(p) > 4:
            active[side] = p[3].split(',')[0]
            effects[side] = set()
            if side == me:
                fresh_sub = None
                pending_sub = False
                hp = _hp(p[4])
                head = p[4].split()[0] if p[4] else ''
                maxhp = int(head.split('/')[1]) if '/' in head and head.split('/')[1].isdigit() else None
            continue
        if kind in ('-damage', '-heal', '-sethp') and len(p) > 3:
            # -sethp may carry both seats (Pain Split). A missing/invalid condition invalidates prior HP evidence.
            positions = range(2, len(p) - 1, 2) if kind == '-sethp' else (2,)
            for pos in positions:
                if _seat(p[pos]) != me:
                    continue
                condition = p[pos + 1]
                hp = _hp(condition)
                head = condition.split()[0] if condition else ''
                if public_target:
                    maximum = head.split('/')[1] if '/' in head else None
                    maxhp = int(maximum) if maximum and maximum.isdigit() else (maxhp if hp == 0 else None)
            continue
        if kind == 'faint' and side == me:
            fresh_sub = None
            pending_sub = False
        if kind == '-activate' and side == me and _substitute(p, kind):
            fresh_sub = None
        if kind in ('-start', '-end') and side and len(p) > 3:
            name = normalize_id(p[3].split(':', 1)[-1])
            if side == me and name == 'substitute':
                fresh_sub = maxhp // 4 if kind == '-start' and pending_sub and maxhp and not public_target else None
                pending_sub = False
            (effects[side].add if kind == '-start' else effects[side].discard)(name)
            continue
        if kind == '-prepare' and side and len(p) > 3:
            effects[side].add('prepare:' + normalize_id(p[3]))
            continue
        if kind in ('-item', '-enditem') and side and any(x.startswith('[from] move:') for x in p[4:]):
            item_changed.add((side, species_key(active.get(side, ''))))
            continue
        if kind != 'move' or side is None or len(p) < 4:
            continue
        mover = side
        move = tier2.canonical_move_id(p[3])
        flags = [x for x in p[5:] if x]
        pending_sub = mover == me and move == 'substitute' and not public_target
        if not any(f == '[still]' for f in flags):   # the move releases its user's charged semi-invulnerable state
            effects[mover] = {e for e in effects[mover] if not e.startswith('prepare:')}
        j, window, closed = i + 1, [], boundary
        while j < count:
            q = lines[j].split('|')
            kq = q[1] if len(q) > 1 else ''
            if lines[j].startswith('|request|') or kq == 't:':
                j += 1
                continue
            if kq in _WINDOW_END or (kq == '-end' and len(q) > 3
                    and normalize_id(q[3].split(':', 1)[-1]) in ('futuresight', 'doomdesire')):
                closed = True
                break
            window.append((j, q, kq))
            j += 1
        if mover == me:
            # Our move: did it hit the foe (Revenge doubles), on its HP or its Substitute?
            if any(len(q) > 3 and _seat(q[2]) == foe and ((kq == '-damage' and not _tagged(q, 4)) or _substitute(q, kq))
                   for _, q, kq in window):
                hit_by[foe] = True
            continue
        damage = [(j, q) for j, q, kq in window if kq == '-damage' and len(q) > 3 and _seat(q[2]) == me and not _tagged(q, 4)]
        substitute = any(_substitute(q, kq) and _seat(q[2]) == me for _, q, kq in window)
        endure = any(kq == '-activate' and len(q) > 3 and _seat(q[2]) == me and q[3] in ('move: Endure', 'item: Focus Band')
                     for _, q, kq in window)
        if not damage and not substitute and not endure:
            continue
        crit = any(kq == '-crit' and _seat(q[2]) == me for _, q, kq in window)
        hitcount = any(kq == '-hitcount' for _, q, kq in window)
        eff = next((kq[1:] for _, q, kq in window if kq in ('-supereffective', '-resisted') and _seat(q[2]) == me), None)
        pursuit = '[from] Pursuit' in flags or any(lines[x].startswith('|-activate|') and _seat(lines[x].split('|')[2]) == me
                                                   and 'move: Pursuit' in lines[x] for x in range(max(0, i - 2), i))
        calls = [f[len('[from]'):].strip() for f in flags if f.startswith('[from]') and f != '[from] Pursuit']
        call_ids = [normalize_id(f.split(':', 1)[1] if f.lower().startswith('move:') else f) for f in calls]
        sleep_talk = call_ids == ['sleeptalk'] and not public_target
        called = bool(calls) and not sleep_talk
        context = contexts.get(i)
        observed, ko = None, False
        if len(damage) == 1 and hp is not None:
            after = _hp(damage[0][1][3])
            if after is not None:
                observed, ko = hp - after, after == 0
        sub_hp = None
        if substitute:
            activations = [q for _, q, kq in window if kq == '-activate' and _seat(q[2]) == me
                           and _substitute(q, kq) and '[damage]' in q[4:]]
            broken = any(kq == '-end' and _seat(q[2]) == me and _substitute(q, kq) for _, q, kq in window)
            if not damage and len(activations) == 1 and not broken and fresh_sub and not public_target:
                sub_hp = fresh_sub
            # Even an unsupported first hit spends the exact fresh capacity. Later hits cannot reuse it.
            fresh_sub = None
        if public_target and not (hp == maxhp == 100):
            continue
        after_percent = _hp(damage[0][1][3]) if public_target and len(damage) == 1 else None
        if public_target and damage:
            condition = damage[0][1][3].split()[0]
            if (after_percent != 0 and not condition.endswith('/100')) or after_percent is None or not 0 <= after_percent <= 100:
                after_percent = None
        strike = dict(line=i, last=window[-1][0] if window else i, attacker=active.get(foe) or (context.attacker_species if context else ''),
                      defender=active.get(me) or (context.defender_species if context else ''), move=move, crit=crit, weather=weather,
                      pursuit=pursuit, hp_before=hp, maxhp=maxhp, observed=observed, ko=ko, eff=eff, context=context)
        if public_target:
            strike['observed'] = None
        strike.update(attacker_seat=foe, defender_seat=me, posterior_key=species_key(strike['defender'] if public_target else strike['attacker']),
                      observation_kind='public-hp' if public_target else ('substitute' if sub_hp is not None else 'exact'),
                      sub_hp=sub_hp, after_percent=after_percent, sleep_talk=sleep_talk)
        strike['attacker_key'] = species_key(strike['attacker'])   # once per strike, not once per (strike, species)
        strike['reason'] = _refusal(strike, closed=closed, called=called, hitcount=hitcount, damage=damage, substitute=substitute,
                                    endure=endure, hit_by=hit_by[foe], effects=effects, me=me, foe=foe, item_changed=item_changed)
        strikes.append(strike)
    return strikes


def _refusal(s, *, closed, called, hitcount, damage, substitute, endure, hit_by, effects, me, foe, item_changed) -> str | None:
    move, context = s['move'], s['context']
    if not closed:
        return 'window-open'   # the move's lines may run past the given lines (not a decision prefix): never guessed
    if called:
        return 'called'
    if move == 'struggle':
        return 'struggle'
    if move in FIXED_DAMAGE_MOVES:
        return 'fixed-damage'
    if move in OHKO_MOVES:
        return 'ohko'
    if move in CONDITIONAL_POWER_MOVES:
        return 'conditional-power:' + move
    if move in ('eruption', 'waterspout') and (context is None or context.attacker_hp_fraction < 1.0):
        return 'conditional-power:' + move
    if move == 'revenge' and hit_by:
        return 'conditional-power:revenge'
    if hitcount or len(damage) > 1 or move in MULTI_HIT_MOVES:
        return 'multi-hit'
    if substitute and s.get('sub_hp') is None:
        return 'hit-substitute'
    if endure:
        return 'endure-or-focusband'
    if context is None:
        return 'no-context'
    if context.attacker_transformed or context.defender_transformed:
        return 'transform'
    if context.attacker_type_changed or context.defender_type_changed:
        return 'type-change'
    if (context.attacker_item_mutated or context.defender_item_mutated or (foe, s['attacker_key']) in item_changed
            or (me, species_key(s['defender'])) in item_changed):
        return 'item-changed'
    if context.attacker_ability_overridden or context.defender_ability_overridden:
        return 'ability-overridden'
    if any(e.startswith('prepare:') and e[len('prepare:'):] in SEMI_INVULNERABLE for e in effects[me]):
        return 'defender-semi-invulnerable'
    if 'charge' in effects[foe]:
        return 'unmodelled:charge'
    if (effects['p1'] | effects['p2']) & SPORTS:
        return 'unmodelled:mud/water-sport'
    if context.defender_screens:
        return 'unvalidated:screen'   # the core models screens, but no r5a strike exercised one (scope §1a)
    if s.get('observation_kind') == 'substitute':
        return None
    if s.get('observation_kind') == 'public-hp':
        if s.get('after_percent') is None:
            return 'hp-unknown'
        return 'no-hp-drop' if s['after_percent'] == 100 else None
    if s['hp_before'] is None or s['observed'] is None:
        return 'hp-unknown'
    if s['observed'] <= 0:
        return 'no-hp-drop'
    return None


# --- Per-variant rolls (tier2's shared core; participants as the census and the world constructor build them). ---

_PARTICIPANTS: dict[tuple[str, str, str], tier2.StrikeParticipant | None] = {}


def _variant_participant(source: Gen3RandbatSource, species: str, variant: Gen3RandbatVariant) -> tier2.StrikeParticipant | None:
    key = (source.metadata.source_hash, species_key(species), variant.variant_id)
    if key not in _PARTICIPANTS:
        result = None
        fixture = _worlds._fixture_from_variant_payload(variant.to_summary(), fallback_species=species, set_source=source)
        if fixture is not None:
            base = _worlds._base_stats_for_species(source, fixture.species)
            if base is not None:
                stats = _worlds._computed_stats(base, evs=fixture.evs, ivs=fixture.ivs, level=fixture.level)
                result = tier2.StrikeParticipant(species_key=species_key(species), level=fixture.level, stats=stats,
                    ability=normalize_id(fixture.ability or ''), item=normalize_id(fixture.item or ''),
                    moves=tuple(tier2.canonical_move_id(m) for m in fixture.moves))
        _PARTICIPANTS[key] = result
    return _PARTICIPANTS[key]


def _own_participant(team, species: str, dex: ShowdownDex, max_hp: int | None):
    rows = [m for m in team if species_key(m.species) == species_key(species)]
    if len(rows) != 1:
        return None, ()
    mon = rows[0]
    stats = dict(mon.stats)
    if 'hp' not in stats and max_hp is not None:
        stats['hp'] = max_hp   # a fainted member's request row shows no maximum; our own switch line does, exactly
    if 'hp' not in stats:
        return None, ()
    info = dex.species_info(species_key(mon.species))
    return (tier2.StrikeParticipant(species_key=species_key(mon.species), level=mon.level, stats=stats,
                                    ability=normalize_id(mon.ability or ''), item=normalize_id(mon.item or ''), moves=tuple(mon.moves)),
            tuple(info.types) if info else ())


def _effectiveness(attacker: tier2.StrikeParticipant, move: str, defender_types: tuple[str, ...], dex: ShowdownDex) -> float:
    """The type effectiveness of this attacker's own move on our mon (Hidden Power resolves to its set's type), from the dex's
    chart, as the core computes it (tier2._strike_damage: dex.effectiveness of the participant's move type)."""
    info = dex.move_info(tier2._participant_move_id(attacker, move))
    return dex.effectiveness(info.type if info else None, defender_types)


def _effectiveness_line(attacker: tier2.StrikeParticipant, move: str, defender_types: tuple[str, ...], dex: ShowdownDex) -> str | None:
    """The effectiveness line this attacker's own move prints on our mon."""
    value = _effectiveness(attacker, move, defender_types, dex)
    return 'supereffective' if value > 1 else 'resisted' if 0 < value < 1 else None


def _counts(strike, species, variants, hard, own_team, *, source, dex):
    """(refusal, None) or (None, per-variant fitting roll counts over the hard support; 0 outside it, not computed)."""
    outgoing = strike.get('observation_kind') == 'public-hp'
    if outgoing:
        own_attacker, _ = _own_participant(own_team, strike['attacker'], dex, None)
        info = dex.species_info(species_key(species))
        types = tuple(info.types) if info else ()
        if own_attacker is None or not types:
            return 'unknown-attacker/defender', None
    else:
        defender, types = _own_participant(own_team, strike['defender'], dex, strike['maxhp'])
        if defender is None:
            return 'unknown-defender', None
        if strike['maxhp'] is not None and defender.stats['hp'] != strike['maxhp']:
            return 'defender-maxhp-disagrees', None
    if not any(hard):
        return 'no-support', None
    token = SimpleNamespace(crit=strike['crit'], weather=strike['weather'], pursuit_intercept=strike['pursuit'])
    counts = []
    for index, variant in enumerate(variants):
        if not hard[index]:
            counts.append(0)
            continue
        if outgoing:
            attacker, defender = own_attacker, _variant_participant(source, species, variant)
        else:
            # A Sleep Talk callee is a literal move in its user's set, even when the belief does not reveal called moves.
            if strike.get('sleep_talk') and not variant.matches(revealed_moves=(strike['move'],)):
                counts.append(0)
                continue
            attacker = _variant_participant(source, species, variant)
        result = None if attacker is None or defender is None else tier2._strike_damage(attacker=attacker, defender=defender, defender_types=types,
            observed_move=strike['move'], token=token, context=strike['context'], dex=dex, require_move_in_set=True,
            pinch_ambiguity_band=_PINCH_BAND, split_choice_band_baseline=False)
        if (result is not None and not result.rolls and (strike.get('sub_hp') is not None or outgoing
                or (strike['observed'] is not None and strike['observed'] > 0))
                and _effectiveness(attacker, strike['move'], types, dex) == 0):
            # The core's no-damage case (gen3_damage_rolls: an empty tuple for an immune type; _strike_damage already
            # returned None for zero base power), confirmed as type immunity on the chart it used: a Hidden Power type
            # that cannot touch our mon. Damage was dealt, so this variant cannot be the attacker: it fits 0 rolls.
            counts.append(0)
            continue
        if result is None or len(result.rolls) != ROLLS:
            return 'core-unsupported:' + strike['move'], None
        if result.ambiguous:
            return 'ambiguous-band', None
        if outgoing:
            maximum = int(defender.stats['hp'])
            lo, hi = _public_hp_bounds(strike['after_percent'], maximum)
            fit = sum(lo <= max(0, maximum - roll) <= hi for roll in result.rolls)
        elif strike.get('sub_hp') is not None:
            fit = sum(0 < roll < strike['sub_hp'] for roll in result.rolls)
        elif strike['ko']:
            fit = sum(1 for roll in result.rolls if roll >= strike['hp_before'])
        else:
            fit = sum(1 for roll in result.rolls if roll == strike['observed'])
        if fit and _effectiveness_line(attacker, strike['move'], types, dex) != strike['eff']:
            return 'effectiveness-disagreement', None
        counts.append(fit)
    return None, tuple(counts)


def _public_hp_bounds(shown: int, maximum: int) -> tuple[int, int]:
    """Inverse Gen3 public ceil percent; below-full ceil100 is displayed as99. Pure integer bounds."""
    if type(maximum) is not int or maximum <= 0 or type(shown) is not int or not 0 <= shown <= 100:
        raise ValueError('positive maximum and public0..100percent required')
    if shown == 0:
        return 0, 0
    if shown == 100:
        return maximum, maximum
    return ((shown - 1) * maximum) // 100 + 1, maximum - 1 if shown == 99 else (shown * maximum) // 100


def _encounter(foe, owner, key, hard, rows, provenance) -> EncounterPosterior:
    weights = [1 if h else 0 for h in hard]
    strikes = []
    for strike, reason, counts in rows:
        common = dict(line=strike['line'], attacker=strike.get('attacker_seat', foe), defender=strike.get('defender_seat', owner), move=strike['move'], observed=strike['observed'],
                      ko=strike['ko'], crit=strike['crit'], observation_kind=strike.get('observation_kind', 'exact'),
                      sub_hp=strike.get('sub_hp'), after_percent=strike.get('after_percent'))
        if reason is not None:
            strikes.append(Strike(admitted=False, reason=reason, counts=None, **common))
            continue
        updated = [w * c for w, c in zip(weights, counts)]
        if not any(updated):   # off-model: no surviving variant explains it; never empty the support
            strikes.append(Strike(admitted=False, reason=OFF_MODEL, counts=None, **common))
            continue
        weights = updated
        strikes.append(Strike(admitted=True, reason='', counts=counts, **common))
    alive = [w for w in weights if w]
    if alive:
        divisor = math.gcd(*alive)
        weights = [w // divisor for w in weights]
    return EncounterPosterior(foe, key, hard, tuple(w > 0 for w in weights), tuple(weights), tuple(strikes), (), provenance)
