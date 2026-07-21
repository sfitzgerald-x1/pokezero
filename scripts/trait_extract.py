"""Pure events -> metrics for the checkpoint trait-tracking plan (versioned).

Consumes the events-*.jsonl.gz produced by trait_eval.py and emits metrics.json for one
(checkpoint x opponent-mode). Adding a trait later means re-running this over stored events,
not regenerating games. Every metric is derived only from the captured event schema (omniscient
protocol log + per-decision active-mon PP + starting movesets + terminal state).

Denominators for move-category rates are "games-present": games in which >=1 acting-seat mon
carries a category move. We report uses/game-present and carrier-use fraction, separating "the
generator didn't deal the move" from "the policy ignores it".
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
import sys
from collections import Counter, defaultdict

# Increment when persisted metric definitions change. Existing event captures
# can be re-extracted, and the version documents the resulting metric schema.
METRICS_VERSION = "trait_extract.v3"

# ---- frozen gen3 move-category lists (move ids: lowercased, no spaces/hyphens) ----
def mid(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())

STAT_BOOST = {mid(x) for x in ["Swords Dance","Dragon Dance","Calm Mind","Bulk Up","Agility","Amnesia",
    "Barrier","Acid Armor","Iron Defense","Cosmic Power","Tail Glow","Belly Drum","Growth","Meditate"]}
CURSE = mid("Curse")  # boost only when non-Ghost (handled via user type at use-time)
HEAL_NON_REST = {mid(x) for x in ["Recover","Softboiled","Milk Drink","Slack Off","Synthesis","Morning Sun","Moonlight"]}
WISH = mid("Wish")
REST = mid("Rest")
WEATHER_PRIMARY = {mid("Sunny Day"): "sun", mid("Rain Dance"): "rain"}
WEATHER_AUX = {mid("Sandstorm"): "sand", mid("Hail"): "hail"}
PHAZE = {mid("Roar"), mid("Whirlwind")}
SPIKES = mid("Spikes")
RAPID_SPIN = mid("Rapid Spin")
TOXIC = mid("Toxic")
PARA_MOVES = {mid("Thunder Wave"), mid("Stun Spore"), mid("Glare")}
SLEEP_MOVES = {mid("Sleep Powder"), mid("Spore"), mid("Hypnosis"), mid("Lovely Kiss"), mid("Sing"), mid("Grass Whistle")}
YAWN = mid("Yawn")
WILL_O_WISP = mid("Will-O-Wisp")
POISON_POWDER = mid("Poison Powder")
# every *dedicated status-only* move (a non-damaging move whose purpose is to inflict a status):
# paralysis + sleep + burn + poison, and Yawn (delayed sleep). This is the aggregate "how much does
# the checkpoint lean on status". Secondary statuses riding on attacking moves (Body Slam paralysis,
# Sludge Bomb poison, etc.) are deliberately excluded — those are not a status *choice*.
STATUS_MOVES = PARA_MOVES | SLEEP_MOVES | {TOXIC, YAWN, WILL_O_WISP, POISON_POWDER}
BOOM = {mid("Explosion"), mid("Self-Destruct")}
BATON_PASS = mid("Baton Pass")
SUBSTITUTE = mid("Substitute")
FOCUS_PUNCH = mid("Focus Punch")
LEECH_SEED = mid("Leech Seed")
SOLAR_BEAM = mid("Solar Beam")
KNOCK_OFF = mid("Knock Off")   # in gen3 randbats (162 carriers / 41 uses in a 2000-game sample)
# NOTE: Thief is NOT in the gen3 randbats pool (0 carriers empirically) — deliberately not tracked.
REVERSAL_MOVES = {mid("Reversal"), mid("Flail")}  # BP scales inversely with the user's HP fraction
BELLY_DRUM = mid("Belly Drum")
# Positive-priority *damaging* moves in the gen3 randbats pool. Priority lets a slower mon strike
# first; using one when the opponent outspeeds you (inferred from turn order) is the skilled
# application, and a priority move that lands the KO is its payoff. Fake Out is +1 too but is a
# first-turn flinch/utility move rather than a speed-circumventing revenge-KO, so it's left out here.
PRIORITY_MOVES = {mid(x) for x in ["Quick Attack", "Extreme Speed", "Mach Punch"]}
# Moves whose gen3 priority is NOT 0 — excluded from turn-order speed inference, since they reorder
# execution independent of speed and would corrupt the "who moved first => who's faster" read.
NONZERO_PRIORITY = PRIORITY_MOVES | PHAZE | {mid(x) for x in
    ["Fake Out", "Protect", "Detect", "Endure", "Counter", "Mirror Coat", "Vital Throw",
     "Focus Punch", "Snatch", "Magic Coat", "Follow Me", "Helping Hand"]}
DESTINY_BOND = mid("Destiny Bond")  # success = it drags the attacker down (an -activate fires)
INTIMIDATE = mid("Intimidate")
# absorb abilities that negate an incoming move AND heal (Volt/Water Absorb) or boost (Flash Fire).
ABSORB_ABILITIES = {mid("Volt Absorb"), mid("Water Absorb"), mid("Flash Fire")}
PROTECT_MOVES = {mid("Protect"), mid("Detect")}  # the boom-blocking protection moves (not Endure)
# v3-only traits (gen3 randbats runs Sleep Clause Mod; see docs/observation_v3_spec.md, PR #779).
AROMATHERAPY = {mid("Aromatherapy"), mid("Heal Bell")}  # party-wide status cure (identical twins)
NATURAL_CURE = mid("Natural Cure")   # ability: cures the mon's status on switch-out
STRUGGLE = mid("Struggle")
# Counter (physical) / Mirror Coat (special) — reactive damage-return moves (both on Wobbuffet in
# gen3 randbats). They fail SILENTLY (no -fail) when there's no matching damage to bounce; a success
# is the direct -damage they deal to the target (no [from] tag, unlike upkeep residuals).
COUNTER_MOVES = {mid("Counter"), mid("Mirror Coat")}


def reversal_bp(cur, mx):
    """gen3 Reversal/Flail base power from the user's HP: lower HP -> higher BP. A rising *average*
    BP over training means the policy learns to fire these at low HP (where they hit hardest)."""
    if not mx:
        return 0
    ratio = (cur * 48) // mx
    if ratio < 2:
        return 200
    if ratio < 6:
        return 150
    if ratio < 13:
        return 100
    if ratio < 22:
        return 80
    if ratio < 43:
        return 40
    return 20


def parse_hp(token):
    """'247/247' / '196/262 tox' / '0 fnt' -> (cur, max), or None."""
    t = (token or "").split()
    if t and "/" in t[0]:
        a, b = t[0].split("/", 1)
        try:
            return int(a), int(b)
        except ValueError:
            return None
    return None


def status_token(cond):
    """condition string -> its trailing status word ('196/262 tox' -> 'tox'), else None."""
    t = (cond or "").split()
    return t[1] if len(t) > 1 else None


def ability_in_args(a):
    """Return the mid() of an ability named in protocol args ('[from] ability: Volt Absorb' or
    'ability: Flash Fire'), else None."""
    for x in a:
        s = str(x)
        if "ability:" in s.lower():
            return mid(s.split("ability:")[-1])
    return None

OTHER_SEAT = {"p1": "p2", "p2": "p1"}

# Per-game trait -> win correlation. One row per (game, behavioral seat) with a decided outcome:
# x = the seat's count of the trait IN THAT GAME, y = 1 if that seat won. n is games, not
# checkpoints, so this has real power. In self-play both seats are the same policy and share the
# game, so the winner-vs-loser comparison is paired: it controls for policy strength AND game
# length (a game-level quantity has zero within-game variance and correctly falls out at r=0).
# Only *chosen* behaviors — nothing outcome-definitional, which would be circular. Deliberately
# excluded: `forced_switch` (a forced switch happens because your active mon fainted, so its count
# is essentially "mons lost" — correlating it with losing restates the outcome and buries the real
# effects at r~-0.6), and the mons-alive/closer metrics for the same reason.
PER_GAME_TRAITS = [
    "cat_stat_boost", "cat_toxic", "cat_substitute", "cat_spikes", "cat_heal", "cat_phaze",
    "cat_rest", "cat_sleep", "cat_para", "cat_leechseed", "cat_boom", "cat_batonpass",
    "cat_solarbeam", "cat_rapidspin_total", "cat_yawn", "cat_wish", "cat_weather_sun",
    "cat_weather_rain", "cat_curse",
    "pivot", "immunity_switchin", "switch_out_sleeping", "switch_out_frozen",
    "cat_phaze_justified", "cat_rapidspin_spikesdown", "cat_solarbeam_sun", "bp_stat_or_sub",
    "focuspunch_executed", "focuspunch_disrupted",
]


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def parse_line(line):
    """Return (tag, args) for a |tag|... protocol line, else (None, [])."""
    if not line.startswith("|"):
        return None, []
    parts = line.split("|")
    return parts[1], parts[2:]


def seat_of(ref):
    """'p1a: Zapdos' -> 'p1'."""
    return ref[:2] if ref[:2] in ("p1", "p2") else None


def species_of(ref):
    return ref.split(":", 1)[1].strip() if ":" in ref else ref


class GameParse:
    """Stateful walk of one game's omniscient protocol; accumulates per-seat trait events."""
    def __init__(self, movesets):
        self.movesets = movesets  # {seat: [{species, moves:[id...]}]}
        self.active = {"p1": None, "p2": None}
        self.boosts = {"p1": defaultdict(int), "p2": defaultdict(int)}
        self.status = {"p1": {}, "p2": {}}    # species -> status
        self.rest_sleep = {"p1": set(), "p2": set()}  # species whose slp came from their own Rest
        self.sub = {"p1": False, "p2": False}
        self.spikes = {"p1": 0, "p2": 0}      # current layers on that side
        self.spikes_max = {"p1": 0, "p2": 0}  # peak layers ever on that side (survives Rapid Spin)
        self.weather = None
        self.ev = defaultdict(lambda: Counter())   # seat -> Counter of trait events
        self.move_counts = {"p1": Counter(), "p2": Counter()}
        self.moves_total = {"p1": 0, "p2": 0}
        self.pending_faint = {"p1": False, "p2": False}
        self.last_move = {"p1": None, "p2": None}
        self.hp = {"p1": None, "p2": None}   # (cur, max) of the active mon, for Reversal/Flail BP
        self.bd = {"p1": None, "p2": None}   # open Belly-Drum KO window per seat: None or KO count
        self._pending_switch_immunity = None  # (seat, incoming) awaiting a resolved move this turn
        self.turn_neutral = []   # seats that made a priority-0 move this turn, in protocol order
        self.slower = {"p1": None, "p2": None}   # observed: is this seat's active mon slower than opp?
        self._pri_ko_seat = None  # a priority move by this seat is awaiting an immediate opponent KO
        self.protected = {"p1": False, "p2": False}  # used Protect/Detect this turn (reset each turn)
        self.tox = {"p1": None, "p2": None}   # active mon's badly-poison counter (None if not toxiced)
        self.seed = {"p1": None, "p2": None}  # turns a leech-seeded mon has stayed active (None if unseeded)
        self._switchin_seats = set()          # seats that switched in this turn (absorb-read window)
        self._boom_target = None              # seat a resolving enemy boom targets, awaiting immunity
        # v3-only trackers
        self.slept_by = {"p1": set(), "p2": set()}  # opp species this seat move-slept (Sleep Clause set)
        self._move_in_flight = None            # seat whose move is resolving, for -fail attribution
        self._nc_switchin = set()              # seats that switched a Natural Cure mon in this turn
        self._cm_pending = None                # seat whose Counter/Mirror Coat awaits its damage/whiff
        # Protect chain: was this seat's immediately-preceding move a *successful* Protect/Detect?
        # (consecutive Protect has a diminishing success chance in gen3, so repeating after a success
        # is the risky play we break out.) Reset by any non-Protect move, a failed Protect, or a switch.
        self.prev_protect_success = {"p1": False, "p2": False}
        self._protect_pending = None           # seat whose Protect awaits its -singleturn/-fail resolution

    def carriers(self, seat, move_id):
        return sum(1 for m in self.movesets.get(seat, []) if move_id in {mid(x) for x in m["moves"]})

    def ability_of(self, seat, species):
        """The captured ability of `species` on this seat's team (v3 movesets), else '' (older data)."""
        for m in self.movesets.get(seat, []):
            if m.get("species") == species:
                return mid(m.get("ability", ""))
        return ""

    def walk(self, protocol):
        for line in protocol:
            tag, a = parse_line(line)
            if tag == "switch" or tag == "drag":
                seat = seat_of(a[0]); sp = species_of(a[0])
                if seat:
                    voluntary = not self.pending_faint[seat] and tag == "switch"
                    # baton pass: previous move by seat was Baton Pass
                    if voluntary and self.last_move[seat] == BATON_PASS:
                        self.ev[seat]["bp_switch"] += 1
                        # a *meaningful* Baton Pass carries a stat boost or a Substitute to the
                        # incoming mon. self.boosts/sub still hold the OUTGOING mon's state here
                        # (reset happens below), so this is what's being passed.
                        if any(v > 0 for v in self.boosts[seat].values()) or self.sub[seat]:
                            self.ev[seat]["bp_stat_or_sub"] += 1
                    elif voluntary:
                        self.ev[seat]["pivot"] += 1
                        # sleeping/frozen switch-out of the OUTGOING mon. Rest-induced sleep is a
                        # self-chosen heal, not the enemy-sleep-preservation behavior we track here.
                        prev = self.active[seat]
                        st = self.status[seat].get(prev)
                        if st == "slp" and prev not in self.rest_sleep[seat]:
                            self.ev[seat]["switch_out_sleeping"] += 1
                        elif st == "frz":
                            self.ev[seat]["switch_out_frozen"] += 1
                    else:
                        self.ev[seat]["forced_switch"] += 1
                    # the outgoing mon leaves: close any open Belly-Drum KO window it held
                    self._close_bd(seat)
                    self._end_tox(seat)   # the outgoing mon's toxic counter resets on switch (gen3)
                    self._end_seed(seat)  # leech seed is cleared on switch — the episode ends here
                    self.prev_protect_success[seat] = False  # a switch breaks the consecutive-Protect chain
                    self.pending_faint[seat] = False
                    self.active[seat] = sp
                    self.boosts[seat] = defaultdict(int)
                    self.sub[seat] = False
                    self.hp[seat] = parse_hp(a[2]) if len(a) > 2 else None  # "Species, L98, F|247/247"
                    self.slower[seat] = None  # fresh mon: its speed vs the opponent is unknown again
                    # a badly-poisoned mon coming (back) in restarts its toxic counter at 0 (gen3)
                    self.tox[seat] = 0 if status_token(a[2] if len(a) > 2 else None) == "tox" else None
                    self._switchin_seats.add(seat)  # absorb-ability reads credited on the switch-in turn
                    if self.ability_of(seat, sp) == NATURAL_CURE:
                        self._nc_switchin.add(seat)  # a Natural Cure mon came in — a status it eats cures free
                    self._move_in_flight = None      # a switch ends any prior move's -fail window
                    # a switch-in may materialize immunity to the opponent's move this turn
                    self._pending_switch_immunity = (seat, sp)
            elif tag == "move":
                seat = seat_of(a[0]); move = mid(a[1]) if len(a) > 1 else None
                # A move re-emitted with `[from] lockedmove` is the FORCED continuation of a
                # multi-turn move, not a chosen action: Solar Beam charges (`[still]` + `-prepare`)
                # and is re-emitted next turn locked. Counting that line double-counts the move —
                # and because a Solar Beam only charges when there is NO sun, the double-count lands
                # entirely on the no-sun side and deflates the in-sun rate (v22-lr3m@2600k read
                # 91.0% but is 95.1% at decision level). The charge is the decision; skip the fire.
                if any("lockedmove" in str(x) for x in a[2:]):
                    continue
                if seat and move:
                    self.moves_total[seat] += 1
                    self.move_counts[seat][a[1]] += 1
                    self.last_move[seat] = move
                    # any newly-resolving move ends a prior priority-KO window (target survived to act)
                    self._pri_ko_seat = None
                    self._cm_pending = None   # a new move ends a prior Counter/Mirror Coat window (whiffed)
                    if move in PRIORITY_MOVES:
                        self._pri_ko_seat = seat            # awaiting an immediate opponent faint
                    elif move not in NONZERO_PRIORITY:
                        self.turn_neutral.append(seat)      # a priority-0 move: usable for speed inference
                    self._classify_move(seat, move, a)
                    self._move_in_flight = seat         # a -fail before the next action = this move failed
            elif tag == "cant":
                # e.g. |cant|p1a: X|Focus Punch  (disrupted) — record attempt+disruption
                seat = seat_of(a[0])
                reason = a[1] if len(a) > 1 else ""
                if seat and mid(reason) == FOCUS_PUNCH:
                    self.ev[seat]["focuspunch_attempt"] += 1
                    self.ev[seat]["focuspunch_disrupted"] += 1
            elif tag == "-boost":
                seat = seat_of(a[0])
                if seat:
                    self.boosts[seat][a[1]] += int(a[2]) if len(a) > 2 else 1
            elif tag == "-unboost":
                seat = seat_of(a[0])
                if seat:
                    self.boosts[seat][a[1]] -= int(a[2]) if len(a) > 2 else 1
            elif tag in ("-setboost",):
                seat = seat_of(a[0])
                if seat:
                    self.boosts[seat][a[1]] = int(a[2]) if len(a) > 2 else 0
            elif tag == "-status":
                seat = seat_of(a[0])
                if seat:
                    sp = species_of(a[0])
                    self.status[seat][sp] = a[1] if len(a) > 1 else None
                    if len(a) > 1 and a[1] == "tox":
                        self.tox[seat] = 0  # badly poisoned; counter ticks up each end-of-turn
                    if len(a) > 1 and a[1] == "slp":
                        # sleep from the mon's own Rest, flagged by the protocol's [from] tag
                        # (fallback: the seat's last resolved move was Rest, and Rest is self-target).
                        from_rest = any("move: rest" in str(x).lower() for x in a[2:]) or \
                            (self.last_move[seat] == REST and sp == self.active[seat])
                        (self.rest_sleep[seat].add if from_rest else self.rest_sleep[seat].discard)(sp)
                        if sp not in self.rest_sleep[seat]:
                            # move-induced sleep (not Rest): the inducer's Sleep-Clause bit turns on
                            self.slept_by[OTHER_SEAT[seat]].add(sp)
            elif tag == "-curestatus":
                seat = seat_of(a[0])
                if seat:
                    sp = species_of(a[0])
                    self.status[seat].pop(sp, None)
                    self.rest_sleep[seat].discard(sp)
                    if len(a) > 1 and a[1] == "slp":
                        self.slept_by[OTHER_SEAT[seat]].discard(sp)  # sleeper woke -> clause bit may clear
                    if len(a) > 1 and a[1] == "tox":
                        self._end_tox(seat)  # cured (Rest/heal bell/natural cure): episode ends
            elif tag == "-cureteam":
                # Aromatherapy cures the whole party in one line (no per-mon -curestatus) — clear the
                # tracked status and the opponent's Sleep-Clause bits for our now-awake slots.
                seat = seat_of(a[0])
                if seat:
                    for sp, st in list(self.status[seat].items()):
                        if st == "slp":
                            self.slept_by[OTHER_SEAT[seat]].discard(sp)
                    self.status[seat] = {}
                    self.rest_sleep[seat] = set()
            elif tag == "-start":
                seat = seat_of(a[0])
                if seat and len(a) > 1 and mid(a[1]) == SUBSTITUTE:
                    self.sub[seat] = True
                if seat and len(a) > 1 and LEECH_SEED in mid(a[1]):
                    self.seed[seat] = 0           # this mon is now leech-seeded; count its turns in
                if seat:
                    self._check_absorb(seat, a)   # Flash Fire boost rides a -start line
            elif tag == "-end":
                seat = seat_of(a[0])
                if seat and len(a) > 1 and mid(a[1]) == SUBSTITUTE:
                    self.sub[seat] = False
                if seat and len(a) > 1 and LEECH_SEED in mid(a[1]):
                    self._end_seed(seat)          # seed removed (e.g. Rapid Spin): episode ends
            elif tag in ("-damage", "-heal", "-sethp"):
                seat = seat_of(a[0])
                hp = parse_hp(a[1]) if len(a) > 1 else None
                if seat and hp:
                    self.hp[seat] = hp
                # a badly-poisoned mon's end-of-turn psn tick escalates its counter (1/16, 2/16, ...).
                if seat and self.tox.get(seat) is not None and any("psn" in str(x).lower() for x in a[2:]):
                    self.tox[seat] += 1
                # each Leech Seed drain (a -damage on the seeded mon; the seeder's -heal is skipped)
                # is one more turn the seeded mon stayed in.
                if (tag == "-damage" and seat and self.seed.get(seat) is not None
                        and any("leech seed" in str(x).lower() for x in a[2:])):
                    self.seed[seat] += 1
                if tag == "-heal" and seat:
                    self._check_absorb(seat, a)   # Volt/Water Absorb heal rides a -heal line
                # Counter/Mirror Coat success: a direct hit on the target (no [from]) after its use.
                if (tag == "-damage" and self._cm_pending is not None
                        and seat == OTHER_SEAT[self._cm_pending] and not any("[from]" in str(x) for x in a[2:])):
                    self.ev[self._cm_pending]["cm_success"] += 1
                    self._cm_pending = None
                # residual damage (poison/burn/Leech Seed/sand — carries a [from]) is not the priority
                # move's hit: don't let a later residual faint be miscredited as a priority KO.
                if (self._pri_ko_seat is not None and seat == OTHER_SEAT[self._pri_ko_seat]
                        and any("[from]" in str(x) for x in a[2:])):
                    self._pri_ko_seat = None
            elif tag == "-activate":
                # Destiny Bond succeeds (drags the attacker down) exactly when its -activate fires,
                # tagged on the DB user's seat — the same seat that chose the move.
                seat = seat_of(a[0])
                if seat and len(a) > 1 and DESTINY_BOND in mid(a[1]):
                    self.ev[seat]["destinybond_success"] += 1
            elif tag == "-weather":
                self.weather = mid(a[0]) if a else None
            elif tag == "-sidestart":
                seat = seat_of(a[0])
                if seat and len(a) > 1 and mid(a[1].replace("move: ", "")) == SPIKES:
                    self.spikes[seat] += 1
                    self.spikes_max[seat] = max(self.spikes_max[seat], self.spikes[seat])
            elif tag == "-sideend":
                seat = seat_of(a[0])
                if seat and len(a) > 1 and mid(a[1].replace("move: ", "")) == SPIKES:
                    self.spikes[seat] = 0
            elif tag == "-immune":
                seat = seat_of(a[0])
                # immunity that materialized right after a switch-in by this seat. An absorb ability
                # negating the hit at full HP surfaces here too — by design it counts as BOTH an
                # immunity switch-in AND (via _check_absorb) the more advantageous heal/boost read.
                if self._pending_switch_immunity and self._pending_switch_immunity[0] == seat:
                    self.ev[seat]["immunity_switchin"] += 1
                    self._pending_switch_immunity = None
                if seat:
                    self._check_absorb(seat, a)   # full-HP Volt/Water Absorb / Flash Fire negation
                    # a Ghost (or otherwise immune mon) eats the enemy's boom -> a boom block
                    if self._boom_target == seat:
                        self.ev[seat]["boom_block"] += 1
                        self._boom_target = None
            elif tag == "-singleturn":
                seat = seat_of(a[0])
                if seat and len(a) > 1 and "protect" in mid(a[1]):
                    self.protected[seat] = True   # Protect/Detect succeeded this turn (blocks a boom)
                    if self._protect_pending == seat:
                        self.prev_protect_success[seat] = True   # this Protect landed
                        self._protect_pending = None
            elif tag == "-ability":
                seat = seat_of(a[0])
                if seat and len(a) > 1 and mid(a[1]) == INTIMIDATE:
                    self.ev[seat]["intimidate_activation"] += 1
            elif tag == "-fail":
                # a move that failed within its action window -> the move in flight failed. Switch-window
                # fails (blocked switch-in Intimidate, etc.) are excluded: a switch clears _move_in_flight.
                if self._move_in_flight is not None:
                    self.ev[self._move_in_flight]["move_failed"] += 1
                    self._move_in_flight = None
                s = seat_of(a[0])
                if s and self._protect_pending == s:
                    self.prev_protect_success[s] = False   # this Protect failed (e.g. the consecutive penalty)
                    self._protect_pending = None
            elif tag == "faint":
                seat = seat_of(a[0])
                if seat:
                    self.pending_faint[seat] = True
                    self._end_tox(seat)   # a mon that fainted while toxiced reached its peak stage
                    self._end_seed(seat)  # a mon that fainted while seeded closes its turns-in count
                    self.slept_by[OTHER_SEAT[seat]].discard(species_of(a[0]))  # fainted sleeper clears clause
                    self.status[seat].pop(species_of(a[0]), None)  # a fainted mon carries no status
                    self.rest_sleep[seat].discard(species_of(a[0]))
                    # a priority move that just landed (no other move since) took this KO
                    if self._pri_ko_seat is not None and seat == OTHER_SEAT[self._pri_ko_seat]:
                        self.ev[self._pri_ko_seat]["cat_priority_ko"] += 1
                        self._pri_ko_seat = None
                    # a mon fainting while the OTHER seat is mid-Belly-Drum-window = a KO credited to
                    # that drummer (proxy: any opp faint while the drummer is the active mon).
                    drummer = OTHER_SEAT[seat]
                    if self.bd[drummer] is not None:
                        self.bd[drummer] += 1
                    # the drummer itself fainting closes its own window
                    self._close_bd(seat)
            elif tag == "turn":
                self._pending_switch_immunity = None  # immunity window is the switch-in turn only
                self._resolve_speed()                 # score the just-finished turn's move order
                self._pri_ko_seat = None
                self.protected = {"p1": False, "p2": False}
                self._switchin_seats = set()
                self._boom_target = None
                self._nc_switchin = set()
                self._move_in_flight = None
                self._protect_pending = None
                self._cm_pending = None
        # game over: flush any Belly-Drum windows still open (drummer survived to the end) and any
        # toxic episode a still-active mon was in (its peak stage is whatever it reached at game end).
        for s in ("p1", "p2"):
            self._close_bd(s)
            self._end_tox(s)
            self._end_seed(s)
            # peak Spikes layers seat s stacked on the opponent (Spikes are laid on the other side)
            self.ev[s]["spikes_max_achieved"] = self.spikes_max[OTHER_SEAT[s]]

    def _close_bd(self, seat):
        # close a Belly-Drum KO window: bank its KOs into the running sum, and — separately — count
        # it as a *success* if it produced at least one KO. The average-KOs figure is skewed by how
        # many opponents remain, so the fraction of uses that convert to any KO is the cleaner "was
        # the setup worth it" read; both are reported.
        if self.bd[seat] is not None:
            self.ev[seat]["bellydrum_ko_sum"] += self.bd[seat]
            if self.bd[seat] >= 1:
                self.ev[seat]["bellydrum_success"] += 1
            self.bd[seat] = None

    def _end_tox(self, seat):
        # record the peak toxic counter this episode reached, then close it. Low peaks across a run =
        # the policy switches toxiced mons out early to preserve HP; high peaks = it leaves them in.
        if self.tox[seat] is not None:
            self.ev[seat]["tox_stage_sum"] += self.tox[seat]
            self.ev[seat]["tox_episodes"] += 1
            self.tox[seat] = None

    def _end_seed(self, seat):
        # record how many turns this leech-seeded mon stayed active, then close the episode. Low
        # averages = the policy pivots seeded mons out (leech seed clears on switch); high = it eats
        # the drain. Read the same way as avg toxic stage.
        if self.seed[seat] is not None:
            self.ev[seat]["seed_turns_sum"] += self.seed[seat]
            self.ev[seat]["seed_episodes"] += 1
            self.seed[seat] = None

    def _check_absorb(self, seat, a):
        # Volt/Water Absorb (heal) or Flash Fire (boost) negating an incoming move. Any activation
        # marks the ability present; one on the switch-in turn is the "read" we credit.
        if ability_in_args(a) in ABSORB_ABILITIES:
            self.ev[seat]["absorb_activation"] += 1
            if seat in self._switchin_seats:
                self.ev[seat]["absorb_switchin"] += 1
                self._switchin_seats.discard(seat)  # credit the read once per switch-in

    def _resolve_speed(self):
        # A turn where both seats used a priority-0 move reveals who is faster: the seat whose move
        # was emitted first. This reflects EFFECTIVE speed in context (paralysis quarter-speed,
        # Swift Swim/Chlorophyll under weather, speed boosts) — exactly what determines whether a
        # priority move was actually needed. Re-observed each turn, so mid-game boosts self-correct.
        tn = self.turn_neutral
        self.turn_neutral = []
        if len(tn) >= 2 and tn[0] != tn[1]:
            self.slower[tn[0]] = False
            self.slower[tn[1]] = True

    def _classify_move(self, seat, move, a):
        opp = OTHER_SEAT[seat]
        E = self.ev[seat]
        if move in STAT_BOOST:
            E["cat_stat_boost"] += 1
        if move in REVERSAL_MOVES:
            # score the BP this use would have had, from the user's HP at decision time
            E["cat_reversal"] += 1
            hp = self.hp.get(seat)
            if hp:
                E["reversal_bp_sum"] += reversal_bp(*hp)
        if move == BELLY_DRUM:
            E["cat_bellydrum"] += 1
            self._close_bd(seat)            # a prior open window (double Belly Drum): close/score it
            self.bd[seat] = 0               # open a KO-attribution window for this drummer
        if move == CURSE:  # boost only for non-Ghost; type unknown from log -> record separately
            E["cat_curse"] += 1
        if move in HEAL_NON_REST:
            E["cat_heal"] += 1
        if move == WISH:
            E["cat_wish"] += 1
        if move == REST:
            E["cat_rest"] += 1
        if move in WEATHER_PRIMARY:
            E["cat_weather_" + WEATHER_PRIMARY[move]] += 1
        if move in WEATHER_AUX:
            E["cat_weather_aux_" + WEATHER_AUX[move]] += 1
        if move in PHAZE:
            E["cat_phaze"] += 1
            justified = (any(v > 0 for v in self.boosts[opp].values()) or self.sub[opp])
            E["cat_phaze_justified" if justified else "cat_phaze_neutral"] += 1
        if move == SPIKES:
            E["cat_spikes"] += 1
        if move == RAPID_SPIN:
            E["cat_rapidspin_total"] += 1
            if self.spikes[seat] >= 1:
                E["cat_rapidspin_spikesdown"] += 1
        if move == TOXIC:
            E["cat_toxic"] += 1
        if move in PARA_MOVES:
            E["cat_para"] += 1
        if move in SLEEP_MOVES:
            E["cat_sleep"] += 1
            # Sleep Clause: if this seat already move-slept an opponent, its sleep-clause bit is on and
            # this click is wasted (the move will fail). Reproduces the v3 encoder's sleep_clause_blocks_self.
            if self.slept_by[seat]:
                E["sleep_clause_active"] += 1
        if move in STATUS_MOVES and opp in self._nc_switchin:
            # opponent threw a status move at a Natural Cure mon we brought in this turn — it eats the
            # status "for free" (cured on its eventual switch-out).
            self.ev[opp]["nc_switchin_on_status"] += 1
        if move in AROMATHERAPY:
            E["cat_aromatherapy"] += 1
            # mons this cure clears = the statused party mons right now. Aromatherapy emits a single
            # -cureteam (no per-mon lines) and Heal Bell's per-mon lines are [silent] with no move tag,
            # so count from tracked status at use time rather than from the cure lines.
            E["aroma_cured"] += sum(1 for s in self.status[seat].values() if s)
        if move == STRUGGLE:
            E["cat_struggle"] += 1       # out of PP on everything (should be very rare)
        if move in PROTECT_MOVES:
            E["cat_protect"] += 1
            if self.prev_protect_success[seat]:
                # Protect used when the previous move was a *successful* Protect — the risky repeat
                # (gen3's consecutive-Protect penalty makes the second land far less often).
                E["cat_protect_consecutive"] += 1
            self._protect_pending = seat            # its -singleturn (ok) / -fail (failed) resolves it
        else:
            self.prev_protect_success[seat] = False  # any non-Protect move breaks the chain
        if move in COUNTER_MOVES:
            E["cat_counter_mirrorcoat"] += 1
            self._cm_pending = seat                 # success = the direct -damage it deals this turn
        if move == YAWN:
            E["cat_yawn"] += 1
        if move == WILL_O_WISP:
            E["cat_burn"] += 1
        if move in STATUS_MOVES:  # aggregate: any dedicated status-only move (overlaps the above)
            E["cat_status_move"] += 1
        if move in BOOM:
            E["cat_boom"] += 1
            E["cat_boom_" + move] += 1
            # the OPPONENT faces this boom: did it block? A successful Protect (its +3-priority
            # -singleturn already fired this turn) or an up Substitute is in place before the boom
            # hits; a Ghost/type immunity shows as an -immune line handled above (awaited here).
            self.ev[opp]["boom_faced"] += 1
            if self.protected[opp] or self.sub[opp]:
                self.ev[opp]["boom_block"] += 1
            else:
                self._boom_target = opp   # await a possible -immune on the target this turn
        if move == BATON_PASS:
            E["cat_batonpass"] += 1
        if move == SUBSTITUTE:
            E["cat_substitute"] += 1
        if move == FOCUS_PUNCH:
            E["focuspunch_attempt"] += 1
            E["focuspunch_executed"] += 1  # reached move resolution (not |cant|); disruption handled at |cant|
        if move == LEECH_SEED:
            E["cat_leechseed"] += 1
        if move == KNOCK_OFF:
            E["cat_knockoff"] += 1
        if move in PRIORITY_MOVES:
            E["cat_priority"] += 1
            if self.slower.get(seat):   # opponent observed to outspeed us -> skilled priority use
                E["cat_priority_vs_faster"] += 1
            # cat_priority_ko is credited in the faint handler (an immediate KO by this move)
        if move == DESTINY_BOND:
            E["cat_destinybond"] += 1
        if move == SOLAR_BEAM:
            E["cat_solarbeam"] += 1
            E["cat_solarbeam_sun" if self.weather == mid("Sunny Day") or self.weather == "sunnyday" else "cat_solarbeam_nosun"] += 1
        # status-absorber switch-in: opponent used a status move that resolved into our just-switched
        # already-statused mon (approximated: the incoming mon has a status when a status move targets it)


def ability_present(g, gp, seat, abilities, activation_key, require_exact=False):
    """Is one of `abilities` on this seat's team? Exact when the eval captured per-mon abilities on
    the movesets (recent checkpoints). On older captures (species+moves only) we either fall back to
    whether the ability fired at all, or — when `require_exact` — report absent so the metric comes
    out None and the old checkpoint is dropped from the report rather than shown wrong.

    The fallback under-counts presence: an ability is only 'seen' when the opponent triggers it. For
    Intimidate that's nearly every time it switches in, so fallback ≈ exact and it's kept. For the
    absorb abilities it fires only when hit by the matching type (rare) — a ~7x under-count that
    badly inflates the rate — so absorb requires exact gating and its pre-capture points are dropped."""
    mons = g.get("movesets", {}).get(seat, [])
    if any("ability" in m for m in mons):
        return any(mid(m.get("ability", "")) in abilities for m in mons)
    return False if require_exact else gp.ev[seat][activation_key] > 0


def extract(files, lineage=None, milestone=None):
    manifest = None
    seats_behavioral = None
    games = []
    skipped = 0
    for path in files:
        try:
            for line in gzip.open(path, "rt"):
                rec = json.loads(line)
                if rec.get("record") == "manifest":
                    manifest = manifest or rec
                    continue
                games.append(rec)
        except (EOFError, OSError, json.JSONDecodeError) as e:
            # a truncated/corrupt shard (e.g. a killed writer) — keep the games read so far and
            # move on rather than failing the whole extraction.
            skipped += 1
            print(f"WARN: skipped rest of {path} ({type(e).__name__})", file=sys.stderr)
    opponent = (manifest or {}).get("opponent", "self")
    # in self-play both seats are the bot; in foulplay only p1 (bot) is behavioral
    behav_seats = ("p1", "p2") if opponent == "self" else ("p1",)

    n = len(games)
    # the observation unit for behavioral rates is the (game, behavioral-seat): a "seat-game".
    # self-play has 2 seat-games/game, foulplay 1 -> per-seat-game rates are directly comparable.
    seat_games = n * len(behav_seats)
    move_counts = Counter()
    total_moves = 0
    turns = []
    pivots = []
    caps = 0
    species_team = Counter()
    species_win = Counter()
    wins_bot = 0
    cat_present_games = defaultdict(int)   # category -> games where a behav seat carried it
    cat_uses = defaultdict(int)
    cat_extra = defaultdict(int)
    switch_ev = defaultdict(int)
    opp_fp_attempt = 0
    opp_fp_disrupted = 0
    # ability-gated metrics: denominator is seat-games where the ability is on the team (exact when
    # the eval captured abilities on the movesets; otherwise inferred from the ability firing at all).
    intim_present = 0; intim_activations = 0
    absorb_present = 0; absorb_switchins = 0
    spikes_present = 0; spikes_max_sum = 0   # avg peak Spikes layers, over games the seat carries Spikes
    nc_present = 0; nc_switchins = 0         # (v3) Natural-Cure-mon switch-ins onto a status move / game
    pg_rows = []   # (per-seat trait counts for one game, 1 if that seat won) — decided games only
    pp_exhaust_bot = []
    pp_exhaust_opp = []
    mons_alive_win = []
    opp_mons_alive_loss = []
    last_active_win = Counter()

    CATS = {"cat_stat_boost","cat_heal","cat_wish","cat_rest","cat_weather_sun","cat_weather_rain",
            "cat_phaze","cat_spikes","cat_rapidspin_total","cat_toxic","cat_para","cat_sleep","cat_yawn",
            "cat_burn","cat_status_move","cat_knockoff","cat_reversal","cat_bellydrum",
            "cat_priority","cat_destinybond","cat_aromatherapy","cat_protect","cat_counter_mirrorcoat",
            "cat_boom","cat_batonpass","cat_substitute","cat_leechseed","cat_solarbeam","cat_curse"}
    CAT_MOVE = {"cat_stat_boost":STAT_BOOST,"cat_heal":HEAL_NON_REST,"cat_wish":{WISH},"cat_rest":{REST},
        "cat_weather_sun":{k for k,v in WEATHER_PRIMARY.items() if v=="sun"},
        "cat_weather_rain":{k for k,v in WEATHER_PRIMARY.items() if v=="rain"},
        "cat_phaze":PHAZE,"cat_spikes":{SPIKES},"cat_rapidspin_total":{RAPID_SPIN},"cat_toxic":{TOXIC},
        "cat_para":PARA_MOVES,"cat_sleep":SLEEP_MOVES,"cat_yawn":{YAWN},
        "cat_burn":{WILL_O_WISP},"cat_status_move":STATUS_MOVES,"cat_boom":BOOM,
        "cat_batonpass":{BATON_PASS},"cat_substitute":{SUBSTITUTE},"cat_leechseed":{LEECH_SEED},
        "cat_solarbeam":{SOLAR_BEAM},"cat_curse":{CURSE},"cat_knockoff":{KNOCK_OFF},
        "cat_reversal":REVERSAL_MOVES,"cat_bellydrum":{BELLY_DRUM},
        "cat_priority":PRIORITY_MOVES,"cat_destinybond":{DESTINY_BOND},"cat_aromatherapy":AROMATHERAPY,
        "cat_protect":PROTECT_MOVES,"cat_counter_mirrorcoat":COUNTER_MOVES}

    for g in games:
        gp = GameParse(g.get("movesets", {}))
        gp.walk(g["protocol"])
        # avg_turns is over DECIDED games only; a timeout (stall that hit the turn cap) is not a
        # game that "lasted N turns", it's a game that never resolved — count it as a timeout.
        if g.get("capped"):
            caps += 1
        else:
            turns.append(g.get("turn_count") or 0)
        winner = g.get("winner")
        bot_won = (winner == "p1")  # p1 is the bot seat in both modes
        if bot_won:
            wins_bot += 1
        for seat in behav_seats:
            move_counts.update(gp.move_counts[seat])
            total_moves += gp.moves_total[seat]
            pivots.append(gp.ev[seat]["pivot"])
            for k in ("immunity_switchin","switch_out_sleeping","switch_out_frozen"):
                switch_ev[k] += gp.ev[seat][k]
            for cat, mids in CAT_MOVE.items():
                carried = any(mids & {mid(x) for x in m["moves"]} for m in g.get("movesets", {}).get(seat, []))
                if carried:
                    cat_present_games[cat] += 1
                    cat_uses[cat] += gp.ev[seat][cat]
            # NOTE: these are UNGATED (every occurrence), whereas move_categories[*].total_uses is
            # gated on the seat's moveset carrying the move. Mixing the two across a ratio is a bug
            # — a move used but not carried (Metronome/Mimic) lifts the numerator without the
            # denominator and can push a "conditional %" above 100. Conditional rates must therefore
            # be computed from an ungated pair here (e.g. sun/(sun+nosun)), never against
            # total_uses. cat_rapidspin_total is carried here for exactly that reason.
            for extra in ("cat_rapidspin_spikesdown","cat_rapidspin_total",
                          "cat_phaze_justified","cat_phaze_neutral",
                          "focuspunch_attempt","focuspunch_executed","focuspunch_disrupted",
                          "cat_solarbeam_sun","cat_solarbeam_nosun","bp_switch","bp_stat_or_sub",
                          # ungated counters for average-over-use metrics (avoid the gated/ungated trap)
                          "cat_reversal","reversal_bp_sum","cat_bellydrum","bellydrum_ko_sum",
                          "bellydrum_success",
                          # priority conditionals + Destiny Bond success, all rated over ungated uses
                          "cat_priority","cat_priority_vs_faster","cat_priority_ko",
                          "cat_destinybond","destinybond_success",
                          # toxic-stage / leech-seed episodes and enemy-boom blocking
                          "tox_stage_sum","tox_episodes","seed_turns_sum","seed_episodes",
                          "boom_faced","boom_block",
                          # v3-only: sleep-clause, move failures, Aromatherapy cures, Struggle
                          "cat_sleep","sleep_clause_active","move_failed",
                          "cat_aromatherapy","aroma_cured","cat_struggle",
                          "cat_protect","cat_protect_consecutive",
                          "cat_counter_mirrorcoat","cm_success"):
                cat_extra[extra] += gp.ev[seat][extra]
            # ability-gated per-game rates (Intimidate activations, absorb switch-in reads). These are
            # per-GAME counts, so a timeout stall — where a weak checkpoint pivots an intimidator in
            # and out for ~1000 turns — would balloon the count (v22-lr3m@100k hit 38.8 intim/game off
            # 49.6% capped games). Restrict to decided games, exactly as avg_turns does.
            if not g.get("capped"):
                if ability_present(g, gp, seat, {INTIMIDATE}, "intimidate_activation"):
                    intim_present += 1
                    intim_activations += gp.ev[seat]["intimidate_activation"]
                # absorb requires exact ability gating (fallback badly inflates it), so pre-capture
                # checkpoints report absorb_present=0 -> rate None -> dropped from the report.
                if ability_present(g, gp, seat, ABSORB_ABILITIES, "absorb_activation", require_exact=True):
                    absorb_present += 1
                    absorb_switchins += gp.ev[seat]["absorb_switchin"]
                # (v3) Natural-Cure-mon switch-ins onto a status move / game, over games where a Natural
                # Cure mon is on the team. Requires exact abilities (v3 captures them).
                if ability_present(g, gp, seat, {NATURAL_CURE}, None, require_exact=True):
                    nc_present += 1
                    nc_switchins += gp.ev[seat]["nc_switchin_on_status"]
            # avg peak Spikes layers achieved, over games where the seat's team carries Spikes (a
            # per-game max, so stalls don't inflate it — no need to restrict to decided games).
            if any(SPIKES in {mid(x) for x in m["moves"]} for m in g.get("movesets", {}).get(seat, [])):
                spikes_present += 1
                spikes_max_sum += gp.ev[seat]["spikes_max_achieved"]
        # opponent Focus Punch, for "bot disrupts opponent's Focus Punch": the OTHER seat's
        # attempts and how many the bot disrupted. In foulplay the opponent is FoulPlay (p2); in
        # self-play both seats are opponents of each other.
        for oseat in {OTHER_SEAT[s] for s in behav_seats}:
            opp_fp_attempt += gp.ev[oseat]["focuspunch_attempt"]
            opp_fp_disrupted += gp.ev[oseat]["focuspunch_disrupted"]
        # per-game rows for the trait->win correlation: decided games only (a timeout has no winner
        # and would silently become a "loss" for both seats).
        if not g.get("capped") and winner in ("p1", "p2"):
            for seat in behav_seats:
                pg_rows.append((gp.ev[seat], 1.0 if winner == seat else 0.0))
        # species vector: every behavioral-seat team, each species labeled by whether that seat won.
        # self-play samples both teams (symmetric); foulplay samples only the bot's team.
        for seat in behav_seats:
            seat_won = (winner == seat)
            for sp in {m["species"] for m in g.get("movesets", {}).get(seat, [])}:
                species_team[sp] += 1
                if seat_won:
                    species_win[sp] += 1
        # endgame from final protocol state
        alive = _alive_counts(g["protocol"])
        if bot_won:
            mons_alive_win.append(alive["p1"])
            la = _last_active(g["protocol"], "p1")
            if la:
                last_active_win[la] += 1
        elif winner == "p2":
            opp_mons_alive_loss.append(alive["p2"])
        # PP exhaustion from pp_track
        be, oe = _pp_exhaustions(g.get("pp_track", []))
        pp_exhaust_bot.append(be)
        pp_exhaust_opp.append(oe)

    def per_game(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    # point-biserial r of each per-game trait count against winning that game
    pg_corr = {}
    for t in PER_GAME_TRAITS:
        xs = [float(ev[t]) for ev, _ in pg_rows]
        ys = [w for _, w in pg_rows]
        r = _pearson(xs, ys)
        if r is not None:
            pg_corr[t] = {"r": round(r, 4), "n": len(xs), "mean": round(sum(xs) / len(xs), 4)}

    metrics = {
        "metrics_version": METRICS_VERSION, "opponent": opponent, "n_games": n,
        "lineage": lineage or (manifest or {}).get("lineage"),
        "milestone": milestone if milestone is not None else (manifest or {}).get("milestone"),
        "checkpoint": (manifest or {}).get("checkpoint"),
        "skipped_shards": skipped,
        "behavioral_seats": list(behav_seats), "seat_games": seat_games, "capped_games": caps,
        "bot_win_rate": round(wins_bot / n, 4) if n else None,
        # Phase 1 basics
        "top5_moves": [{"move": m, "share": round(c / total_moves, 4), "uses_per_game": round(c / (n or 1), 3)}
                       for m, c in move_counts.most_common(5)],
        "move_distribution": {m: c for m, c in move_counts.most_common()},
        # per-game trait -> win correlation within THIS checkpoint (n = decided seat-games, not
        # checkpoints). Self-play is a paired winner-vs-loser design: same policy, same game.
        "per_game_correlations": pg_corr,
        "per_game_rows": len(pg_rows),
        "avg_turns": per_game(turns),            # decided games only (timeouts excluded)
        "decided_games": len(turns),
        "timeout_rate": round(caps / n, 4) if n else 0.0,
        "avg_pivots": per_game(pivots),
        # Phase 2 move categories. Denominator is seat-games in which the acting side carried a
        # category move (games-present), separating "not dealt the move" from "policy ignores it".
        "move_categories": {cat: {"seat_games_present": cat_present_games[cat],
                                   "uses_per_seat_game_present": round(cat_uses[cat] / cat_present_games[cat], 4) if cat_present_games[cat] else 0.0,
                                   "carrier_rate": round(cat_present_games[cat] / (seat_games or 1), 4),
                                   "total_uses": cat_uses[cat]} for cat in sorted(CATS)},
        "move_category_extras": {k: cat_extra[k] for k in sorted(cat_extra)},
        # bot's own Focus Punch: fraction of attempts that landed (were not disrupted first).
        "focus_punch_attempts": cat_extra["focuspunch_attempt"],
        "focus_punch_success_rate": (round(cat_extra["focuspunch_executed"] /
                                     (cat_extra["focuspunch_executed"] + cat_extra["focuspunch_disrupted"]), 4)
                                     if (cat_extra["focuspunch_executed"] + cat_extra["focuspunch_disrupted"]) else None),
        # bot disrupting the OPPONENT's Focus Punch: fraction of opponent attempts the bot broke.
        "opp_focus_punch_attempts": opp_fp_attempt,
        "opp_focus_punch_disruption_rate": (round(opp_fp_disrupted / opp_fp_attempt, 4)
                                            if opp_fp_attempt else None),
        # Reversal/Flail: average BP over uses. Higher = firing them at lower HP (the correct play).
        "reversal_uses": cat_extra["cat_reversal"],
        "reversal_avg_bp": (round(cat_extra["reversal_bp_sum"] / cat_extra["cat_reversal"], 2)
                            if cat_extra["cat_reversal"] else None),
        # Belly Drum: average opponent mons KO'd by the drummer after the move (payoff of the setup).
        "bellydrum_uses": cat_extra["cat_bellydrum"],
        "bellydrum_avg_kos": (round(cat_extra["bellydrum_ko_sum"] / cat_extra["cat_bellydrum"], 3)
                              if cat_extra["cat_bellydrum"] else None),
        # fraction of Belly Drum uses that converted to >=1 KO — "was the setup worth it", robust to
        # how many opponents happened to remain (which skews the average-KOs figure).
        "bellydrum_ko_rate": (round(cat_extra["bellydrum_success"] / cat_extra["cat_bellydrum"], 4)
                              if cat_extra["cat_bellydrum"] else None),
        # Priority moves (Quick Attack / Extreme Speed / Mach Punch): raw usage is in move_categories;
        # here the two conditionals rated over ungated uses — used when the opponent outspeeds us
        # (skilled) and used to land the KO (payoff).
        "priority_uses": cat_extra["cat_priority"],
        "priority_vs_faster_rate": (round(cat_extra["cat_priority_vs_faster"] / cat_extra["cat_priority"], 4)
                                    if cat_extra["cat_priority"] else None),
        "priority_ko_rate": (round(cat_extra["cat_priority_ko"] / cat_extra["cat_priority"], 4)
                             if cat_extra["cat_priority"] else None),
        # Destiny Bond: fraction of uses that dragged the attacker down (an -activate fired).
        "destinybond_uses": cat_extra["cat_destinybond"],
        "destinybond_success_rate": (round(cat_extra["destinybond_success"] / cat_extra["cat_destinybond"], 4)
                                     if cat_extra["cat_destinybond"] else None),
        # Intimidate activations per game, among seat-games where an intimidator is on the team.
        "intimidate_present_seat_games": intim_present,
        "intimidate_activations_per_game": (round(intim_activations / intim_present, 4)
                                            if intim_present else None),
        # Volt/Water Absorb + Flash Fire switch-in "reads" (absorb triggered the turn you brought the
        # mon in), per game among seat-games where such an ability is on the team.
        "absorb_present_seat_games": absorb_present,
        "absorb_switchins_per_game": (round(absorb_switchins / absorb_present, 4)
                                      if absorb_present else None),
        # Average peak toxic stage reached before a badly-poisoned mon leaves/cures/faints/game-ends.
        # Lower over training = the policy switches toxiced mons out early to preserve HP.
        "toxic_episodes": cat_extra["tox_episodes"],
        "avg_toxic_stage": (round(cat_extra["tox_stage_sum"] / cat_extra["tox_episodes"], 3)
                            if cat_extra["tox_episodes"] else None),
        # Leech Seed: average turns a seeded mon stays active before it leaves (seed clears on switch),
        # read like avg toxic stage — lower = pivots seeded mons out rather than eating the drain.
        "leechseed_episodes": cat_extra["seed_episodes"],
        "avg_leechseed_turns": (round(cat_extra["seed_turns_sum"] / cat_extra["seed_episodes"], 3)
                                if cat_extra["seed_episodes"] else None),
        # Spikes: average peak layers stacked on the opponent, over games the seat carries Spikes.
        # Rising toward 3 = the policy learns to fully stack the hazard rather than lay one and move on.
        "spikes_present_seat_games": spikes_present,
        "spikes_avg_max_layers": (round(spikes_max_sum / spikes_present, 3) if spikes_present else None),
        # ---- v3-only traits (gen3 Sleep Clause; see docs/observation_v3_spec.md) ----
        # Sleep: total sleep-move clicks and the fraction thrown while this seat's Sleep-Clause bit is
        # already set (a wasted click the clause blocks). Rate ~0 = the policy doesn't redundantly sleep.
        "sleep_uses": cat_extra["cat_sleep"],
        "sleep_clause_active_uses": cat_extra["sleep_clause_active"],
        "sleep_clause_active_rate": (round(cat_extra["sleep_clause_active"] / cat_extra["cat_sleep"], 4)
                                     if cat_extra["cat_sleep"] else None),
        # Move-failure rate: moves that hit a -fail within their action window / all moves.
        "move_fail_rate": (round(cat_extra["move_failed"] / total_moves, 4) if total_moves else None),
        "moves_failed": cat_extra["move_failed"],
        # Aromatherapy / Heal Bell: usage is in move_categories; here the payoff — avg statused mons
        # cured per use.
        "aromatherapy_uses": cat_extra["cat_aromatherapy"],
        "aromatherapy_avg_cured": (round(cat_extra["aroma_cured"] / cat_extra["cat_aromatherapy"], 3)
                                   if cat_extra["cat_aromatherapy"] else None),
        # Natural Cure: switch-ins of a Natural Cure mon onto an incoming status move, per game, over
        # games where a Natural Cure mon is on the team (exact ability gating — v3 captures abilities).
        "nc_present_seat_games": nc_present,
        "nc_switchin_on_status_per_game": (round(nc_switchins / nc_present, 4) if nc_present else None),
        # Struggle: total occurrences (should be very rare — everything out of PP).
        "struggle_uses": cat_extra["cat_struggle"],
        "struggle_per_game": round(cat_extra["cat_struggle"] / (n or 1), 5),
        # Protect/Detect: total uses (also in move_categories, gated), and the fraction thrown right
        # after a *successful* Protect — the risky repeat gen3's consecutive-Protect penalty punishes.
        "protect_uses": cat_extra["cat_protect"],
        "protect_after_success_uses": cat_extra["cat_protect_consecutive"],
        "protect_after_success_rate": (round(cat_extra["cat_protect_consecutive"] / cat_extra["cat_protect"], 4)
                                       if cat_extra["cat_protect"] else None),
        # Counter + Mirror Coat (grouped): total uses and the fraction that actually landed damage
        # (they whiff silently when there's no matching physical/special hit to bounce).
        "counter_mirrorcoat_uses": cat_extra["cat_counter_mirrorcoat"],
        "counter_mirrorcoat_success_rate": (round(cat_extra["cm_success"] / cat_extra["cat_counter_mirrorcoat"], 4)
                                            if cat_extra["cat_counter_mirrorcoat"] else None),
        # Boom blocks: of the enemy Explosion/Self-Destruct the bot faced, the fraction it neutralized
        # via Protect, an absorbing Substitute, or a Ghost/type immunity.
        "boom_faced": cat_extra["boom_faced"],
        "boom_blocks": cat_extra["boom_block"],
        "boom_block_rate": (round(cat_extra["boom_block"] / cat_extra["boom_faced"], 4)
                            if cat_extra["boom_faced"] else None),
        # Phase 2 switch behavior — per seat-game so self-play and foulplay are comparable
        "switch_behavior": {k: {"total": v, "per_seat_game": round(v / (seat_games or 1), 4)} for k, v in switch_ev.items()},
        # Phase 2 resource / endgame
        "pp_exhaustion_bot_per_game": per_game(pp_exhaust_bot),
        "pp_exhaustion_opp_per_game": per_game(pp_exhaust_opp),
        "avg_bot_mons_alive_on_win": per_game(mons_alive_win),
        "avg_opp_mons_alive_on_loss": per_game(opp_mons_alive_loss),
        "top5_last_active_on_win": last_active_win.most_common(5),
        # Phase 2 species vector
        "species_vector": _species_vector(species_team, species_win, wins_bot, n, opponent),
    }
    return metrics


def _alive_counts(protocol):
    """Mons per side that never fainted = 6 - faints for that side (gen3 teams are 6)."""
    faints = {"p1": 0, "p2": 0}
    for line in protocol:
        tag, a = parse_line(line)
        if tag == "faint":
            s = seat_of(a[0])
            if s:
                faints[s] += 1
    return {"p1": 6 - faints["p1"], "p2": 6 - faints["p2"]}


def _last_active(protocol, seat):
    last = None
    for line in protocol:
        tag, a = parse_line(line)
        if tag in ("switch", "drag") and seat_of(a[0]) == seat:
            last = species_of(a[0])
    return last


def _pp_exhaustions(pp_track):
    """Distinct (active-mon, move) pairs that reached 0 PP, per seat. bot=p1, opp=p2.

    Dedup is by (mon, move-id), NOT by turn: a move sitting at 0 PP over many turns is one
    exhausted move, not many. Falls back to move-id alone when the snapshot predates the `mon`
    field (older captures)."""
    zeroed = {"p1": set(), "p2": set()}
    for snap in pp_track:
        seat = snap.get("seat")
        if seat not in zeroed:
            continue
        mon = snap.get("mon")
        for m in snap.get("moves", []):
            if m.get("pp") == 0:
                zeroed[seat].add((mon, m.get("id")))
    return len(zeroed["p1"]), len(zeroed["p2"])


def _species_vector(team, win, wins_bot, n, opponent):
    base = wins_bot / n if n else 0.0
    out = {}
    for sp, t in team.items():
        w = win.get(sp, 0)
        p_win_given = w / t if t else 0.0
        out[sp] = {"team_instances": t, "win_instances": w,
                   "p_win_given_on_team": round(p_win_given, 4),
                   "win_delta": round(p_win_given - (0.5 if opponent == "self" else base), 4),
                   "low_n": t < 50}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", required=True, help="events-*.jsonl.gz (globs ok)")
    ap.add_argument("--lineage", default=None)
    ap.add_argument("--milestone", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = []
    for e in args.events:
        files.extend(sorted(glob.glob(e)) or [e])
    metrics = extract(files, lineage=args.lineage, milestone=args.milestone)
    json.dump(metrics, open(args.out, "w"), indent=1)
    print(f"WROTE {args.out} n_games={metrics['n_games']} opponent={metrics['opponent']}")
    print("top5:", [(m["move"], m["share"]) for m in metrics["top5_moves"]])
    print("avg_turns:", metrics["avg_turns"], "avg_pivots:", metrics["avg_pivots"])


if __name__ == "__main__":
    main()
