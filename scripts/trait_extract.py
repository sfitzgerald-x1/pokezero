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

METRICS_VERSION = "trait_extract.v1"

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
BOOM = {mid("Explosion"), mid("Self-Destruct")}
BATON_PASS = mid("Baton Pass")
SUBSTITUTE = mid("Substitute")
FOCUS_PUNCH = mid("Focus Punch")
LEECH_SEED = mid("Leech Seed")
SOLAR_BEAM = mid("Solar Beam")

OTHER_SEAT = {"p1": "p2", "p2": "p1"}


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
        self.spikes = {"p1": 0, "p2": 0}      # layers on that side
        self.weather = None
        self.ev = defaultdict(lambda: Counter())   # seat -> Counter of trait events
        self.move_counts = {"p1": Counter(), "p2": Counter()}
        self.moves_total = {"p1": 0, "p2": 0}
        self.pending_faint = {"p1": False, "p2": False}
        self.last_move = {"p1": None, "p2": None}
        self._pending_switch_immunity = None  # (seat, incoming) awaiting a resolved move this turn

    def carriers(self, seat, move_id):
        return sum(1 for m in self.movesets.get(seat, []) if move_id in {mid(x) for x in m["moves"]})

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
                    self.pending_faint[seat] = False
                    self.active[seat] = sp
                    self.boosts[seat] = defaultdict(int)
                    self.sub[seat] = False
                    # a switch-in may materialize immunity to the opponent's move this turn
                    self._pending_switch_immunity = (seat, sp)
            elif tag == "move":
                seat = seat_of(a[0]); move = mid(a[1]) if len(a) > 1 else None
                if seat and move:
                    self.moves_total[seat] += 1
                    self.move_counts[seat][a[1]] += 1
                    self.last_move[seat] = move
                    self._classify_move(seat, move, a)
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
                    if len(a) > 1 and a[1] == "slp":
                        # sleep from the mon's own Rest, flagged by the protocol's [from] tag
                        # (fallback: the seat's last resolved move was Rest, and Rest is self-target).
                        from_rest = any("move: rest" in str(x).lower() for x in a[2:]) or \
                            (self.last_move[seat] == REST and sp == self.active[seat])
                        (self.rest_sleep[seat].add if from_rest else self.rest_sleep[seat].discard)(sp)
            elif tag == "-curestatus":
                seat = seat_of(a[0])
                if seat:
                    sp = species_of(a[0])
                    self.status[seat].pop(sp, None)
                    self.rest_sleep[seat].discard(sp)
            elif tag == "-start":
                seat = seat_of(a[0])
                if seat and len(a) > 1 and mid(a[1]) == SUBSTITUTE:
                    self.sub[seat] = True
            elif tag == "-end":
                seat = seat_of(a[0])
                if seat and len(a) > 1 and mid(a[1]) == SUBSTITUTE:
                    self.sub[seat] = False
            elif tag == "-weather":
                self.weather = mid(a[0]) if a else None
            elif tag == "-sidestart":
                seat = seat_of(a[0])
                if seat and len(a) > 1 and mid(a[1].replace("move: ", "")) == SPIKES:
                    self.spikes[seat] += 1
            elif tag == "-sideend":
                seat = seat_of(a[0])
                if seat and len(a) > 1 and mid(a[1].replace("move: ", "")) == SPIKES:
                    self.spikes[seat] = 0
            elif tag == "-immune":
                seat = seat_of(a[0])
                # immunity that materialized right after a switch-in by this seat
                if self._pending_switch_immunity and self._pending_switch_immunity[0] == seat:
                    self.ev[seat]["immunity_switchin"] += 1
                    self._pending_switch_immunity = None
            elif tag == "faint":
                seat = seat_of(a[0])
                if seat:
                    self.pending_faint[seat] = True
            elif tag == "turn":
                self._pending_switch_immunity = None  # immunity window is the switch-in turn only

    def _classify_move(self, seat, move, a):
        opp = OTHER_SEAT[seat]
        E = self.ev[seat]
        if move in STAT_BOOST:
            E["cat_stat_boost"] += 1
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
        if move == YAWN:
            E["cat_yawn"] += 1
        if move in BOOM:
            E["cat_boom"] += 1
            E["cat_boom_" + move] += 1
        if move == BATON_PASS:
            E["cat_batonpass"] += 1
        if move == SUBSTITUTE:
            E["cat_substitute"] += 1
        if move == FOCUS_PUNCH:
            E["focuspunch_attempt"] += 1
            E["focuspunch_executed"] += 1  # reached move resolution (not |cant|); disruption handled at |cant|
        if move == LEECH_SEED:
            E["cat_leechseed"] += 1
        if move == SOLAR_BEAM:
            E["cat_solarbeam"] += 1
            E["cat_solarbeam_sun" if self.weather == mid("Sunny Day") or self.weather == "sunnyday" else "cat_solarbeam_nosun"] += 1
        # status-absorber switch-in: opponent used a status move that resolved into our just-switched
        # already-statused mon (approximated: the incoming mon has a status when a status move targets it)


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
    pp_exhaust_bot = []
    pp_exhaust_opp = []
    mons_alive_win = []
    opp_mons_alive_loss = []
    last_active_win = Counter()

    CATS = {"cat_stat_boost","cat_heal","cat_wish","cat_rest","cat_weather_sun","cat_weather_rain",
            "cat_phaze","cat_spikes","cat_rapidspin_total","cat_toxic","cat_para","cat_sleep","cat_yawn",
            "cat_boom","cat_batonpass","cat_substitute","cat_leechseed","cat_solarbeam","cat_curse"}
    CAT_MOVE = {"cat_stat_boost":STAT_BOOST,"cat_heal":HEAL_NON_REST,"cat_wish":{WISH},"cat_rest":{REST},
        "cat_weather_sun":{k for k,v in WEATHER_PRIMARY.items() if v=="sun"},
        "cat_weather_rain":{k for k,v in WEATHER_PRIMARY.items() if v=="rain"},
        "cat_phaze":PHAZE,"cat_spikes":{SPIKES},"cat_rapidspin_total":{RAPID_SPIN},"cat_toxic":{TOXIC},
        "cat_para":PARA_MOVES,"cat_sleep":SLEEP_MOVES,"cat_yawn":{YAWN},"cat_boom":BOOM,
        "cat_batonpass":{BATON_PASS},"cat_substitute":{SUBSTITUTE},"cat_leechseed":{LEECH_SEED},
        "cat_solarbeam":{SOLAR_BEAM},"cat_curse":{CURSE}}

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
            for extra in ("cat_rapidspin_spikesdown","cat_phaze_justified","cat_phaze_neutral",
                          "focuspunch_attempt","focuspunch_executed","focuspunch_disrupted",
                          "cat_solarbeam_sun","cat_solarbeam_nosun","bp_switch","bp_stat_or_sub"):
                cat_extra[extra] += gp.ev[seat][extra]
        # opponent Focus Punch, for "bot disrupts opponent's Focus Punch": the OTHER seat's
        # attempts and how many the bot disrupted. In foulplay the opponent is FoulPlay (p2); in
        # self-play both seats are opponents of each other.
        for oseat in {OTHER_SEAT[s] for s in behav_seats}:
            opp_fp_attempt += gp.ev[oseat]["focuspunch_attempt"]
            opp_fp_disrupted += gp.ev[oseat]["focuspunch_disrupted"]
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
