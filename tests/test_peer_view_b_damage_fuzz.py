"""Fuzz of view b's damage rule (constructor_worlds.peer_damage_facts) against the real env parser, from both seats.

Random Gen 3 games are written as omniscient protocol lines (both seats' HP exact): leads, voluntary switches, drags, |cant|,
confusion self-hits (untagged, after the -activate latch), multi-hit moves, misses, Protect, recoil, drain, Pain Split's two
-sethp lines, Leftovers, poison, Spikes on arrival, faints with replacements, the 99 cap and 1-HP losses. Each seat in turn is
the owner: its own wire shows its HP exact and the other seat's at Showdown's public percent, the peer's wire the reverse, by
this module's own percent rule (Showdown's JavaScript, Math.ceil(100 * hp / maxhp) on doubles with the 99 cap: never
constructor_worlds.public_percent). At every |turn| boundary and at random mid-turn cuts, the rule on the owner's wire must
equal what pokezero.showdown's own parser reads on the peer's wire (its last_damage_taken of the owner's seat and
last_damage_dealt of the peer's), exactly (both sum the same fractions in the same order). Two mutants of the rule (no 99 cap,
no confusion latch) must be caught."""
import math
import random
import unittest
from unittest.mock import patch

from gpu_engine import constructor_worlds as cw
from pokezero import showdown

SPECIES = ('Snorlax', 'Skarmory', 'Starmie', 'Gengar', 'Zapdos', 'Tyranitar', 'Blissey', 'Metagross', 'Salamence', 'Aerodactyl',
    'Swampert', 'Celebi')
GAMES = 1500


def shown(hp, maximum):
    """Showdown's sim/pokemon.ts getHealth (HP Percentage Mod), as JavaScript computes it: Math.ceil(100 * hp / maxhp) in doubles,
    99 for a damaged Pokemon that would show 100."""
    value = math.ceil(100 * hp / maximum)
    return 99 if value == 100 and hp < maximum else value


def wire(lines, exact_seat):
    """`lines` as `exact_seat`'s player channel carries them: the other seat's HP at the public percent, '0 fnt' kept."""
    out = []
    for line in lines:
        p = line.split('|')
        if len(p) > 3 and p[1] in ('switch', 'drag', '-damage', '-heal', '-sethp'):
            i = 4 if p[1] in ('switch', 'drag') else 3
            if p[2][:2] != exact_seat and p[i] != '0 fnt':
                hp, top = map(int, p[i].split('/'));p[i] = f'{shown(hp, top)}/100'
        out.append('|'.join(p))
    return out


class Game:
    """One random battle's omniscient lines; `cuts` are prefix lengths to compare at."""
    def __init__(self, rng):
        self.rng, self.lines, self.cuts, self.turn = rng, ['|player|p1|Alice', '|player|p2|Bob', '|start'], [], 0
        names = rng.sample(SPECIES, 12)
        self.team = {side: [dict(name=n, max=rng.randint(60, 460)) for n in names[k * 6:k * 6 + rng.randint(2, 6)]] for k, side in enumerate(('p1', 'p2'))}
        for mons in self.team.values():
            for m in mons:m['hp'] = m['max'] if rng.random() < .7 else rng.randint(1, m['max'])
        self.active, self.spikes = {}, {'p1': rng.random() < .3, 'p2': rng.random() < .3}

    def ident(self, side):return f'{side}a: {self.active[side]["name"]}'

    def cond(self, mon):return '0 fnt' if mon['hp'] == 0 else f'{mon["hp"]}/{mon["max"]}'

    def foe(self, side):return 'p2' if side == 'p1' else 'p1'

    def alive(self, side):return [m for m in self.team[side] if m['hp'] > 0]

    def bench(self, side):return [m for m in self.alive(side) if m is not self.active.get(side)]

    def arrive(self, side, mon, kind='switch'):
        self.active[side] = mon;self.lines.append(f'|{kind}|{self.ident(side)}|{mon["name"]}, L80|{self.cond(mon)}')
        if self.spikes[side] and self.rng.random() < .6:self.damage(side, max(1, mon['max'] // 8), '[from] Spikes')

    def amount(self, side):
        mon, r = self.active[side], self.rng.random()
        if r < .15:return 1
        if r < .25:return mon['hp']   # a knockout
        return self.rng.randint(1, max(1, mon['max'] // 2))

    def damage(self, side, amount, tag=None):
        mon = self.active[side]
        if mon['hp'] == 0:return
        mon['hp'] = max(0, mon['hp'] - amount);self.lines.append(f'|-damage|{self.ident(side)}|{self.cond(mon)}' + (f'|{tag}' if tag else ''))
        if mon['hp'] == 0:self.lines.append(f'|faint|{self.ident(side)}')

    def heal(self, side, amount, tag):
        mon = self.active[side]
        if 0 < mon['hp'] < mon['max']:mon['hp'] = min(mon['max'], mon['hp'] + amount);self.lines.append(f'|-heal|{self.ident(side)}|{self.cond(mon)}|{tag}')

    def act(self, side):
        rng, foe, mon = self.rng, self.foe(side), self.active[side]
        if mon['hp'] == 0 or self.active[foe]['hp'] == 0:return
        r = rng.random()
        if r < .12 and self.bench(side):self.arrive(side, rng.choice(self.bench(side)));return
        if r < .18:self.lines.append(f'|cant|{self.ident(side)}|par');return
        if r < .26:   # a confusion self-hit: the latch, then untagged damage on itself (or tagged, as later gens write it)
            self.lines.append(f'|-activate|{self.ident(side)}|confusion')
            self.damage(side, self.amount(side), None if rng.random() < .7 else '[from] confusion');return
        self.lines.append(f'|move|{self.ident(side)}|Tackle|{self.ident(foe)}');r = rng.random()
        if r < .08:self.lines.append(f'|-miss|{self.ident(side)}|{self.ident(foe)}');return
        if r < .14:self.lines.append(f'|-activate|{self.ident(foe)}|move: Protect');return
        if r < .22 and self.bench(foe):self.arrive(foe, rng.choice(self.bench(foe)), 'drag');return   # Roar / Whirlwind
        if r < .30:   # Pain Split: two single-target -sethp lines
            total = mon['hp'] + self.active[foe]['hp'];half = max(1, total // 2)
            for who in (foe, side):
                m = self.active[who];m['hp'] = min(m['max'], half);tags = '|[from] move: Pain Split' + ('|[silent]' if who == foe else '')
                self.lines.append(f'|-sethp|{self.ident(who)}|{self.cond(m)}{tags}')
            return
        hits = rng.choice((1, 1, 1, 2, 3, 5))
        for _ in range(hits):self.damage(foe, self.amount(foe) if hits == 1 else rng.randint(1, max(1, self.active[foe]['max'] // 6)))
        if hits > 1:self.lines.append(f'|-hitcount|{self.ident(foe)}|{hits}')
        if rng.random() < .15:self.damage(side, max(1, mon['max'] // rng.choice((3, 4, 8))), f'[from] Recoil|[of] {self.ident(foe)}')
        elif rng.random() < .1:self.heal(side, rng.randint(1, mon['max'] // 2), f'[from] drain|[of] {self.ident(foe)}')

    def play(self):
        rng = self.rng
        for side in ('p1', 'p2'):self.arrive(side, self.team[side][0])
        while self.turn < rng.randint(3, 40):
            self.turn += 1;self.lines.append(f'|turn|{self.turn}');self.cuts.append(len(self.lines))
            for side in rng.sample(('p1', 'p2'), 2):
                self.act(side)
                if rng.random() < .1:self.cuts.append(len(self.lines))   # a mid-turn prefix
            for side in ('p1', 'p2'):   # residuals
                if self.active[side]['hp'] > 0 and rng.random() < .3:self.heal(side, max(1, self.active[side]['max'] // 16), '[from] item: Leftovers')
                if self.active[side]['hp'] > 0 and rng.random() < .15:self.damage(side, max(1, self.active[side]['max'] // 8), '[from] psn')
            if any(self.active[s]['hp'] == 0 for s in ('p1', 'p2')):
                self.lines.append('|upkeep')
                for side in ('p1', 'p2'):
                    if self.active[side]['hp'] == 0:
                        if not self.bench(side):return
                        self.arrive(side, rng.choice(self.bench(side)))
                if any(self.active[s]['hp'] == 0 for s in ('p1', 'p2')):return   # a replacement fainted to Spikes: stop here
            if not self.alive('p1') or not self.alive('p2'):return


def env_cells(lines, owner, cuts):
    """What the env parser reads on the PEER's wire at each cut: (last_damage_taken of the owner's seat, last_damage_dealt of the
    peer's)."""
    own, peer = f'p{owner + 1}', f'p{2 - owner}'
    parser = showdown._ReplayParser(battle_id='fuzz', complete_prefix=True, hp_visibility={peer: 'exact', own: 'percentage'})
    theirs = wire(lines, peer);out = [];fed = 0
    for cut in cuts:
        parser.feed(theirs[fed:cut]);fed = cut;state = parser.snapshot()
        out.append((float(state.last_damage_taken.get(own, 0.)), float(state.last_damage_dealt.get(peer, 0.))))
    return out


def rule_cells(lines, owner, cuts):
    ours = wire(lines, f'p{owner + 1}')
    return [(d.owner_taken, d.peer_dealt) for d in (cw.peer_damage_facts(ours[:cut], owner) for cut in cuts)]


def mismatches(games, seed=20261109):
    rng = random.Random(seed);bad = [];compared = 0;nonzero = 0
    for g in range(games):
        game = Game(rng);game.play()
        for owner in (0, 1):
            want, got = env_cells(game.lines, owner, game.cuts), rule_cells(game.lines, owner, game.cuts)
            compared += len(want);nonzero += sum(any(w) for w in want)
            bad += [(g, owner, cut, w, x) for cut, w, x in zip(game.cuts, want, got) if w != x]
    return bad, compared, nonzero


class DamageRuleFuzzTests(unittest.TestCase):
    def test_the_rule_on_our_wire_is_the_env_parser_on_the_peers_wire(self):
        bad, compared, nonzero = mismatches(GAMES)
        self.assertGreater(compared, 20000);self.assertGreater(nonzero, compared // 4)
        self.assertEqual(bad[:5], [], f'{len(bad)} of {compared} cuts differ')

    def test_the_fuzz_catches_a_rule_without_the_99_cap_or_the_confusion_latch(self):
        uncapped = lambda hp, maximum: (100 * max(0, hp) + max(maximum, 1) - 1) // max(maximum, 1)
        with patch.object(cw, 'public_percent', uncapped):self.assertTrue(mismatches(200)[0])
        with patch.object(cw, '_confusion', lambda effect: False):self.assertTrue(mismatches(200)[0])


if __name__ == '__main__':unittest.main()
