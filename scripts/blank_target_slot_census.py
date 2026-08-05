"""How often does the engine blank a `|move|` line's TARGET SLOT, per move?

`sim/battle.ts:3155-3159` erases that slot whenever a move's animation is
suppressed -- "If no animation plays, the target should never be known". Belief
used to read Pressure's foe-targeted precondition off that slot, so every
blanked line silently lost its double charge. This census is the evidence for
how large that class is, and it exists as a script rather than as a number in a
comment because a number in a comment cannot be re-run.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/blank_target_slot_census.py [GAMES] [SEED]

Note when reading the output: self-targeted moves (Recover, Protect, Wish, Baton
Pass, Refresh) blank often and it is harmless -- they are in
`belief._NEVER_PRESSURED_POOL_MOVES` and were never pressured anyway. The rows
that mattered are the foe-targeted ones.
"""
import collections, os, random, sys
from pokezero.local_showdown import (
    DEFAULT_SHOWDOWN_ROOT as default_showdown_root_const,
    LocalShowdownConfig,
    LocalShowdownEnv,
)


def default_showdown_root():
    return default_showdown_root_const

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 60
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 4711
# Falls back to the repo's default root rather than hard-requiring the env var: this script is
# quoted as a reproducer in belief.py and tests/test_pressure_pp_charge.py, and a reproducer that
# exits 1 in the default configuration is no better than the unreproducible number it replaced.
ROOT = os.environ.get("POKEZERO_SHOWDOWN_ROOT") or str(default_showdown_root())
if not os.path.isdir(ROOT):
    raise SystemExit(f"no Showdown checkout at {ROOT}; set POKEZERO_SHOWDOWN_ROOT")

total = collections.Counter()
blank = collections.Counter()
env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=ROOT, set_belief_source=True))
try:
    for game in range(GAMES):
        rng = random.Random(SEED * 1_000_003 + game)
        env.reset(seed=SEED + game)
        steps = 0
        while steps < 400 and env.terminal() is None:
            req = env.requested_players()
            if not req:
                break
            actions, ok = {}, True
            for p in req:
                mask = env.observe(p).legal_action_mask
                legal = [i for i, a in enumerate(mask) if a]
                if not legal:
                    ok = False
                    break
                mv = [i for i in legal if i < 4]
                actions[p] = rng.choice(mv) if mv and rng.random() < 0.9 else rng.choice(legal)
            if not ok or len(actions) != len(req):
                break
            env.step(actions)
            steps += 1
        for line in env.protocol_lines:
            parts = line.split("|")
            # |move|ACTOR|Move Name|TARGET   -> parts = ['', 'move', actor, name, target, ...]
            if len(parts) < 4 or parts[1] != "move":
                continue
            mid = parts[3].lower().replace(" ", "").replace("-", "").replace("'", "")
            total[mid] += 1
            if len(parts) < 5 or not parts[4].strip():
                blank[mid] += 1
finally:
    env.close()

print(f"games={GAMES} seed={SEED}")
print(f"{'move':16s} {'blank':>6s}/{'total':<6s}")
for mid, b in blank.most_common(15):
    print(f"{mid:16s} {b:6d}/{total[mid]:<6d}")
print(f"TOTAL blank={sum(blank.values())} of {sum(total.values())} move lines")
