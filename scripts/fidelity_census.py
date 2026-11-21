"""Decision-share census over the GPU regression's capture shards (numpy + json only, no engine import).

Each lane-step (one engine decision of the capture: 964 shards, 1,124,864 lane-steps) is tested for whether a cause can
fire at that decision. Speeds are the actives' Speed stat with stat stages and paralysis (x1/4); weather speed abilities
and items are ignored (an approximation, stated in the report).
usage: scripts/fidelity_census.py CAPTURE_DIR POOL.npz OUT.json"""
import json, sys, glob, os, collections
import numpy as np
cap = sys.argv[1]; meta = json.loads(np.load(sys.argv[2])['metadata_utf8'].tobytes()); V = meta['vocabulary']
paths = json.load(open(os.path.join(cap, 'capture.json')))['paths']; leaf = {p: f'l{i:03d}' for i, p in enumerate(paths)}
idx = lambda kind, names: np.isin(np.arange(len(V[kind])), [V[kind].index(n) for n in names if n in V[kind]])
HEAL = idx('items', ['leftovers', 'petayaberry', 'salacberry', 'liechiberry', 'ganlonberry', 'apicotberry', 'starfberry'])
CHOICE = idx('items', ['choiceband'])
SPEEDBOOST = idx('abilities', ['speedboost']); CUTE = idx('abilities', ['cutecharm']); TRACE = idx('abilities', ['trace'])
FORECAST = idx('abilities', ['forecast'])
DRAG = idx('moves', ['roar', 'whirlwind'])
SHEDSKIN = idx('abilities', ['shedskin'])
UPDATE_AB = idx('abilities', ['magmaarmor', 'limber', 'insomnia', 'vitalspirit', 'owntempo', 'immunity', 'waterveil', 'oblivious'])
UPDATE_IT = idx('items', ['lumberry', 'liechiberry', 'petayaberry', 'salacberry'])
BOOST_ITEM_TYPE = np.zeros(len(V['items']), int)
for name, t in (('silkscarf', 'normal'), ('twistedspoon', 'psychic')):BOOST_ITEM_TYPE[V['items'].index(name)] = V['types'].index(t)
MOVE_TYPE = np.array([int((r or {}).get('type') or 0) for r in meta['moves']])
HIDDEN = np.array([m.startswith('hiddenpower') for m in V['moves']])
TRANSFORM = idx('moves', ['transform']); STICK = idx('items', ['stick']); MARVEL = idx('abilities', ['marvelscale'])
DITTO = np.isin(np.arange(len(V['species'])), [V['species'].index('ditto')])
STAGE = np.array([2/8, 2/7, 2/6, 2/5, 2/4, 2/3, 1, 3/2, 4/2, 5/2, 6/2, 7/2, 8/2])
counts = collections.Counter(); total = 0
for f in sorted(glob.glob(os.path.join(cap, 'shards', 'shard-*.npz'))):
    z = np.load(f); g = lambda p: z[leaf[p]]
    done = g('[0].done').astype(bool); live = ~done
    L = len(done); r = np.arange(L)[:, None]; s = np.arange(2)[None, :]
    act = g('[0].active'); hp = g('[0].hp'); stats = g('[0].stats'); boosts = g('[0].boosts'); status = g('[0].status')
    item = g('[0].item'); ability = g('[0].ability'); moves = g('[0].moves'); actions = g('[1]')
    a_hp = hp[r, s, act]; alive = (a_hp > 0).all(1)
    spe = stats[r, s, act, 5] * STAGE[np.clip(boosts[r, s, act, 4] + 6, 0, 12)] * np.where(status[r, s, act] == 1, 0.25, 1.0)
    tie = np.floor(spe[:, 0]) == np.floor(spe[:, 1])
    a_item = item[r, s, act]; a_ab = ability[r, s, act]; a_st = status[r, s, act]
    sw = (actions >= 4) & (actions <= 8); mv = actions < 4
    chosen = np.where(mv, moves[r, s, act, np.clip(actions, 0, 3)], 0)
    slot = np.clip(actions - 4, 0, 4); target = slot + (slot >= act)
    entering_trace = (sw & TRACE[ability[r, s, target]]).any(1)
    base = live & alive
    hpt = g('[0].hidden_power_type')[r, s, act]
    ctype = np.where(HIDDEN[chosen], np.where(hpt > 0, hpt, V['types'].index('dark')), MOVE_TYPE[chosen])
    boost_item = mv & (BOOST_ITEM_TYPE[a_item] > 0) & (ctype == BOOST_ITEM_TYPE[a_item]) & (chosen > 0)
    forced = g('[0].forced_switch').astype(bool)
    turn = g('[0].turn')
    c = {
        'residual heal tie (Leftovers / pinch berries, equal Speed)': base & tie & HEAL[a_item].all(1),
        'residual status tie (brn/psn/tox both, equal Speed)': base & tie & np.isin(a_st, [2, 3, 4]).all(1),
        'residual Speed Boost tie': base & tie & SPEEDBOOST[a_ab].all(1),
        'both players switch at equal Speed': base & tie & sw.all(1),
        'equal-Speed actives (any order tie can arise)': base & tie,
        'Cute Charm holder active': live & CUTE[a_ab].any(1),
        'Cute Charm holder in either party': live & CUTE[ability].any((1, 2)),
        'Trace holder switching in (chosen switch)': live & entering_trace,
        'Roar / Whirlwind chosen': live & (mv & DRAG[chosen]).any(1),
        'Choice Band holder under Encore': live & (CHOICE[a_item] & (g('[0].encore_counter')[r, s, act] > 0)).any(1),
        'Castform (Forecast) active': live & FORECAST[a_ab].any(1),
        'Silk Scarf / Twisted Spoon holder using a matching-type move': live & boost_item.any(1),
        'Transform chosen': live & (mv & TRANSFORM[chosen]).any(1),
        'Ditto active': live & DITTO[g('[0].species')[r, s, act]].any(1),
        'Stick holder active': live & STICK[a_item].any(1),
        'Marvel Scale holder active with a major status': live & (MARVEL[a_ab] & (a_st > 0)).any(1),
        'Shed Skin holder active with a major status': live & (SHEDSKIN[a_ab] & (a_st > 0)).any(1),
        "eachEvent('Update') tie (equal Speed, an Update handler on either active)": base & tie & (UPDATE_AB[a_ab] | UPDATE_IT[a_item]).any(1),
        'double replacement (both sides forced) at equal Speed': live & forced.all(1) & tie,
        'lead turn (turn 0) at equal Speed': live & (turn == 0) & tie,
        'Castform (Forecast) in either party': live & FORECAST[ability].any((1, 2)),
    }
    total += int(live.sum())
    for k, v in c.items():counts[k] += int(v.sum())
out = dict(source=cap, lane_steps_live=total, causes={k: dict(lane_steps=n, share=n / total) for k, n in counts.items()})
json.dump(out, open(sys.argv[3], 'w'), indent=1)
for k, n in counts.most_common():print(f'{n:9d} {n / total:8.4%}  {k}')
