"""Literal Native team sets from frozen pool indices and constructor operands.

No team generation, stat calculation, sampling, engine initialization or
history inference occurs. The caller must bind the exact pool NPZ and sealed
construction manifest before calling this conversion.
"""
import numpy as np

STAT_NAMES = ('hp','atk','def','spa','spd','spe')
FROZEN_FIELDS = ('species','level','types','stats','moves','pp','item','ability','hidden_power_type','hidden_power_power')
VOCABULARIES = dict(species='species',moves='moves',item='items',ability='abilities',nature='natures')


def literal_teams(pool, metadata, raw, *, row, team_indices):
    if (type(row) is not int or not 0 <= row < len(raw.species) or not isinstance(team_indices, (list, tuple))
            or len(team_indices) != 2 or any(type(x) is not int or x < 0 for x in team_indices)):
        raise ValueError('literal sealed constructor row/team-index pair required')
    vocabulary = metadata['vocabulary'];teams = []
    for side, index in enumerate(team_indices):
        for name in (*FROZEN_FIELDS,'nature','ivs','evs'):
            value = np.asarray(pool[name])
            if value.dtype.kind not in 'iu' or not index < len(value):raise ValueError('literal constructor pool field missing: '+name)
        for name in FROZEN_FIELDS:
            if not np.array_equal(np.asarray(pool[name])[index],np.asarray(getattr(raw,name))[row,side]):
                raise ValueError('frozen pool/constructor differs: '+name)
        ivs=np.asarray(pool['ivs'])[index];evs=np.asarray(pool['evs'])[index]
        if (ivs.shape != (6,6) or evs.shape != (6,6) or np.any(ivs < 0) or np.any(ivs > 31) or np.any(evs < 0) or np.any(evs > 255)
                or np.any(np.sum(evs,axis=1) > 510)):
            raise ValueError('literal six-member IV/EV sets required')
        gender=np.asarray(raw.gender)[row,side];happiness=np.asarray(raw.happiness)[row,side]
        if (gender.shape != (6,) or gender.dtype.kind not in 'iu' or np.any(gender < 0) or np.any(gender > 2)
                or happiness.shape != (6,) or happiness.dtype.kind not in 'iu' or np.any(happiness < 0) or np.any(happiness > 255)):
            raise ValueError('sealed literal constructor gender/happiness required')
        team=[]
        for slot in range(6):
            def name(field,number,*,allow_zero=False):
                values=vocabulary[VOCABULARIES[field]]
                if not 0 <= number < len(values) or not allow_zero and number == 0 or type(values[number]) is not str:
                    raise ValueError('literal constructor vocabulary identity missing: '+field)
                return values[number]
            moves=np.asarray(pool['moves'])[index,slot]
            if moves.shape != (4,) or np.all(moves == 0):raise ValueError('complete literal move slots required')
            member=dict(species=name('species',int(np.asarray(pool['species'])[index,slot])),
                level=int(np.asarray(pool['level'])[index,slot]),
                ability=name('ability',int(np.asarray(pool['ability'])[index,slot])),
                item=name('item',int(np.asarray(pool['item'])[index,slot]),allow_zero=True),
                nature=name('nature',int(np.asarray(pool['nature'])[index,slot])),
                moves=[name('moves',int(move)) for move in moves if move != 0],
                ivs=dict(zip(STAT_NAMES,map(int,ivs[slot]))),evs=dict(zip(STAT_NAMES,map(int,evs[slot]))),
                gender=('N','M','F')[int(gender[slot])],happiness=int(happiness[slot]))
            if not 1 <= member['level'] <= 100:raise ValueError('literal constructor level required')
            team.append(member)
        teams.append(team)
    return teams
