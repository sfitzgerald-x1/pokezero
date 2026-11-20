"""Literal pinned Native reference decoding; no projection or admission.

Native serializes a six-character original-to-current team permutation.
Pokémon references name CURRENT positions, so species/name matching is not an
identity join. These helpers consume only Native JSON and the literal original
constructor species roster. They neither accept GPU state nor fill history.
"""
import re


def to_id(value):
    if type(value) is not str:raise ValueError('literal Native identity string required')
    result = re.sub('[^a-z0-9]', '', value.lower())
    if not result:raise ValueError('empty Native identity')
    return result


def reference_identifier(value, kind, *, allow_none=False):
    if kind not in ('Move', 'Species', 'Ability', 'Item', 'Condition'):raise ValueError('unsupported Native reference kind')
    if value is None or value == '':
        if allow_none:return None
        raise ValueError('required Native identity missing')
    if isinstance(value, dict):
        if kind != 'Move' or not ('move' in value or 'id' in value):raise ValueError('unsupported Native reference object')
        result = reference_identifier(value.get('move', value.get('id')), kind)
        if 'move' in value and 'id' in value and reference_identifier(value['id'], kind) != result:
            raise ValueError('Native ActiveMove identities disagree')
        return result
    if type(value) is not str:raise ValueError('literal Native reference required')
    if re.fullmatch('[a-z0-9]+', value):return value
    match = re.fullmatch(r'\['+kind+r':([a-z0-9]+)\]', value)
    if match is None:raise ValueError('wrong or malformed Native reference kind')
    return match[1]


class PartyMap:
    """Decode one frozen singles boundary, preserving original physical slots.

    Complete constructor-set and physical-field validation remain mandatory
    separate checks. This class establishes identity only, including duplicate
    original species and changed current effective species after Transform.
    """
    def __init__(self, battle, original_species):
        if (not isinstance(battle, dict) or not isinstance(battle.get('sides'), list) or len(battle['sides']) != 2
                or len(original_species) != 2 or any(len(side) != 6 for side in original_species)):
            raise ValueError('literal two-side six-member Native boundary required')
        self.battle = battle;self.original_to_current = [];self.current_to_original = []
        for side, saved in enumerate(battle['sides']):
            if not isinstance(saved, dict):raise ValueError('literal Native side object required')
            team = saved.get('team');pokemon = saved.get('pokemon');active = saved.get('active')
            if (type(team) is not str or len(team) != 6 or set(team) != set('123456')
                    or not isinstance(pokemon, list) or len(pokemon) != 6
                    or not isinstance(active, list) or len(active) != 1):
                raise ValueError('complete Native team permutation/pokemon/active required')
            if 'id' in saved and saved['id'] != 'p'+str(side+1):raise ValueError('Native side identity differs')
            if 'n' in saved and (type(saved['n']) is not int or saved['n'] != side):raise ValueError('Native side ordinal differs')
            permutation = tuple(int(char)-1 for char in team)
            inverse = tuple(permutation.index(current) for current in range(6))
            self.original_to_current.append(permutation);self.current_to_original.append(inverse)
            for original, current in enumerate(permutation):
                mon = pokemon[current]
                if (not isinstance(mon, dict) or not isinstance(mon.get('set'), dict)
                        or to_id(mon['set'].get('species')) != to_id(original_species[side][original])):
                    raise ValueError('Native original constructor species/permutation differs')
        for side in (0, 1):
            actual_side, _ = self.resolve_pokemon(battle['sides'][side]['active'][0])
            if actual_side != side:raise ValueError('Native active reference crosses sides')

    def resolve_pokemon(self, reference):
        if type(reference) is not str:raise ValueError('literal Native Pokemon reference required')
        match = re.fullmatch(r'\[Pokemon:p([12])([a-f])\]', reference)
        if match is None:raise ValueError('malformed Native Pokemon reference')
        side = int(match[1])-1;current = ord(match[2])-ord('a')
        return side, self.current_to_original[side][current]

    def mon(self, side, original):
        if type(side) is not int or side not in (0, 1) or type(original) is not int or not 0 <= original < 6:
            raise ValueError('original Native party coordinate required')
        return self.battle['sides'][side]['pokemon'][self.original_to_current[side][original]]

    def active(self, side):
        if type(side) is not int or side not in (0, 1):raise ValueError('Native side required')
        actual_side, original = self.resolve_pokemon(self.battle['sides'][side]['active'][0])
        if actual_side != side:raise ValueError('Native active reference crosses sides')
        return original
