"""Remote replay of eight retained roots; no new games or model forwards."""
import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from gpu_engine import phase_d_protocol_world as protocol, combat, transition
from gpu_engine.phase_d_protocol_constructor import ProtocolConstructor, load_inputs


class Batch2Replay(unittest.TestCase):
    constructor = None
    baseline = False
    outcomes = []
    fixtures = Path(__file__).parent / 'fixtures' / 'phase_d_constructor_walls_batch2'

    def setUp(self):
        shared = type(self).constructor
        self.constructor = ProtocolConstructor(catalog=shared.catalog,source=shared.source,metadata=shared.metadata,
            owning_config=shared.owning_config,count=16,seed=20261103,functions=shared.functions)

    def replay(self, name):
        data = json.loads((self.fixtures / (name + '.json')).read_text())
        request = data['request']
        self.assertEqual(hashlib.sha256(''.join(x+'\n' for x in request['lines']).encode()).hexdigest(), data['prefix_sha256'])
        owner = int(request['owner'][1])-1
        foe = 1-owner
        vocab = self.constructor.metadata['vocabulary']
        original = protocol.map_spec
        planted = 'planted_non_trapping_foe' in name
        calls = []
        def plant(*a, **kw):
            result = original(*a, **kw)
            if result.state is not None:
                state = result.state
                ability = 'pressure' if name.startswith('shadow') else 'sandveil'
                active = int(state.active[0,foe])
                state = state._replace(ability=state.ability.at[0,foe,active].set(vocab['abilities'].index(ability)))
                calls.append(ability)
                return replace(result, state=state)
            return result
        made = None
        error = None
        with patch.object(protocol, 'map_spec', side_effect=plant) if planted else nullcontext():
            try:
                _, made, _ = self.constructor.prepare(request, request['decision_index'])
            except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
                error = type(exc).__name__ + ':' + str(exc)
        outcome = dict(fixture=name, built=0 if made is None else sum(w.state is not None for w in made.worlds),
                       error=error, refusals=dict(Counter(r for w in (() if made is None else made.worlds) for r in w.refusals)))
        self.outcomes.append(outcome)
        self.assertEqual({k:outcome[k] for k in ('built','error','refusals')},data['main' if self.baseline else 'patched'])
        if self.baseline:
            return
        if planted:
            self.assertEqual(len(calls),16)
        if made is not None:
            self.assertEqual(len(made.worlds),16)
        if not outcome['built']:
            return
        table = SimpleNamespace(combat=combat.from_metadata(self.constructor.metadata))
        for w, expected in zip(made.worlds,data['checks']['per_world'],strict=True):
            state = w.state
            self.assertEqual(w.index,expected['world'])
            oa,fa = int(state.active[0,owner]),int(state.active[0,foe])
            facts = {}
            if 'species' in expected:
                side = owner if name.startswith('castform') else foe
                active = int(state.active[0,side])
                facts['species'] = vocab['species'][int(state.species[0,side,active])]
                for key in ('types','base_types'):
                    facts[key] = [vocab['types'][int(x)] if int(x) else None for x in getattr(state,key)[0,side,active]]
            if 'foe_ability' in expected:
                facts['foe_ability'] = vocab['abilities'][int(state.ability[0,foe,fa])]
                facts['own_switch_lanes'] = np.asarray(transition.action_mask(state,table))[0,owner,4:].tolist()
                if 'own' in expected:
                    facts.update(own=vocab['species'][int(state.species[0,owner,oa])],foe=vocab['species'][int(state.species[0,foe,fa])],
                                 own_types=[vocab['types'][int(x)] if int(x) else None for x in state.types[0,owner,oa]],
                                 own_trapped_flag=bool(state.trapped[0,owner]),own_trap_source=int(state.trap_source[0,owner,oa]))
            if 'pp' in expected:
                slot = expected['slot']; matches = [i for i,m in enumerate(state.moves[0,foe,slot]) if vocab['moves'][int(m)]==expected['move']]
                self.assertEqual(len(matches),1)
                facts.update(slot=slot,move=expected['move'],pp=int(state.pp[0,foe,slot,matches[0]]))
                checks = next(x['checks'] for x in made.mapper_diagnostics if x['world']==w.index)
                self.assertTrue(any(x.get('pressure_floor') and x['status']=='CLEAR' for x in checks))
            self.assertEqual(facts,{k:v for k,v in expected.items() if k!='world'})

    def test_kecleon(self): self.replay('kecleon_color_change')
    def test_castform(self): self.replay('castform_forecast')
    def test_pressure_floor(self): self.replay('pressure_pp_floor')
    def test_pressure_overflow_refused(self): self.replay('pressure_pp_floor_plus_one_use')
    def test_shadow_tag(self): self.replay('shadow_tag_trapped')
    def test_shadow_tag_wrong_ability_refused(self): self.replay('shadow_tag_trapped_planted_non_trapping_foe')
    def test_maybe_trapped(self): self.replay('maybe_trapped_614_0')
    def test_maybe_trapped_non_trapping_ability(self): self.replay('maybe_trapped_614_0_planted_non_trapping_foe')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('metadata','vocabulary','showdown-root','node-binary','expected-metadata-sha256','expected-vocabulary-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--baseline',action='store_true')
    args = parser.parse_args()
    catalog,source,metadata,config = load_inputs(args)
    Batch2Replay.constructor = ProtocolConstructor(catalog=catalog,source=source,metadata=metadata,owning_config=config,count=16,seed=20261103)
    Batch2Replay.baseline = args.baseline
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Batch2Replay))
    print(json.dumps(dict(passed=result.wasSuccessful(),outcomes=Batch2Replay.outcomes),sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
