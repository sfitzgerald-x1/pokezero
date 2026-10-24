"""Two authored two-turn traces, not fresh-source or full-fidelity evidence.

Slaking uses Slack Off from HP100/max1000 against a Splash-only Ditto. The
ability-none control differs only in Slaking's Truant ability. Both are legal
engine specifications, not sampled random-battle sets or production policies.
All finite critical/damage cases are enumerated despite being inactive here;
other chance inputs cannot affect these unstatused, itemless, tie-free turns.
Native full serialized states/volatiles and every JAX field are retained. The
common HP/PP projection does not erase or translate native-only TRUANT state.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as j
import numpy as np

from .benchmark import load_bound_pool
from .oracle_anchor import finite_rows
from .state import empty_state
from . import transition as t


def jax_root(metadata, *, truant):
    vocab=metadata['vocabulary'];state=empty_state(1,seed=246810)
    species=j.array([[vocab['species'].index('slaking'),vocab['species'].index('ditto')]])
    ability=j.array([[vocab['abilities'].index('truant') if truant else 0,0]])
    moves=j.array([[vocab['moves'].index('slackoff'),vocab['moves'].index('splash')]])
    stats=state.stats.at[:,:,0,:].set(200).at[:,:,0,0].set(1000)
    stats=stats.at[:,0,0,5].set(301).at[:,1,0,5].set(199)
    return state._replace(species=state.species.at[:,:,0].set(species),
        ability=state.ability.at[:,:,0].set(ability),stats=stats,
        types=state.types.at[:,:,0,0].set(vocab['types'].index('normal')),
        level=state.level.at[:,:,0].set(100),max_hp=state.max_hp.at[:,:,0].set(1000),
        hp=state.hp.at[:,0,0].set(100).at[:,1,0].set(1000),
        moves=state.moves.at[:,:,0,0].set(moves),pp=state.pp.at[:,:,0,0].set(8),
        done=j.array([False]))


def native_root(*, truant):
    if os.environ.get('POKEZERO_ENUMERATE_ROLLS')!='1':
        raise ValueError('caller must opt into exact native rolls in a fresh process')
    import poke_engine as engine
    from pokezero.poke_engine_adapter import BattleSpec,SideSpec,PokemonSpec,MoveSpec,build_poke_engine_state
    mons=[]
    for side,(name,move) in enumerate((('slaking','slackoff'),('ditto','splash'))):
        mons.append(PokemonSpec(id=name,level=100,types=('normal',),hp=100 if side==0 else 1000,
            maxhp=1000,attack=200,defense=200,special_attack=200,special_defense=200,
            speed=301 if side==0 else 199,ability='truant' if side==0 and truant else 'none',
            moves=(MoveSpec(move,8),)))
    return build_poke_engine_state(BattleSpec(*(SideSpec(pokemon=(mon,)) for mon in mons)),module=engine)


def native_step(state):
    import poke_engine as engine
    before=state.to_string();branches=engine.generate_instructions(state,'slackoff','splash')
    outcomes=defaultdict(float);records=[]
    for branch in branches:
        probability=float(branch.percentage)/100
        if not np.isfinite(probability) or probability<0:raise ValueError('invalid native mass')
        if probability==0:continue
        after=state.apply_instructions(branch)
        sides=(after.side_one,after.side_two);mons=[side.pokemon[0] for side in sides]
        projected=tuple(int(x) for x in (*[mon.hp for mon in mons],*[mon.moves[0].pp for mon in mons]))
        outcomes[projected]+=probability
        records.append(dict(probability=probability,hp_pp=list(projected),full_state=after.to_string(),
            active_abilities=[str(mon.ability) for mon in mons],
            volatile_statuses=[sorted(map(str,side.volatile_statuses)) for side in sides],
            instructions=list(map(str,branch.instruction_list))))
    if state.to_string()!=before or abs(sum(outcomes.values())-1)>1e-6:
        raise ValueError('native parent mutated or branch mass invalid')
    # The declared deterministic domain has ONE full future state, not merely
    # one HP/PP projection. Never pick a favorable branch or erase a flag.
    if len({r['full_state'] for r in records})!=1:raise ValueError('unexpected native full-state branching')
    return state.apply_instructions(next(b for b in branches if float(b.percentage)>0)),outcomes,records


def measure(pool_path,expected_pool_sha):
    start=time.monotonic();_,metadata,digest=load_bound_pool(pool_path)
    if digest!=expected_pool_sha:raise ValueError('metadata source binding differs')
    if len(jax.devices())!=1 or jax.devices()[0].platform!='cpu':raise ValueError('one CPU device required; no GPU overlap')
    table=t.tables_from_metadata(metadata);draws,weights=finite_rows(100);n=len(weights)
    chance=t.ChanceInputs(j.asarray(draws),*(j.zeros((n,2),j.int32) for _ in range(4)),j.zeros(n,j.int32))
    advance=jax.jit(lambda state:t.step_with_chance(state,j.zeros((n,2),j.int32),table,chance))
    cases=[]
    for enabled in (False,True):
        state=jax.tree.map(lambda x:j.repeat(x,n,axis=0),jax_root(metadata,truant=enabled))
        native=native_root(truant=enabled)
        case=dict(truant=enabled,initial_native_full_state=native.to_string(),turns=[])
        for turn in (1,2):
            out=jax.block_until_ready(advance(state));state=out.state
            if np.asarray(out.invalid_actions).any():raise ValueError('authored action invalid')
            # Keep every field, including latent queues/flags and RNG addresses.
            # All outcomes must coincide on the FULL JAX state in this domain.
            arrays={name:np.asarray(value) for name,value in state._asdict().items()}
            if any(not np.all(a==a[:1]) for a in arrays.values()):raise ValueError('unexpected JAX full-state branching')
            projected=tuple(map(int,(*arrays['hp'][0,:,0],*arrays['pp'][0,:,0,0])))
            gpu={projected:sum(weights,Fraction())}
            native,ref,records=native_step(native)
            support=set(ref)|set(gpu)
            case['turns'].append(dict(turn=turn,jax_finite_cases=n,jax_exact_mass=str(sum(gpu.values())),
                jax_full_state={name:a[0].tolist() for name,a in arrays.items()},native_branches=records,
                projected_outcomes=[dict(hp_pp=list(key),jax_probability=str(gpu.get(key,Fraction())),native_probability=ref.get(key,0)) for key in sorted(support)],
                total_variation_half_l1=sum(abs(float(gpu.get(key,0))-ref.get(key,0)) for key in support)/2))
        cases.append(case)
    native_path=Path(importlib.import_module('poke_engine.poke_engine').__file__)
    return dict(schema='pokezero.gpu_truant_counterexample.v1',cases=cases,
        elapsed_seconds=time.monotonic()-start,team_pool_sha256=digest,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        native_module_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),jax=jax.__version__,numpy=np.__version__,
        projection='Common active HP/selected-slot PP plus unmodified native full state/volatiles and all JAX fields. No assertion of full cross-schema state equivalence; native-only TRUANT is an explicit unsupported latent state, never normalized away.',
        chance='Two authored cases x two turns; 1024 exact crit/damage semantic cases/turn, total rational mass1. Nondamaging always-hit moves, no item/status/KO/tie/switch/secondary/ability-start chance. Other draw columns inactive by declared domain. Not PRNG-key enumeration.',
        interpretation='An authored mechanic counterexample/control only; not fresh source roots, a causal explanation of species scores, a structural impossibility proof, pool-frequency correct-effect coverage, strength or production qualification.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--team-pool',required=True);p.add_argument('--expected-pool-sha',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('exclusive new measurement output required')
    result=measure(a.team_pool,a.expected_pool_sha)
    with a.out.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(cases=len(result['cases']),elapsed_seconds=result['elapsed_seconds'],tv_by_case=[[t['total_variation_half_l1'] for t in c['turns']] for c in result['cases']]),flush=True))
