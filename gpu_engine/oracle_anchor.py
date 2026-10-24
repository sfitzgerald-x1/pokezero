"""Exact finite-law ordinary-damage anchor, NOT covered-pool fidelity.

The narrow authored domain has no items, abilities, statuses, boosts, switches,
speed ties or KO. Only critical, accuracy and damage draws affect HP and PP.
Both engines consume the same immutable input specification. Poke-engine is a
pinned comparison implementation, not independent Showdown ground truth.
"""
from collections import defaultdict
from fractions import Fraction
import argparse
import hashlib
import itertools
import importlib
import json
import os
from pathlib import Path
import random
import time

import jax
import jax.numpy as j
import numpy as np

from . import transition as t
from .state import empty_state


def finite_rows(accuracy=95):
    """Exact unit mass; reused roll columns stay shared, zero mass excluded."""
    if type(accuracy)is not int or not 1<=accuracy<=100:raise ValueError('integer accuracy required')
    hit_accuracy=accuracy
    rows=[];weights=[]
    accuracy_cases=range(1 if accuracy==100 else 2)
    for c0,d0,a0,c1,d1,a1 in itertools.product(range(2),range(16),accuracy_cases,
                                              range(2),range(16),accuracy_cases):
        row=np.zeros((2,12),np.int32)
        mass=Fraction(1)
        for side,crit,damage,accuracy in ((0,c0,d0,a0),(1,c1,d1,a1)):
            row[side,4]=crit  # 0 crit, 1 is representative of rolls 1..15
            row[side,5]=damage
            row[side,6]=0 if accuracy==0 else 99
            mass*=Fraction(1 if crit==0 else 15,16)*Fraction(1,16)
            mass*=Fraction(hit_accuracy if accuracy==0 else 100-hit_accuracy,100)
        rows.append(row);weights.append(mass)
    assert sum(weights)==1
    return np.asarray(rows),tuple(weights)


def anchor_input(index,seed=246810,move='tackle'):
    """Fresh authored values; not a source root, accepted team or game."""
    rng=random.Random(seed+index)
    # Unique speeds and high HP rule out ordering/KO-dependent domains.
    if move not in ('tackle','earthquake'):raise ValueError('bounded anchor move required')
    return dict(index=index,level=100,hp=1000,
        attack=[rng.randrange(100,301),rng.randrange(100,301)],
        defense=[rng.randrange(100,301),rng.randrange(100,301)],
        speed=[301,199],move=move,power=35 if move=='tackle' else 100,
        accuracy=95 if move=='tackle' else 100,pp=8)


def jax_distribution(spec,draws,weights):
    n=len(weights);state=empty_state(n)
    stats=np.full((n,2,6,6),200,np.int32);stats[...,0]=spec['hp']
    for side in range(2):
        stats[:,side,:,1]=spec['attack'][side]
        stats[:,side,:,2]=spec['defense'][side]
        stats[:,side,:,5]=spec['speed'][side]
    state=state._replace(stats=j.asarray(stats),level=j.full_like(state.level,spec['level']),
        hp=j.full_like(state.hp,spec['hp']),max_hp=j.full_like(state.hp,spec['hp']),
        moves=state.moves.at[...,0].set(1),pp=state.pp.at[...,0].set(spec['pp']),
        types=state.types.at[...,0].set(1),done=j.zeros_like(state.done))
    table=t.MoveTable(j.array([0,spec['power'],50]),j.array([0,1 if spec['move']=='tackle' else 2,0]),j.array([False,True,True]),
        j.array([100,spec['accuracy'],100]),j.array([True,False,True]),j.zeros(3,j.int32),
        j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(2))
    chance=t.ChanceInputs(j.asarray(draws),*(j.zeros((n,2),j.int32) for _ in range(4)),
        j.zeros(n,j.int32))
    result=jax.jit(t.step_with_chance)(state,j.zeros((n,2),j.int32),table,chance)
    hp,pp=np.asarray(result.state.hp)[:,:,0],np.asarray(result.state.pp)[:,:,0,0]
    if np.any(hp<=0) or np.asarray(result.invalid_actions).any():
        raise ValueError('anchor domain violated: KO or invalid action')
    distribution=defaultdict(Fraction)
    for h,p,weight in zip(hp,pp,weights,strict=True):
        distribution[tuple(map(int,(*h,*p)))]+=weight
    return distribution


def native_distribution(spec):
    # Native reads this in a OnceLock; the CLI checks it BEFORE importing native.
    if os.environ.get('POKEZERO_ENUMERATE_ROLLS')!='1':
        raise ValueError('isolated process must set POKEZERO_ENUMERATE_ROLLS=1 before native import')
    import poke_engine as engine
    from pokezero.poke_engine_adapter import BattleSpec,SideSpec,PokemonSpec,MoveSpec,build_poke_engine_state
    sides=[]
    for side in range(2):
        mon=PokemonSpec(id='ditto',level=100,types=('normal',),hp=spec['hp'],maxhp=spec['hp'],
            attack=spec['attack'][side],defense=spec['defense'][side],
            special_attack=200,special_defense=200,speed=spec['speed'][side],
            moves=(MoveSpec(spec['move'],spec['pp']),))
        sides.append(SideSpec(pokemon=(mon,)*6))
    state=build_poke_engine_state(BattleSpec(*sides),module=engine)
    before=state.to_string();distribution=defaultdict(float)
    branches=engine.generate_instructions(state,spec['move'],spec['move'])
    if any(not np.isfinite(float(branch.percentage)) or float(branch.percentage)<0
           for branch in branches):raise ValueError('invalid native branch probability')
    total=sum(float(branch.percentage) for branch in branches)/100
    if not np.isfinite(total) or abs(total-1)>1e-6:
        raise ValueError(f'native branch mass {total}')
    for branch in branches:
        if float(branch.percentage)==0:continue
        after=state.apply_instructions(branch)
        mons=[getattr(after,seat).pokemon[0] for seat in ('side_one','side_two')]
        key=tuple([int(mon.hp) for mon in mons]+[int(mon.moves[0].pp) for mon in mons])
        distribution[key]+=float(branch.percentage)/100
    if state.to_string()!=before:raise ValueError('native oracle mutated the parent')
    return distribution,len(branches),total


def measure(*,count,move='tackle'):
    if type(count)is not int or not 1<=count<=32:raise ValueError('anchor cap is 32')
    draws,weights=finite_rows(95 if move=='tackle' else 100);rows=[];start=time.monotonic()
    for index in range(count):
        spec=anchor_input(index,move=move)
        native,branches,total=native_distribution(spec)
        gpu=jax_distribution(spec,draws,weights)
        support=set(native)|set(gpu)
        rows.append(dict(input=spec,native_branches=branches,native_mass=total,
            jax_cases=len(weights),jax_exact_mass=str(sum(gpu.values())),
            native_only_support=len(set(native)-set(gpu)),jax_only_support=len(set(gpu)-set(native)),
            total_variation_half_l1=sum(abs(float(gpu.get(key,0))-native.get(key,0)) for key in support)/2,
            outcomes=[dict(hp_pp=list(key),jax_probability=str(gpu.get(key,Fraction())),
                native_probability=native.get(key,0)) for key in sorted(support)]))
    return dict(schema='pokezero.gpu_native_damage_anchor.v1',rows=rows,
        elapsed_seconds=time.monotonic()-start,
        projection='active HP and selected-slot PP only; not full future-state equivalence',
        domain=f'Authored ordinary {move} mirror, PP=8 (native only emits PP below 10), no abilities/items/status/boosts/ties/KO; not fresh source roots or pool-frequency coverage',
        probability='Exact rational JAX semantic finite-draw masses versus native f32 percentages; not PRNG-key enumeration',
        interpretation='Mechanic anchor only. No game strength, full fidelity, or 85% ledger credit.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--count',type=int,default=32)
    parser.add_argument('--move',choices=('tackle','earthquake'),default='tackle')
    args=parser.parse_args()
    if os.environ.get('POKEZERO_ENUMERATE_ROLLS')!='1':raise ValueError('exact native rolls required')
    result=measure(count=args.count,move=args.move)
    result['driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    native=Path(importlib.import_module('poke_engine.poke_engine').__file__)
    result['bindings']=dict(jax=jax.__version__,numpy=np.__version__,
        native_module_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        gpu_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))})
    with args.out.open('x') as stream:json.dump(result,stream,sort_keys=True)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
