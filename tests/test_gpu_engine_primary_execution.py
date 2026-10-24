"""Direct pinned-Showdown/JAX primary effects; finite authored subset, not games."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE = os.environ.get('NODE') or shutil.which('node')
SHOWDOWN = os.environ.get('POKEZERO_SHOWDOWN_ROOT')


def specifications():
    """Fixed before either engine executes; no selection from passing outcomes."""
    cases = []
    def add(move, actor=0, turns=1, **kw):
        choices = [['move 1' if side == actor else 'move 1' for side in range(2)]]
        choices += [['move splash', 'move splash'] for _ in range(turns - 1)]
        cases.append(dict(id=f'{len(cases):03d}-{move}', move=move, actor=actor,
                          choices=choices, **kw))
    for actor in (0, 1):
        for move in ('toxic', 'thunderwave', 'willowisp'):
            add(move, actor, turns=2)
            for guard in ({'substitute': True}, {'foe': 'protect'}, {'targetStatus': 'brn'}, {'roll': 99}):
                add(move, actor, **guard)
        for move, typ in (('toxic', 'Poison'), ('toxic', 'Steel'), ('willowisp', 'Fire'),
                          ('thunderwave', 'Ground'), ('thunderwave', 'Electric'), ('stunspore', 'Ground')):
            add(move, actor, type=typ)
        for move, ability in (('toxic', 'Immunity'), ('thunderwave', 'Limber'), ('willowisp', 'Water Veil')):
            add(move, actor, ability=ability)
        for duration in (2, 3, 4, 5):
            add('sleeppowder', actor, turns=5, duration=duration)
        for guard in ({'substitute': True}, {'foe': 'protect'}, {'targetStatus': 'par'}):
            add('sleeppowder', actor, **guard)
        add('toxic', actor, maxhp=1001, targetStatus='tox', toxicStage=14)
        add('toxic', actor, maxhp=1001, turns=3)
        case = cases[-1]
        case['choices'][1][1-actor] = 'switch 2'
        case['choices'][2][1-actor] = 'switch 2'  # Showdown reorders party on switch
        for move in ('calmmind', 'swordsdance', 'bulkup', 'agility', 'dragondance'):
            for stage in (-6, 0, 5, 6): add(move, actor, stage=stage)
        add('rest', actor, turns=4, hp=17, status='brn')
        add('rest', actor, status='brn')
        add('rest', actor, hp=17, item='lumberry', turns=2)
        for move in ('recover', 'softboiled'):
            for hp in (1, 75, 101): add(move, actor, hp=hp)
        add('spikes', actor, turns=4)
        cases[-1]['choices'] = [['move 1', 'move 1'] for _ in range(4)]
    # Uncontaminated status-only accuracy fixtures; exactly one accuracy event.
    for a, e in ((0,0), (1,0), (-1,0), (0,1), (0,-1), (6,6), (-6,-6), (6,-6), (-6,6), (3,2), (-3,-2)):
        add('toxic', accuracyStage=a, evasionStage=e, accuracyOnly=True)
    return cases


@unittest.skipUnless(importlib.util.find_spec('jax') and NODE and SHOWDOWN,
                     'requires JAX, Node and pinned Gen3 Showdown')
class PrimaryExecutionTests(unittest.TestCase):
    def test_actual_primary_execution_and_accuracy_boundaries(self):
        import jax
        import jax.numpy as j
        from gpu_engine import effects, entry, items, mechanics as m, secondary, transition as tr
        from gpu_engine.state import empty_state
        cases = specifications()
        names = ['_none', 'struggle', 'splash', 'protect', 'toxic', 'thunderwave', 'stunspore',
                 'willowisp', 'sleeppowder', 'rest', 'recover', 'softboiled', 'calmmind',
                 'swordsdance', 'bulkup', 'agility', 'dragondance', 'spikes', 'growl']
        payload = {'names': names[1:], 'cases': cases}
        proc = subprocess.run([NODE, str(Path(__file__).with_name('gpu_primary_execution_oracle.cjs')), SHOWDOWN],
                              input=json.dumps(payload), text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        oracle = json.loads(proc.stdout)
        self.assertEqual(len(oracle['cases']),len(cases))
        evidence = os.environ.get('POKEZERO_PRIMARY_EXECUTION_EVIDENCE')
        if evidence:
            # Explicit evidence destination, exclusive create; never replace a result.
            with open(evidence, 'x') as f: json.dump({'input': payload, 'oracle': oracle}, f, sort_keys=True)
        catalog = [None] + oracle['catalog']
        types = ['_none','normal','fighting','flying','poison','ground','rock','bug','ghost',
                 'steel','fire','water','grass','electric','psychic','ice','dragon','dark']
        abilities = ['_none', 'runaway', 'immunity', 'limber', 'waterveil']
        meta = {'vocabulary': {'moves': names, 'types': types, 'abilities': abilities,
                              'items': ['_none', 'lumberry'], 'species': ['_none', 'mew', 'ditto']},
                'moves': catalog}
        n = len(names)
        table = tr.MoveTable(j.zeros(n,j.int32), j.array([0]+[types.index(x['type']) for x in catalog[1:]]),
            j.zeros(n,bool), j.array([100]+[100 if x['accuracy'] is True else x['accuracy'] for x in catalog[1:]]),
            j.array([True]+[x['accuracy'] is True for x in catalog[1:]]), j.array([0]+[x['priority'] for x in catalog[1:]]),
            j.ones((18,18),j.int32), j.ones((18,18),j.int32), j.array(1), effects.from_metadata(meta),
            items=items.from_metadata(meta), entry=entry.from_metadata(meta), secondary=secondary.from_metadata(meta))
        statuses = ['', 'par', 'brn', 'psn', 'tox', 'slp', 'frz']
        b = len(cases); s = empty_state(b, seed=20261024)
        hp = j.zeros_like(s.hp).at[:,:,:2].set(101)
        s = s._replace(hp=hp, max_hp=hp, level=j.full_like(s.level,100),
            stats=j.full_like(s.stats,200), moves=j.full_like(s.moves,names.index('splash')),
            pp=j.full_like(s.pp,20), types=s.types.at[...,0].set(types.index('normal')),
            ability=j.full_like(s.ability,1), done=j.zeros_like(s.done))
        draws = j.zeros((b,2,12),j.int32).at[:,:,1:3].set(99).at[:,:,4:6].set(15)
        for i, spec in enumerate(cases):
            a=spec['actor']; target=1-a; maxhp=spec.get('maxhp',101)
            s=s._replace(hp=s.hp.at[i,:,:2].set(maxhp),max_hp=s.max_hp.at[i,:,:2].set(maxhp),
                stats=s.stats.at[i,a,:,5].set(1000).at[i,target,:,5].set(100),
                moves=s.moves.at[i,a,0,0].set(names.index(spec['move'])).at[i,target,0,0].set(names.index(spec.get('foe','splash')))
                    .at[i,target,0,1].set(names.index('splash' if spec.get('foe') else 'growl')))
            for field, side in (('status',a),('targetStatus',target)):
                s=s._replace(status=s.status.at[i,side,0].set(statuses.index(spec.get(field,''))))
            if 'hp' in spec: s=s._replace(hp=s.hp.at[i,a,0].set(spec['hp']))
            if 'type' in spec: s=s._replace(types=s.types.at[i,target,0,0].set(types.index(spec['type'].lower())))
            if 'ability' in spec: s=s._replace(ability=s.ability.at[i,target,0].set(abilities.index(spec['ability'].lower().replace(' ',''))))
            if spec.get('substitute'): s=s._replace(substitute_hp=s.substitute_hp.at[i,target,0].set(25))
            if 'stage' in spec: s=s._replace(boosts=s.boosts.at[i,a,0].set(spec['stage']))
            if 'item' in spec: s=s._replace(item=s.item.at[i,a,0].set(1))
            if 'toxicStage' in spec: s=s._replace(status_counter=s.status_counter.at[i,target,0].set(spec['toxicStage']))
            if 'accuracyStage' in spec: s=s._replace(boosts=s.boosts.at[i,a,0,5].set(spec['accuracyStage']).at[i,target,0,6].set(spec['evasionStage']))
            draws=draws.at[i,a,6].set(spec.get('roll',0)).at[i,a,7].set(spec.get('duration',2)-2)
        chance=tr.ChanceInputs(draws,j.zeros((b,2),j.int32),j.full((b,2),99,j.int32),
            j.zeros((b,2),j.int32),j.zeros((b,2),j.int32),j.zeros(b,j.int32))
        step=jax.jit(tr.step_with_chance)
        snapshots=[]
        for turn in range(max(len(x['choices']) for x in cases)):
            actions=j.ones((b,2),j.int32)
            for i,spec in enumerate(cases):
                if turn>=len(spec['choices']):
                    s=s._replace(done=s.done.at[i].set(True))
                    continue
                for side,choice in enumerate(spec['choices'][turn]):
                    splash_slot=1 if side==spec['actor'] or spec.get('foe') else 0
                    # Bench Ditto has Splash in slot0, including switch-return turns.
                    if int(s.active[i,side])==1: splash_slot=0
                    actions=actions.at[i,side].set(4 if choice.startswith('switch') else splash_slot if choice=='move splash' else 0)
            result=step(s,actions,table,chance); jax.block_until_ready(result)
            self.assertFalse(bool(j.any(result.invalid_actions)))
            s=result.state
            snapshots.append(jax.tree.map(lambda x:x.tolist(),s))
        mismatches=[]
        for i,(spec,row) in enumerate(zip(cases,oracle['cases'])):
            self.assertEqual(row['id'],spec['id'])
            self.assertEqual(len(row['rows']),len(spec['choices']))
            actor=spec['actor']; target=1-actor
            own=row['rows'][0]['mons'][actor][0]
            foe=row['rows'][0]['mons'][target][0]
            self.assertEqual(own['pp'][0],19,spec)
            self.assertTrue(any(e.startswith(f'|move|p{actor+1}a:') for e in row['rows'][0]['events']),spec)
            # Nonvacuity guards bind real execution and failed uses, not just
            # two engines agreeing on untouched fields or catalog metadata.
            if spec['move']=='sleeppowder' and 'duration' in spec:
                clocks=[r['mons'][target][0]['counter'] for r in row['rows']]
                self.assertEqual(clocks,[max(0,spec['duration']-1-t) for t in range(5)])
                self.assertEqual(foe['pp'][0],20)  # sleep vetoed the queued action
                self.assertLess(row['rows'][-1]['mons'][target][0]['pp'][0],20)  # wake acted
                self.assertTrue(any('|cant|' in e and '|slp' in e for e in row['rows'][0]['events']))
                self.assertTrue(any('|-curestatus|' in e and '|slp' in e for r in row['rows'] for e in r['events']))
            if spec['move'] in ('thunderwave','willowisp') and len(row['rows'])==2:
                self.assertEqual(foe['status'],'par' if spec['move']=='thunderwave' else 'brn')
                self.assertEqual(foe['pp'][0],19)  # selected paralysis draw allowed slower action
            if spec.get('substitute') or spec.get('foe')=='protect' or 'ability' in spec:
                self.assertEqual(foe['status'],'',spec)
            if 'targetStatus' in spec:
                self.assertEqual(foe['status'],spec['targetStatus'],spec)
            if spec.get('roll')==99 and spec['move'] in ('toxic','willowisp'):
                self.assertEqual(foe['status'],'',spec)
            if spec['move']=='toxic' and len(row['rows'])==2:
                self.assertEqual((foe['hp'],foe['counter']),(95,1))
                self.assertEqual((row['rows'][1]['mons'][target][0]['hp'],
                                  row['rows'][1]['mons'][target][0]['counter']),(83,2))
            if 'toxicStage' in spec:
                self.assertEqual((foe['hp'],foe['counter']),(71,15))
            if spec['move']=='toxic' and len(row['rows'])==3:
                self.assertEqual(row['rows'][-1]['mons'][target][0]['counter'],1)
                self.assertEqual(row['rows'][-1]['mons'][target][0]['hp'],877)
            if 'stage' in spec and spec['stage']==0:
                self.assertTrue(any(v>0 for v in own['boosts']))
                self.assertEqual(foe['boosts'],[0]*7)
            if spec['move']=='rest':
                self.assertEqual(own['pp'][0],19)
                if 'hp' in spec:
                    self.assertEqual(own['hp'],101)
                    self.assertEqual((own['status'],own['counter']),('',0) if 'item' in spec else ('slp',3))
                    self.assertEqual(own['item'],'')
                    self.assertEqual(row['rows'][-1]['mons'][actor][0]['status'],'')
                else: self.assertEqual((own['status'],own['hp']),('brn',89))
            if spec['move'] in ('recover','softboiled'):
                self.assertEqual(own['hp'],min(101,spec['hp']+50))
                self.assertEqual(own['pp'][0],19)
            if spec['move']=='spikes':
                self.assertEqual([r['spikes'][target] for r in row['rows']],[1,2,3,3])
                self.assertEqual(row['rows'][-1]['mons'][actor][0]['pp'][0],16)
                self.assertTrue(any('|-fail|' in e for e in row['rows'][-1]['events']))
            for turn, expected in enumerate(row['rows']):
                state=snapshots[turn]
                actual={'active':state.active[i], 'spikes':state.spikes[i], 'mons':[]}
                for side in range(2):
                    party=[]
                    for mon in range(2):
                        status=state.status[i][side][mon]
                        party.append(dict(hp=state.hp[i][side][mon],status=statuses[status],
                            counter=state.status_counter[i][side][mon] if status in (m.TOX,m.SLP) else 0,
                            boosts=state.boosts[i][side][mon],item='lumberry' if state.item[i][side][mon] else '',
                            pp=state.pp[i][side][mon][:2 if mon==0 else 1]))
                    actual['mons'].append(party)
                compare=copy.deepcopy({k:v for k,v in expected.items() if k!='events'})
                # Native Showdown resets Toxic on entry; JAX resets on departure.
                # Only an INACTIVE counter is unobservable before that reset.
                # Preserve the raw oracle counter in the evidence above.
                for side in range(2):
                    for mon in range(2):
                        if compare['mons'][side][mon]['status']=='tox' and mon!=compare['active'][side]:
                            compare['mons'][side][mon]['counter']=0
                if actual!=compare: mismatches.append({'id':spec['id'],'turn':turn,'actual':actual,'expected':expected})
            if spec.get('accuracyOnly'):
                accuracy=[d for d in row['draws'] if d['event']=='accuracy']
                self.assertEqual(len(accuracy),1,spec)
                threshold,den=accuracy[0]['args']; self.assertEqual(den,100)
                mask=m.accuracy_hit(85,spec['accuracyStage'],spec['evasionStage'],j.arange(100))
                self.assertEqual(mask.tolist(),[r<threshold for r in range(100)],spec)
        if evidence:
            with open(evidence+'.comparisons.json','x') as f: json.dump({'cases':b,'mismatches':mismatches},f,sort_keys=True)
        self.assertEqual(mismatches,[],json.dumps(mismatches[:3],sort_keys=True))


if __name__=='__main__': unittest.main()
