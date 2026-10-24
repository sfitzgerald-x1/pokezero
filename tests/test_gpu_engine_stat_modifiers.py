"""Measured missing stat families on authored inputs; not game qualification."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE=os.environ.get('NODE') or shutil.which('node')
SHOWDOWN=os.environ.get('POKEZERO_SHOWDOWN_ROOT')


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated spike JAX')
class StatModifierTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import combat,items,mechanics,pursuit,transition
        from gpu_engine.state import empty_state
        self.jax,self.j,self.c,self.i,self.m,self.p,self.t,self.empty=jax,j,combat,items,mechanics,pursuit,transition,empty_state
        self.species=['_none','latias','latios','mew','azumarill','medicham']
        self.items=['_none','souldew','choiceband']
        self.abilities=['_none','runaway','hugepower','purepower','guts']
        types=['_none','normal','fighting','flying','poison','ground','rock','bug','ghost',
            'steel','fire','water','grass','electric','psychic','ice','dragon','dark']
        self.meta={'vocabulary':dict(species=self.species,items=self.items,abilities=self.abilities,
            types=types,moves=['_none','tackle','watergun','pursuit','splash','struggle'])}
        self.it=self.i.from_metadata(self.meta);self.ct=self.c.from_metadata(self.meta)
        self.table=self.t.MoveTable(j.array([0,50,50,40,0,50]),j.array([0,1,11,17,1,0]),
            j.array([False,True,False,False,False,True]),j.full(6,100),j.ones(6,bool),j.zeros(6,j.int32),
            j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(5),items=self.it,combat=self.ct,
            pursuit=self.p.from_metadata(self.meta))

    def row(self,**changes):
        return dict(dict(attacker='Latias',defender='Mew',item='Soul Dew',defender_item='',ability='Run Away',
            attack=103,defense=101,attack_stage=0,defense_stage=0,physical=False,critical=False,
            burned=False,confusion=False,pursuit=False),**changes)

    def factors(self,rows):
        j=self.j
        idx=lambda names,field:j.array([names.index(r[field].lower().replace(' ','')) if r[field] else 0 for r in rows])
        physical=j.array([r['physical'] or r['confusion'] for r in rows])
        item,species,ability=idx(self.items,'item'),idx(self.species,'attacker'),idx(self.abilities,'ability')
        num,den=self.i.attack_modifier(self.it,item,species,physical)
        num,den=self.c.attack_modifier(self.ct,ability,j.array([self.m.BRN if r['burned'] else 0 for r in rows]),
            physical,item_num=num,item_den=den)
        dn,dd=self.i.defense_modifier(self.it,idx(self.items,'defender_item'),idx(self.species,'defender'),physical)
        return num,den,dn,dd

    def test_category_species_and_physical_power_event_chain(self):
        rows=[self.row(),self.row(attacker='Latios'),self.row(attacker='Mew'),self.row(physical=True),
            self.row(defender='Latios',defender_item='Soul Dew'),
            self.row(attacker='Azumarill',item='Choice Band',ability='Huge Power',physical=True,burned=True),
            self.row(attacker='Medicham',item='',ability='Pure Power',physical=True),
            self.row(attacker='Azumarill',item='',ability='Huge Power',physical=False)]
        num,den,dn,dd=self.factors(rows)
        self.assertEqual((num.tolist(),den.tolist()),([3,3,1,1,3,6,2,1],[2,2,1,1,2,2,1,1]))
        self.assertEqual((dn.tolist(),dd.tolist()),([1,1,1,1,3,1,1,1],[1,1,1,1,2,1,1,1]))

    def state(self,rows):
        j=self.j;s=self.empty(len(rows),seed=71)
        index=lambda names,field:j.array([names.index(r[field].lower().replace(' ','')) if r[field] else 0 for r in rows])
        stats=j.full_like(s.stats,200).at[...,0].set(10000)
        for stat in (1,3):stats=stats.at[:,0,:,stat].set(j.array([r['attack'] for r in rows])[:,None])
        for stat in (2,4):stats=stats.at[:,:, :,stat].set(j.array([r['defense'] for r in rows])[:,None,None])
        boosts=s.boosts
        for stat in (0,2):boosts=boosts.at[:,0,0,stat].set(j.array([r['attack_stage'] for r in rows]))
        for stat in (1,3):boosts=boosts.at[:,1,0,stat].set(j.array([r['defense_stage'] for r in rows]))
        return s._replace(stats=stats,level=j.full_like(s.level,73),hp=j.full_like(s.hp,10000),
            max_hp=j.full_like(s.hp,10000),pp=j.full_like(s.pp,10),done=j.zeros_like(s.done),
            types=s.types.at[...,0].set(7),boosts=boosts,
            species=s.species.at[:,0,0].set(index(self.species,'attacker')).at[:,1,0].set(index(self.species,'defender')),
            ability=s.ability.at[:,0,0].set(index(self.abilities,'ability')),
            item=s.item.at[:,0,0].set(index(self.items,'item')).at[:,1,0].set(index(self.items,'defender_item')),
            status=s.status.at[:,0,0].set(j.array([self.m.BRN if r['burned'] else 0 for r in rows])),
            moves=j.full_like(s.moves,4).at[:,0,0,0].set(j.array([1 if r['physical'] else 2 for r in rows])))

    @unittest.skipUnless(NODE and SHOWDOWN,'requires pinned Showdown')
    def test_native_damage_odd_stats_boosts_crit_burn_and_confusion(self):
        rows=[self.row(),self.row(attacker='Latios'),self.row(attacker='Mew'),self.row(physical=True),
            self.row(item='',defender='Latios',defender_item='Soul Dew'),
            self.row(defender='Latios',defender_item='Soul Dew',defense=103),
            self.row(defender='Latios',defender_item='Soul Dew',attack_stage=-2,defense_stage=2,critical=True),
            self.row(defender='Latios',defender_item='Soul Dew',attack_stage=2,defense_stage=-2,critical=True),
            self.row(attacker='Azumarill',item='',ability='Huge Power',physical=True),
            self.row(attacker='Medicham',item='',ability='Pure Power',physical=True),
            self.row(attacker='Medicham',item='Choice Band',ability='Pure Power',physical=True,burned=True),
            self.row(attacker='Azumarill',item='Choice Band',ability='Huge Power',physical=True,confusion=True,burned=True),
            self.row(attacker='Medicham',item='Choice Band',ability='Pure Power',physical=True,confusion=True),
            self.row(attacker='Azumarill',item='',ability='Huge Power',physical=False),
            self.row(confusion=True,physical=True)]
        result=subprocess.run([NODE,str(Path(__file__).with_name('gpu_stat_modifiers_oracle.cjs')),SHOWDOWN],
            input=json.dumps(rows),text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr);native=json.loads(result.stdout)
        j=self.j;num,den,dn,dd=self.factors(rows)
        kwargs=dict(level=73,power=j.array([40 if r['confusion'] else 50 for r in rows]),
            attack=j.array([r['attack'] for r in rows]),defense=j.array([r['defense'] for r in rows]),
            attack_stage=j.array([r['attack_stage'] for r in rows]),defense_stage=j.array([r['defense_stage'] for r in rows]),
            physical=j.array([r['physical'] or r['confusion'] for r in rows]),
            critical=j.array([r['critical'] for r in rows]),typeless=j.array([r['confusion'] for r in rows]),
            burned=j.array([r['burned'] for r in rows]),attack_num=num,attack_den=den,defense_num=dn,defense_den=dd)
        actual=self.m.gen3_damage(**kwargs)
        self.assertEqual(actual.tolist(),[r['damage'] for r in native])
        self.assertEqual(int(native[0]['modified_spa']),154)
        self.assertEqual(int(native[2]['modified_spa']),103)
        self.assertEqual(int(native[4]['modified_spd']),151)

    def test_ordinary_step_and_pursuit_consume_defender_modifiers(self):
        rows=[self.row(item='',defender='Latios',defender_item='Soul Dew'),
            self.row(item='',defender='Mew',defender_item='Soul Dew'),
            self.row(item='',defender='Latios',defender_item='Soul Dew',physical=True),
            self.row(attacker='Azumarill',item='Choice Band',ability='Huge Power',physical=True,burned=True)]
        j=self.j;s=self.state(rows);a=j.zeros((len(rows),2),j.int32)
        eager=self.t.step(s,a,self.table);compiled=self.jax.jit(self.t.step)(s,a,self.table)
        self.jax.block_until_ready(compiled)
        for x,y in zip(self.jax.tree.leaves(eager),self.jax.tree.leaves(compiled)):
            self.assertTrue(bool(j.array_equal(x,y)))
        split=self.jax.vmap(lambda key:self.jax.random.split(key,2))(s.keys)[:,1]
        draws=self.jax.vmap(lambda k:self.jax.random.randint(k,(2,12),0,j.array([2,100,100,100,16,16,100,4,8,2,2,2])))(split)
        num,den,dn,dd=self.factors(rows)
        damage=self.m.gen3_damage(level=73,power=50,attack=103,defense=101,
            physical=j.array([r['physical'] for r in rows]),critical=draws[:,0,4]==0,
            burned=j.array([r['burned'] for r in rows]),attack_num=num,attack_den=den,defense_num=dn,defense_den=dd,
            roll=85+draws[:,0,5])
        self.assertEqual((10000-eager.state.hp[:,1,0]).tolist(),damage.tolist())
        # Both Soul Dew chains apply to the outgoing target before its switch;
        # Huge Power cannot accidentally amplify this Special Dark move.
        s=s._replace(moves=s.moves.at[:,0,0,0].set(3))
        chance=j.zeros((len(rows),12),j.int32).at[:,4].set(15).at[:,5].set(15)
        out,used,_=self.p.intercept(s,0,j.zeros(len(rows),j.int32),j.full(len(rows),3),chance,self.table,
            eligible=True,contact_roll=j.full(len(rows),59,j.int32))
        special=[dict(r,physical=False) for r in rows];n,d,n2,d2=self.factors(special)
        expected=self.m.gen3_damage(level=73,power=80,attack=103,defense=101,physical=False,
            attack_num=n,attack_den=d,defense_num=n2,defense_den=d2)
        self.assertEqual((10000-out.hp[:,1,0]).tolist(),expected.tolist());self.assertTrue(bool(j.all(used)))

    @unittest.skipUnless(NODE and SHOWDOWN,'requires pinned Showdown')
    def test_native_outgoing_pursuit_uses_both_soul_dew_chains_not_power_ability(self):
        rows=[self.row(pursuit=True),self.row(pursuit=True,item='',defender='Latios',defender_item='Soul Dew'),
            self.row(pursuit=True,defender='Latios',defender_item='Soul Dew',critical=True,attack_stage=-2,defense_stage=2),
            self.row(pursuit=True,attacker='Azumarill',ability='Huge Power',item='Choice Band')]
        result=subprocess.run([NODE,str(Path(__file__).with_name('gpu_stat_modifiers_oracle.cjs')),SHOWDOWN],
            input=json.dumps(rows),text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr);native=json.loads(result.stdout)
        j=self.j;s=self.state(rows);chance=j.zeros((len(rows),12),j.int32).at[:,5].set(15)
        chance=chance.at[:,4].set(j.array([0 if r['critical'] else 15 for r in rows]))
        out,_,_=self.p.intercept(s,0,j.zeros(len(rows),j.int32),j.full(len(rows),3),chance,self.table,
            eligible=True,contact_roll=j.full(len(rows),59,j.int32))
        self.assertEqual((10000-out.hp[:,1,0]).tolist(),[r['damage'] for r in native])

    def test_confusion_power_band_burn_and_invalid_terminal_identity(self):
        j=self.j
        rows=[self.row(attacker='Azumarill',item='Choice Band',ability='Huge Power',physical=True,burned=True,confusion=True)]*16
        s=self.state(rows)._replace(confusion_counter=j.full((16,2,6),2,j.int32).at[:,1,:].set(0))
        a=j.zeros((16,2),j.int32)
        result=self.t.step(s,a,self.table)
        keys=self.jax.vmap(lambda key:self.jax.random.split(key,2))(s.keys)[:,1]
        draws=self.jax.vmap(lambda k:self.jax.random.randint(k,(2,12),0,j.array([2,100,100,100,16,16,100,4,8,2,2,2])))(keys)
        hit=draws[:,0,3]>=50
        damage=self.m.gen3_damage(level=73,power=40,attack=103,defense=101,physical=True,
            typeless=True,burned=True,attack_num=6,attack_den=2,roll=85+draws[:,0,5])
        self.assertTrue(bool(j.any(hit)));self.assertTrue(bool(j.any(~hit)))
        self.assertEqual(result.state.hp[:,0,0].tolist(),(10000-j.where(hit,damage,0)-1250).tolist())
        s=self.jax.tree.map(lambda x:x[:4],s)._replace(done=j.array([False,False,True,False]))
        actions=j.array([[0,0],[0,0],[0,0],[9,0]])
        eager=self.t.step(s,actions,self.table)
        compiled=self.jax.jit(self.t.step)(s,actions,self.table);self.jax.block_until_ready(compiled)
        for old,new,fast in zip(self.jax.tree.leaves(s),self.jax.tree.leaves(eager.state),self.jax.tree.leaves(compiled.state)):
            self.assertTrue(bool(j.array_equal(new,fast)));self.assertTrue(bool(j.array_equal(old[2:],new[2:])))
        prefix=self.t.step(self.jax.tree.map(lambda x:x[:2],s),actions[:2],self.table)
        for full,small in zip(self.jax.tree.leaves(eager),self.jax.tree.leaves(prefix)):
            self.assertTrue(bool(j.array_equal(full[:2],small)))


if __name__=='__main__':unittest.main()
