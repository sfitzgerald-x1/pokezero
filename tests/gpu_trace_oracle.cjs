// Authored handler controls against the pinned Gen3 source; no scored roster.
const path=require('node:path');
const root=process.argv[2];
const {Battle}=require(path.join(root,'dist/sim/battle.js'));
const {Dex}=require(path.join(root,'dist/sim/dex.js'));
const dex=Dex.mod('gen3');
function fixture(foe='Static') {
  const b=new Battle({formatid:'gen3customgame',seed:[11,22,33,44]});
  const set=(species,ability)=>({species,ability,moves:['splash'],level:100});
  b.setPlayer('p1',{team:[set('Porygon2','Trace')]});
  b.setPlayer('p2',{team:[set('Ditto',foe)]});
  const p=b.p1.active[0],q=b.p2.active[0];
  p.ability=p.baseAbility='trace';q.ability=q.baseAbility=dex.abilities.get(foe).id;
  return {b,p,q};
}
function start(b,p) {b.singleEvent('Start',dex.abilities.get('trace'),p.abilityState,p);}
const copies=[];
for(const foe of ['Static','Intimidate','Drizzle','Drought','Sand Stream','Truant','Trace']) {
  const {b,p}=fixture(foe);let starts=0;
  const original=b.singleEvent.bind(b);
  b.singleEvent=function(id,effect,...args) {
    if(id==='Start' && effect.id!=='trace') starts++;
    return original(id,effect,...args);
  };
  start(b,p);
  copies.push({foe:dex.abilities.get(foe).id,current:p.ability,original:p.baseAbility,copied_start_calls:starts});
  b.destroy();
}
const cures=[];
for(const [foe,status] of [['Water Veil','brn'],['Magma Armor','frz'],['Limber','par'],
    ['Immunity','psn'],['Immunity','tox'],['Insomnia','slp'],['Vital Spirit','slp']]) {
  const {b,p}=fixture(foe);p.status=status;p.statusState={id:status,target:p,time:3};
  start(b,p);b.eachEvent('Update');cures.push({foe,status:p.status});b.destroy();
}
const own=fixture('Own Tempo');
own.p.volatiles.confusion={id:'confusion',target:own.p,duration:3};
start(own.b,own.p);own.b.eachEvent('Update');
const own_tempo_confusion=!!own.p.volatiles.confusion;own.b.destroy();
const failed=fixture();failed.q.fainted=true;failed.q.hp=0;
start(failed.b,failed.p);const no_retry=[failed.p.ability];
failed.q.fainted=false;failed.q.hp=failed.q.maxhp;
failed.b.eachEvent('Update');no_retry.push(failed.p.ability);failed.b.destroy();
const nc=fixture('Natural Cure');nc.p.status='psn';nc.p.statusState={id:'psn',target:nc.p};
start(nc.b,nc.p);nc.b.eachEvent('Update');const natural_cure=[nc.p.status];
nc.b.singleEvent('SwitchOut',dex.abilities.get('naturalcure'),nc.p.abilityState,nc.p);
natural_cure.push(nc.p.status);nc.p.clearVolatile();natural_cure.push(nc.p.ability);nc.b.destroy();
process.stdout.write(JSON.stringify({copies,cures,own_tempo_confusion,no_retry,natural_cure}));
