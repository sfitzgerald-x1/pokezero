// Authored pinned Gen3 party-cure cases, not a game/divergence panel.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
function sample(userSound=false,foeMove='splash',sub=false) {
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  b.setPlayer('p1',{name:'own',team:Array.from({length:6},(_,i)=>({species:'Mew',
    ability:(i===2||i===0&&userSound)?'Soundproof':'Run Away',moves:['healbell'],level:100}))});
  b.setPlayer('p2',{name:'foe',team:[{species:'Mew',ability:'Soundproof',moves:[foeMove],level:100}]});
  const statuses=['brn','slp','tox','par','frz','psn'];
  b.p1.pokemon.forEach((p,i)=>{p.baseMaxhp=p.maxhp=p.hp=1000;p.setStatus(statuses[i]);});
  b.p1.pokemon[5].hp=0;b.p1.pokemon[5].fainted=true;
  const p=b.p1.active[0],q=b.p2.active[0];p.storedStats.spe=300;q.storedStats.spe=100;
  q.setStatus('psn');p.addVolatile('confusion');
  // Confusion remains but does not turn this status-cure fixture into a miss.
  b.randomChance=(n,d)=>(n===1&&d===2)||n>=d;
  if(sub){q.addVolatile('substitute');q.volatiles.substitute.hp=19;}
  b.makeChoices('move 1','move 1');
  const result={status:b.p1.pokemon.map(p=>p.status),foe:q.status,
    confused:!!p.volatiles.confusion,pp:p.moveSlots[0].pp};
  b.destroy();return result;
}
process.stdout.write(JSON.stringify({normal:sample(),user_sound:sample(true),
  protect:sample(false,'protect'),sub:sample(false,'splash',true)}));
