// Authored pinned Gen 3 boundaries, not the 200-game divergence panel.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
function sample(options={}) {
  const name=options.move||'explosion';
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let side=0;side<2;side++) b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
    {species:'Mew',ability:(side?options.foeAbility:options.ability)||'Run Away',
      moves:side?[options.foeMove||'splash']:(options.talk?['sleeptalk',name,'solarbeam','focuspunch']:[name]),level:100},
    ...(options.last?[]:[{species:'Ditto',moves:['splash'],ability:'Run Away'}]),
  ]});
  const p=b.p1.active[0],q=b.p2.active[0];
  p.baseMaxhp=p.maxhp=p.hp=1000;q.baseMaxhp=q.maxhp=q.hp=options.targetHP||1000;
  p.storedStats.atk=200;p.storedStats.spe=300;q.storedStats.def=201;q.storedStats.spe=100;
  p.types=['Psychic'];q.types=[options.ghost?'Ghost':'Water'];
  if(options.miss)p.boosts.accuracy=-6;
  if(options.talk){p.setStatus('slp');p.statusState.time=5;b.sample=values=>values[0];}
  if(options.sub){q.addVolatile('substitute');q.volatiles.substitute.hp=19;q.hp=1000;}
  // Preserve accuracy/Protect threshold logic, reject crits, choose maximum roll.
  b.randomChance=(numerator,denominator)=>numerator>=denominator;
  b.randomizer=value=>value;
  b.makeChoices('move 1','move 1');
  const result={hp:[p.hp,q.hp],pp:[p.moveSlots[0].pp,q.moveSlots[0].pp],
    sub:q.volatiles.substitute?.hp||0,ended:b.ended,winner:b.winner||'',
    request:b.requestState,calledPP:p.moveSlots[1]?.pp};
  b.destroy();return result;
}
const cases={normal:sample(),selfdestruct:sample({move:'selfdestruct'}),
  miss:sample({miss:true}),protect:sample({foeMove:'protect'}),ghost:sample({ghost:true}),
  damp:sample({foeAbility:'Damp'}),self_damp:sample({ability:'Damp'}),
  last_ko:sample({last:true,targetHP:1}),last_survive:sample({last:true}),
  endure:sample({foeMove:'endure',targetHP:1}),sub:sample({sub:true}),
  foe_attack:sample({foeMove:'tackle'}),pressure:sample({foeAbility:'Pressure'}),
  talk:sample({talk:true}),talk_pressure:sample({talk:true,foeAbility:'Pressure'}),
  talk_damp:sample({talk:true,foeAbility:'Damp'})};
function damageGrid() {
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let side=0;side<2;side++)b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
    {species:'Mew',moves:['splash'],ability:'Run Away',level:100}]});
  const p=b.p1.active[0],q=b.p2.active[0];p.types=['Psychic'];q.types=['Water'];
  p.storedStats.atk=200;q.storedStats.def=201;
  const rows=[];
  for(const [atk,def,critical,burn]of [[0,0,false,false],[-2,2,true,false],[2,-1,false,true]]) {
    p.boosts.atk=atk;q.boosts.def=def;p.status=burn?'brn':'';
    for(const name of ['explosion','selfdestruct','tackle']) {
      const values=[];
      for(let roll=85;roll<=100;roll++) {
        b.randomizer=value=>Math.floor(value*roll/100);
        const move=b.dex.getActiveMove(name);move.willCrit=critical;
        values.push(b.actions.getDamage(p,q,move));
      }
      rows.push({name,atk,def,critical,burn,values});
    }
  }
  b.destroy();return rows;
}
process.stdout.write(JSON.stringify({cases,damage:damageGrid()}));
