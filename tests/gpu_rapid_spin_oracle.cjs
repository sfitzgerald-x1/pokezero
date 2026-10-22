// Authored pinned Gen3 hit/cleanup boundaries, not a game panel.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
function sample(options={}) {
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let side=0;side<2;side++)b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
    {species:'Mew',ability:'Run Away',moves:side?[options.foe||'splash']:['rapidspin'],level:100},
    {species:'Ditto',ability:'Run Away',moves:['splash']}]});
  const p=b.p1.active[0],q=b.p2.active[0];
  p.baseMaxhp=p.maxhp=p.hp=1000;q.baseMaxhp=q.maxhp=q.hp=options.one?1:1000;
  p.storedStats.spe=300;q.storedStats.spe=100;
  p.addVolatile('leechseed',q);p.side.addSideCondition('spikes',q,b.dex.moves.get('spikes'));
  p.side.addSideCondition('spikes',q,b.dex.moves.get('spikes'));
  p.addVolatile('partiallytrapped',q,b.dex.moves.get('bind'));
  p.addVolatile('trapped',q,b.dex.moves.get('meanlook'));
  if(options.ghost)q.types=['Ghost'];
  if(options.sub){q.addVolatile('substitute');q.volatiles.substitute.hp=19;}
  if(options.miss)p.boosts.accuracy=-6;
  b.randomChance=(n,d)=>n>=d;b.randomizer=x=>x;
  b.makeChoices('move 1','move 1');
  const r={seed:!!p.volatiles.leechseed,spikes:p.side.sideConditions.spikes?.layers||0,
    partial:!!p.volatiles.partiallytrapped,trapped:!!p.volatiles.trapped,
    speed:p.boosts.spe,pp:p.moveSlots[0].pp,targetHP:q.hp,sub:q.volatiles.substitute?.hp||0};
  b.destroy();return r;
}
process.stdout.write(JSON.stringify({normal:sample(),miss:sample({miss:true}),ghost:sample({ghost:true}),
  protect:sample({foe:'protect'}),sub:sample({sub:true}),ko:sample({one:true}),
  endure:sample({one:true,foe:'endure'})}));
