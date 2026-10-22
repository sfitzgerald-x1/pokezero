// Pinned Gen3 authored charge/hit/PP boundaries, not a game panel.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
function sample(options={}) {
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let side=0;side<2;side++)b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
    {species:'Mew',ability:'Run Away',item:side?'':(options.band?'Choice Band':''),
      moves:side?[options.foe||'splash']:['focuspunch','tackle'],level:100},
    {species:'Ditto',ability:'Run Away',moves:['splash']}]});
  const p=b.p1.active[0],q=b.p2.active[0];p.baseMaxhp=p.maxhp=p.hp=1000;q.baseMaxhp=q.maxhp=q.hp=1000;
  p.storedStats.spe=300;q.storedStats.spe=100;p.types=['Psychic'];q.types=[options.ghost?'Ghost':'Water'];
  if(options.sub){p.addVolatile('substitute');p.volatiles.substitute.hp=options.sub;}
  if(options.sleep){p.setStatus('slp');p.statusState.time=5;}
  if(options.miss)q.boosts.accuracy=-6;
  if(options.immune)p.types=['Ghost'];
  if(options.endure){p.hp=1;p.addVolatile('endure');}
  p.lastMove=b.dex.moves.get('tackle');
  // Hits at ordinary accuracy, misses at -6; reject critical/secondary rolls.
  b.randomChance=(n,d)=>n/d>=.8;b.randomizer=x=>x;
  b.makeChoices('move 1',options.switch?'switch 2':'move 1');
  const result={hp:[p.hp,q.hp],sub:p.volatiles.substitute?.hp||0,pp:p.moveSlots[0].pp,
    last:p.lastMove?.id||'',lock:p.volatiles.choicelock?.move||'',
    focused:b.log.some(s=>s.includes('|-singleturn|')&&s.includes('Focus Punch')),
    failed:b.log.some(s=>s.includes('|cant|')&&s.includes('Focus Punch')),
    active:b.p2.active[0].species.id,activeHP:b.p2.active[0].hp,
    tacklePP:p.moveSlots[1].pp,foePP:q.moveSlots[0].pp};
  b.destroy();return result;
}
process.stdout.write(JSON.stringify({normal:sample(),hit:sample({foe:'tackle'}),
  miss:sample({foe:'tackle',miss:true}),immune:sample({foe:'tackle',immune:true}),
  sub:sample({foe:'tackle',sub:200}),broken_sub:sample({foe:'tackle',sub:1}),
  status:sample({foe:'toxic'}),protect:sample({foe:'protect'}),ghost:sample({ghost:true}),
  sleep:sample({sleep:true}),hit_band:sample({foe:'tackle',band:true}),switch:sample({switch:true}),
  flinch:sample({foe:'fakeout'}),encore:sample({foe:'encore'}),
  zero_damage:sample({foe:'tackle',endure:true}),fixed_damage:sample({foe:'seismictoss'})}));
