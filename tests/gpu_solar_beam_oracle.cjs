// Pinned Gen3 Solar Beam boundaries, not a seeded strength or parity panel.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
function copiedCharge() {
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let side=0;side<2;side++)b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
    {species:'Mew',ability:'Run Away',moves:['solarbeam','splash']},
    {species:'Ditto',ability:'Run Away',moves:['splash']}]});
  const p=b.p1.active[0],q=b.p2.active[0],next=b.p1.pokemon[1];
  // Authored primitive boundary, NOT proof of a naturally reachable sequence.
  p.addVolatile('twoturnmove',q,b.dex.moves.get('solarbeam'));
  p.volatiles.twoturnmove.duration=1;
  next.copyVolatileFrom(p,'copyvolatile');
  // Event handlers inspect only active holders; install this authored recipient.
  p.isActive=false;next.isActive=true;next.position=0;b.p1.active[0]=next;
  const locked=next.getLockedMove();
  const result={charge:!!next.volatiles.solarbeam,duration:next.volatiles.twoturnmove?.duration,
    locked,request:next.getMoves(locked).map(x=>x.id),known:next.moves,oldCharge:!!p.volatiles.solarbeam};
  b.destroy();return result;
}
function sample(options={}) {
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let side=0;side<2;side++)b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
    {species:'Mew',ability:side?(options.cloud?'Cloud Nine':options.pressure?'Pressure':'Run Away'):'Run Away',
      moves:side?['splash','protect','encore']:['solarbeam','splash'],level:100},
    {species:'Ditto',ability:'Run Away',moves:['splash']}]});
  const p=b.p1.active[0],q=b.p2.active[0];
  for(const mon of [p,q]){mon.baseMaxhp=mon.maxhp=mon.hp=1000;mon.types=['Psychic'];
    for(const stat of Object.keys(mon.storedStats))mon.storedStats[stat]=200;}
  p.storedStats.spe=300;q.storedStats.spe=100;
  b.randomChance=(n,d)=>n/d>=.8;b.randomizer=x=>x;
  if(options.weather){b.field.setWeather(options.weather,p);if(options.expire)b.field.weatherState.duration=1;}
  if(options.onePP)p.moveSlots[0].pp=1;
  b.makeChoices('move solarbeam',options.protect?'move protect':'move splash');
  const first={hp:q.hp,pp:p.moveSlots[0].pp,charge:!!p.volatiles.solarbeam,
    locked:p.getLockedMove()||'',trapped:!!b.p1.activeRequest.active[0].trapped};
  if(options.sleep){p.setStatus('slp');p.statusState.time=5;}
  if(options.paralysis){p.setStatus('par');b.randomChance=(n,d)=>n/d>=.8||n/d===.25;}
  if(options.protect)q.removeVolatile('stall');
  b.makeChoices('move solarbeam',options.protect?'move protect':'move splash');
  const second={hp:q.hp,pp:p.moveSlots[0].pp,charge:!!p.volatiles.solarbeam,
    locked:p.getLockedMove()||'',weather:b.field.weather};
  b.destroy();return {first,second};
}
process.stdout.write(JSON.stringify({clear:sample(),sun:sample({weather:'sunnyday'}),
  rain:sample({weather:'raindance'}),sand:sample({weather:'sandstorm'}),hail:sample({weather:'hail'}),
  cloud:sample({weather:'sunnyday',cloud:true}),expire:sample({weather:'sunnyday',expire:true}),
  sleep:sample({sleep:true}),paralysis:sample({paralysis:true}),onePP:sample({onePP:true}),
  protect:sample({protect:true}),pressure:sample({pressure:true}),copied:copiedCharge()}));
