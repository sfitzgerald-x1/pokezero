// Authored Gen3 Encore timing/override boundaries, not the divergence panel.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
function sample(options={}) {
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let side=0;side<2;side++)b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
    {species:'Mew',ability:'Run Away',moves:side?['tackle','splash','protect','sleeptalk']:['encore','splash'],level:100},
    {species:'Ditto',ability:'Run Away',moves:['splash'],level:100}]});
  const p=b.p1.active[0],q=b.p2.active[0];
  p.hp=p.maxhp=p.baseMaxhp=q.hp=q.maxhp=q.baseMaxhp=10000;
  p.storedStats.spe=options.slow?50:300;q.storedStats.spe=100;
  const rolls=[];
  b.random=(a,z)=>{if(a===3&&z===7){rolls.push([a,z]);return 3+(options.roll||0);}return z===undefined?0:a;};
  b.randomChance=(n,d)=>n>=d;b.randomizer=x=>x;
  if(!options.noPrevious)b.makeChoices('move splash','move tackle');
  if(options.ppZero)q.moveSlots[0].pp=0;
  if(options.last)q.lastMove=b.dex.getActiveMove(options.last);
  if(options.sub){q.addVolatile('substitute');q.volatiles.substitute.hp=19;q.hp=10000;}
  if(options.ghost)q.types=['Ghost'];
  if(options.sleep){q.setStatus('slp');q.statusState.time=5;}
  if(options.fallback){q.addVolatile('encore');q.moveSlots[0].pp=0;
    b.runEvent('DisableMove',q);b.makeRequest('move');}
  const hpBefore=p.hp;
  try {
    b.makeChoices(`move ${options.fallback?'splash':'encore'}`,
      `move ${options.fallback?'struggle':options.protect?'protect':'splash'}`);
  } catch(error) {
    throw new Error(JSON.stringify({message:error.message,errors:[b.p1.choice.error,b.p2.choice.error],
      moves:b.p2.activeRequest?.active?.[0]?.moves,encore:q.volatiles.encore?.move}));
  }
  const snap=()=>({move:q.volatiles.encore?.move||'',duration:q.volatiles.encore?.duration||0,
    last:q.lastMove?.id||'',pp:q.moveSlots.map(x=>x.pp)});
  const initial=snap();
  if(options.repeat)b.makeChoices('move encore','move tackle');
  if(options.switch)b.makeChoices('move splash','switch 2');
  const result={initial,after:snap(),rolls,hpLoss:hpBefore-p.hp,
    called:b.log.some(line=>line.includes('[from] move: Sleep Talk'))};b.destroy();return result;
}
function choiceSleepTalk() {
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  b.setPlayer('p1',{name:'caller',team:[{species:'Mew',ability:'Run Away',item:'Choice Band',
    moves:['sleeptalk','tackle','solarbeam','focuspunch'],level:100}]});
  b.setPlayer('p2',{name:'target',team:[{species:'Mew',ability:'Run Away',moves:['splash'],level:100}]});
  const p=b.p1.active[0],q=b.p2.active[0];q.hp=q.maxhp=q.baseMaxhp=10000;
  p.setStatus('slp');p.statusState.time=5;b.sample=values=>values[0];
  b.makeChoices('move sleeptalk','move splash');const firstHP=q.hp;
  b.makeChoices('move sleeptalk','move splash');
  const result={firstHit:firstHP<10000,secondHit:q.hp<firstHP,pp:p.moveSlots[0].pp,
    lock:p.volatiles.choicelock?.move};b.destroy();return result;
}
const cases={fast:sample(),slow:sample({slow:true}),none:sample({noPrevious:true}),
  zero:sample({ppZero:true}),missing:sample({last:'icebeam'}),excluded:sample({last:'encore'}),
  sub:sample({sub:true}),ghost:sample({ghost:true}),protect:sample({protect:true}),
  repeat:sample({repeat:true}),switch:sample({switch:true}),
  asleep:sample({sleep:true,last:'sleeptalk'}),forced_protect:sample({last:'protect'}),
  fallback:sample({fallback:true}),
  choice:choiceSleepTalk(),
  durations:[0,1,2,3].map(roll=>sample({roll}).initial.duration)};
process.stdout.write(JSON.stringify(cases));
