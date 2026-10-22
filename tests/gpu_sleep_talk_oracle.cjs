// Authored Gen3 Sleep Talk PP, selection and sleep-clock fixtures.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
function create({time=5,awake=false,pp=7,ability='Run Away',targetAbility='Run Away',item='',
  moves=['sleeptalk','tackle','solarbeam','focuspunch']}={}) {
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let i=0;i<2;i++)b.setPlayer(`p${i+1}`,{name:`seat${i}`,team:[
    {species:'Mew',ability:i?targetAbility:ability,item:i?'':item,
      moves:i?['splash']:moves,level:100},
    {species:'Ditto',ability:'Run Away',moves:['splash'],level:100}]});
  const p=b.p1.active[0],q=b.p2.active[0];
  p.baseMaxhp=p.maxhp=p.hp=q.baseMaxhp=q.maxhp=q.hp=1000;
  if(!awake){p.setStatus('slp');p.statusState.time=time;}
  p.moveSlots[1].pp=pp;
  b.randomizer=x=>x;
  const choices=[];
  b.sample=values=>{choices.push(values.map(x=>x.move||x));return values[0];};
  return {b,p,q,choices};
}
function snapshot(f) {
  return {status:f.p.status,time:f.p.statusState.time||0,skipped:f.p.statusState.skippedTime||0,
    pp:f.p.moveSlots.map(m=>m.pp),damaged:f.q.hp<1000,choices:f.choices,
    nopp:f.b.log.some(line=>line.includes('|cant|')&&line.includes('|nopp|')),
    lock:f.p.volatiles.choicelock?.move||''};
}
function once(options={}) {
  const f=create(options);f.b.makeChoices('move sleeptalk','move splash');
  const out=snapshot(f);f.b.destroy();return out;
}
function switching(reset=false) {
  const f=create();f.b.makeChoices('move sleeptalk','move splash');
  if(reset)f.b.makeChoices('move tackle','move splash');
  else f.b.makeChoices('move sleeptalk','move splash');
  const before=snapshot(f);
  f.b.makeChoices('switch 2','move splash');
  f.b.makeChoices(`switch ${f.p.position+1}`,'move splash');
  const after=snapshot(f);f.b.destroy();return {before,after};
}
function hazard() {
  const f=create();f.b.makeChoices('move sleeptalk','move splash');
  f.b.makeChoices('switch 2','move splash');f.p.hp=1;
  f.b.p1.addSideCondition('spikes',f.q,f.b.dex.moves.get('spikes'));
  f.b.makeChoices(`switch ${f.p.position+1}`,'move splash');
  const out={...snapshot(f),hp:f.p.hp};f.b.destroy();return out;
}
process.stdout.write(JSON.stringify({normal:once(),zero:once({pp:0}),wake:once({time:1}),
  awake:once({awake:true}),early:once({ability:'Early Bird'}),
  pressure:once({targetAbility:'Pressure'}),choice:once({item:'Choice Band'}),
  no_eligible:once({moves:['sleeptalk','solarbeam','focuspunch','fly']}),
  switch_back:switching(),reset_switch:switching(true),hazard:hazard()}));
