// Authored pinned Gen 3 fixtures; not the bounded game divergence panel.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
function sample(move,weather,ability='Run Away',maxhp=101,hp=1) {
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let side=0;side<2;side++) b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
    {species:'Mew',ability:side?ability:'Run Away',moves:side?['splash']:[move],level:100}]});
  const p=b.p1.active[0],q=b.p2.active[0];
  p.baseMaxhp=p.maxhp=maxhp;p.hp=hp;p.types=['Rock','Ice'];
  p.storedStats.spe=300;q.storedStats.spe=100;
  if(weather)b.field.setWeather(weather,p,{effectType:'Move'});
  b.makeChoices('move 1','move 1');
  const result={move,weather,ability,maxhp,start:hp,hp:p.hp,pp:p.moveSlots[0].pp};
  b.destroy();return result;
}
const cases=[];
for(const move of ['synthesis','moonlight','morningsun'])
  for(const weather of ['','sunnyday','raindance','sandstorm','hail'])cases.push(sample(move,weather));
for(const ability of ['Air Lock','Cloud Nine'])
  for(const weather of ['sunnyday','raindance'])cases.push(sample('synthesis',weather,ability));
cases.push(sample('synthesis','','Run Away',101,101),sample('synthesis','sunnyday','Run Away',2,1));
process.stdout.write(JSON.stringify(cases));
