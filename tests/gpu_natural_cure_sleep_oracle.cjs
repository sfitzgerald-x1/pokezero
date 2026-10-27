// Authored source-law controls; no scientific roster or coverage credit.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
function run(side,time) {
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let s=0;s<2;s++) b.setPlayer(`p${s+1}`,{name:`seat${s}`,team:[
    {species:s===side?'Altaria':'Mew',ability:s===side?'Natural Cure':'Synchronize',
      moves:s===side?['rest','splash']:['spore','splash'],level:100},
    {species:'Ditto',ability:'Limber',moves:['splash'],level:100}]});
  const sides=[b.p1,b.p2],p=sides[side].active[0];
  p.baseMaxhp=p.maxhp=1000;p.hp=50;
  const originalRandom=b.random.bind(b);
  b.random=(m,n)=>m===2 && n===6 ? time : originalRandom(m,n);
  const snap=()=>({status:p.status,time:p.statusState.time||0,
    skipped:p.statusState.skippedTime||0,hp:p.hp,pp:p.moveSlots.map(x=>x.pp)});
  const choose=(own,other)=>b.makeChoices(...(side===0?[own,other]:[other,own]));
  choose('move rest','move splash');const afterRest=snap();
  choose('switch 2','move splash');const afterCure=snap();
  choose(`switch ${p.position+1}`,'move spore');const freshNaturalSleep=snap();
  const attempts=[];
  for(let i=0;i<5;i++){choose('move splash','move splash');attempts.push(snap());}
  const out={side,forcedNaturalDuration:time,afterRest,afterCure,freshNaturalSleep,attempts};
  b.destroy();return out;
}
process.stdout.write(JSON.stringify([0,1].flatMap(side=>[2,3,4,5].map(time=>run(side,time)))));
