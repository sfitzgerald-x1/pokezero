// Authored Rest/refund source controls; no random-battle root collection.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
const rows=[];
for(const side of [0,1])for(const variant of ['consecutive','discard','early','naturalcure']){
 const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
 for(let s=0;s<2;s++)b.setPlayer(`p${s+1}`,{name:`seat${s}`,team:[
  {species:'Mew',ability:s===side?(variant==='early'?'Early Bird':variant==='naturalcure'?'Natural Cure':'Run Away'):'Run Away',
   moves:s===side?['rest','sleeptalk','splash','solarbeam']:['splash'],level:100},
  {species:'Ditto',ability:'Run Away',moves:['splash'],level:100}]});
 const player=b.sides[side],p=player.active[0];p.baseMaxhp=p.maxhp=101;p.hp=50;
 const snapshots=[],calledPools=[];
 const snap=()=>snapshots.push({status:p.status,time:p.statusState.time||0,skipped:p.statusState.skippedTime||0,hp:p.hp});
 const play=action=>{const req=['move splash','move splash'];req[side]=action;b.makeChoices(...req);snap();};
 b.sample=values=>{calledPools.push(values.map(x=>x.move||x));return values.find(x=>(x.move||x)==='splash')||values[0];};
 play('move rest');play('move sleeptalk');
 if(variant==='consecutive'||variant==='discard')play(variant==='discard'?'move splash':'move sleeptalk');
 play('switch 2');play(`switch ${p.position+1}`);play('move splash');
 rows.push({side,variant,snapshots,calledPools});b.destroy();
}
process.stdout.write(JSON.stringify(rows));
