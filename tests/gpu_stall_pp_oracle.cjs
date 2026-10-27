// Authored pinned-source PP controls, not random-battle collection or a replay.
const path=require('node:path');const {execFileSync}=require('node:child_process');
const root=path.resolve(process.argv[2]);
const commit=execFileSync('git',['-C',root,'rev-parse','HEAD'],{encoding:'utf8'}).trim();
const compiled=require('../gpu_engine/gender_policy.cjs').compiledTree(path.join(root,'dist'));
if(commit!=='f76228a1354b5d0f307ca2d16101294ad3a2308b'||compiled.files!==500||compiled.sha256!=='d22c0cc51aa32b983c2946108834418e824b0056a727e121e19dc56919350a01')throw Error('pinned simulator source changed');
const {Battle}=require(path.join(root,'dist/sim/battle.js'));
const rows=[];
for(const side of [0,1])for(const move of ['protect','detect','endure'])for(const failed of [false,true])for(const foe of ['splash','switch']){
 const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
 for(let s=0;s<2;s++)b.setPlayer(`p${s+1}`,{name:`seat${s}`,team:[
  {species:'Mew',ability:'Run Away',moves:s===side?[move,'sleeptalk']:['splash'],level:100},
  {species:'Ditto',ability:'Run Away',moves:['splash'],level:100}]});
 const p=b.sides[side].active[0];p.storedStats.spe=300;b.sides[1-side].active[0].storedStats.spe=100;const actions=['move splash','move splash'];actions[side]=`move ${move}`;
 b.makeChoices(...actions);const before=p.moveSlots[0].pp;
 const rolls=[];b.randomChance=(n,d)=>{rolls.push([n,d]);return !failed;};
 actions[1-side]=foe==='switch'?'switch 2':'move splash';b.makeChoices(...actions);
 rows.push({side,move,failed,foe,pp_before:before,pp_after:p.moveSlots[0].pp,stall:p.volatiles.stall?.counter||0,rolls});b.destroy();
}
for(const side of [0,1])for(const failed of [false,true]){
 const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
 for(let s=0;s<2;s++)b.setPlayer(`p${s+1}`,{name:`seat${s}`,team:[{species:'Mew',ability:'Run Away',moves:s===side?['protect','sleeptalk']:['splash'],level:100}]});
 const p=b.sides[side].active[0];p.storedStats.spe=300;b.sides[1-side].active[0].storedStats.spe=100;const actions=['move splash','move splash'];actions[side]='move protect';b.makeChoices(...actions);
 p.setStatus('slp');p.statusState.time=10;p.statusState.startTime=10;
 b.sample=xs=>xs.find(x=>(x.move||x)==='protect');b.randomChance=()=>!failed;
 const before=p.moveSlots.map(m=>m.pp);actions[side]='move sleeptalk';b.makeChoices(...actions);
 rows.push({side,called_protect:true,failed,pp_before:before,pp_after:p.moveSlots.map(m=>m.pp),stall:p.volatiles.stall?.counter||0});b.destroy();
}
if(rows.some(r=>r.called_protect?r.pp_after[0]!==r.pp_before[0]||r.pp_after[1]!==r.pp_before[1]-1:r.pp_after!==r.pp_before-1))throw Error('PP law differs');
console.log(JSON.stringify({source_commit:commit,compiled_tree:compiled,cases:rows.length,rows}));
