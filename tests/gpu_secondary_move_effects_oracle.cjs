// Authored pinned Gen3 secondary boundaries; no scientific roster replay.
const fs=require('node:fs'),path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
const {Dex}=require(path.join(process.argv[2],'dist/sim/dex.js'));
const specs=JSON.parse(fs.readFileSync(0,'utf8'));
const names=['silverwind','ancientpower','mudshot','sacredfire'];
const dex=Dex.mod('gen3');
const catalog=names.map(id=>({id,secondaries:dex.moves.get(id).secondaries}));
const cases=specs.map(spec=>{
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let side=0;side<2;side++)b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
    {species:'Mew',ability:(side===spec.side?spec.source:spec.target)||'Run Away',moves:['splash'],level:100},
    {species:'Ditto',ability:'Run Away',moves:['splash'],level:100}]});
  const own=b.sides[spec.side].active[0],foe=b.sides[1-spec.side].active[0];
  for(const side of b.sides)for(const p of side.pokemon){
    p.baseMaxhp=p.maxhp=p.hp=1000;for(const k of Object.keys(p.storedStats))p.storedStats[k]=200;
    p.types=['Normal'];p.updateSpeed();
  }
  foe.types=[spec.type||'Normal'];
  if(spec.stage!==undefined)for(const k of ['atk','def','spa','spd','spe'])own.boosts[k]=spec.stage;
  if(spec.targetStage!==undefined)foe.boosts.spe=spec.targetStage;
  if(spec.sub){foe.addVolatile('substitute');foe.volatiles.substitute.hp=spec.sub;}
  if(spec.ko||spec.endure)foe.hp=1;
  if(spec.endure)foe.addVolatile('endure');
  if(spec.protect)foe.addVolatile('protect');
  if(spec.status)foe.setStatus(spec.status);
  const roll=spec.roll||0;
  b.randomChance=(n,d)=>d===100?roll<n:n>=d;
  b.random=(n,d)=>d===undefined&&n===100?roll:0;b.randomizer=x=>x;
  const move=b.dex.getActiveMove(spec.move);move.accuracy=spec.miss?0:true;move.willCrit=false;
  b.actions.useMove(move,own,foe);
  const result={ownBoosts:{...own.boosts},foeBoosts:{...foe.boosts},foeHP:foe.hp,
    foeStatus:foe.status,subHP:foe.volatiles.substitute?.hp||0};
  b.destroy();return result;
});
process.stdout.write(JSON.stringify({catalog,cases}));
