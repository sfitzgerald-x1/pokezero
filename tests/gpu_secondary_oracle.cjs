// Selected Gen 3 source events on authored states, not the 200-game panel.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
const {Dex}=require(path.join(process.argv[2],'dist/sim/dex.js'));
const names=['rockslide','icebeam','icepunch','thunderbolt','thunderpunch','thunder',
  'bodyslam','sludgebomb','fireblast','flamethrower','firepunch','shadowball','crunch','psychic','sacredfire'];
const dex=Dex.mod('gen3');
const catalog=names.map(id=>{
  const s=dex.moves.get(id).secondaries[0];
  return {id,chance:s.chance,status:s.status||'',spd:!!s.boosts?.spd,flinch:s.volatileStatus==='flinch'};
});
function sample(name,options={}) {
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let i=0;i<2;i++) b.setPlayer(`p${i+1}`,{name:`seat${i}`,team:[
    {species:'Mew',ability:(i?options.target:options.source)||'Run Away',
      moves:['splash'],level:100,item:i?(options.item||''):''},
    {species:'Ditto',ability:'Run Away',moves:['splash'],level:100}]});
  const p=b.p1.active[0],q=b.p2.active[0];
  p.baseMaxhp=p.maxhp=p.hp=q.baseMaxhp=q.maxhp=q.hp=1000;
  q.types=[options.type||'Water'];
  if(options.status) q.setStatus(options.status);
  if(options.sub) {q.addVolatile('substitute');q.volatiles.substitute.hp=1;q.hp=1000;}
  if(options.sun) b.field.setWeather('sunnyday',p,b.dex.moves.get('sunnyday'));
  b.random=(a,z)=>z===undefined && a===100?(options.roll||0):0;
  b.randomizer=x=>x;
  const move=b.dex.getActiveMove(name);move.accuracy=true;move.willCrit=false;
  b.actions.useMove(move,p,q);
  const result={status:q.status,spd:q.boosts.spd,flinch:!!q.volatiles.flinch,
    substitute:!!q.volatiles.substitute,item:q.item};
  b.destroy();return result;
}
const cases={
  flinch:sample('rockslide'),focus:sample('rockslide',{target:'Inner Focus'}),
  dust_status:sample('icebeam',{target:'Shield Dust'}),dust_drop:sample('crunch',{target:'Shield Dust'}),
  clear:sample('crunch',{target:'Clear Body'}),smoke:sample('shadowball',{target:'White Smoke'}),
  limber:sample('bodyslam',{target:'Limber'}),veil:sample('firepunch',{target:'Water Veil'}),
  magma:sample('icebeam',{target:'Magma Armor'}),immunity:sample('sludgebomb',{target:'Immunity'}),
  broken_sub:sample('thunderbolt',{sub:true}),sun:sample('icebeam',{sun:true}),
  suppressed_sun:sample('icebeam',{sun:true,target:'Cloud Nine'}),
  grace:sample('bodyslam',{source:'Serene Grace',roll:55}),ordinary:sample('bodyslam',{roll:55}),
  thaw_burn:sample('firepunch',{status:'frz'}),electric:sample('bodyslam',{type:'Electric'}),
  lum:sample('bodyslam',{item:'Lum Berry'}),ice:sample('icebeam',{type:'Ice'}),
};
process.stdout.write(JSON.stringify({catalog,cases}));
