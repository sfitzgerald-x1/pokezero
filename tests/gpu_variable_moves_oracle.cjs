// Canonical battle move, not Dex alias metadata; authored source fixtures only.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
function create() {
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let i=0;i<2;i++)b.setPlayer(`p${i+1}`,{name:`seat${i}`,team:[
    {species:'Mew',ability:i?'Flash Fire':'Limber',moves:['hiddenpowerfire','return','frustration'],level:73},
    {species:'Ditto',ability:'Limber',moves:['splash'],level:73}]});
  for(const side of b.sides){const p=side.active[0];p.baseMaxhp=p.maxhp=p.hp=1000;}
  return b;
}
function hit(frozen) {
  const b=create(),p=b.p1.active[0],q=b.p2.active[0];
  p.hpType='Fire';p.hpPower=33;if(frozen)q.setStatus('frz');
  b.actions.useMove('hiddenpower',p,q);
  const result={damage:q.hp<1000,status:q.status,flash:!!q.volatiles.flashfire};b.destroy();return result;
}
const b=create(),p=b.p1.active[0];
const defaults=[p.happiness,b.dex.moves.get('return').basePowerCallback(p),b.dex.moves.get('frustration').basePowerCallback(p)];
p.happiness=0;const endpoints=[b.dex.moves.get('return').basePowerCallback(p),b.dex.moves.get('frustration').basePowerCallback(p)];
const hidden=[];
for(const [type,power] of [['Fire',33],['Fighting',40]]){
  p.hpType=type;p.hpPower=power;
  const move=b.dex.getActiveMove('hiddenpower');move.onModifyMove.call(b,move,p);
  const dynamic=b.dex.moves.get('hiddenpower').basePowerCallback.call(b,p);
  const target=b.p2.active[0];target.setAbility('Limber');
  target.hp=1000;b.actions.useMove('hiddenpower',p,target);const actual=1000-target.hp;
  const reference=create(),r=reference.p1.active[0],t=reference.p2.active[0];
  t.setAbility('Limber');
  // Force the same damage roll; the check concerns resolved type/power, not PRNG parity.
  b.randomizer=x=>x;reference.randomizer=x=>x;
  target.hp=1000;const controlledMove=b.dex.getActiveMove('hiddenpower');controlledMove.willCrit=false;
  b.actions.useMove(controlledMove,p,target);const controlled=1000-target.hp;
  const staticMove=reference.dex.getActiveMove(type==='Fire'?'flamethrower':'brickbreak');
  staticMove.basePower=power;staticMove.willCrit=false;staticMove.secondary=null;staticMove.secondaries=null;
  reference.actions.useMove(staticMove,r,t);
  hidden.push([move.type,dynamic,controlled===1000-t.hp && actual>0]);reference.destroy();
}
// Exhaust the HP lattice at small, odd and realistic maximum HP values. This
// calls the pinned Gen3 callback; it does not duplicate its six-band formula.
const hpBands=[];
for(const name of ['reversal','flail'])for(const max of [1,47,48,321,714]){
  for(let hp=1;hp<=max;hp++){
    p.hp=hp;p.maxhp=max;
    hpBands.push([name,hp,max,b.dex.moves.get(name).basePowerCallback.call(b,p)]);
  }
}
b.destroy();process.stdout.write(JSON.stringify({defaults,endpoints,hidden,hpBands,frozen:hit(true),awake:hit(false)}));
