// Authored pinned Gen 3 event fixtures, not a full-game fidelity panel.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
const {Dex}=require(path.join(process.argv[2],'dist/sim/dex.js'));
const dex=Dex.mod('gen3');
function create() {
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let side=0;side<2;side++) b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
    {species:'Mew',ability:'Limber',moves:['splash','leechseed','wish'],level:100},
    {species:'Ditto',ability:'Limber',moves:['splash'],level:100},
  ]});
  for(const side of b.sides) for(const p of side.pokemon) {
    p.baseMaxhp=p.maxhp=101;p.hp=80;
    p.storedStats.spe=200;p.updateSpeed();
  }
  return b;
}
function seed(b) {
  const move=b.dex.getActiveMove('leechseed');move.accuracy=true;
  b.actions.useMove(move,b.p1.active[0],b.p2.active[0]);
}
function onlyActives(b) {
  for(const side of b.sides){side.pokemon[1].hp=0;side.pokemon[1].fainted=true;side.pokemonLeft=1;}
}
function seedCases() {
  let b=create();b.p1.active[0].hp=101;seed(b);b.fieldEvent('Residual');
  const full=[b.p1.active[0].hp,b.p2.active[0].hp];b.destroy();
  b=create();seed(b);b.p1.active[0].faint();b.faintMessages();b.fieldEvent('Residual');
  const dead=b.p2.active[0].hp;b.destroy();
  b=create();b.p1.active[0].hp=50;b.p1.pokemon[1].hp=10;seed(b);
  b.makeChoices('switch 2','move 1');
  const switched=[b.p1.pokemon.find(p=>p.species.id==='mew').hp,b.p1.active[0].hp,b.p2.active[0].hp];b.destroy();
  b=create();onlyActives(b);b.p1.active[0].hp=50;b.p1.active[0].setStatus('psn');b.p2.active[0].hp=3;
  seed(b);b.fieldEvent('Residual');
  const ko=[b.p1.active[0].hp,b.p2.active[0].hp,b.ended];b.destroy();
  const blocked={};
  for(const kind of ['grass','substitute','protect','miss']) {
    b=create();const target=b.p2.active[0];
    if(kind==='grass') target.types=['Grass'];
    if(kind==='substitute'||kind==='protect') target.addVolatile(kind);
    if(kind==='miss') {const m=b.dex.getActiveMove('leechseed');m.accuracy=0;b.actions.useMove(m,b.p1.active[0],target);}
    else seed(b);
    blocked[kind]=!!target.volatiles.leechseed;b.destroy();
  }
  const ordered=[];
  for(const speed of [200,300]) {
    b=create();const left=b.p1.active[0],right=b.p2.active[0];
    left.hp=1;left.setItem('leftovers');left.setStatus('psn');right.hp=50;
    left.storedStats.spe=speed;left.updateSpeed();seed(b);b.fieldEvent('Residual');
    ordered.push([left.hp,right.hp]);b.destroy();
  }
  return {full,dead,switched,ko,blocked,ordered};
}
function wishCases() {
  let b=create(),p=b.p1.active[0];p.hp=10;
  b.actions.useMove('wish',p);const initial=b.p1.slotConditions[0].wish.duration;
  b.fieldEvent('Residual');const first=[p.hp,b.p1.slotConditions[0].wish.duration];
  b.actions.useMove('wish',p);const retry=b.p1.slotConditions[0].wish.duration;
  b.fieldEvent('Residual');const second=[p.hp,!!b.p1.slotConditions[0].wish];b.destroy();
  b=create();p=b.p1.active[0];p.baseMaxhp=p.maxhp=301;p.hp=50;b.p1.pokemon[1].hp=10;
  b.actions.useMove('wish',p);b.fieldEvent('Residual');b.makeChoices('switch 2','move 1');
  const switched=b.p1.active[0].hp;b.destroy();
  b=create();p=b.p1.active[0];b.actions.useMove('wish',p);b.fieldEvent('Residual');
  p.faint();b.faintMessages();b.fieldEvent('Residual');
  const dead=[p.hp,!!b.p1.slotConditions[0].wish];b.destroy();
  b=create();p=b.p1.active[0];p.hp=1;b.actions.useMove('wish',p);b.fieldEvent('Residual');
  onlyActives(b);b.field.setWeather('sandstorm',p);b.fieldEvent('Residual');
  const beforeWeather=[p.hp,b.ended];b.destroy();
  return {initial,first,retry,second,switched,dead,before_weather:beforeWeather};
}
process.stdout.write(JSON.stringify({seed:seedCases(),wish:wishCases(),
  seed_order:[dex.conditions.get('leechseed').onResidualOrder,dex.conditions.get('leechseed').onResidualSubOrder],
  wish_order:dex.conditions.get('wish').onResidualOrder}));
