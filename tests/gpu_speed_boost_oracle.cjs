// Authored pinned Gen 3 timing/queue fixtures, not a full-game fidelity panel.
const path = require('node:path');
const {Battle} = require(path.join(process.argv[2],'dist/sim/battle.js'));
const {Dex} = require(path.join(process.argv[2],'dist/sim/dex.js'));
function create() {
  const b = new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let side=0;side<2;side++) b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
    {species:'Mew',ability:side?'Limber':'Speed Boost',moves:['splash'],level:100},
    {species:'Ditto',ability:side?'Limber':'Speed Boost',moves:['splash'],level:100},
  ]});
  return b;
}
const b=create();
const stages=[b.p1.active[0].boosts.spe];
b.makeChoices('move 1','move 1');stages.push(b.p1.active[0].boosts.spe);
b.makeChoices('switch 2','move 1');stages.push(b.p1.active[0].boosts.spe);
b.makeChoices('move 1','move 1');stages.push(b.p1.active[0].boosts.spe);
b.destroy();
const forced=create();
forced.p1.active[0].faint();forced.faintMessages();forced.checkFainted();
forced.midTurn=true;forced.makeRequest('switch');forced.makeChoices('switch 2','');
const forcedStages=[forced.p1.active[0].boosts.spe];
forced.makeChoices('move 1','move 1');forcedStages.push(forced.p1.active[0].boosts.spe);
forced.destroy();
const queue=create();
const left=queue.p1.active[0],right=queue.p2.active[0];
left.setAbility('limber');left.setItem('leftovers');left.maxhp=left.baseMaxhp=100;left.hp=50;
right.setAbility('speedboost');right.maxhp=right.baseMaxhp=100;right.hp=1;right.setStatus('psn');
for(const side of queue.sides) {
  side.pokemon[1].hp=0;side.pokemon[1].fainted=true;side.pokemonLeft=1;
  side.active[0].storedStats.spe=200;side.active[0].updateSpeed();side.active[0].activeTurns=1;
}
queue.fieldEvent('Residual');
const equalSpeedHeal=left.hp;
queue.destroy();
const ability=Dex.mod('gen3').abilities.get('speedboost');
process.stdout.write(JSON.stringify({stages,forced_stages:forcedStages,equal_speed_heal:equalSpeedHeal,
  residual_order:[ability.onResidualOrder,ability.onResidualSubOrder]}));
