// Authored Gen 3 chance/cancellation fixtures, not complete-game parity.
const path = require('node:path');
const {Battle} = require(path.join(process.argv[2], 'dist/sim/battle.js'));
const {Dex} = require(path.join(process.argv[2], 'dist/sim/dex.js'));
const dex = Dex.mod('gen3');
const names = ['aeroblast','aircutter','blazekick','crabhammer','crosschop',
  'karatechop','leafblade','poisontail','razorleaf','slash'];
function sample(name, ability) {
  const b = new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for (let i=0; i<2; i++) b.setPlayer(`p${i+1}`,{name:`seat${i}`,team:[
    {species:'Mew',ability:i?ability:'Limber',moves:['splash'],level:100}]});
  const p=b.p1.active[0],q=b.p2.active[0],move=b.dex.getActiveMove(name),draws=[];
  b.randomChance=(a,z)=>{draws.push([a,z]);return true;};
  b.randomizer=x=>x;
  const damage=b.actions.getDamage(p,q,move);
  const crit=q.getMoveHitData(move).crit;
  b.destroy();return {draws,crit,damage};
}
process.stdout.write(JSON.stringify({ratios:names.map(name=>dex.moves.get(name).critRatio),
  ordinary:sample('tackle','Limber'),high:sample('slash','Limber'),
  battle:sample('slash','Battle Armor'),shell:sample('pursuit','Shell Armor')}));
