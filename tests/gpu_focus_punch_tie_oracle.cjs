const path = require('node:path');
const root = process.argv[2];
const {Battle} = require(path.join(root,'dist/sim/battle.js'));
const {PRNG} = require(path.join(root,'dist/sim/prng.js'));
const {install} = require('../gpu_engine/chance_hooks.cjs');

let control = null;
let events = [];
install(Battle,PRNG,()=>control,event=>events.push(event),()=>null,path.join(root,'dist/sim/pokemon.js'));

function play(preparationFirst, p2Speed) {
  const battle = new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  const team = () => [{species:'Sudowoodo',ability:'Rock Head',moves:['focuspunch']}];
  battle.setPlayer('p1',{name:'one',team:team()});
  battle.setPlayer('p2',{name:'two',team:team()});
  const p1 = battle.p1.active[0], p2 = battle.p2.active[0];
  for (const mon of [p1,p2]) mon.baseMaxhp = mon.maxhp = mon.hp = 1000;
  p1.storedStats.spe = 100;
  p2.storedStats.spe = p2Speed;
  events = [];
  control = {
    schema:'pokezero.gpu_chance_tape.v6',nonce:`${preparationFirst}:${p2Speed}`,
    draws:[Array(12).fill(0),Array(12).fill(0)],secondary:[0,0],sleep_talk:[0,0],
    effect:[Array(5).fill(0),Array(5).fill(0)],phazing:[0,0],entry_ties:[0,0],resuming:false,
  };
  control.draws[0][11] = preparationFirst;
  // Independently select p1 for ordinary move order on the tied rows.
  battle.makeChoices('move 1','move 1');
  const result = {
    starts:battle.log.filter(s=>s.includes('|move: Focus Punch')).map(s=>s.split('|')[2].slice(0,2)),
    attacks:battle.log.filter(s=>s.startsWith('|move|')).map(s=>s.split('|')[2].slice(0,2)),
    hp:[p1.hp,p2.hp],
    events,
  };
  control = null;
  battle.destroy();
  return result;
}

process.stdout.write(JSON.stringify([play(0,100),play(1,100),play(0,101)]));
