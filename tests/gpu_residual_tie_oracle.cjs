const assert = require('node:assert/strict');
const path = require('node:path');
const root = process.argv[2];
const {Battle} = require(path.join(root,'dist/sim/battle.js'));
const {PRNG} = require(path.join(root,'dist/sim/prng.js'));
const {install} = require('../gpu_engine/chance_hooks.cjs');

let control = null;
let events = [];
install(Battle,PRNG,()=>control,event=>events.push(event),()=>null,path.join(root,'dist/sim/pokemon.js'));

function tape(nonce, first=0) {
  const draws = [Array(12).fill(0),Array(12).fill(0)];
  draws[0][11] = first;
  return {
    schema:'pokezero.gpu_chance_tape.v6',nonce,draws,secondary:[0,0],sleep_talk:[0,0],
    effect:[Array(5).fill(0),Array(5).fill(0)],phazing:[0,0],entry_ties:[0,0],resuming:false,
  };
}

function fixture(moves) {
  const b = new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  b.setPlayer('p1',{name:'one',team:[{species:'Sudowoodo',moves:moves[0]}]});
  b.setPlayer('p2',{name:'two',team:[{species:'Sudowoodo',moves:moves[1]}]});
  return b;
}

function wish(first) {
  const b = fixture([['wish','splash'],['wish','splash']]);
  const holders = b.sides.map(s=>s.active[0]);
  for (const [side,mon] of holders.entries()) {
    mon.storedStats.spe = 100;
    mon.baseMaxhp = mon.maxhp = 200 + 100*side;
    mon.hp = 30 + 30*side;
  }
  b.makeChoices('move 1','move 1');
  const shuffle = b.prng.shuffle.bind(b.prng);
  if (first === null) {
    events = [];
    control = tape('wish');
  } else {
    b.prng.shuffle = (items,start=0,end=items.length) => {
      const tied = items.slice(start,end);
      if (tied.length===2 && tied.every(h=>h.effect?.id==='wish')) {
        tied.sort((a,b)=>Number(b.effectHolder.side.n===first)-Number(a.effectHolder.side.n===first));
        items.splice(start,2,...tied);
      } else shuffle(items,start,end);
    };
  }
  b.makeChoices('move 2','move 2');
  const result = {
    hp:holders.map(p=>p.hp), wish:b.sides.map(s=>Boolean(s.slotConditions[0].wish)),
    order:b.log.filter(line=>line.includes('|[from] move: Wish')).map(line=>line.split('|')[2].slice(0,2)),
  };
  if (control) assert.equal(events.filter(e=>e.kind==='INERT_WISH_TIE').length,1);
  control = null;
  b.destroy();
  return result;
}

function counterPair(first) {
  const b = fixture([['counter'],['mirrorcoat']]);
  const preparations = [];
  const runAction = b.runAction;
  b.runAction = function(action) {
    if (action.choice==='beforeTurnMove') preparations.push(action.move.id);
    return runAction.call(this,action);
  };
  for (const s of b.sides) s.active[0].storedStats.spe = 100;
  events = [];
  control = tape(`counter:${first}`,first);
  b.makeChoices('move 1','move 1');
  assert.deepEqual(events.filter(e=>e.role==='before_turn_move_tie').map(e=>e.raw),[first]);
  assert.ok(events.some(e=>e.kind==='INERT_RESIDUAL_TIE'&&e.effects.slice().sort().join(',')==='counter,mirrorcoat'));
  control = null;
  b.destroy();
  return preparations;
}

const p1 = wish(0), p2 = wish(1), coupled = wish(null);
assert.deepEqual(p1,{hp:[130,210],wish:[false,false],order:['p1','p1','p2','p2']});
assert.deepEqual(p2,{...p1,order:['p2','p2','p1','p1']});
assert.deepEqual(coupled,p1);
assert.deepEqual(counterPair(0),['counter','mirrorcoat']);
assert.deepEqual(counterPair(1),['mirrorcoat','counter']);
