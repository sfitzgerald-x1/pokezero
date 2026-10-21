// Execute the actual pinned Gen 3 handlers on authored inputs. Not game parity.
const path = require('node:path');
const {Dex} = require(path.join(process.argv[2], 'dist/sim/dex.js'));
const dex = Dex.mod('gen3');
const stall = dex.conditions.get('stall');
const context = {effectState:{}, effect:stall, randomChance:(n,d) => {
  context.denominator = d; return false;
}, debug:()=>{}};
stall.onStart.call(context);
const start = context.effectState.counter;
const after_restart = [], successes = [];
for (let i=0;i<3;i++) {
  stall.onStallMove.call(context);
  successes.push(8/context.denominator);
  stall.onRestart.call(context);
  after_restart.push(context.effectState.counter);
}
const move = dex.moves.get('substitute');
const target = {hp:101,maxhp:101,volatiles:{substitute:{}},
  removeVolatile:id=>delete target.volatiles[id],
  addVolatile:id=>{target.volatiles[id]={};}};
const subContext = {effectState:target.volatiles.substitute,add:()=>{},
  directDamage:amount=>{target.hp -= Math.floor(amount);}};
move.condition.onStart.call(subContext,target);
move.onHit.call(subContext,target);
const cost = 101-target.hp, sub_hp = target.volatiles.substitute.hp;
const source = {};
const hitContext = {actions:{getDamage:()=>1000},add:()=>{},runEvent:()=>{},HIT_SUBSTITUTE:'sub'};
dex.conditions.get('substitute').onTryPrimaryHit.call(hitContext,target,source,dex.moves.get('tackle'));
process.stdout.write(JSON.stringify({stall:{start,after_restart,successes},
  substitute:{cost,sub_hp,remaining_sub_hp:target.volatiles.substitute?.hp || 0,damage:source.lastDamage},
  priorities:Object.fromEntries(['protect','detect','endure'].map(id=>[id,dex.moves.get(id).priority]))}));
