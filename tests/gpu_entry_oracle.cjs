// A single authored switch turn and pinned handlers, not a scientific roster.
const path = require('node:path');
const {Dex} = require(path.join(process.argv[2],'dist/sim/dex.js'));
const {Battle} = require(path.join(process.argv[2],'dist/sim/battle.js'));
const dex = Dex.mod('gen3');
const set = (species,ability)=>({species,ability,moves:['splash'],level:100});
const battle = new Battle({formatid:'gen3customgame',seed:[11,22,33,44]});
battle.setPlayer('p1',{name:'a',team:[set('Ditto','Limber'),set('Salamence','Intimidate')]});
battle.setPlayer('p2',{name:'b',team:[set('Ditto','Limber'),set('Salamence','Intimidate')]});
battle.p1.active[0].storedStats.spe=300;
battle.p2.active[0].storedStats.spe=100;
battle.updateSpeed();
battle.makeChoices('switch 2','switch 2');
const stages = [battle.p1.active[0].boosts.atk,battle.p2.active[0].boosts.atk];
const weatherBattle = new Battle({formatid:'gen3customgame',seed:[11,22,33,44]});
const fast = {...set('Kyogre','Drizzle'),nature:'Timid',evs:{spe:252}};
const slow = {...set('Groudon','Drought'),nature:'Brave',ivs:{spe:0}};
weatherBattle.setPlayer('p1',{name:'a',team:[fast]});
weatherBattle.setPlayer('p2',{name:'b',team:[slow]});
const spikes = dex.moves.get('spikes').condition;
const damages=[1,2,3].map(layers=>{
  let damage=0;
  spikes.onEntryHazard.call({effectState:{layers},damage:n=>{damage=Math.max(1,Math.floor(n));}},
    {isGrounded:()=>true,hasItem:()=>false,maxhp:101});
  return damage;
});
const cured={status:'tox',counter:4,clearStatus(){this.status='';this.counter=0;}};
dex.abilities.get('naturalcure').onSwitchOut.call({add:()=>{}},cured);
let drops=0;
dex.abilities.get('intimidate').onStart.call({hint:()=>{},add:()=>{},boost:()=>{drops++;}},
  {adjacentFoes:()=>[{volatiles:{substitute:{}}}],side:{}});
process.stdout.write(JSON.stringify({double_switch_attack_stages:stages,weather_start:weatherBattle.field.weather,
  spikes_damage:damages,natural_cure:{status:cured.status,counter:cured.counter},intimidate_substitute_drop_calls:drops}));
battle.destroy();
weatherBattle.destroy();
