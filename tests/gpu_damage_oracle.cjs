// Bounded arithmetic oracle, not a battle or the 200-game divergence check.
// Execute the pinned Showdown implementations, with resolved event modifiers.
const fs = require('node:fs');
const path = require('node:path');
const root = process.argv[2];
const {Dex} = require(path.join(root, 'dist/sim/dex.js'));
const {Battle} = require(path.join(root, 'dist/sim/battle.js'));
const {Pokemon} = require(path.join(root, 'dist/sim/pokemon.js'));
const damage = Dex.mod('gen3').data.Scripts.actions.modifyDamage;
const rows = JSON.parse(fs.readFileSync(0, 'utf8'));
const result = rows.map(r => {
  if (r.immune || r.power <= 0) return 0;
  const battle = {
    field:{pseudoWeather:{}}, trunc:Dex.trunc, modify:Battle.prototype.modify,
    debug() {},
    clampIntRange:(v,lo,hi=Infinity)=>Math.max(lo,Math.min(hi,Math.floor(v))),
    random:()=>100-r.roll, randomizer:Battle.prototype.randomizer,
    runEvent(name,...args) {
      const value = args.at(-1);
      if (name === 'WeatherModifyDamage' && (r.weather_num !== 1 || r.weather_den !== 1)) {
        return this.modify(value,r.weather_num,r.weather_den);
      }
      if (name === 'ModifyDamagePhase1') {
        const chain = {trunc:Dex.trunc,event:{modifier:1},chainModify:Battle.prototype.chainModify};
        if (r.screen && !r.critical) chain.chainModify(1,2);
        if (r.phase1_num !== 1 || r.phase1_den !== 1) {
          chain.chainModify(r.phase1_num,r.phase1_den);
        }
        return this.modify(value,chain.event.modifier);
      }
      return value;
    },
  };
  const stat = (value,stage) => Pokemon.prototype.calculateStat.call(
    {storedStats:{atk:value},battle},'atk',stage,1);
  const atkStage = r.critical && r.attack_stage < 0 ? 0 : r.attack_stage;
  const defStage = r.critical && r.defense_stage > 0 ? 0 : r.defense_stage;
  const attack = battle.modify(stat(r.attack,atkStage),r.attack_num,r.attack_den);
  const defense = Math.max(1,battle.modify(stat(r.defense,defStage),r.defense_num,r.defense_den));
  const tr = Dex.trunc;
  const base = tr(tr(tr(tr(2*r.level/5+2)*r.power*attack)/defense)/50);
  const pokemon = {status:r.burned?'brn':'',hasAbility:name=>name==='guts'&&r.guts,
    hasType:()=>r.stab};
  const typeMod = Math.log2(r.effectiveness_num/r.effectiveness_den);
  if (!Number.isInteger(typeMod)) throw new Error('oracle expects power-of-two type modifiers');
  const target = {getMoveHitData:()=>({crit:r.critical}),runEffectiveness:()=>typeMod};
  return damage.call({battle},base,pokemon,target,
    {type:r.typeless?'???':'Water',category:r.physical?'Physical':'Special'},true);
});
const q12 = (v,n,d)=>Battle.prototype.modify.call({trunc:Dex.trunc},v,n,d);
process.stdout.write(JSON.stringify({damage:result,
  large_modifier:q12(2000000,3,2),
  paralysis_speed:[101,103,1].map(v=>q12(v,1,4))}));
