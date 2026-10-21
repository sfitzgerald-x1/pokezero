// Authored pinned Gen 3 battle events, not the 200-game fidelity panel.
const path = require('node:path');
const {Battle} = require(path.join(process.argv[2],'dist/sim/battle.js'));
const {Dex} = require(path.join(process.argv[2],'dist/sim/dex.js'));
const dex = Dex.mod('gen3');
function battle(ability0,ability1,item='') {
  const b = new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for (let side=0;side<2;side++) b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
    {name:'active',species:'Mew',ability:side?ability1:ability0,moves:['splash','flamethrower'],level:100,item:side?'':item},
    {name:'bench',species:'Ditto',ability:'Limber',moves:['splash'],level:100},
  ]});
  b.randomizer=damage=>damage;  // authored full-roll diagnostic, no sampled outcome
  return b;
}
function guts(item) {
  const b=battle('Guts','Thick Fat',item),p=b.p1.active[0];
  p.setStatus('brn'); p.storedStats.atk=103; p.storedStats.def=100;
  const attack=b.runEvent('ModifyAtk',p,p,b.dex.getActiveMove('tackle'),103);
  const confusion=b.actions.getDamage(p,p,40);
  b.destroy(); return {attack,confusion};
}
function trap(ability0,ability1,types) {
  const b=battle(ability0,ability1),p=b.p1.active[0];
  p.types=types; p.trapped=false;
  b.runEvent('TrapPokemon',p);
  const trapped=!!p.trapped;
  b.destroy(); return trapped;
}
const trapCases=[
  ['Limber','Arena Trap',['Normal']],['Limber','Arena Trap',['Flying']],
  ['Levitate','Arena Trap',['Normal']],['Limber','Magnet Pull',['Steel']],
  ['Limber','Magnet Pull',['Ghost']],['Shadow Tag','Shadow Tag',['Ghost']],
  ['Limber','Shadow Tag',['Flying']],['Limber','Magnet Pull',['Flying','Steel']],
];
const fat = ['Fire','Ice','Water'].map(type=>{
  const ctx={event:{modifier:1},trunc:Dex.trunc,chainModify:Battle.prototype.chainModify};
  dex.abilities.get('thickfat').onSourceBasePower.call(ctx,95,{},{},{type});
  return Battle.prototype.modify.call(ctx,95,ctx.event.modifier);
});
process.stdout.write(JSON.stringify({guts:guts(''),guts_band:guts('Choice Band'),
  thick_fat_power:fat,trap_cases:trapCases,trap:trapCases.map(row=>trap(...row))}));
