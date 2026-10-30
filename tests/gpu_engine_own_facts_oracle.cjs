// Literal component objects invoke pinned source methods/callbacks directly.
// No Battle/Pokemon constructor, team generator, transition or game is called.
const fs=require('node:fs'),path=require('node:path');
const {execFileSync}=require('node:child_process');
const root=process.argv[2];
if(execFileSync('git',['-C',root,'rev-parse','HEAD'],{encoding:'utf8'}).trim()!==
  'f76228a1354b5d0f307ca2d16101294ad3a2308b')throw new Error('source drift');
const {Dex,toID}=require(path.join(root,'dist/sim/dex.js'));
const {Pokemon}=require(path.join(root,'dist/sim/pokemon.js'));
const {Battle}=require(path.join(root,'dist/sim/battle.js'));
const {Side}=require(path.join(root,'dist/sim/side.js'));
const dex=Dex.mod('gen3');
function encode(input){
 const sides=input.sides.map((row,side)=>{
  const s={id:'p'+(side+1),name:'literal',pokemonLeft:row.party.filter(p=>p.hp>0).length};
  s.pokemon=row.party.map((p,i)=>{
   const mon={side:s,position:i,isActive:i===0,fainted:p.hp===0,fullname:s.id+': literal',
    details:p.species+', L'+p.level+(p.gender===1?', M':p.gender===2?', F':''),
    species:dex.species.get(p.species),hp:p.hp,baseAbility:p.base_ability,ability:p.ability,
    item:p.item,pokeball:'pokeball',baseStoredStats:p.stats,
    types:p.types,knownType:true,volatiles:{},trapped:false,maybeTrapped:false,maybeLocked:false,maybeDisabled:false,
    hpType:p.hp_type,hpPower:70,canMegaEvo:false,canMegaEvoX:false,canMegaEvoY:false,canUltraBurst:false,
    switchFlag:row.forced,
    getHealth:()=>({secret:p.hp?`${p.hp}/${p.maximum}${p.status?' '+p.status:''}`:'0 fnt'}),
    getTypes:()=>p.types,hasType:t=>p.types.includes(t),hasAbility:a=>p.ability===a,
    isAdjacent:()=>true,runStatusImmunity:kind=>p.hp>0&&dex.getImmunity(kind,p.types),
    isGrounded:()=>p.rooted||(!p.types.includes('Flying')&&p.ability!=='levitate'),
    getItem:()=>dex.items.get(p.item),ignoringItem:()=>false,
    moves:p.moves.map(m=>toID(dex.moves.get(m.id).id)),
    moveSlots:p.moves.map(m=>({id:toID(dex.moves.get(m.id).id),move:dex.moves.get(m.id).name,
      target:dex.moves.get(m.id).target,pp:m.pp,maxpp:m.maxpp,disabled:false})),
    getLockedMove:()=>null,getSemiLockedMove:()=>null,getDynamaxRequest:()=>undefined};
   mon.tryTrap=Pokemon.prototype.tryTrap;mon.getMoves=Pokemon.prototype.getMoves;
   mon.getMoveRequestData=Pokemon.prototype.getMoveRequestData;
   mon.getSwitchRequestData=Pokemon.prototype.getSwitchRequestData;
   mon.isLastActive=Pokemon.prototype.isLastActive;mon.disableMove=Pokemon.prototype.disableMove;
   mon.hasMove=id=>mon.moves.includes(id);mon.removeVolatile=id=>{delete mon.volatiles[id];};
   mon._literal=p;return mon;
  });
  s.active=[s.pokemon[0]];s.getRequestData=Side.prototype.getRequestData;return s;
 });
 const battle={dex,gen:3,sides,supportCancel:true,actions:{canZMove:()=>undefined},
  canSwitch:s=>s.pokemon.slice(1).filter(p=>!p.fainted).length};
 for(const s of sides){for(const p of s.pokemon)p.battle=battle;}
 for(let i=0;i<2;i++){
  const p=sides[i].active[0],foe=sides[1-i].active[0],v=p._literal;
  if(v.rooted||v.linked||v.visible_trap)p.tryTrap();
  if(v.choice){const c=dex.conditions.get('choicelock');
   c.onDisableMove.call({effectState:{move:v.choice,sourceEffect:dex.items.get('choiceband')}},p);}
  const current=dex.abilities.get(foe.ability);
  function invoke(ability,maybe){const fn=maybe?
   (ability.onFoeMaybeTrapPokemon||ability.onAnyMaybeTrapPokemon):
   (ability.onFoeTrapPokemon||ability.onAnyTrapPokemon);
   if(fn)fn.call({effectState:{target:foe}},p,foe);}
  invoke(current,false);invoke(current,true);
  // Exactly the permitted public alternate loop in pinned nextTurn.
  const species=dex.species.get(input.public_species[1-i]);
  const rule=dex.formats.getRuleTable(dex.formats.get('gen3randombattle'));
  for(const [slot,name] of Object.entries(species.abilities)){
   const id=toID(name);if(id===foe.ability||(slot==='H'&&species.unreleasedHidden)||rule.has('-ability:'+id))continue;
   const alternate=dex.abilities.get(id).onFoeMaybeTrapPokemon;
   if(alternate)alternate.call({effectState:{}},p,foe);
  }
 }
 return Battle.prototype.getRequests.call(battle,input.replacement?'switch':'move');
}
const inputs=JSON.parse(fs.readFileSync(0,'utf8'));
process.stdout.write(JSON.stringify(inputs.map(encode))+'\n');
