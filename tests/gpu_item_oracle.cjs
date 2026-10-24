// Invoke actual pinned Gen 3 item/status handlers on authored inputs, not games.
const path = require('node:path');
const {Dex} = require(path.join(process.argv[2],'dist/sim/dex.js'));
const {Battle} = require(path.join(process.argv[2],'dist/sim/battle.js'));
const dex = Dex.mod('gen3');
function factor(id,callback,species) {
  const fn = dex.items.get(id)[callback];
  if (!fn) return 1;
  return fn.call({chainModify:x=>x},200,{species:{name:species},volatiles:{}}) || 1;
}
const leftovers = dex.items.get('leftovers');
const poison = dex.conditions.get('psn');
let requested;
leftovers.onResidual.call({heal:n=>{requested=n;}},{baseMaxhp:101});
const lum = dex.items.get('lumberry');
let eaten=0;
const pokemon = {status:'tox',volatiles:{confusion:{}},
  eatItem:()=>{eaten++;lum.onEat.call({},pokemon);},
  cureStatus:()=>{pokemon.status='';},removeVolatile:id=>{delete pokemon.volatiles[id];}};
lum.onAfterSetStatus.call({},'tox',pokemon);
const lockContext = {effectState:{}};
dex.conditions.get('choicelock').onStart.call(lockContext,{lastMove:{id:'tackle'}});
function residualOrder(speed0,speed1) {
  const handlers = [{id:'heal0',order:10,subOrder:4,speed:speed0},
    {id:'heal1',order:10,subOrder:4,speed:speed1},
    {id:'status0',order:10,subOrder:6,speed:speed0},
    {id:'status1',order:10,subOrder:6,speed:speed1}];
  return handlers.sort(Battle.prototype.comparePriority).map(h=>h.id);
}
const pinch=[],pinch_handlers={};
for(const name of ['petayaberry','salacberry']){
  const item=dex.items.get(name);
  pinch_handlers[name]=[item.onResidualOrder,item.onResidualSubOrder,!!item.onUpdate];
  for(const seat of [0,1])for(const [hp,maxhp,stage] of
      [[0,101,0],[25,101,0],[26,101,0],[1,3,0],[1,4,0],[1,1,0],[25,101,6],[25,101,-6]]){
    const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
    for(let s=0;s<2;s++)b.setPlayer(`p${s+1}`,{name:`seat${s}`,team:[
      {species:'Mew',ability:'Limber',item:s===seat?name:'',moves:['splash']}]});
    const p=b.sides[seat].active[0],key=name==='petayaberry'?'spa':'spe';
    p.baseMaxhp=p.maxhp=maxhp;p.hp=hp;p.boosts[key]=stage;
    const fields=['atk','def','spa','spd','spe','accuracy','evasion'];
    const before=fields.map(k=>p.boosts[k]);
    b.singleEvent('Residual',item,p.itemState,p);
    pinch.push({name,seat,hp,maxhp,before,after:fields.map(k=>p.boosts[k]),item:p.item});
    b.destroy();
  }
}
process.stdout.write(JSON.stringify({multipliers:{band:factor('choiceband','onModifyAtk','Ditto'),
  club_cubone:factor('thickclub','onModifyAtk','Cubone'),club_marowak:factor('thickclub','onModifyAtk','Marowak'),
  club_other:factor('thickclub','onModifyAtk','Ditto'),ball_pikachu_spa:factor('lightball','onModifySpA','Pikachu'),
  ball_other_spa:factor('lightball','onModifySpA','Ditto'),ball_atk_handler:!!dex.items.get('lightball').onModifyAtk},
  residual_order:{leftovers:[leftovers.onResidualOrder,leftovers.onResidualSubOrder],
    psn:[poison.onResidualOrder,poison.onResidualSubOrder]},leftovers_requested_heal:requested,
  lum:{eaten,status:pokemon.status,confused:!!pokemon.volatiles.confusion},choice_lock:lockContext.effectState.move,
  unequal_speed_events:residualOrder(300,200),
  equal_speed_event_kinds:residualOrder(200,200).map(id=>id.replace(/[01]/,'')),pinch,pinch_handlers}));
