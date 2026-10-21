// Invoke actual pinned Gen 3 item/status handlers on authored inputs, not games.
const path = require('node:path');
const {Dex} = require(path.join(process.argv[2],'dist/sim/dex.js'));
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
process.stdout.write(JSON.stringify({multipliers:{band:factor('choiceband','onModifyAtk','Ditto'),
  club_cubone:factor('thickclub','onModifyAtk','Cubone'),club_marowak:factor('thickclub','onModifyAtk','Marowak'),
  club_other:factor('thickclub','onModifyAtk','Ditto'),ball_pikachu_spa:factor('lightball','onModifySpA','Pikachu'),
  ball_other_spa:factor('lightball','onModifySpA','Ditto'),ball_atk_handler:!!dex.items.get('lightball').onModifyAtk},
  residual_order:{leftovers:[leftovers.onResidualOrder,leftovers.onResidualSubOrder],
    psn:[poison.onResidualOrder,poison.onResidualSubOrder]},leftovers_requested_heal:requested,
  lum:{eaten,status:pokemon.status,confused:!!pokemon.volatiles.confusion},choice_lock:lockContext.effectState.move}));
