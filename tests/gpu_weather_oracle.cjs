// Invoke pinned Gen 3 weather handlers on authored inputs, not a game panel.
const path = require('node:path');
const {Dex} = require(path.join(process.argv[2],'dist/sim/dex.js'));
const {Battle} = require(path.join(process.argv[2],'dist/sim/battle.js'));
const {Field} = require(path.join(process.argv[2],'dist/sim/field.js'));
const dex = Dex.mod('gen3');
const names = ['raindance','sunnyday','sandstorm','hail'];
const conditions = names.map(n=>dex.conditions.get(n));
const abilityDurations = conditions.map(c=>{
  const ctx = {gen:3,effectState:{duration:5},add:()=>{}};
  c.onFieldStart.call(ctx,{}, {},{effectType:'Ability'});
  return ctx.effectState.duration;
});
function modifiers(weather) {
  return ['Water','Fire','Normal'].map(type=>{
    const ctx = {chainModify:x=>x,debug:()=>{}};
    return dex.conditions.get(weather).onWeatherModifyDamage.call(ctx,100,{},
      {effectiveWeather:()=>weather},{type}) ?? null;
  });
}
function expiry(duration) {
  let calls=0;
  const handler = {state:{duration},end:()=>{},endCallArgs:[],callback:()=>{},effect:{effectType:'Weather'},effectHolder:{}};
  const ctx = {field:{},sides:[],speedSort:()=>{},findFieldEventHandlers:()=>[handler],
    singleEvent:()=>{calls++;},faintMessages:()=>{},ended:false};
  Battle.prototype.fieldEvent.call(ctx,'Residual');
  return calls;
}
process.stdout.write(JSON.stringify({orders:conditions.map(c=>c.onFieldResidualOrder),
  durations:conditions.map(c=>c.duration),ability_durations:abilityDurations,
  same_weather_move:names.map(weather=>Field.prototype.setWeather.call({weather,battle:{dex,gen:3}},weather,
    {},{effectType:'Move'})),rain_modifiers:modifiers('raindance'),sun_modifiers:modifiers('sunnyday'),
  thunder_accuracy:['raindance','sunnyday',''].map(weather=>{
    const move={accuracy:70};
    dex.moves.get('thunder').onModifyMove.call({},move,{}, {effectiveWeather:()=>weather});
    return move.accuracy;
  }),expiry_callback_counts:[expiry(1),expiry(2)],
  rock_spdef_handler:!!dex.conditions.get('sandstorm').onModifySpD}));
