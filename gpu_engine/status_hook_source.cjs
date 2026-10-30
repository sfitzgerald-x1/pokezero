// Static effective Dex identity inventory only; no Teams/Battle/RNG invocation.
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const source = require('./request_source_rules.cjs');
const stable = value => JSON.stringify(value, (_, x) => x && !Array.isArray(x) &&
  typeof x === 'object' ? Object.fromEntries(Object.keys(x).sort().map(k => [k,x[k]])) : x);
const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const families = ['Damage','Residual','Update','SetStatus','AfterSetStatus',
  'UseItem','TryEatItem','Eat','EatItem','AfterUseItem'];
const dispatch = Object.fromEntries(families.map(name => [name,
  name === 'Residual' ? ['onResidual','onSideResidual','onFieldResidual'] :
  ['Update','Eat'].includes(name) ? ['on'+name] :
  ['','Ally','Foe','Any','Source'].map(prefix => 'on'+prefix+name)]));
const relevant = new Map(Object.entries(dispatch).flatMap(([name, keys],i) => keys.map(k => [k,1<<i])));

function scan(value) {
  const entries=[], carriers=[]; let flags=0, known=true;
  const seen = new WeakSet();
  function walk(object, location) {
    if (!object || typeof object !== 'object' || seen.has(object)) return;
    seen.add(object);
    for (const key of Object.keys(object).sort()) {
      const x=object[key], where=location.concat(key);
      if (key.startsWith('on')) {
        const metadata=/(?:Order|Priority|SubOrder)$/.test(key);
        const type=typeof x;
        let encoded;
        if (type==='function') encoded=Function.prototype.toString.call(x);
        else if (type==='undefined') encoded='undefined';
        else if (x===null || ['string','boolean'].includes(type) ||
                 (type==='number' && Number.isFinite(x))) encoded=stable(x);
        else {encoded=Object.prototype.toString.call(x);known=false;}
        const entry={path:where.join('.'),key,type,metadata,sha256:hash(encoded)};
        if (!metadata) {
          // Absence, not body-text inspection, determines the relevant mask.
          if (type!=='undefined' && relevant.has(key)) flags |= relevant.get(key);
          if (type!=='undefined' && type!=='function' && type!=='boolean' &&
              !(type==='number' && Number.isFinite(x))) known=false;
          for (const suffix of ['Order','Priority','SubOrder']) {
            const v=object[key+suffix];
            entry[suffix]=v===undefined ? null : v;
            if (v!==undefined && typeof v!=='boolean' &&
                !(typeof v==='number' && Number.isFinite(v))) known=false;
          }
        } else if (type!=='undefined' && type!=='boolean' &&
                   !(type==='number' && Number.isFinite(x))) known=false;
        entries.push(entry);
      }
      // Residual's duration path can call End without onResidual. This is a
      // conservative carrier obligation, not proof of an active condition.
      if (['duration','durationCallback'].includes(key) && x!==undefined && x!==0) {
        carriers.push({path:where.join('.'),type:typeof x,sha256:hash(String(x))}); flags |= 2;
      }
      if (x && typeof x==='object') walk(x,where);
    }
  }
  walk(value,[]);
  return {entries,carriers,flags,known};
}

function row(effect,requested,index,kind,toID) {
  if (index===0 && requested==='_none') return {requested,id:'',index,kind,
    effect_type:'Absent',exists:true,absent:true,known:true,flags:0,entries:[],carriers:[]};
  const identity=toID(effect.id)===requested || (kind==='moves' &&
    typeof effect.realMove==='string' && toID(effect.realMove)===effect.id &&
    toID(effect.name)===requested);
  const found=!!effect?.exists && effect.gen<=3 && identity;
  const callbacks=scan(effect);
  return {requested,id:effect.id,index,kind,effect_type:effect.effectType,
    source_name:effect.name,real_move:effect.realMove||null,
    exists:found,absent:false,...callbacks,known:found&&callbacks.known};
}

function build(root,metadata) {
  // Re-run the ORIGINAL loader's complete pinned revision/tree/vocabulary cut.
  const rules=source.build(root,metadata);
  for (const kind of ['abilities','items','species','moves']) {
    const ids=metadata.vocabulary[kind];
    if (!Array.isArray(ids) || ids[0]!=='_none' || new Set(ids).size!==ids.length ||
        ids.slice(1).some(x=>typeof x!=='string'||!x||/[^a-z0-9]/.test(x)))
      throw Error('exact unique canonical vocabulary required: '+kind);
  }
  const {Dex,toID}=require(path.join(root,'dist/sim/dex.js'));
  const dex=Dex.mod('gen3'), format=dex.formats.get('gen3randombattle');
  const table=dex.formats.getRuleTable(format);
  const rows=Object.fromEntries(['abilities','items','species','moves'].map(kind => [kind,
    metadata.vocabulary[kind].map((id,i)=>row(dex[kind].get(id),id,i,kind,toID))]));
  // Enumerate the finite effective condition namespace, including nested move,
  // item, ability, side/slot/field carriers. No active-carrier history is minted.
  const conditionIDs=new Set(Object.keys(dex.data.Conditions));
  for (const kind of ['Moves','Abilities','Items'])
    for (const [id,value] of Object.entries(dex.data[kind])) if (value?.condition) conditionIDs.add(id);
  const conditions=[...conditionIDs].sort().map(id => {
    const effect=dex.conditions.get(id), callbacks=scan(effect);
    return {requested:id,id:effect.id,kind:'conditions',effect_type:effect.effectType,
      exists:!!effect.exists,...callbacks};
  });
  const skip=['onBegin','onTeamPreview','onBattleStart','onValidateRule','onValidateTeam','onChangeSet','onValidateSet'];
  // Preserve actual RuleTable iteration order: Battle's constructor installs
  // these rows as pseudoWeather carriers using this exact lexical predicate.
  const ruleRows=[...table.keys()].filter(id=>!/^[+*!-]/.test(id)).map(id => {
    const effect=dex.formats.get(id);
    return {requested:id,id:effect.id,kind:'rules',effect_type:effect.effectType,
      exists:!!effect.exists,installed_as_pseudo_weather:!!effect.exists &&
        Object.keys(effect).some(key=>key.startsWith('on')&&!skip.includes(key)),...scan(effect)};
  });
  return {schema:'pokezero.gen3-status-hook-source.static.v1',revision:source.PIN,
    format:'gen3randombattle',source_tree:rules.source_tree,compiled_tree:rules.compiled_tree,
    request_generator_sha256:rules.generator_sha256,rules_sha256:hash(stable(rules)),
    generator_sha256:hash(fs.readFileSync(__filename)),
    metadata_sha256:hash(stable(metadata)),vocabulary_sha256:hash(stable(metadata.vocabulary)),
    vocabulary:metadata.vocabulary,dispatch,rows,conditions,
    format_row:{id:format.id,effect_type:format.effectType,...scan(format)},
    rule_table:[...table.entries()].map(([id,value])=>[id,value]),rule_carrier_skip_keys:skip,rules:ruleRows,
    scope:'STATIC_IDENTITY_ONLY_NO_OWNING_HISTORY_NO_ADMISSION'};
}
module.exports={build,dispatch};
if (require.main===module) process.stdout.write(JSON.stringify(build(process.argv[2],
  JSON.parse(fs.readFileSync(0,'utf8'))))+'\n');
