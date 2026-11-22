#!/usr/bin/env node
'use strict';
// Validation reference only: condition a large unmodified randomTeam sample.
// Production inference uses the DP and never this acceptance loop.
const fs = require('node:fs');
const crypto = require('node:crypto');
const {loadGenerator, setKey, PIN, EXECUTED_PIN} = require('./gen3_generator_weights.cjs');

function conditionDigest(facts) {
  if (Object.keys(facts).some(k => !['evidence','fixed_slots','initial_wobbuffet','terminal_weights'].includes(k)) ||
      typeof facts.initial_wobbuffet !== 'boolean') throw new Error('invalid conditioning facts');
  const terminal = facts.terminal_weights || [1, 1];
  if (terminal.length !== 2 || terminal.some(w => w !== 1)) {
    throw new Error('reference does not support reverse Wobbuffet terminal likelihoods');
  }
  const evidence = Object.entries(facts.evidence).sort(([a], [b]) => a.localeCompare(b));
  if (evidence.some(([, row]) => !Array.isArray(row) || !row.length || row.some(w => w !== 0 && w !== 1))) {
    throw new Error('reference requires Boolean evidence likelihoods');
  }
  const fixed = Object.entries(facts.fixed_slots).map(([slot, species]) => {
    if (!/^[0-5]$/.test(slot) || typeof species !== 'string') throw new Error('invalid generation slot');
    return [Number(slot), species];
  }).sort(([a], [b]) => a-b);
  const canonical = ['gen3-joint-condition-v1', evidence, fixed, facts.initial_wobbuffet, terminal];
  return crypto.createHash('sha256').update(JSON.stringify(canonical)).digest('hex');
}

function features(team, catalog) {
  let context = 0;
  const counts = {Water: 0, Flying: 0, Psychic: 0};
  let electric = 0, water = 0, rain = 0;
  for (let i = 0; i < team.length; i++) {
    const mon = team[i];
    const profile = catalog[mon.speciesId];
    for (const type of profile.types) if (type in counts) counts[type]++;
    electric += Number(profile.weak.includes('Electric'));
    water += Number(profile.weak.includes('Water'));
    rain += Number(mon.moves.includes('raindance') || mon.ability === 'Drizzle');
    if (i === team.length - 1) break;
    context |= Number(mon.moves.includes('raindance') || mon.ability === 'Drizzle');
    context |= Number(mon.moves.includes('sunnyday') || mon.ability === 'Drought') << 1;
    context |= Number(mon.moves.some(m => ['aromatherapy', 'healbell'].includes(m))) << 2;
    context |= Number(mon.moves.includes('rapidspin')) << 3;
  }
  return {second_species: team[1].speciesId, last_species: team.at(-1).speciesId,
    last_context: String(context), type_counts: [counts.Water, counts.Flying, counts.Psychic].join(','),
    weakness_counts: [electric, water].join(','), second_set: `${team[1].speciesId}:${team[1].variantIndex}`,
    lead_rain: `${team[0].variantIndex}:${rain}`};
}

function sample(root, input, teams, seed) {
  if (!Number.isSafeInteger(teams) || teams < 1) throw new Error('invalid sample size');
  const condition_sha256 = conditionDigest(input.facts);
  const generator = loadGenerator(root); generator.setSeed(seed);
  const indices = {};
  for (const [species, variants] of Object.entries(input.manifest.species)) {
    indices[species] = new Map();
    variants.forEach((v, i) => {const key=setKey(v); if (!indices[species].has(key)) indices[species].set(key,i);});
  }
  const catalog = {};
  for (const b of input.model.bases) for (const s of b.forms) catalog[s]=b;
  const counts = {}; let accepted = 0;
  for (let i = 0; i < teams; i++) {
    generator.battleHasWobbuffet = input.facts.initial_wobbuffet;
    const team = generator.randomTeam();
    if (team.length !== 6) throw new Error('incomplete generator team');
    for (const mon of team) {
      const vi=indices[mon.speciesId]?.get(setKey(mon));
      if (vi === undefined) throw new Error(`generated set missing: ${setKey(mon)}`);
      mon.variantIndex=vi;
    }
    const allowed = Object.entries(input.facts.evidence).every(([s,w]) => team.some(m => m.speciesId===s && w[m.variantIndex]>0)) &&
      Object.entries(input.facts.fixed_slots).every(([slot,s]) => team[Number(slot)].speciesId===s);
    if (allowed) {
      accepted++;
      for (const [name,value] of Object.entries(features(team,catalog))) {
        const hist=(counts[name] ||= {});hist[value]=(hist[value]||0)+1;
      }
    }
  }
  return {schema:'gen3-joint-reference-v2', showdown_commit:PIN, executed_sha256:EXECUTED_PIN, seed, teams,
    condition_sha256, facts:input.facts,
    sets:teams*6, accepted, source_hash:input.manifest.source_hash, model_sha256:input.model.sha256, counts};
}
module.exports = {features, sample, conditionDigest};
if (require.main === module) {
  const [root,input,teams,seed,output]=process.argv.slice(2);
  if (!output) throw new Error('usage: gen3_joint_reference.cjs ROOT INPUT TEAMS SEED OUTPUT');
  fs.writeFileSync(output,JSON.stringify(sample(root,JSON.parse(fs.readFileSync(input)),Number(teams),seed))+'\n');
}
