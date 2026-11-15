#!/usr/bin/env node
'use strict';

// Statistical validation uses unmodified randomTeam randomness, recording the
// context seen at each call to randomSet. No teams or private battle data kept.
const fs = require('node:fs');
const {PIN, loadGenerator, contextIndex, setKey, executedDigest} = require('./gen3_generator_weights.cjs');

function sample(root, minimumSets, seed) {
  if (!Number.isSafeInteger(minimumSets) || minimumSets < 1) throw new Error('invalid sample size');
  const generator = loadGenerator(root);
  generator.setSeed(seed);
  const counts = {};
  const original = generator.randomSet;
  let sets = 0, teams = 0;
  generator.randomSet = function(species, context, ...rest) {
    const index = contextIndex(context);
    const set = original.call(this, species, context, ...rest);
    const key = setKey(set);
    const row = (counts[set.speciesId] ||= {});
    const histogram = (row[index] ||= {});
    histogram[key] = (histogram[key] || 0) + 1;
    sets++;
    return set;
  };
  while (sets < minimumSets) {
    // One generator per battle: carry Wobbuffet exclusion across two teams.
    if (teams % 2 === 0) generator.battleHasWobbuffet = generator.battleHasDitto = false;
    generator.randomTeam();
    teams++;
    if (teams % 10000 === 0) process.stderr.write(`${teams} teams / ${sets} sets\n`);
  }
  return {schema: 'gen3-generator-sample-v1', showdown_commit: PIN,
    executed_sha256: executedDigest(generator.enumerationRoot), seed, sets, teams, counts};
}

if (require.main === module) {
  const [root, minimumSets, seed, output] = process.argv.slice(2);
  if (!output) throw new Error('usage: gen3_generator_sample.cjs ROOT MINIMUM_SETS SEED OUTPUT');
  fs.writeFileSync(output, JSON.stringify(sample(root, Number(minimumSets), seed)) + '\n');
}
module.exports = {sample};
