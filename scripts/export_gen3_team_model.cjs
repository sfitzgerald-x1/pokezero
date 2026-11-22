#!/usr/bin/env node
'use strict';

const fs = require('node:fs');
const {PIN, EXECUTED_PIN, loadGenerator, sha256} = require('./gen3_generator_weights.cjs');

function exportModel(root) {
  const generator = loadGenerator(root);
  if (generator.maxTeamSize !== 6 || generator.forceMonotype || generator.adjustLevel) {
    throw new Error('only the pinned standard six-member format is modeled');
  }
  const [pool, tokens] = generator.getPokemonPool('Normal', [], false, Object.keys(generator.randomSets));
  const types = generator.dex.types.names();
  const bases = Object.entries(pool).map(([name, forms]) => {
    const profiles = forms.map(form => {
      const species = generator.dex.species.get(form);
      return {
        types: species.types.slice().sort(),
        weak: types.filter(type => generator.dex.getEffectiveness(type, species) > 0),
        double_weak: types.filter(type => generator.dex.getEffectiveness(type, species) > 1),
        level100: generator.getLevel(species) === 100,
      };
    });
    if (profiles.some(profile => JSON.stringify(profile) !== JSON.stringify(profiles[0]))) {
      throw new Error(`formes with different legality need rejected-token state: ${name}`);
    }
    const incompatible = Object.entries(pool).filter(([other, otherForms]) => {
      const outcomes = forms.flatMap(form => otherForms.map(otherForm => generator.getPokemonCompatibility(
        generator.dex.species.get(form), [{speciesId: otherForm}])));
      if (outcomes.some(value => value !== outcomes[0])) throw new Error(`forme compatibility differs: ${name}/${other}`);
      return !outcomes[0];
    }).map(([other]) => other);
    return {name, forms, tokens: tokens.filter(value => value === name).length, ...profiles[0], incompatible};
  });
  const payload = {schema: 'gen3-joint-team-model-v1', showdown_commit: PIN, executed_sha256: EXECUTED_PIN,
    slots: 6, types, bases};
  return {...payload, sha256: sha256(JSON.stringify(payload))};
}

module.exports = {exportModel};
if (require.main === module) {
  const [root, output] = process.argv.slice(2);
  if (!output) throw new Error('usage: export_gen3_team_model.cjs BUILT_SHOWDOWN OUTPUT');
  fs.writeFileSync(output, JSON.stringify(exportModel(root)) + '\n');
}
