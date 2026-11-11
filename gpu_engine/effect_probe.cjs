// Showdown side of the per-move two-engine probe (gpu_engine/effect_probe.py).
// Diagnostic only: one Battle per enumerated case, driven turn by turn with the
// same semantic chance tape the JAX engine receives (chance_hooks.cjs, tape v2).
// JSON lines on stdin/stdout; usage: node effect_probe.cjs SHOWDOWN_ROOT
const fs = require('node:fs');
const path = require('node:path');
const readline = require('node:readline');
const rows = require('./team_rows.cjs');

const BOOSTS = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'];
const DEFAULT_EVS = {hp:85, atk:85, def:85, spa:85, spd:85, spe:85};   // gen3 randbats (teams.ts)
const DEFAULT_IVS = {hp:31, atk:31, def:31, spa:31, spd:31, spe:31};

function setup(root) {
  const {Battle} = require(path.join(root, 'dist/sim/battle.js'));
  const {PRNG} = require(path.join(root, 'dist/sim/prng.js'));
  const pool = rows.load(root);
  const levels = JSON.parse(fs.readFileSync(path.join(root, 'dist/data/random-battles/gen3/sets.json'), 'utf8'));
  let tape = null, audit = [], initializing = null;
  require('./chance_hooks.cjs').install(Battle, PRNG, () => tape, row => audit.push(row), () => initializing,
    path.join(root, 'dist/sim/pokemon.js'));
  // Probe-only (fidelity M1): Roar/Whirlwind's getRandomSwitchable samples the drag target. With exactly one
  // switchable candidate the result is forced, so it is audited as inert instead of needing a tape role; the
  // engine's choose_bench forces the same mon. Two or more candidates stay with the hooks (unresolved).
  const hooked = Battle.prototype.sample;
  Battle.prototype.sample = function (items) {
    const move = this.activeMove && this.activeMove.id;
    if (tape && items.length === 1 && (move === 'roar' || move === 'whirlwind')) {
      audit.push({kind:'INERT_FORCED_SAMPLE', move, candidates:1});
      return items[0];
    }
    return hooked.call(this, items);
  };
  return {Battle, pool, levels, setTape(value) {tape = value;}, setInitializing(value) {initializing = value;},
    takeAudit() {const a = audit; audit = []; return a;}};
}

// Authored set -> complete set: the randbats level for the species unless given,
// randbats EVs/IVs unless given. The same object feeds Showdown and the pool row.
function fill(env, set) {
  const species = env.pool.dex.species.get(set.species);
  if (!species.exists) throw new Error(`unknown species: ${set.species}`);
  const level = set.level ?? env.levels[species.id]?.level;
  if (!Number.isInteger(level)) throw new Error(`no randbats level for ${species.id}; give level`);
  const out = {species:species.name, level, moves:set.moves.map(x => env.pool.dex.moves.get(x).id),
    item:set.item || '', ability:set.ability, evs:{...DEFAULT_EVS, ...(set.evs || {})},
    ivs:{...DEFAULT_IVS, ...(set.ivs || {})}};
  if (set.nature) out.nature = set.nature;
  if (set.gender) out.gender = set.gender;
  for (const move of out.moves) if (!env.pool.dex.moves.get(move).exists) throw new Error(`unknown move: ${move}`);
  return out;
}

function choice(battle, side, text) {
  if (!text) return '';
  const [kind, value] = text.split(/\s+/, 2);
  if (kind !== 'switch') return text;
  // Scripts name party members by original (authored) 1-based position.
  const target = Number(value) - 1;
  const index = battle.sides[side].pokemon.findIndex(p => p.pzIndex === target);
  if (index < 0) throw new Error(`no party member ${value}`);
  return `switch ${index + 1}`;
}

function requestMask(battle, side) {
  const s = battle.sides[side], req = s.activeRequest, mask = Array(9).fill(false);
  if (!req || req.wait || battle.ended) return mask;
  const active = s.active[0], order = s.pokemon.slice().sort((a, b) => a.pzIndex - b.pzIndex);
  const bench = order.filter(p => p !== active);
  const forced = !!(req.forceSwitch && req.forceSwitch[0]);
  if (!forced && req.active) {
    const moves = req.active[0].moves || [];
    const slots = active.moveSlots;
    if (moves.length === 1 && (slots.length !== 1 || moves[0].id !== slots[0].id)) {
      // A lock (charge, recharge, Struggle): every slot holding the move, else slot 0.
      let any = false;
      slots.forEach((m, i) => {if (m.id === moves[0].id) {mask[i] = true; any = true;}});
      if (!any) mask[0] = true;
    } else {
      moves.forEach((m, i) => {mask[i] = !m.disabled;});
    }
  }
  const trapped = !forced && !!(req.active && (req.active[0].trapped || req.active[0].maybeTrapped));
  bench.forEach((p, i) => {mask[4 + i] = !trapped && !p.fainted && p.hp > 0;});
  return mask;
}

function originalIndex(pokemon) {return pokemon && pokemon.pzIndex !== undefined ? pokemon.pzIndex : null;}

function snapshot(battle) {
  const sides = battle.sides.map(side => {
    const mons = side.pokemon.slice().sort((a, b) => a.pzIndex - b.pzIndex).map(p => {
      const v = p.volatiles;
      return {hp:p.hp, maxhp:p.maxhp, status:p.status || '', fainted:!!p.fainted,
        status_time:p.status === 'slp' ? (p.statusState.time ?? null) : null,
        tox_stage:p.status === 'tox' ? (p.statusState.stage ?? null) : null,
        boosts:BOOSTS.map(b => p.boosts[b]), item:p.item || '', item_knocked:!!p.itemKnockedOff,
        moves:p.moveSlots.map(m => m.id), pp:p.moveSlots.map(m => m.pp),
        volatiles:{
          confusion:v.confusion ? v.confusion.time : null,
          choicelock:v.choicelock ? v.choicelock.move : null,
          encore:v.encore ? {duration:v.encore.duration ?? null, move:v.encore.move} : null,
          perishsong:v.perishsong ? v.perishsong.duration : null,
          yawn:v.yawn ? v.yawn.duration : null,
          partiallytrapped:v.partiallytrapped ? {duration:v.partiallytrapped.duration,
            source:originalIndex(v.partiallytrapped.source)} : null,
          mustrecharge:v.mustrecharge ? v.mustrecharge.duration : null,
          destinybond:!!v.destinybond,
          substitute:v.substitute ? v.substitute.hp : null,
          names:Object.keys(v).sort(),
        }};
    });
    return {active:originalIndex(side.active[0]), mons, mask:requestMask(battle, side.n)};
  });
  const winner = !battle.ended ? null : battle.winner === 'p1' ? 0 : battle.winner === 'p2' ? 1 : 2;
  return {turn:battle.turn, ended:!!battle.ended, winner, sides, log_tail:battle.log.slice(-40)};
}

function main() {
  const root = process.argv[2];
  if (!root) throw new Error('usage: effect_probe.cjs SHOWDOWN_ROOT');
  const env = setup(root);
  let battles = [];
  const reply = value => process.stdout.write(JSON.stringify(value) + '\n');
  const handlers = {
    prepare(cmd) {
      const sets = cmd.teams.map(team => team.map(set => fill(env, set)));
      const teamRows = sets.map(team => team.map(set => rows.memberRow(env.pool.dex, env.pool.id, set)));
      return {sets, rows:teamRows, vocabulary:env.pool.vocabulary};
    },
    start(cmd) {
      env.setTape(null); env.takeAudit();
      // Start-of-battle randomness uses the seeded PRNG but is audited by the chance
      // hooks: an M/F gender draw is inert unless a set uses Attract or Cute Charm,
      // the end-of-turn Quick Claw roll is inert without a Quick Claw holder, and
      // any other draw (a lead Speed tie) is reported as INITIAL_RANDOM.
      const genderSensitive = cmd.sets.some(team => team.some(set =>
        set.moves.includes('attract') || env.pool.toID(set.ability) === 'cutecharm'));
      env.setInitializing({allow_inert_gender: !genderSensitive});
      try {
        battles = Array.from({length:cmd.cases}, () => {
          const b = new env.Battle({formatid:cmd.format, seed:cmd.seed});
          cmd.sets.forEach((team, s) => b.setPlayer(`p${s + 1}`, {name:`p${s + 1}`, team:team.map(x => ({...x}))}));
          for (const side of b.sides) side.pokemon.forEach((p, i) => {p.pzIndex = i;});
          return b;
        });
      } finally {env.setInitializing(null);}
      return {snapshots:battles.map(snapshot), audit:env.takeAudit()};
    },
    step(cmd) {
      const results = battles.map((b, i) => {
        if (!cmd.live[i]) return null;
        env.setTape(cmd.tapes[i]); env.takeAudit();
        let error = null;
        const ignored = [];
        try {
          // A side Showdown is not asking (wait) gets no choice; a requested side
          // without a scripted choice is a script error, never an automatic one.
          const choices = cmd.choices[i].map((text, s) => {
            const req = b.sides[s].activeRequest, asked = !!req && !req.wait;
            if (!asked) {if (text) ignored.push(s); return '';}
            if (!text) throw new Error(`SCRIPT_MISSING_CHOICE: side ${s} is requested`);
            return choice(b, s, text);
          });
          b.makeChoices(...choices);
        } catch (e) {error = String(e && e.message || e);}
        env.setTape(null);
        return {snapshot:snapshot(b), audit:env.takeAudit(), error, ignored};
      });
      return {results};
    },
  };
  const lines = readline.createInterface({input:process.stdin});
  lines.on('line', line => {
    if (!line.trim()) return;
    let cmd;
    try {
      cmd = JSON.parse(line);
      const handler = handlers[cmd.cmd];
      if (!handler) throw new Error(`unknown command ${cmd.cmd}`);
      reply({ok:true, ...handler(cmd)});
    } catch (e) {reply({ok:false, error:String(e && e.stack || e)});}
  });
}

module.exports = {fill, requestMask, snapshot, choice};
if (require.main === module) main();
