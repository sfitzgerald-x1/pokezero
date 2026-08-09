#!/usr/bin/env python
"""C154: re-adjudicate every UNREACHABLE verdict in the known-gaps ledger's section 4.

WHY THIS EXISTS. C153 adopted a rule for unreachability claims and wrote it into
``scripts/c153_wide_negative_census.py`` above ``CENSUS_CANNOT_REACH``:

  > TRACE THE RAISE SITE TO THE CALLER THAT ACTUALLY REACHES IT, not to a plausible
  > sentence about it.

Applied to that map's seven entries the rule corrected THREE -- one wrong verdict
(``public_effect_blocked``, filed unreachable while ``blocked_slots`` comes from the
production ``_public_effect_signals`` on a live observation) and two wrong reasons
(``deferred_opponent_action`` and ``rest_sleep_refund_pending_precounts_legacy``, both
verdicts right and both stated mechanisms false). A FOURTH instance of the same class
appeared outside that map, in a granularity split written one commit after the rule was
adopted. The class is not confined to the category that earned the rule.

``reports/c138_known_gaps_ledger.md`` section 4 holds 26 UNREACHABLE verdicts (R1-R27,
R26 withdrawn) that had never been through it. They were carried on prior work's word.
At 3-in-7 that is not a formality, and it was not.

⚠ HOW MANY CORRECTIONS THIS PASS FILES IS NOT WRITTEN HERE, and that is deliberate. Two
earlier revisions of this docstring said SEVEN in one paragraph and TEN in another while
the artifact carried THIRTEEN -- the fifth and sixth instances of the defect this PR's own
ledger edits are about, in the generator that produces the number. The count lives in the
artifact, is derived by ``correction_counts()`` below, and is asserted against the verdict
records by ``tests/test_unreachable_readjudication.py``. Every row keeps its UNREACHABLE
verdict and the corrected ones lose only their stated mechanism, which is the
``deferred_opponent_action`` shape rather than the ``public_effect_blocked`` one -- section
4 has no wrong VERDICT that this pass could find.

WHAT THIS SCRIPT IS. One artifact generator. It re-derives, from source and from the
vendored Showdown checkout, every measurement any section 4 row rests on, and pairs each
row with the citations its demonstration needs. ``tests/test_unreachable_readjudication.py``
then holds the ledger against the artifact.

THE THREE VERDICT WORDS, and none of them is "verified unreachable, measured":

  * ``UNREACHABLE_TRACED``      -- with the call path and the specific statement that
                                   forecloses it. ⚠ Every row additionally carries a
                                   ``foreclosure`` field, because the word alone
                                   over-claims: 24 rows are ``ALL_CALLERS`` ("X builds its
                                   own payload at f:N with neither argument and hands it
                                   to Y, which raises at :M, so it cannot fire FOR ANY
                                   CALLER of X"), and three are ``RANDBATS_POPULATION`` --
                                   foreclosed over section 4's stated population and NOT
                                   over every caller. R23's counter fires today on the
                                   scenario corpus. See ``FORECLOSURES``.
  * ``NOT_OBSERVED_AT_SCOPE``   -- reachable in principle, measured zero, with the scope,
                                   the bound and the denominator that MATCHES THE
                                   EMISSION SITE.
  * ``WRONG``                   -- with what actually happens.

CITATIONS ARE RESOLVED, NOT TYPED. C153's final round watched FIFTEEN citations go stale
in a single merge (#1202). Its cure is ``_raise_line`` by AST, ``_anchor`` with
UNIQUENESS REQUIRED, and ``_anchor_after`` for position-only anchors, and this module
IMPORTS those three rather than copying them. Importing also imports that module's
top-level anchor resolution, which is deliberate: a stale anchor anywhere in the traced
set is then one loud failure at import, not two modules disagreeing about a line number.

Three things that machinery learned only by running, all of which bite here too:

  * the first version took the FIRST match and resolved three citations to docstring
    mentions hundreds of lines above the code they meant -- so uniqueness is the default
    and an ambiguous anchor is an error;
  * one refusal reason has THREE raise sites, so a bare citation is inadmissible -- and
    in this pass ``skip:world_unsupported:`` likewise has TWO emission sites, recorded as
    two;
  * a flag occurs SIX times, so an index is as brittle as a literal -- hence
    ``_anchor_after``.

THE DENOMINATOR MATCHES THE EMISSION SITE. Section 4's rows are pool and structural
verdicts, so most need no denominator at all. Four of them (R1, R7, R8, R23) close a
``skip:world_unsupported:<reason>`` counter and one (R2) closes
``world_prestate_mismatch:weather_HAIL``; for those the artifact records the AST-derived
emission granularity and the DENOMINATOR NAME, because a later reader who needs a bound
must not guess between the three scalars in use. All five emit inside ``_prepare_boundary``,
whose refusals fire BEFORE ``boundaries_measured`` increments, so the denominator is
``boundaries_full_round`` -- 658,559 on C153's strict arm against 641,866 measured
boundaries, and quoting the wrong one understated a result by ~80x once already. The
granularity is derived by AST over the enclosing scope and never by loop depth:
``abort:no_legal_action`` is per-game because its statement RETURNS out of ``run_game``,
which a depth heuristic gets backwards.

NO NEW SWEEP WAS RUN, AND NONE WAS NEEDED. Every one of the 26 resolves by tracing or by
pool census; nothing in this pass reduced to "measured zero somewhere". Where a scope
sentence is quoted it is C153's committed strict arm (8,000 games, unregistered seeds
1,001,000-1,008,999), cited as the widest scope the program has, never as a new
measurement.

INSTRUMENTS, from ``reports/c138_known_gaps_ledger.md`` section 1.2, which is where the
choice of instrument is adjudicated and not re-litigated here:

  * is a MOVE reachable?     union of every set's ``movepool`` in
                             ``data/random-battles/gen3/sets.json``
  * is an ABILITY reachable? union of every set's ``abilities`` in the same file
  * is an ITEM reachable?    generation, NOT ``sets.json`` -- a gen3 set has no item
                             field at all
  * does it exist in gen3?   ``Dex.mod('gen3')``

⚠ A WHOLE-POOL "0 of 220" IS SIDE-INDEPENDENT; A PER-SPECIES ONE IS NOT. R26 was wrong
because it scoped a movepool check to TWO SPECIES for a ``target: normal`` move whose
user is the opponent. The failure mode is the SCOPING, not the instrument: if no species
in the pool has the move, no side can use it and no side can be hit by it. Eighteen of
these rows are whole-pool absences and are safe for that reason, which is worth saying
out loud because the rule as written in section 8 reads as though every movepool check
were suspect.

⚠ AND A GENERATION-TIME ITEM CENSUS IS NOT A RUNTIME ONE. R6, R10 and R27 all closed on
"``getItem`` cannot return it", which is a statement about team generation and says
nothing about acquisition in play -- the exact gap that made R26 wrong, in the same
document, about the same mechanic. The item universe is closed at runtime too, but for a
different reason, and this artifact measures it: the pool's only item-moving moves are
``trick`` (2 species) and ``knockoff`` (4), which SWAP and REMOVE; ``thief``, ``covet``,
``recycle``, ``switcheroo`` and ``bugbite`` are each 0 of 220, and no gen3 mechanism
CREATES an item. A closed set stays closed under permutation and deletion.

WHERE THIS ARTIFACT LIVES, AND THE CLAIM THAT PUT IT SOMEWHERE ELSE FIRST.

It lives in ``reports/artifacts/``, inside ``counter_artifacts()``'s glob, so
``tests/test_never_fired_counter_census.py`` shape-checks it on every run like every other
committed measurement.

⚠ A first revision put it in ``tests/data/`` and argued the move at length: the artifact is
keyed by refusal-reason names -- ``nature_not_neutral``, ``weather_unsupported``,
``volatile_unsupported``, ``future_sight_pending`` -- and carries pool counts beside them,
so under ``reports/`` it would supposedly read as four counters firing across the corpus.
**Measured, that is false.** Copied into ``reports/artifacts/`` with
``_EXPECTED_COUNTER_ARTIFACTS`` bumped, the census reports ``Ran 22 tests ... OK``. The
names appear only as string VALUES inside prose, and ``_evidence_in`` admits exactly two
shapes -- a counter name as a token in the leaf's dotted PATH, or C43's string-field-plus-
numeric-sibling -- and its docstring says in terms that "A name merely mentioned inside
prose is NOT evidence", an exclusion that predates this pass and is load-bearing for two
other rows.

So the placement bought nothing, and it COST the guard that census's own header warns
about by name: "A future census written to ``tests/data/`` would leave the corpus and lose
the check with no test going red." One PR earlier a convenience field of exactly that shape
nearly inverted 46 verdicts and the shape-matching census is what caught it. The rule this
leaves behind is the one this whole pass is about: **a hazard asserted is not a hazard
measured**, and it was asserted in the docstring of the instrument built to stop that.

Regenerate with::

    python scripts/c154_unreachable_readjudication.py --write \\
        reports/artifacts/c154_unreachable_readjudication.json

from a machine with a pokemon-showdown checkout resolvable by
``pokezero.local_showdown.default_showdown_root`` (``POKEZERO_SHOWDOWN_ROOT`` wins) and a
built ``dist/``. CI builds no Showdown checkout, so nothing in CI re-derives the pool
half against a live pool; the artifact records the commit it was taken at, exactly as
``scripts/c152_pool_reachability_census.py`` does, so the staleness is bounded and
nameable. The CITATION half needs no checkout and IS re-derived on every run of the pin.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pokezero.local_showdown import default_showdown_root  # noqa: E402

# The citation machinery, IMPORTED rather than reimplemented. See the module docstring.
from c153_wide_negative_census import (  # noqa: E402
    _DENOMINATOR_BY_FUNCTION,
    _anchor,
    _anchor_after,
    _raise_line,
    emission_sites,
)

EW = "src/pokezero/engine_world.py"
LS = "src/pokezero/local_showdown.py"
ETD = "scripts/engine_transition_differential.py"
EV = "rust/pokezero-search/src/events.rs"
TF = "src/pokezero/transitions_fold.py"


# ---------------------------------------------------------------------------
# Emission granularity, for the five rows that close a differential counter.
# ---------------------------------------------------------------------------


def counter_emission(pattern: str) -> dict[str, Any]:
    """Every emission site for ``pattern``, with its derived denominator.

    A LIST, never a single site. ``skip:world_unsupported:`` has TWO increments in
    ``_prepare_boundary`` and ``world_prestate_mismatch`` has a static and a dynamic one;
    a row that cited "the" emission site would be inadmissible for the same reason a
    reason with three raise sites cannot be cited by line.
    """

    sites = [s for s in emission_sites() if s["pattern"] == pattern]
    if not sites:
        raise SystemExit(
            f"no `counts[{pattern!r}]` emission site in the differential. The counter this "
            "row closes has been renamed or removed; re-trace it rather than re-typing it."
        )
    denominators = sorted({_DENOMINATOR_BY_FUNCTION[s["function"]] for s in sites})
    if len(denominators) != 1:
        raise SystemExit(
            f"{pattern!r} emits from functions with different denominators {denominators}; "
            "a single bound is not admissible for it."
        )
    return {
        "pattern": pattern,
        "sites": [
            {
                "function": s["function"],
                "line": s["line"],
                "per_game": s["ends_the_game"],
                "loop_depth": len(s["loops"]),
            }
            for s in sorted(sites, key=lambda s: s["line"])
        ],
        "denominator": denominators[0],
    }


# ---------------------------------------------------------------------------
# The pool census, by node against the vendored Showdown.
# ---------------------------------------------------------------------------

#: Every move id any section 4 row's verdict depends on, with the row that needs it.
#: Held as one flat list so a row cannot quietly stop measuring a move it cites.
POOL_MOVES: tuple[str, ...] = (
    # R1
    "futuresight", "doomdesire",
    # R2 / R4
    "hail", "sandstorm", "raindance", "sunnyday",
    # R6 / R10 / R27 -- runtime item movement, which a generation census does not see
    "trick", "thief", "covet", "knockoff", "recycle", "switcheroo", "bugbite",
    # R10 -- every gen3 drain move, not just the one the row names
    "absorb", "megadrain", "gigadrain", "leechlife", "dreameater", "drainpunch",
    # R11
    "bonemerang",
    # R12 / R13 / R25
    "rest", "bellydrum", "sleeptalk", "haze", "psychup", "roar", "whirlwind", "batonpass",
    # R22 -- the OTHER half of `move_fails_encore`, which the row does not name and which
    # is reachable
    "encore",
    # R14 -- the routes that would strip Wonder Guard, which the row never checked
    "skillswap", "roleplay", "transform",
    # R15
    "magiccoat", "ingrain",
    # R16
    "dragonrage", "psywave", "nightshade",
    # R17
    "eruption", "waterspout",
    # R18
    "outrage", "petaldance", "thrash", "solarbeam", "hyperbeam",
    # R19
    "snore",
    # R20
    "lowkick",
    # R21
    "reflect", "lightscreen", "safeguard", "mist", "spikes",
    # R22
    "mimic", "imprison", "metronome", "assist", "naturepower", "sketch", "mirrormove",
    # R23 -- including `odorsleuth`, the SECOND producer of the `foresight` volatile,
    # which the row does not mention
    "focusenergy", "mudsport", "watersport", "taunt", "torment", "disable", "nightmare",
    "foresight", "odorsleuth",
    # R24
    "attract",
    # R9
    "leechseed",
)

#: Every ability name a section 4 row depends on.
POOL_ABILITIES: tuple[str, ...] = (
    "Rain Dish",      # R3
    "Dry Skin",       # R5
    "Sand Stream",    # R4
    "Snow Warning",   # R2
    "Liquid Ooze",    # R9
    "Shell Armor",    # control: a common ability, so a zero row is not the only shape
    "Cute Charm",     # R24
    "Insomnia",       # R12
    "Vital Spirit",   # R12
    "Wonder Guard",   # R13 / R14
    "Trace",          # R3 -- the only pool route to an ability a set does not list
)

#: Every item id a section 4 row asserts is absent, plus the row that asserts it.
NAMED_ITEMS: tuple[str, ...] = (
    "sitrusberry",    # R6
    "shellbell",      # R10
    "lansatberry",    # R23 -- the SECOND producer of the `focusenergy` volatile
    "chestoberry", "quickclaw", "kingsrock", "brightpowder", "laxincense",  # R27
    "focusband", "scopelens", "berryjuice", "leppaberry", "oranberry",      # R27
    "pechaberry", "rawstberry", "aspearberry", "persimberry", "cheriberry", # R27
    "leftovers",      # control: the item the generator returns on its terminal path
)

#: Volatiles whose every gen3 producer must be enumerated, not assumed. R23 names eight
#: moves and stops; two of the eight volatiles have a producer that is not the
#: same-named move.
NAMED_VOLATILES: tuple[str, ...] = (
    "foresight", "focusenergy", "mudsport", "watersport", "taunt", "torment",
    "disable", "nightmare", "attract", "imprison",
)

#: Abilities and dex entries whose gen3 EXISTENCE is the claim.
NAMED_DEX_ABILITIES: tuple[str, ...] = ("dryskin", "snowwarning", "raindish", "sandstream")

#: R14's defending type. Shedinja is the pool's only species at or below the 47 maxhp
#: ceiling c129 measured, and Wonder Guard is what the row's corrected argument rests on.
WONDER_GUARD_TYPES: tuple[str, str] = ("Bug", "Ghost")

#: The generative census. 4,000 teams = 24,000 Pokemon, the size section 1.3 used, under
#: a DIFFERENT seed scheme -- section 8 requires presence/absence facts to reproduce
#: across schemes and records that two per-species tallies did not.
GENERATION_TEAMS = 4000
GENERATION_SEED_SCHEME = "[1, 2, 3, i] for i in range(4000)"

_NODE = r"""
const {Dex} = require(process.argv[1] + '/dist/sim/dex');
const {Teams} = require(process.argv[1] + '/dist/sim/index');
const fs = require('fs');
const args = JSON.parse(process.argv[2]);
const sets = JSON.parse(fs.readFileSync(
    process.argv[1] + '/data/random-battles/gen3/sets.json', 'utf8'));
const dex = Dex.mod('gen3');

const moveSpecies = new Map();
const abilitySpecies = new Map();
const setRows = [];
for (const species of Object.keys(sets)) {
  for (const set of sets[species].sets) {
    setRows.push({species, moves: set.movepool || [], abilities: set.abilities || []});
    for (const m of (set.movepool || [])) {
      if (!moveSpecies.has(m)) moveSpecies.set(m, new Set());
      moveSpecies.get(m).add(species);
    }
    for (const a of (set.abilities || [])) {
      if (!abilitySpecies.has(a)) abilitySpecies.set(a, new Set());
      abilitySpecies.get(a).add(species);
    }
  }
}
const speciesFor = id => moveSpecies.has(id) ? [...moveSpecies.get(id)].sort() : [];

// --- moves and abilities, as SPECIES LISTS, not bare counts: a row that names its
// carriers (R3 Ludicolo, R4 Tyranitar, R9 Swalot/Tentacruel) must be checkable.
const move_species = {};
for (const id of args.moves) move_species[id] = speciesFor(id);
const ability_species = {};
for (const name of args.abilities) {
  ability_species[name] = abilitySpecies.has(name) ? [...abilitySpecies.get(name)].sort() : [];
}

// --- per-SET co-occurrence, which is the instrument for a gap needing two things at
// once. Section 8's first added clause.
const setsWith = pred => setRows.filter(pred).length;
const co_occurrence = {
  rest_sets: setsWith(r => r.moves.includes('rest')),
  rest_with_insomnia_or_vital_spirit: setsWith(r => r.moves.includes('rest')
      && r.abilities.some(a => a === 'Insomnia' || a === 'Vital Spirit')),
  sleeptalk_sets: setsWith(r => r.moves.includes('sleeptalk')),
  sleeptalk_with_haze_psychup_roar_whirlwind_batonpass: setsWith(r =>
      r.moves.includes('sleeptalk')
      && r.moves.some(m => ['haze','psychup','roar','whirlwind','batonpass'].includes(m))),
  bellydrum_sets: setsWith(r => r.moves.includes('bellydrum')),
  // R10. The `heal_drain_or_shellbell` bucket's ONLY production emitter is the Sleep
  // Talk ambiguous-tail path, so the bucket needs `sleeptalk` and a drain move on the
  // SAME set -- a requirement the row never states.
  drain_sets: setsWith(r => r.moves.some(m => args.drain_moves.includes(m))),
  sleeptalk_with_a_drain_move: setsWith(r => r.moves.includes('sleeptalk')
      && r.moves.some(m => args.drain_moves.includes(m))),
  // R3 / R4. Carrier-level counts, so a row that names its carriers is checkable.
  tyranitar_sets: setsWith(r => r.species === 'tyranitar'),
  tyranitar_sets_with_sand_stream: setsWith(r => r.species === 'tyranitar'
      && r.abilities.includes('Sand Stream')),
  ludicolo_sets: setsWith(r => r.species === 'ludicolo'),
  ludicolo_sets_with_rain_dish: setsWith(r => r.species === 'ludicolo'
      && r.abilities.includes('Rain Dish')),
  shedinja_sets_with_bellydrum: setsWith(r => r.species === 'shedinja'
      && r.moves.includes('bellydrum')),
};

// --- move classes over the POOL's move union, by the real dex.
const poolMoves = [...moveSpecies.keys()].sort();
const move_classes = {multihit: [], fixed_damage: [], ohko: [], partiallytrapped: [], drain: []};
for (const id of poolMoves) {
  const mv = dex.moves.get(id);
  if (mv.multihit) move_classes.multihit.push([id, JSON.stringify(mv.multihit)]);
  if (mv.damage) move_classes.fixed_damage.push(id);
  if (mv.ohko) move_classes.ohko.push(id);
  if (mv.volatileStatus === 'partiallytrapped') move_classes.partiallytrapped.push(id);
  if (mv.drain) move_classes.drain.push(id);
}

// --- R14. Every pool move SUPER-EFFECTIVE against Wonder Guard's defending types, with
// its multihit. The row's corrected argument is that every one of them is single-hit.
const [t1, t2] = args.wonder_guard_types;
const super_effective = [];
for (const id of poolMoves) {
  const mv = dex.moves.get(id);
  if (mv.category === 'Status') continue;
  if (!dex.getImmunity(mv.type, [t1, t2]) ) continue;
  if (dex.getEffectiveness(mv.type, [t1, t2]) > 0) {
    super_effective.push([id, mv.type, mv.multihit ? JSON.stringify(mv.multihit) : null]);
  }
}
const bonemerang = dex.moves.get('bonemerang');

// --- R22. Showdown's own `failencore` flag, which is what `encore.condition.onStart`
// tests. `Future` entries are excluded: they do not exist in a gen3 battle, so including
// them would make the set equality against the crate's arm fail for the wrong reason.
const failencore_flagged_gen3 = [];
for (const mv of dex.moves.all()) {
  if (mv.flags && mv.flags.failencore && mv.isNonstandard !== 'Future') {
    failencore_flagged_gen3.push(mv.id.toUpperCase());
  }
}
failencore_flagged_gen3.sort();

// --- gen3 EXISTENCE, which is a different question from pool membership.
const dex_abilities = {};
for (const id of args.dex_abilities) {
  const a = dex.abilities.get(id);
  dex_abilities[id] = {exists: !!a.exists, gen: a.gen || null, isNonstandard: a.isNonstandard || null};
}
const dex_items = {};
for (const id of args.items) {
  const it = dex.items.get(id);
  dex_items[id] = {exists: !!it.exists, gen: it.gen || null, isNonstandard: it.isNonstandard || null};
}

// --- every gen3 species carrying an ability, and which of them are in the pool. R3's
// species-level check, generalised so R3 is not the only row that gets one.
const gen3_carriers = {};
for (const name of args.abilities) {
  const carriers = [];
  for (const sp of dex.species.all()) {
    if (sp.isNonstandard) continue;
    if (Object.values(sp.abilities || {}).includes(name)) carriers.push(sp.name);
  }
  gen3_carriers[name] = {
    all: carriers.sort(),
    in_pool: carriers.filter(n => sets[dex.species.get(n).id]).sort(),
  };
}

// --- every gen3 producer of each named volatile, from the DEX DATA (`volatileStatus`).
// Handler-installed volatiles are NOT visible here -- `JSON.stringify` drops functions,
// and a first version of this scan silently returned nothing for every ability and item
// because of exactly that. The source-text scan that covers them is done in Python and
// recorded beside this, under `volatile_producers_by_source`.
const volatile_producers_by_data = {};
for (const v of args.volatiles) volatile_producers_by_data[v] = [];
for (const mv of dex.moves.all()) {
  if (mv.volatileStatus && volatile_producers_by_data[mv.volatileStatus]) {
    volatile_producers_by_data[mv.volatileStatus].push(mv.id);
  }
}
for (const v of args.volatiles) volatile_producers_by_data[v].sort();

// --- moves that set weather, so "the only ability that sets hail" has a partner claim.
const weather_setters = {};
for (const mv of dex.moves.all()) {
  if (!mv.weather) continue;
  const w = String(mv.weather).toLowerCase();
  (weather_setters[w] = weather_setters[w] || []).push(mv.id);
}

// --- move targets. A `target: normal` move is used BY THE OPPONENT; R26 was wrong for
// want of this column.
const targets = {};
for (const id of args.moves) {
  const mv = dex.moves.get(id);
  if (mv.exists) targets[id] = mv.target;
}

// --- the GENERATIVE census: items, natures, levels. `sets.json` has no item field, so
// this is the only instrument that answers R6/R10/R27, and it is the instrument section
// 1.2 names for items.
const items = {};
let pokemon = 0, nature_set = 0, nature_unset = 0;
let level_min = 1e9, level_max = 0, maxhp_min = 1e9;
const low_maxhp = {};
for (let i = 0; i < args.teams; i++) {
  const team = Teams.generate('gen3randombattle', {seed: [1, 2, 3, i]});
  for (const s of team) {
    pokemon++;
    const item = s.item || '(none)';
    items[item] = (items[item] || 0) + 1;
    if (s.nature && String(s.nature).trim()) nature_set++; else nature_unset++;
    if (s.level < level_min) level_min = s.level;
    if (s.level > level_max) level_max = s.level;
    const sp = dex.species.get(s.species);
    const bh = sp.baseStats.hp;
    const iv = (s.ivs && s.ivs.hp !== undefined) ? s.ivs.hp : 31;
    const ev = (s.evs && s.evs.hp) || 0;
    const maxhp = bh === 1 ? 1
        : Math.floor((2 * bh + iv + Math.floor(ev / 4)) * s.level / 100) + s.level + 10;
    if (maxhp < maxhp_min) maxhp_min = maxhp;
    if (maxhp <= args.maxhp_ceiling) low_maxhp[sp.name] = (low_maxhp[sp.name] || 0) + 1;
  }
}

process.stdout.write(JSON.stringify({
  species: Object.keys(sets).length,
  sets: setRows.length,
  distinct_moves: moveSpecies.size,
  distinct_abilities: abilitySpecies.size,
  move_species,
  ability_species,
  co_occurrence,
  move_classes,
  super_effective_against_wonder_guard: super_effective,
  bonemerang_effectiveness: dex.getEffectiveness('Ground', [t1, t2]),
  bonemerang_immune: !dex.getImmunity('Ground', [t1, t2]),
  bonemerang_multihit: bonemerang.multihit,
  failencore_flagged_gen3,
  shedinja_movepool: (sets['shedinja'] ? sets['shedinja'].sets.map(s => s.movepool) : null),
  dex_abilities,
  dex_items,
  gen3_carriers,
  volatile_producers_by_data,
  weather_setters,
  targets,
  generative: {
    teams: args.teams,
    pokemon,
    items,
    distinct_items: Object.keys(items).length,
    nature_set,
    nature_unset,
    level_min,
    level_max,
    maxhp_min,
    species_at_or_below_maxhp_ceiling: low_maxhp,
    maxhp_ceiling: args.maxhp_ceiling,
  },
}));
"""

#: Every gen3 drain move, so R10's "the pool's only drain move is gigadrain" is a
#: measured complement rather than an assertion.
GEN3_DRAIN_MOVES = ("absorb", "megadrain", "gigadrain", "leechlife", "dreameater", "drainpunch")

#: c129 measured every N5 overshoot state at or below this max HP. R14's whole argument
#: is that Shedinja is the pool's only species that reaches it.
N5_MAXHP_CEILING = 47


#: Where the R10 caller graph has to terminate. Named rather than discovered, so a route
#: that reaches `heal_subcase` from somewhere ELSE is a loud failure instead of a quietly
#: wider graph.
HEAL_SUBCASE_ROOT = "render_move_phase"


def rust_call_graph(relative: str, target: str, root: str) -> dict[str, Any]:
    """Every PRODUCTION path from `root` to `target` in a Rust file, by reverse reachability.

    ⚠ THIS EXISTS BECAUSE R10'S CORRECTION MADE R10'S MISTAKE. The correction opens "this
    cell reasoned from its NAME without tracing its caller" and then asserted, untraced,
    that `heal_subcase` is reached ONLY through `ambiguous_unrenderable_slug_with_protect`.
    There are TWO routes and review found the second: the `sleeptalk_refusal_is_unsafe_
    with_protect` predicate at the head of the same block reaches it as well. The
    CONCLUSION survives -- both roots are `render_move_phase`'s Sleep Talk block and only
    one of the two emits a slug -- but a sentence nothing re-derives is exactly what this
    pass exists to remove, so the graph is derived here and pinned.

    Callers are attributed to the enclosing top-level `fn`, and everything at or after
    `mod tests` is excluded: five of the six functions on this graph have a thin non-
    `_with_protect` wrapper whose only callers are tests, and counting those would report a
    production graph that does not exist.
    """

    lines = (ROOT / relative).read_text(encoding="utf-8").splitlines()
    try:
        test_start = next(n for n, line in enumerate(lines, 1) if line.strip().startswith("mod tests"))
    except StopIteration:
        raise SystemExit(f"no `mod tests` boundary in {relative}; the production filter is inert")
    defs = [
        (n, m.group(1))
        for n, line in enumerate(lines, 1)
        if (m := re.match(r"(?:pub )?fn (\w+)", line))
    ]

    def enclosing(number: int) -> str | None:
        found = None
        for n, name in defs:
            if n <= number:
                found = name
            else:
                break
        return found

    def callers(name: str) -> list[dict[str, Any]]:
        pattern = re.compile(r"(?<![\w])" + re.escape(name) + r"\s*\(")
        out = []
        for n, line in enumerate(lines, 1):
            if n >= test_start or "///" in line or line.lstrip().startswith("//"):
                continue
            if re.match(r"\s*(?:pub )?fn " + re.escape(name) + r"\s*\(", line):
                continue
            if pattern.search(line):
                out.append({"line": n, "in": enclosing(n)})
        return out

    edges: dict[str, list[dict[str, Any]]] = {}
    frontier, seen = [target], {target}
    while frontier:
        current = frontier.pop()
        found = callers(current)
        edges[current] = found
        for site in found:
            parent = site["in"]
            if parent and parent != current and parent not in seen:
                seen.add(parent)
                frontier.append(parent)
    roots = sorted(name for name, found in edges.items() if not found)
    # ⚠ CLASSIFY THE ROOTS, or "roots" is as vague as the sentence this replaces. Five of
    # them are thin non-`_with_protect` wrappers whose only callers are tests: they are
    # graph roots because they are DEAD in production, not because they are entry points.
    # Discriminated structurally -- a dead wrapper's `<name>_with_protect` twin is also on
    # the graph -- rather than by matching on the suffix, which would also accept a real
    # entry point that happened to be named that way.
    dead_wrappers = sorted(r for r in roots if f"{r}_with_protect" in edges)
    live_entry_points = sorted(set(roots) - set(dead_wrappers))
    # Edges FROM the chokepoint INTO the subgraph. `callee != root` drops
    # `render_move_phase`'s own tail recursion at :2058, which is not a way in -- a first
    # version reported three edges and the sentence said two.
    through_root = sorted(
        site["line"]
        for callee, found in edges.items()
        for site in found
        if site["in"] == root and callee != root
    )
    return {
        "target": target,
        "edges": {k: edges[k] for k in sorted(edges)},
        "roots": roots,
        "dead_wrappers_with_no_production_caller": dead_wrappers,
        "live_entry_points": live_entry_points,
        "chokepoint": root,
        "edges_out_of_the_chokepoint": through_root,
    }


def _showdown_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _volatile_producers_by_source(root: Path) -> dict[str, list[str]]:
    """Every ``addVolatile('<name>')`` in Showdown's gen3-relevant data and sim.

    ⚠ THIS EXISTS BECAUSE THE OBVIOUS VERSION ASSERTED NOTHING. A first pass enumerated
    producers by walking ``Dex.mod('gen3')`` and regexing ``JSON.stringify`` of each
    ability and item. ``JSON.stringify`` DROPS FUNCTIONS, so every handler-installed
    volatile -- which is all of them, on abilities and items -- came back empty, and the
    scan "confirmed" that the same-named move is the only producer of all ten. It is not:
    Cute Charm installs ``attract`` and Lansat Berry installs ``focusenergy``, both from
    handler bodies. A scan that can only find what is already in the data field is a
    check that cannot fail, and this repo has shipped ten of those.

    Later-generation mods are excluded because a gen3 battle never loads them; ``data/``
    itself is the gen3 inheritance base and ``data/mods/gen3/`` its override, so both are
    in scope. Directories are recorded on the result so the glob is never quoted wider
    than it was run.
    """

    roots = [root / "data", root / "sim", root / "data" / "mods" / "gen3"]
    skip = re.compile(r"/mods/(?!gen3(/|$))")
    found: dict[str, set[str]] = {name: set() for name in NAMED_VOLATILES}
    patterns = {
        name: re.compile(r"addVolatile\(\s*['\"]" + re.escape(name) + r"['\"]")
        for name in NAMED_VOLATILES
    }
    seen: set[Path] = set()
    for base in roots:
        for path in sorted(base.rglob("*.ts")):
            if path in seen or skip.search(path.as_posix()):
                continue
            seen.add(path)
            text = path.read_text(encoding="utf-8", errors="replace")
            for name, pattern in patterns.items():
                for match in pattern.finditer(text):
                    line = text.count("\n", 0, match.start()) + 1
                    found[name].add(f"{path.relative_to(root).as_posix()}:{line}")
    if not seen:
        raise SystemExit(
            f"the addVolatile scan walked no files under {root}; it would have reported "
            "every volatile as having no producer, which is the vacuous-green shape it "
            "exists to avoid."
        )
    return {name: sorted(hits) for name, hits in sorted(found.items())}


def census(root: Path) -> dict[str, Any]:
    if not (root / "data" / "random-battles" / "gen3" / "sets.json").is_file():
        raise SystemExit(
            f"ERROR: no gen3 randbats sets.json under {root}. Set POKEZERO_SHOWDOWN_ROOT "
            "to a pokemon-showdown checkout."
        )
    if not (root / "dist" / "sim" / "dex.js").is_file():
        raise SystemExit(
            f"ERROR: {root} has no built dist/sim/dex.js. Move classification and team "
            "generation need the real dex."
        )
    payload = json.dumps(
        {
            "moves": list(POOL_MOVES),
            "abilities": list(POOL_ABILITIES),
            "items": list(NAMED_ITEMS),
            "volatiles": list(NAMED_VOLATILES),
            "dex_abilities": list(NAMED_DEX_ABILITIES),
            "wonder_guard_types": list(WONDER_GUARD_TYPES),
            "drain_moves": list(GEN3_DRAIN_MOVES),
            "teams": GENERATION_TEAMS,
            "maxhp_ceiling": N5_MAXHP_CEILING,
        }
    )
    result = subprocess.run(
        ["node", "-e", _NODE, "--", str(root), payload],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise SystemExit(f"ERROR: census failed\n{result.stdout}\n{result.stderr}")
    out = json.loads(result.stdout)
    out["volatile_producers_by_source"] = _volatile_producers_by_source(root)
    out["heal_subcase_call_graph"] = rust_call_graph(EV, "heal_subcase", HEAL_SUBCASE_ROOT)
    out["volatile_source_scan_roots"] = ["data/**/*.ts (mods: gen3 only)", "sim/**/*.ts"]
    out["showdown_commit"] = _showdown_commit(root)
    out["generation_seed_scheme"] = GENERATION_SEED_SCHEME
    return out


# ---------------------------------------------------------------------------
# The 26 verdicts.
# ---------------------------------------------------------------------------

#: The only three verdict words. A row that needs a fourth is a row this pass got wrong.
VERDICTS = ("UNREACHABLE_TRACED", "NOT_OBSERVED_AT_SCOPE", "WRONG")

#: ⚠ THE SCOPE OF THE FORECLOSURE, and it is not decoration.
#:
#: `UNREACHABLE_TRACED` was first documented as "cannot fire FOR ANY CALLER of X", which is
#: the shape the `deferred_opponent_action` demonstration has and the shape a reader will
#: assume. It is WRONG for at least two rows and the pass shipped saying otherwise.
#: `volatile_unsupported` DOES fire, today, for a caller -- `struggle_taunt_stall`, a
#: hand-written Custom Game fixture -- and R1's raise is one keyword argument from firing.
#: Section 4's own population is "cannot be reached in gen3 randbats", so both verdicts are
#: right; the WORD was over-claiming.
#:
#: Recorded per row rather than fixed by softening the definition for all 26, because 24 of
#: them really are all-callers foreclosures and flattening that would throw away the
#: stronger result. Refusing a fourth VERDICT category and then quietly widening the third
#: is the same error in a different place.
FORECLOSURES = ("ALL_CALLERS", "RANDBATS_POPULATION")

#: The rows whose foreclosure holds only over section 4's population. Each is a row whose
#: correction records a live firing or a committed construction outside sampled randbats.
NARROW_FORECLOSURE = {
    "R1": (
        "constructible from committed code -- `golden_corpus_scenarios.py`'s "
        "`future_sight_pending` spec -- and foreclosed only by which spec list the harnesses "
        "default to"
    ),
    "R23": (
        "`volatile_unsupported: side 'p1': ['taunt']` FIRES today on the scenario corpus "
        "(`struggle_taunt_stall`); the counter is already filed REACHABLE at H5b"
    ),
    "R24": (
        "the scenario corpus produces the `attract` volatile from the move itself "
        "(`attract_snorlax`)"
    ),
}

#: How the LEDGER'S OWN stated reason fared, which is a separate judgement from the
#: verdict and is the one that moved here. C153's three-in-seven were one wrong verdict and
#: two wrong reasons; this pass found no wrong verdict and a number of wrong-or-incomplete
#: reasons that is DERIVED rather than typed -- see `correction_counts()`, and see the
#: docstring for the two stale figures that were typed here before it existed. Without the
#: distinction the result reads as "26 confirmed".
REASON_STATUSES = ("SOUND", "INCOMPLETE", "FALSE")

#: The engine source under `third_party/poke-engine-src/` is GITIGNORED and regenerated,
#: so section 1.3 cites it by SYMBOL. A symbol citation cannot be resolved by `_anchor`,
#: so where a row rests on it the demonstration names the symbol AND the committed patch
#: that introduces it -- and the patch IS anchored, so the citation still breaks loudly if
#: the patch stops carrying the code.
PATCHES = "third_party"


def _patch_match_arm(name: str, needle: str) -> tuple[str, ...]:
    """The `Choices::X | Choices::Y | ...` alternatives of the `matches!` under `needle`.

    Returned as a SET so a comparison against gen3's `failencore`-flagged move set is an
    equality rather than a membership sweep. Reading the added (`+`) lines of the patch
    rather than the materialized tree, because `third_party/poke-engine-src/` is gitignored
    and regenerated -- so this is committed evidence, which a symbol citation is not.
    """

    text = (ROOT / f"{PATCHES}/{name}").read_text(encoding="utf-8")
    start = text.index(needle)
    # From the `matches!(` to its closing paren, NOT from the fn signature -- whose own
    # `(move_id: &Choices)` closes first and made a first version of this read an empty arm.
    opening = text.index("matches!(", start)
    closing = text.index("\n+    )", opening)
    arm = text[opening:closing]
    # `+` lines only: the arm is what the patch ADDS, and a context line that merely
    # mentions a `Choices::` variant is not part of the shipped set.
    added = "\n".join(line for line in arm.splitlines() if line.startswith("+"))
    found = tuple(sorted(set(re.findall(r"Choices::(\w+)", added))))
    if not found:
        raise SystemExit(
            f"no `Choices::` alternatives found under {needle!r} in {name}; the arm has "
            "moved and the set-equality claim on it is unverified, not merely unchecked."
        )
    return found


def _patch(name: str, needle: str) -> str:
    """`<patch>:<line>` for `needle` in a committed poke-engine patch."""

    return f"{PATCHES}/{name}:{_anchor(f'{PATCHES}/{name}', needle)}"


def _weather_id_keys() -> list[str]:
    """`_WEATHER_IDS`'s KEYS, imported rather than transcribed -- which is the whole point
    of R8's correction, so retyping them here would repeat the error being corrected."""

    from pokezero.engine_world import _WEATHER_IDS  # noqa: PLC0415

    return list(_WEATHER_IDS)


def correction_counts(records: dict[str, Any]) -> dict[str, int]:
    """The corrections tally, DERIVED from the records rather than typed in a docstring.

    ⚠ Two revisions of this module's docstring said SEVEN in one paragraph and TEN in
    another while the artifact carried THIRTEEN -- the fifth and sixth instances of the
    exact defect this pass files against the ledger, inside the generator that produces the
    number. There is now one place the count exists and a pin that holds the prose to it.
    R26 is excluded from `rows_corrected` because it was withdrawn before this pass and has
    no verdict of its own.
    """

    live = {k: v for k, v in records.items() if k != "R26"}
    tally: dict[str, int] = {"rows": len(live)}
    for status in REASON_STATUSES:
        tally[status.lower()] = sum(
            1 for v in live.values() if v["ledger_reason_status"] == status
        )
    tally["rows_corrected"] = tally["incomplete"] + tally["false"]
    tally["all_callers"] = sum(1 for v in live.values() if v["foreclosure"] == "ALL_CALLERS")
    tally["randbats_population"] = sum(
        1 for v in live.values() if v["foreclosure"] == "RANDBATS_POPULATION"
    )
    return tally


def build_verdicts(pool: dict[str, Any]) -> dict[str, Any]:
    """One record per section 4 row, with every citation RESOLVED at build time."""

    moves = pool["move_species"]
    abilities = pool["ability_species"]
    co = pool["co_occurrence"]
    gen = pool["generative"]

    def absent(*ids: str) -> str:
        """`0 of 220` for each id, WITH the denominator, or a loud failure."""

        present = {i: moves[i] for i in ids if moves[i]}
        if present:
            raise SystemExit(
                f"a row asserts these moves are absent from the pool and they are NOT: "
                f"{present}. Re-adjudicate the row; do not relax the assertion."
            )
        listing = ", ".join(f"`{i}`" for i in ids)
        return f"{listing} {'are' if len(ids) > 1 else 'is'} each **0 of {pool['species']}** species"

    # The five rows that close a differential counter. Both `skip:world_unsupported:`
    # sites and both `world_prestate_mismatch` sites are recorded; a single citation
    # would be inadmissible.
    world_unsupported = counter_emission("skip:world_unsupported:")
    prestate = counter_emission("world_prestate_mismatch:")

    # --- resolved citations, every one of them ----------------------------------
    reject_globals = _raise_line(EW, "future_sight_pending")
    world_payload = _anchor(EW, "payload = _public_materialization_payload(state)")
    reject_call = _anchor(EW, "_reject_unsupported_globals(payload)")
    fs_payload = _anchor(LS, '"futureSight": dict(replay.future_sight),')
    fs_write = _anchor("src/pokezero/showdown.py", "future_sight[target_side] = turn_number + _FUTURE_SIGHT_DELAY")
    fs_gate = _anchor("src/pokezero/showdown.py", "if _side_condition_identifier(parts[3]) not in _FUTURE_MOVES:")

    weather_raise = _raise_line(EW, "weather_unsupported")
    weather_map = _anchor(EW, "_WEATHER_IDS = {")
    weather_lookup = _anchor(EW, "weather = _WEATHER_IDS.get(weather_id)")
    weather_payload = _anchor(LS, '"weather": replay.weather,')
    weather_parse = _anchor("src/pokezero/showdown.py", "def _update_weather(parts: Sequence[str], weather: Optional[str]) -> Optional[str]:")
    weather_norm = _anchor("src/pokezero/showdown.py", "def _normalize_identifier(value: str) -> str:")

    nature_raise = _raise_line(EW, "nature_not_neutral")
    nature_read = _anchor(EW, 'nature = normalize_id(mon.nature) if mon.nature else ""')
    neutral_set = _anchor(EW, "_NEUTRAL_NATURES = frozenset({")
    unpack = _anchor(EW, "nature=nature or \"\",")
    packed_source = _anchor(ETD, "packed = {slot: true_teams[slot]")

    volatiles_read = _anchor(EW, 'volatiles = [normalize_id(str(v)) for v in side_payload.get("volatiles") or ()]')
    volatiles_diff = _anchor(EW, "unsupported = sorted(set(volatiles) - supported)")
    volatiles_raise = _raise_line(EW, "volatile_unsupported")
    supported_set = _anchor(EW, "_SUPPORTED_VOLATILES = frozenset({")
    tracked_set = _anchor("src/pokezero/showdown.py", "TRACKED_VOLATILES = frozenset({")
    tracked_gate = _anchor("src/pokezero/showdown.py", "if name not in TRACKED_VOLATILES:")
    volatiles_payload = _anchor(LS, '"volatiles": list(replay.volatiles.get(player, ())),')

    side_cond_raise = _raise_line(EW, "side_condition_unsupported")
    side_cond_map = _anchor(EW, "_SIDE_CONDITION_IDS = {")
    side_cond_payload = _anchor(LS, '"sideConditions": dict(replay.side_condition_counts.get(player, {})),')

    render_fn = _anchor(EV, "fn render_residual_instruction(")
    # POSITION-ONLY: `if heal.heal_amount < 0 {` occurs TWICE in events.rs -- once in the
    # move-phase renderer and once here -- and an occurrence index would be as brittle as a
    # literal, so it is anchored relative to the function that contains it.
    ooze_intercept = _anchor_after(EV, "if heal.heal_amount < 0 {", render_fn)
    heal_fallback = _anchor(EV, ".unwrap_or_else(|| residual_heal_cause(sim.state, side, next_ins));")
    cause_fn = _anchor(EV, "fn residual_heal_cause(")
    cause_leftovers = _anchor(EV, "if s.get_active_immutable().item == Items::LEFTOVERS {")
    cause_wish = _anchor(EV, 'return "move: Wish".to_string();')
    # POSITION-ONLY, and this is the citation the row got wrong: the SAME conjunct text
    # appears twice, once in `ResidualPlan`'s `drains_opponent` (live, pinned in both
    # directions) and once in `residual_heal_cause`. Citing "the Liquid Ooze guard" without
    # saying which is precisely how the row collapsed two guards into one.
    ooze_guard = _anchor_after(
        EV, "&& opponent.get_active_immutable().ability != Abilities::LIQUIDOOZE", cause_fn
    )
    subcase_fn = _anchor(EV, "fn heal_subcase(tail: &[Instruction], index: usize, attacker: SideReference) -> &'static str {")
    subcase_bucket = _anchor(EV, '} else if tail_damages_the_foe(tail, attacker) {')

    gc = "src/pokezero/golden_corpus_scenarios.py"
    fs_scenario = _anchor(gc, '"future_sight_pending",')
    registry_fn = _anchor(gc, "def interaction_registry_specs() -> tuple[ScenarioSpec, ...]:")
    corpus_fn = _anchor(gc, "def scenario_specs() -> tuple[ScenarioSpec, ...]:")
    fallback_fn = _anchor(gc, "def run_scenario_fallback_sweep(")
    rv = "src/pokezero/randbat_vocab.py"
    item_vocab = _anchor(rv, "GEN3_RANDBAT_ITEMS = (")
    sleeptalk_emit = _anchor(EV, "&ambiguous_unrenderable_slug_with_protect(")
    sleeptalk_guard = _anchor(EV, "if !sleeptalk_refusal_is_unsafe_with_protect(")
    ooze_plan_guard = _anchor(EV, "drains_opponent[i] = opponent")
    ooze_pin = _anchor(EV, "fn liquid_ooze_on_the_seeder_means_a_heal_here_is_not_the_drain() {")
    move_phase_ooze = _anchor(EV, "[from] ability: Liquid Ooze|[of] {defender_ident}")
    weather_none_return = _anchor(EW, 'return "none", -1')
    ss = "src/pokezero/scenario_studio/domain.py"
    scenario_weather_ids = _anchor(ss, "SCENARIO_WEATHER_IDS = (")
    scenario_nature = _anchor(ss, 'nature=self.nature,')
    fidelity_caller = _anchor("src/pokezero/engine_fidelity.py", "_build_pokemon_spec(mon, None, dex=dex, slot=slot, is_self=False)")
    scenario_sidestart = _anchor(LS, 'lines.append(f"|-sidestart|{player}: scenario|{name}")')

    hidden_flags = _anchor(EW, '_HIDDEN_INFORMATION_REQUEST_FLAGS = frozenset({"maybeTrapped"')
    hidden_filter = _anchor(EW, 'if flag not in _HIDDEN_INFORMATION_REQUEST_FLAGS and flag != "trapped"')
    failencore_patch = _patch("poke-engine-gen3-encore-failencore.patch", "fn move_fails_encore")
    # ⚠ PARSED, not listed. The report claimed this set equality was machine-checked while
    # the pin asserted six MEMBERSHIPS -- so adding `Choices::TACKLE` to the patch's match
    # arm left the module green, which review demonstrated. A membership check on a set
    # equality claim is a check that cannot fail in the direction that matters.
    failencore_shipped = _patch_match_arm(
        "poke-engine-gen3-encore-failencore.patch", "fn move_fails_encore"
    )

    records: dict[str, Any] = {}

    def row(
        name: str,
        candidate: str,
        verdict: str,
        reason_status: str,
        demonstration: str,
        instruments: tuple[str, ...],
        measurements: dict[str, Any],
        correction: str | None = None,
        counter: dict[str, Any] | None = None,
    ) -> None:
        if verdict not in VERDICTS:
            raise SystemExit(f"{name}: {verdict!r} is not one of {VERDICTS}")
        if reason_status not in REASON_STATUSES:
            raise SystemExit(f"{name}: {reason_status!r} is not one of {REASON_STATUSES}")
        if reason_status != "SOUND" and not correction:
            raise SystemExit(
                f"{name}: reason status {reason_status} with no correction recorded. A "
                "row whose stated mechanism is wrong and whose replacement is not written "
                "down is the shape this whole pass exists to remove."
            )
        records[name] = {
            "row": name,
            "candidate": candidate,
            "verdict": verdict,
            "foreclosure": "RANDBATS_POPULATION" if name in NARROW_FORECLOSURE else "ALL_CALLERS",
            "foreclosure_note": NARROW_FORECLOSURE.get(name),
            "ledger_reason_status": reason_status,
            "correction": correction,
            "demonstration": demonstration,
            "instruments": list(instruments),
            "measurements": measurements,
            **({"counter": counter} if counter else {}),
        }

    # -- R1 --------------------------------------------------------------------
    row(
        "R1",
        "Future Sight / Doom Desire -- residual order 11 and the `future_sight_pending` refusal",
        "UNREACHABLE_TRACED",
        "INCOMPLETE",
        f"`_reject_unsupported_globals` raises at `{EW}:{reject_globals}` when "
        "`payload['futureSight']` has any nonzero value. THE PAYLOAD IS NOT THE "
        "DIFFERENTIAL'S: `world_battle_spec` builds its own at "
        f"`{EW}:{world_payload}` and reaches the guard at `:{reject_call}`, and "
        f"`_public_materialization_payload` fills the key from `replay.future_sight` at "
        f"`{LS}:{fs_payload}`. That mapping has exactly one writer, "
        f"`_update_future_sight` at `src/pokezero/showdown.py:{fs_write}`, gated at "
        f"`:{fs_gate}` on the protocol effect id being in `_FUTURE_MOVES` = "
        "`{futuresight, doomdesire}`. Both are 0 of "
        f"{pool['species']} species, so the mapping is EMPTY for every gen3 randbats "
        "battle and an empty dict makes `any(...)` false -- the guard cannot fire FOR ANY "
        "CALLER of `world_battle_spec`. Residual order 11 has the same single trigger "
        "class.",
        ("pool_move", "source_trace"),
        {"futuresight": moves["futuresight"], "doomdesire": moves["doomdesire"]},
        correction=(
            "⚠ Two things the row does not say, and the second is the one that decides "
            "the verdict. (1) The payload ALWAYS CARRIES the key -- "
            f"`{LS}:{fs_payload}` is unconditional -- so the closure is 'the mapping is "
            "always EMPTY', never 'the key is absent'. That is the exact distinction "
            "C153 drew for `deferred_opponent_action` and did not carry over to this "
            "row. (2) **The repo ships a Custom Game scenario that casts Future Sight** "
            f"(`{gc}:{fs_scenario}`, an Alakazam with the move and a scripted preference "
            "for it), so the raise is constructible from committed code. What keeps it "
            f"unreached is that the spec sits in `interaction_registry_specs()` "
            f"(`:{registry_fn}`) and not `scenario_specs()` (`:{corpus_fn}`), and every "
            f"world-building harness -- including `run_scenario_fallback_sweep` "
            f"(`:{fallback_fn}`) -- defaults to the latter. That is a ONE-KEYWORD-ARGUMENT "
            "closure: passing `specs=interaction_registry_specs()` fires the raise "
            "immediately. 'Dead code in this format' is right; 'retired' would not be."
        ),
        counter=world_unsupported,
    )

    # -- R2 --------------------------------------------------------------------
    row(
        "R2",
        "Hail, and everything downstream -- the ICE branch of `weather_chips`, hail item/ability "
        "interactions, `world_prestate_mismatch:weather_HAIL`",
        "UNREACHABLE_TRACED",
        "SOUND",
        f"{absent('hail')} as a move, and the gen3 weather-producer set is ENUMERATED "
        "rather than asserted: the only `Dex.mod('gen3')` moves carrying a `weather` "
        f"field are {sorted(pool['weather_setters'])}, and the only non-`Future` "
        "abilities that set weather are Drizzle, Drought and Sand Stream. Snow Warning, "
        "the sole hail-setting ability, reports "
        f"`gen: {pool['dex_abilities']['snowwarning']['gen']}, isNonstandard: "
        f"{pool['dex_abilities']['snowwarning']['isNonstandard']!r}` -- it does not exist "
        "in gen3 at all. NO gen3 item sets weather. Two independent routes, both closed, "
        "and the downstream counter is closed with them: "
        f"`world_prestate_mismatch:` is emitted at `{ETD}:{prestate['sites'][-1]['line']}` "
        "with the suffix taken from the mismatch text, so a `weather_HAIL` suffix requires "
        "a HAIL prestate that no route can produce.",
        ("pool_move", "dex", "source_trace"),
        {
            "hail": moves["hail"],
            "snowwarning": pool["dex_abilities"]["snowwarning"],
            "weather_setting_moves": pool["weather_setters"],
        },
        counter=prestate,
    )

    # -- R3 --------------------------------------------------------------------
    row(
        "R3",
        "Rain Dish's `maxhp/16` rain heal and its missing `ResidualPlan` slot",
        "UNREACHABLE_TRACED",
        "SOUND",
        f"Rain Dish is **0 of {pool['sets']} sets**, and the check is made at the SPECIES "
        "level too because a set list is an upper bound on nothing here: the gen3 species "
        f"carrying Rain Dish are {pool['gen3_carriers']['Rain Dish']['all']}, of which "
        f"only {pool['gen3_carriers']['Rain Dish']['in_pool']} is in the pool at all, and "
        f"{co['ludicolo_sets_with_rain_dish']} of Ludicolo's {co['ludicolo_sets']} sets list "
        "Rain Dish. Trace is the only pool "
        "mechanism that can import an ability a set does not list and it COPIES THE "
        "OPPONENT'S, so it cannot manufacture one absent from the whole pool.",
        ("pool_ability", "dex", "generation"),
        {
            "rain_dish_sets": len(abilities["Rain Dish"]),
            "gen3_carriers": pool["gen3_carriers"]["Rain Dish"],
            "trace_species": abilities["Trace"],
        },
    )

    # -- R4 --------------------------------------------------------------------
    row(
        "R4",
        "Weather-expiry sand/hail chip truncation",
        "UNREACHABLE_TRACED",
        "FALSE",
        f"{absent('sandstorm', 'hail')} as moves, so the only sand writer left is Sand "
        f"Stream -- carried by {abilities['Sand Stream']} alone, on all "
        f"{co['tyranitar_sets_with_sand_stream']} of its {co['tyranitar_sets']} sets -- "
        "which writes "
        "`WEATHER_ABILITY_TURNS = -1` (`src/gen3/abilities.rs`, symbol "
        "`ability_on_switch_in`, arm `Abilities::SANDSTREAM`). The expiry block in "
        "`add_end_of_turn_instructions` is guarded on "
        "`state.weather.turns_remaining > 0`, which `-1` fails, so permanent sand is "
        "never decremented and never reaches the `== 0` clear. The chip gate is a "
        "SEPARATE expression, `weather_survives_upkeep = turns_remaining != 1`, and "
        "`weather_chips` returns `Some` only for HAIL or SAND -- so `turns_remaining == 1` "
        "and a chipping weather cannot coincide.",
        ("pool_move", "pool_ability", "source_trace"),
        {
            "sandstorm": moves["sandstorm"],
            "hail": moves["hail"],
            "raindance": moves["raindance"],
            "sunnyday": moves["sunnyday"],
            "sand_stream_species": abilities["Sand Stream"],
        },
        correction=(
            "⚠ **The row's sentence \"The expiry path has no trigger\" is FALSE.** The "
            "order-8 decrement-and-clear block fires on every Rain Dance ("
            f"{moves['raindance']} species) and Sunny Day ({moves['sunnyday']}), and "
            "`weather_survives_upkeep` evaluates FALSE on their expiring turn -- the "
            "expiry path is exercised in ordinary play. What has no trigger is the "
            "narrower thing the row is titled after: a chipping weather with a FINITE "
            "counter. Rain and sun do not chip; sand and hail are the only chipping "
            "weathers and neither can hold a finite counter in this pool. The verdict "
            "survives; the sentence that carried it does not. Two neighbouring "
            "over-readings are corrected with it: permanent (`-1`) weather is NOT "
            "Tyranitar-only -- Kyogre's Drizzle and Groudon's Drought write it too -- and "
            "the payload-seeding lane cannot inject finite sand either, but by a "
            "different mechanism the row does not cite. ⚠ Stated precisely, because a first "
            f"revision of this correction wrote \"`_weather_fields` returns `-1` ONLY under "
            "`weatherFromAbility`\" and that is false -- it also returns `(\"none\", -1)` at "
            f"`{EW}:{weather_none_return}` for absent or `none` weather, which is the "
            "commonest case in the corpus. The accurate statement is narrower: `-1` for a "
            "NAMED weather requires `weatherFromAbility`, and a prefix that saw only "
            "`[upkeep]` lines fails closed at `weather_turns_unknown` rather than "
            "fabricating a counter, so the seeding lane cannot inject finite sand."
        ),
    )

    # -- R5 --------------------------------------------------------------------
    row(
        "R5",
        "Dry Skin's rain heal at order 10.3",
        "UNREACHABLE_TRACED",
        "SOUND",
        "`Dex.mod('gen3').abilities.get('dryskin')` reports "
        f"`gen: {pool['dex_abilities']['dryskin']['gen']}, isNonstandard: "
        f"{pool['dex_abilities']['dryskin']['isNonstandard']!r}` -- it does not exist in "
        "gen3, which is stronger than 'dead code for gen3 randbats' because it closes the "
        f"format change as well. Measured 0 of {pool['sets']} sets for completeness.",
        ("dex", "pool_ability"),
        {"dryskin": pool["dex_abilities"]["dryskin"], "dry_skin_sets": len(abilities["Dry Skin"])},
    )

    # -- R6 --------------------------------------------------------------------
    row(
        "R6",
        "Sitrus Berry, and the monotonicity break it causes in the residual mirror's bisection",
        "UNREACHABLE_TRACED",
        "INCOMPLETE",
        f"The item universe is measured, not read: {gen['pokemon']:,} generated Pokemon "
        f"under a second seed scheme ({GENERATION_SEED_SCHEME}) carry exactly "
        f"{gen['distinct_items']} distinct items, and Sitrus Berry is not among them, "
        "reproducing section 1.3's 13 under a scheme section 8 requires presence facts to "
        "survive. Sitrus DOES exist in gen3 ("
        f"`gen: {pool['dex_items']['sitrusberry']['gen']}`), so the closure is the "
        "generator's, not the dex's.",
        ("generation", "pool_move", "dex"),
        {
            "distinct_items": gen["distinct_items"],
            "items": sorted(gen["items"]),
            "sitrusberry": pool["dex_items"]["sitrusberry"],
            "trick": moves["trick"],
            "knockoff": moves["knockoff"],
            "thief": moves["thief"],
            "covet": moves["covet"],
            "recycle": moves["recycle"],
            "switcheroo": moves["switcheroo"],
            "bugbite": moves["bugbite"],
        },
        correction=(
            "⚠ **The stated reason is a GENERATION-TIME argument and the row needs a "
            "RUNTIME one too.** \"`getItem` cannot return it\" says what a team starts "
            "with and nothing about acquisition in play -- which is the exact gap that "
            "made R26 wrong, in this document, about this mechanic, one row after this "
            f"one. Closed here by measurement: `trick` is {moves['trick']} of "
            f"{pool['species']} species and `knockoff` {moves['knockoff']}, and both only "
            "SWAP or REMOVE; `thief`, `covet`, `recycle`, `switcheroo` and `bugbite` are "
            "each 0, and no gen3 mechanism CREATES an item. A closed set of 13 stays "
            "closed under permutation and deletion, so Sitrus cannot enter mid-battle "
            "either."
        ),
    )

    # -- R7 --------------------------------------------------------------------
    row(
        "R7",
        "`nature_not_neutral`",
        "UNREACHABLE_TRACED",
        "INCOMPLETE",
        f"`_build_pokemon_spec` raises at `{EW}:{nature_raise}` when "
        f"`normalize_id(mon.nature)` (read at `:{nature_read}`) is outside "
        f"`_NEUTRAL_NATURES` (`:{neutral_set}`). `mon` is a `FixturePokemon` and the "
        "production path builds it in `unpack_pokemon` from field 6 of the PACKED TEAM "
        f"STRING at `{EW}:{unpack}` -- the differential's own "
        f"`{ETD}:{packed_source}` hands it `true_teams[slot]['packed']` from the bridge "
        f"snapshot. Natures are unset on {gen['nature_unset']:,} of {gen['pokemon']:,} "
        "generated Pokemon, so field 6 is empty and `nature` is `\"\"`.",
        ("generation", "source_trace"),
        {"nature_unset": gen["nature_unset"], "pokemon": gen["pokemon"]},
        correction=(
            "⚠ **The stated reason is HALF the demonstration, and the missing half is "
            "load-bearing in the opposite direction.** \"Generated sets carry no nature "
            "field at all\" is true and measured, but on its own it does not close the "
            "refusal -- it would OPEN it, because an absent field makes `mon.nature` falsy "
            "and `nature` the empty string. What closes it is that `\"\"` is a MEMBER of "
            f"`_NEUTRAL_NATURES` (`{EW}:{neutral_set}`). Without that membership the "
            "refusal would fire on every Pokemon in every battle. ⚠ Second, the row's "
            "instrument is the generator, and the generator is not the only producer that "
            f"reaches the guard. `_build_pokemon_spec` has a SECOND caller that never "
            f"touches a packed team at all (`src/pokezero/engine_fidelity.py:{fidelity_caller}`), "
            f"and `scenario_studio` parses `nature` out of scenario JSON with no "
            f"vocabulary check and hands it to `FixturePokemon` at `{ss}:{scenario_nature}`. "
            "Neither can fire today -- the fidelity fixtures pass no `nature=`, and the "
            "scenario-studio service never builds an engine world -- but for reasons the "
            "row does not state, so the pool measurement alone does not carry it."
        ),
        counter=world_unsupported,
    )

    # -- R8 --------------------------------------------------------------------
    row(
        "R8",
        "`weather_unsupported`",
        "UNREACHABLE_TRACED",
        "INCOMPLETE",
        f"`_weather_fields` raises at `{EW}:{weather_raise}` when "
        f"`_WEATHER_IDS.get(weather_id)` (`:{weather_lookup}`) misses. `weather_id` is "
        f"`normalize_id(str(payload['weather']))`; the payload key is filled from "
        f"`replay.weather` at `{LS}:{weather_payload}`, whose only writer is "
        f"`_update_weather` at `src/pokezero/showdown.py:{weather_parse}`, which returns "
        f"`_normalize_identifier(parts[2])` (`:{weather_norm}`, "
        "`re.sub(r'[^a-z0-9]+', '', lower())`) of the second field of a `|-weather|` "
        "line. Showdown emits the CONDITION NAME there -- `RainDance`, `SunnyDay`, "
        "`Sandstorm`, `Hail`, or `none` -- so the four reachable ids are exactly "
        "`raindance`, `sunnyday`, `sandstorm`, `hail`, and `none` is early-returned before "
        "the lookup. All four are KEYS of `_WEATHER_IDS`, so the lookup cannot miss.",
        ("source_trace", "dex"),
        {
            "weather_id_keys": sorted(_weather_id_keys()),
            "protocol_forms": ["RainDance", "SunnyDay", "Sandstorm", "Hail", "none"],
            "reachable_ids": ["raindance", "sunnyday", "sandstorm"],
        },
        correction=(
            "⚠ **The stated reason names the wrong side of the mapping, and it survives "
            "only by coincidence.** \"All four gen3 weathers are in `_WEATHER_IDS` "
            "(`rain`, `sun`, `sand`, `hail`)\" reads as a claim about the dict\'s VALUES; "
            f"the lookup at `{EW}:{weather_lookup}` uses its KEYS. The four strings named "
            "happen to be keys as well -- they are the engine-side ALIASES -- so the "
            "membership sentence is technically true. It is still not the demonstration, "
            "because **three of the four strings it names are never what the lookup "
            f"receives.** The seven keys (`{EW}:{weather_map}`) are `raindance, rain, "
            "sunnyday, sun, sandstorm, sand, hail`, and the protocol only ever produces "
            "the condition-name forms, which normalise to `raindance`, `sunnyday`, "
            "`sandstorm`, `hail`. The closure rests on the three ALIAS keys the sentence "
            "does not mention; a map holding only the four it does name would refuse every "
            "rain, sun and sand battle. ⚠ Second, \"all four gen3 weathers\" is broader "
            "than the pool: hail is unreachable (R2), so only three can ever occur, and "
            f"the one lane with no Python-side vocabulary check -- the scenario seed at "
            f"`{LS}:{_anchor(LS, 'parser.weather = weather')}` -- is closed by the "
            "bridge\'s `SCENARIO_WEATHER_IDS` allow-list, a different mechanism the row "
            "does not cite."
        ),
        counter=world_unsupported,
    )

    # -- R9 --------------------------------------------------------------------
    row(
        "R9",
        "Liquid Ooze mislabelled by the residual-heal renderer",
        "UNREACHABLE_TRACED",
        "FALSE",
        f"`render_residual_instruction` (`{EV}:{render_fn}`) intercepts a negative "
        f"`Instruction::Heal` at `:{ooze_intercept}` and renders it as "
        "`|-damage|…|[from] ability: Liquid Ooze` in the THEN arm; `plan.take(side, true)` "
        f"and the `residual_heal_cause` fallback (`:{heal_fallback}`) are the ELSE arm, "
        f"and `residual_heal_cause` (`:{cause_fn}`) has exactly ONE call site in the "
        "crate. So the drain-reversal damage can never be labelled by the fallback, which "
        "is what the row's headline gap needed. Liquid Ooze itself IS reachable -- "
        f"{abilities['Liquid Ooze']} carry it, `leechseed` is {moves['leechseed']} of "
        f"{pool['species']} and both moves are `target: normal`, so the seeder-versus-"
        "holder pairing is cross-side and ordinary.",
        ("pool_ability", "pool_move", "source_trace"),
        {
            "liquid_ooze": abilities["Liquid Ooze"],
            "leechseed": moves["leechseed"],
            "gigadrain": moves["gigadrain"],
        },
        correction=(
            "⚠ **The row's second sentence is FALSE and is a NON SEQUITUR from its "
            "first.** \"The Liquid Ooze guard inside `residual_heal_cause` is therefore "
            "dead code\" does not follow from the negative-heal interception, because the "
            f"guard is not in a negative-heal branch. It is the conjunct "
            f"`ability != Abilities::LIQUIDOOZE` at `{EV}:{ooze_guard}`, inside the "
            "LEECHSEED arm, which only runs on a POSITIVE heal -- exactly the heals the "
            "interception lets through. `residual_heal_cause` takes no heal amount at all "
            "(`(state, side, next_ins)`), so the sign of the heal is not a fact it can "
            "see. It is not dead in the literal sense either: the crate PINS it, at "
            f"`{EV}:{ooze_pin}`, with a positive heal of 6 and no Leftovers, and that "
            "pin's own note records that it exists because deleting the guard once left "
            "the suite green. What actually forecloses the conjunct IN THIS FORMAT is the "
            f"two EARLIER returns plus the item universe: a resolving Wish returns at "
            f"`:{cause_wish}` and a Leftovers holder at `:{cause_leftovers}`, so the "
            "conjunct is only consulted for a positive residual heal on a side holding "
            "NEITHER -- and the engine's residual heal producers are Leftovers, Sitrus "
            "(outside the 13-item universe, R6), Wish (already returned) and the Leech "
            "Seed drain, which under Liquid Ooze is NEGATIVE and intercepted. That "
            "enumeration, not the interception, is the demonstration. ⚠ Two further scope "
            "errors in the same cell. The row says \"what makes this row UNREACHABLE is "
            "the renderer interception ALONE\": there are THREE ooze-aware sites, not "
            f"one. The residual plan's own conjunct at `{EV}:{ooze_plan_guard}` is "
            "separately pinned in both directions and is live. And the MOVE-PHASE ooze "
            f"path is a different renderer with its own interception at "
            f"`{EV}:{move_phase_ooze}`, which is genuinely REACHABLE -- `gigadrain` on "
            f"{moves['gigadrain']} against Liquid Ooze on {abilities['Liquid Ooze']} -- so "
            "the mechanic half of this row survives only because that renderer handles it "
            "too, which the row does not say. `reports/c131` §5 carries the same false "
            "sentence and is corrected by reference."
        ),
    )

    # -- R10 -------------------------------------------------------------------
    row(
        "R10",
        "Shell Bell, and the `heal_drain_or_shellbell` ambiguity",
        "UNREACHABLE_TRACED",
        "INCOMPLETE",
        f"`heal_subcase` (`{EV}:{subcase_fn}`) routes a positive ATTACKER-side heal with "
        f"foe damage in the same tail to `heal_drain_or_shellbell` at `:{subcase_bucket}`. "
        "⚠ THE BUCKET HAS ONE PRODUCTION EMITTER AND THE ROW DOES NOT NAME IT. The "
        "caller graph is DERIVED, not asserted (see `heal_subcase_call_graph`): the crate "
        f"entry point `{pool['heal_subcase_call_graph']['live_entry_points']}` reaches "
        f"`heal_subcase` only through `{HEAL_SUBCASE_ROOT}`, whose remaining "
        f"{len(pool['heal_subcase_call_graph']['dead_wrappers_with_no_production_caller'])} "
        "graph roots are thin non-`_with_protect` wrappers with ZERO production callers. "
        f"`{HEAL_SUBCASE_ROOT}` has exactly "
        f"{len(pool['heal_subcase_call_graph']['edges_out_of_the_chokepoint'])} edges into "
        "the subgraph, both inside its Sleep Talk block -- the "
        f"`sleeptalk_refusal_is_unsafe_with_protect` predicate at `{EV}:{sleeptalk_guard}` "
        f"and the slug emit at `:{sleeptalk_emit}`. Only the second produces a KEY; the "
        "first is a boolean test whose result is discarded. So the tail is always a SLEEP "
        "TALK CALLEE'S tail, and the bucket needs `sleeptalk` and a drain move ON THE SAME "
        "SET. Measured per set, the instrument section 8 requires for a two-thing gap: "
        f"{co['sleeptalk_sets']} of {pool['sets']} sets carry `sleeptalk`, "
        f"{co['drain_sets']} carry a drain move, and "
        f"{co['sleeptalk_with_a_drain_move']} carry both. The bucket is therefore "
        "UNEMITTABLE in this format, which is strictly stronger than the row's "
        "\"unambiguous\". Shell Bell, the other half of the name, is implemented AS drain "
        "(`src/gen3/items.rs`, arm `Items::SHELLBELL`, sets `attacking_choice.drain`), so "
        "it produces a byte-identical `Heal` -- the ambiguity is real by construction and "
        "not by enumeration -- and it is absent from the "
        f"{gen['distinct_items']}-item universe.",
        ("generation", "pool_move", "dex", "source_trace"),
        {
            "shellbell": pool["dex_items"]["shellbell"],
            "pool_drain_moves": pool["move_classes"]["drain"],
            "gigadrain_species": moves["gigadrain"],
            "sleeptalk_sets": co["sleeptalk_sets"],
            "drain_sets": co["drain_sets"],
            "sleeptalk_with_a_drain_move": co["sleeptalk_with_a_drain_move"],
        },
        correction=(
"⚠ Three upgrades, and the first changes the strength of the verdict. (1) "
            "The row argues the bucket is UNAMBIGUOUS; it is in fact UNEMITTABLE, because "
            "every production path into `heal_subcase` starts in "
            f"`{HEAL_SUBCASE_ROOT}`'s Sleep Talk block and no set pairs `sleeptalk` with a "
            f"drain move ({co['sleeptalk_with_a_drain_move']} of {pool['sets']}). The row "
            "reasons from the bucket's NAME and never traces its caller -- the exact shape "
            "the C153 rule forbids. ⚠ **And the first revision of THIS correction did it "
            "too**, in the same sentence that names the failure: it asserted, untraced, "
            "that `heal_subcase` is reached only through "
            "`ambiguous_unrenderable_slug_with_protect`. There are two routes; review found "
            "the second. The conclusion held, the sentence did not, and nothing re-derived "
            "it -- so the graph is now built by `rust_call_graph` and pinned. (2) \"the pool's only drain move is `gigadrain`\" was an "
            "assertion; it is now the `drain`-flagged subset of the pool's "
            f"{pool['distinct_moves']} moves, derived from the dex, with the five gen3 "
            "drain moves it excludes named. (3) The Shell Bell half rests on the same "
            "generation-time item argument R6 does and needs the same runtime clause: the "
            "pool's only item-moving moves SWAP (`trick`) or REMOVE (`knockoff`) and no "
            "gen3 mechanism creates an item. ⚠ One site the row's sentence does not "
            f"cover survives as a note: the NAMED (non-Sleep-Talk) renderer tags a heal "
            f"`[from] drain` whenever `choice.drain.is_some()`, and Shell Bell sets that "
            "field, so a Shell Bell holder would be mislabelled there too -- same "
            "unreachability, second site."
        ),
    )

    # -- R11 -------------------------------------------------------------------
    row(
        "R11",
        "The `TwoToFiveHits` flat-3.2 approximation",
        "UNREACHABLE_TRACED",
        "SOUND",
        "The pool's `multihit` move set is derived from the dex over all "
        f"{pool['distinct_moves']} pool moves and is exactly "
        f"{pool['move_classes']['multihit']} -- one move, whose `multihit` is the SCALAR "
        f"{json.dumps(pool['bonemerang_multihit'])}, not a `[2,5]` range. Every `[2,5]` "
        f"move is therefore 0 of {pool['species']}. Whole-pool absence, so it is "
        "side-independent: no side can use one and no side can be hit by one.",
        ("pool_move", "dex"),
        {"multihit": pool["move_classes"]["multihit"], "bonemerang": moves["bonemerang"]},
    )

    # -- R12 -------------------------------------------------------------------
    row(
        "R12",
        "Rest's Insomnia / Vital Spirit fail clauses",
        "UNREACHABLE_TRACED",
        "SOUND",
        f"PER-SET co-occurrence, which is the instrument section 8 requires for a gap "
        f"needing two things at once: {co['rest_sets']} of {pool['sets']} sets carry "
        f"`rest` and {co['rest_with_insomnia_or_vital_spirit']} of those "
        f"{co['rest_sets']} list Insomnia or Vital Spirit. Rest is `target: self` and the ability is the user's own, so "
        "same-side is the CORRECT instrument here and the `target: normal` caution does "
        "not apply. Comatose is gen7.",
        ("pool_set_co_occurrence",),
        {
            "rest_sets": co["rest_sets"],
            "rest_with_insomnia_or_vital_spirit": co["rest_with_insomnia_or_vital_spirit"],
        },
    )

    # -- R13 -------------------------------------------------------------------
    row(
        "R13",
        "Belly Drum's Shedinja `maxhp === 1` fail clause",
        "UNREACHABLE_TRACED",
        "SOUND",
        f"Shedinja has {len(pool['shedinja_movepool'])} set, movepool "
        f"{pool['shedinja_movepool']}, and {co['shedinja_sets_with_bellydrum']} of its "
        f"{len(pool['shedinja_movepool'])} sets carry `bellydrum` (against "
        f"{co['bellydrum_sets']} Belly Drum sets pool-wide). Belly Drum is `target: self`, "
        "so the same-side instrument is the right one. It ships for source parity only.",
        ("pool_set_co_occurrence",),
        {
            "shedinja_movepool": pool["shedinja_movepool"],
            "shedinja_sets_with_bellydrum": co["shedinja_sets_with_bellydrum"],
            "bellydrum_sets": co["bellydrum_sets"],
        },
    )

    # -- R14 -------------------------------------------------------------------
    row(
        "R14",
        "N5 -- the residual ceiling overshooting into a move-KO",
        "UNREACHABLE_TRACED",
        "INCOMPLETE",
        f"N5 needs `maxhp <= {N5_MAXHP_CEILING}` AND `hit_count > 1`. Over "
        f"{gen['pokemon']:,} generated Pokemon the minimum max HP is {gen['maxhp_min']} "
        f"and the only species at or below {N5_MAXHP_CEILING} is "
        f"{sorted(gen['species_at_or_below_maxhp_ceiling'])}. The pool's only multi-hit "
        f"move is Bonemerang (Ground); `getEffectiveness('Ground', ['Bug','Ghost'])` is "
        f"{pool['bonemerang_effectiveness']} -- resisted, not immune -- so Wonder Guard "
        "blocks it. Enumerated exhaustively over the pool's "
        f"{pool['distinct_moves']} moves: "
        f"{len(pool['super_effective_against_wonder_guard'])} are super-effective against "
        "Bug/Ghost and EVERY ONE is single-hit.",
        ("generation", "pool_move", "dex"),
        {
            "maxhp_min": gen["maxhp_min"],
            "species_at_or_below_ceiling": gen["species_at_or_below_maxhp_ceiling"],
            "super_effective": pool["super_effective_against_wonder_guard"],
            "bonemerang_effectiveness": pool["bonemerang_effectiveness"],
            "skillswap": moves["skillswap"],
            "roleplay": moves["roleplay"],
        },
        correction=(
            "⚠ **The corrected argument still had a hole, in the same cross-side family "
            "the row already carries a ⚠ for.** It rests entirely on Wonder Guard and "
            "never asks whether the OPPONENT can take Wonder Guard away. In gen3 the two "
            f"moves that can are Skill Swap ({moves['skillswap']} of {pool['species']}) "
            f"and Role Play ({moves['roleplay']}); Gastro Acid, Worry Seed, Entrainment, "
            "Simple Beam and Mold Breaker are all gen4+. Both are 0, so the route is "
            "closed -- but nothing had closed it until it was measured. Transform is in "
            f"the pool ({moves['transform']} species) and is not a route either: it copies "
            "the target's ability onto the user and does not copy HP. ⚠ **And the "
            "exhaustive type enumeration the row is proud of is VACUOUS.** \"Of the pool's "
            "125 moves, every move that is super-effective against Bug/Ghost is "
            "single-hit\" is true, but so is 'every move in the pool except Bonemerang is "
            "single-hit' -- the multi-hit census already gave the whole result and the "
            f"type scan adds nothing. It is kept here ({len(pool['super_effective_against_wonder_guard'])} "
            "moves, all single-hit) as a control, not as the argument. ⚠ The argument the "
            "row does not make is stronger and simpler, and c129 made it: "
            "`residual_disjoint_bands` admits a band only under `0 < threshold < ceiling`, "
            "and at the TWO KILL sites -- which are the two the N5 ceiling sits in -- that "
            "`ceiling` is `defender_active.hp`. So at `hit_count == 1` the arm deals "
            "strictly less than the defender's HP by construction and cannot overshoot "
            "into a KO, and Shedinja's `hp == 1` makes `0 < threshold < 1` unsatisfiable, "
            "so no band is ever built for the one species that motivated the row. ⚠ Scoped "
            "to the kill sites deliberately: a first revision wrote `ceiling = "
            "defender.hp` flat, and it is `i16::MAX` at the other TWO of the four call "
            "sites (the survive mirrors). The argument holds where it is used; the "
            "sentence did not hold as written. ⚠ Finally, the row "
            "says \"the N5 code\" as though there were one site: the same "
            "`(threshold + hit_count - 1) / hit_count` ceiling appears TWICE in "
            "`src/gen3/generate_instructions.rs`, at `residual_per_hit` and at "
            "`crit_residual_per_hit` on the crit-straddle branch, so a reader checking one "
            "of them has checked half."
        ),
    )

    # -- R15 -------------------------------------------------------------------
    row(
        "R15",
        "Magic Coat / reflect path, and Ingrain blocking phazing",
        "UNREACHABLE_TRACED",
        "SOUND",
        f"{absent('magiccoat', 'ingrain')}. Whole-pool absence, so side-independent. "
        f"Both ARE in `TRACKED_VOLATILES` (`src/pokezero/showdown.py:{tracked_set}`) and "
        f"neither is in `_SUPPORTED_VOLATILES` (`{EW}:{supported_set}`), so the refusal at "
        f"`{EW}:{volatiles_raise}` genuinely IS keyed to them -- it simply has no "
        "producer. The `reflectable: true` flag Roar and Whirlwind carry upstream is inert "
        "for want of a Magic Coat user.",
        ("pool_move", "source_trace"),
        {"magiccoat": moves["magiccoat"], "ingrain": moves["ingrain"]},
    )

    # -- R16..R20 --------------------------------------------------------------
    row(
        "R16", "Dragon Rage and Psywave emit no instructions", "UNREACHABLE_TRACED", "SOUND",
        f"{absent('dragonrage', 'psywave', 'nightshade')}. Whole-pool absence, so "
        "side-independent -- these are `target: normal` moves and the verdict holds "
        "anyway, because no side has them to use. Night Shade was fixed by the "
        "fixed-damage-pipeline patch regardless. The pool's `damage`-flagged move set is "
        f"exactly {pool['move_classes']['fixed_damage']}.",
        ("pool_move", "dex"),
        {"dragonrage": moves["dragonrage"], "psywave": moves["psywave"],
         "nightshade": moves["nightshade"], "fixed_damage": pool["move_classes"]["fixed_damage"]},
    )
    row(
        "R17", "Eruption / Water Spout one-ULP ordering divergence", "UNREACHABLE_TRACED", "SOUND",
        f"{absent('eruption', 'waterspout')}. Whole-pool absence, side-independent.",
        ("pool_move",), {"eruption": moves["eruption"], "waterspout": moves["waterspout"]},
    )
    row(
        "R18", "Locked-continuation PP on Outrage / Petal Dance / Thrash",
        "UNREACHABLE_TRACED", "SOUND",
        f"{absent('outrage', 'petaldance', 'thrash')}. The reachable half of that patch is "
        f"Solar Beam ({moves['solarbeam']} species) and Hyper Beam "
        f"({moves['hyperbeam']}), which is the control that keeps this from being a "
        "vacuous absence.",
        ("pool_move",),
        {"outrage": moves["outrage"], "petaldance": moves["petaldance"],
         "thrash": moves["thrash"], "solarbeam": moves["solarbeam"], "hyperbeam": moves["hyperbeam"]},
    )
    row(
        "R19", "Snore treated as not sleep-usable", "UNREACHABLE_TRACED", "SOUND",
        f"{absent('snore')}. Sleep Talk, the other sleep-usable move, IS in the pool at "
        f"{co['sleeptalk_sets']} of {pool['sets']} sets, so the absence is a fact about "
        "Snore and not about sleep moves.",
        ("pool_move",), {"snore": moves["snore"], "sleeptalk_sets": co["sleeptalk_sets"]},
    )
    row(
        "R20", "Low Kick's weight-based base power, which Transform does not copy",
        "UNREACHABLE_TRACED", "SOUND",
        f"{absent('lowkick')}. Transform IS in the pool ({moves['transform']} species), so "
        "the row's second clause is about a reachable copier and an unreachable move, not "
        "about two absences.",
        ("pool_move",), {"lowkick": moves["lowkick"], "transform": moves["transform"]},
    )

    # -- R21 -------------------------------------------------------------------
    row(
        "R21",
        "Reflect / Light Screen keeping a trailing float position in the damage pipeline",
        "UNREACHABLE_TRACED",
        "INCOMPLETE",
        f"{absent('reflect', 'lightscreen', 'safeguard', 'mist')}. No Pokemon in the pool "
        "can set a screen, and the construction capability the row flags is traced rather "
        f"than waved at: `_build_side_spec` maps side conditions through "
        f"`_SIDE_CONDITION_IDS` (`{EW}:{side_cond_map}`), which DOES carry `reflect`, "
        f"`lightscreen`, `safeguard` and `mist`, and refuses an unmapped one at "
        f"`:{side_cond_raise}`. The values it maps come from "
        f"`side_payload['sideConditions']`, filled from `replay.side_condition_counts` at "
        f"`{LS}:{side_cond_payload}`, whose live writer is the `-sidestart` arm of the "
        f"protocol fold. A screen therefore needs a `|-sidestart|…|move: Reflect` line, "
        "which needs a pool Pokemon to use the move. `spikes` is the control: it is the "
        f"one mapped condition the pool CAN set ({moves['spikes']} species), so the map is "
        "not dead.",
        ("pool_move", "source_trace"),
        {"reflect": moves["reflect"], "lightscreen": moves["lightscreen"],
         "safeguard": moves["safeguard"], "mist": moves["mist"], "spikes": moves["spikes"]},
        correction=(
            "⚠ **The row flags itself for re-checking and then answers with an "
            "assertion.** \"`engine_world` *can construct* screens as side conditions -- "
            "but no battle path reaches that state\" is the shape this pass exists to "
            "replace, and re-tracing it moves where the closure sits. It is NOT "
            f"`side_condition_unsupported` (`{EW}:{side_cond_raise}`): that fires for a "
            "condition OUTSIDE the map, and `reflect`, `lightscreen`, `safeguard` and "
            "`mist` are all INSIDE it, with turn counters derived rather than copied. If a "
            "screen ever appeared, engine_world would build it. The only thing that keeps "
            "it from appearing is the protocol: a mapped condition enters solely through a "
            "`|-sidestart|` line, and no pool Pokemon can emit one for a screen -- with no "
            "copier route either, since `metronome`, `assist`, `mirrormove`, `mimic`, "
            "`sketch`, `naturepower` and `magiccoat` are each 0 and Sleep Talk draws from "
            f"the user's own slots. `spikes` ({moves['spikes']} species) is the live "
            "control on the same map. ⚠ One capability the row does not mention and that "
            f"is not a battle path: the scenario harness injects arbitrary side conditions "
            f"at `{LS}:{scenario_sidestart}`. No committed scenario under `scenarios/` "
            "sets one, so it is the same shape as the row's own parenthetical -- recorded "
            "rather than left to be found again."
        ),
    )

    # -- R22 -------------------------------------------------------------------
    row(
        "R22",
        "Mimic, Imprison, Psych Up, Metronome, Assist, Nature Power, Sketch, Mirror Move",
        "UNREACHABLE_TRACED",
        "FALSE",
        f"{absent('mimic', 'imprison', 'psychup', 'metronome', 'assist', 'naturepower', 'sketch', 'mirrormove')}"
        ". Whole-pool absence, side-independent -- which matters for `imprison`, whose "
        "producer runs on the FOE (`onFoeDisableMove`), because both teams draw from the "
        f"same {pool['species']}-species pool. `maybeDisabled` has exactly one `= true` "
        "site in the whole Showdown tree, inside `imprison.condition.onFoeDisableMove`, "
        "and gen3 does not override that condition; `maybeLocked` has no `= true` site "
        "outside the gen1 partial-trapping mod and in gen3 is definitionally "
        "`maybeLocked || maybeDisabled`. G32 closes with `psychup` at 0, because the only "
        f"in-pool move-caller is Sleep Talk ({co['sleeptalk_sets']} sets) and it selects "
        "from the user's OWN move slots.",
        ("pool_move", "source_trace"),
        {
            "mimic": moves["mimic"], "imprison": moves["imprison"], "psychup": moves["psychup"],
            "metronome": moves["metronome"], "assist": moves["assist"],
            "naturepower": moves["naturepower"], "sketch": moves["sketch"],
            "mirrormove": moves["mirrormove"], "encore": moves["encore"],
            "transform": moves["transform"],
        },
        correction=(
            "⚠ **Two clauses are wrong, one of them flatly.** (1) \"This closes … the "
            "`failencore` move-list edge cases\" is FALSE. `move_fails_encore` "
            f"({failencore_patch}, symbol in `src/gen3/generate_instructions.rs`) matches "
            "`ENCORE | MIMIC | MIRRORMOVE | SKETCH | STRUGGLE | TRANSFORM`, and R22's "
            f"eight names cover only three of the six: `encore` is {moves['encore']} of "
            f"{pool['species']} species, `transform` is {moves['transform']}, and Struggle "
            "is reachable by PP exhaustion. Half that list is live in ordinary play. "
            "**Nothing opens**, and that is a SET EQUALITY, measured on both sides: the "
            f"arm's alternatives parsed out of the committed patch are {list(failencore_shipped)}, "
            "and the non-`Future` gen3 moves carrying Showdown's `failencore` flag -- the "
            "condition `encore.condition.onStart` actually tests -- are "
            f"{pool['failencore_flagged_gen3']}. Equal, so the shipped list is right for "
            "its three reachable members too. ⚠ A first revision called this "
            "machine-checked while the pin asserted six MEMBERSHIPS, which review defeated "
            "by adding `Choices::TACKLE` to the arm and watching the module stay green. "
            "The clause is withdrawn; the patch is not. (2) \"closes `_HIDDEN_INFORMATION_REQUEST_FLAGS`'s "
            "`maybeDisabled`/`maybeLocked`\" uses the wrong verb. That frozenset "
            f"(`{EW}:{hidden_flags}`) is a TOLERATE-list: its members are filtered OUT of "
            f"the refusal binding at `:{hidden_filter}`, so those two flags never caused a "
            "refusal to begin with. What unreachability protects is a SILENT one -- under "
            "Imprison the singles request reports the blocked moves as `disabled: false`, "
            "engine_world tolerates the flag, and search would plan a move Showdown "
            "rejects."
        ),
    )

    # -- R23 -------------------------------------------------------------------
    row(
        "R23",
        "Focus Energy, Mud Sport, Water Sport, Taunt, Torment, Disable, Nightmare, Foresight",
        "UNREACHABLE_TRACED",
        "FALSE",
        f"{absent('focusenergy', 'mudsport', 'watersport', 'taunt', 'torment', 'disable', 'nightmare', 'foresight')}"
        f". The refusal IS keyed to them, traced: `_build_side_spec` reads the token list "
        f"at `{EW}:{volatiles_read}`, subtracts `supported` at `:{volatiles_diff}` and "
        f"raises at `:{volatiles_raise}`. The tokens that can appear are exactly "
        f"`TRACKED_VOLATILES` (`src/pokezero/showdown.py:{tracked_set}`, enforced at "
        f"`:{tracked_gate}`, carried into the payload at `{LS}:{volatiles_payload}`), all "
        "eight are members, and none is admitted by `_SUPPORTED_VOLATILES` "
        f"(`{EW}:{supported_set}`) or by any contextual expansion -- so each WOULD raise "
        "if it ever appeared.",
        ("pool_move", "dex", "generation", "source_trace"),
        {
            "volatile_producers_by_data": pool["volatile_producers_by_data"],
            "volatile_producers_by_source": pool["volatile_producers_by_source"],
            "odorsleuth": moves["odorsleuth"],
            "lansatberry": pool["dex_items"]["lansatberry"],
            "distinct_items": gen["distinct_items"],
        },
        correction=(
            "⚠ **\"Each 0 of 220 as moves\" is not the whole producer set for two of the "
            "eight, and \"cannot fire from play\" is false at the scope it is written.** "
            "(1) The `foresight` VOLATILE has a second gen3 move producer, `odorsleuth` "
            f"(`volatileStatus: 'foresight'`, no condition of its own), also 0 of "
            f"{pool['species']} -- the row checks one move id where two exist, which is "
            "the R26 shape. (2) `focusenergy` has a live gen3 NON-MOVE producer: Lansat "
            f"Berry (`gen: {pool['dex_items']['lansatberry']['gen']}`, `onEat` calls "
            "`addVolatile('focusenergy')`). A move census cannot see it; what forecloses "
            f"it is the item universe -- Lansat is not among the {gen['distinct_items']} "
            "the generator returns, and no pool move creates an item. (3) The refusal DOES "
            "fire, on this repo's own scenario corpus: `struggle_taunt_stall` is refused "
            "with `volatile_unsupported: side 'p1': ['taunt']`, recorded in "
            "`docs/belief_edge_case_matrix.md` and in `tests/test_struggle_only_move_state.py`. "
            "The correct scope is \"cannot fire from SAMPLED RANDBATS play\"; it is "
            "measured live from hand-written Custom Game fixtures. ⚠ And the enumeration "
            "that found (1) and (2) is itself a corrected instrument: a first version "
            "walked `Dex.mod('gen3')` and regexed `JSON.stringify` of each ability and "
            "item, which DROPS FUNCTIONS and therefore reported that the same-named move "
            "is the only producer of all ten -- a scan that could not fail. Both scans are "
            "recorded on this row, `by_data` and `by_source`."
        ),
        counter=world_unsupported,
    )

    # -- R24 -------------------------------------------------------------------
    row(
        "R24",
        "Attract as a *move*",
        "UNREACHABLE_TRACED",
        "INCOMPLETE",
        f"{absent('attract')}. The gen3 producers of the `attract` VOLATILE are "
        "enumerated: the move (0), the ability Cute Charm (gen3, in pool on "
        f"{abilities['Cute Charm']}), the item Destiny Knot (gen4) and G-Max Cuddle "
        "(gen8). Baton Pass cannot move it -- the condition carries `noCopy: true`. So the "
        "volatile is reachable only through Cute Charm, which is G37's correctly-scoped "
        "row.",
        ("pool_move", "pool_ability", "dex", "source_trace"),
        {"attract": moves["attract"], "cute_charm": abilities["Cute Charm"],
         "producers": pool["volatile_producers_by_source"]["attract"]},
        correction=(
            "⚠ Two scope notes the row's phrasing invites a reader to get wrong. `attract` "
            f"is a MEMBER of `_SUPPORTED_VOLATILES` (`{EW}:{supported_set}`), so the "
            "volatile is expressed and searched, not refused -- unlike the eight in R23, "
            "which is the neighbouring row. And \"reachable only through Cute Charm\" is "
            "randbats-scoped: the scenario corpus produces it from the move itself "
            "(`attract_snorlax`)."
        ),
    )

    # -- R25 -------------------------------------------------------------------
    row(
        "R25",
        "Sleep Talk calling Haze / Psych Up / Roar / Whirlwind / Baton Pass",
        "UNREACHABLE_TRACED",
        "SOUND",
        f"PER-SET co-occurrence on this `sets.json`: {co['sleeptalk_sets']} of "
        f"{pool['sets']} sets carry `sleeptalk` and "
        f"{co['sleeptalk_with_haze_psychup_roar_whirlwind_batonpass']} of those "
        f"{co['sleeptalk_sets']} also carry any of the five. Sleep Talk selects from the USER'S OWN move slots, so same-side "
        "co-occurrence is the correct instrument and a whole-pool count would be the wrong "
        "one. Measured on this file, not carried from the crate's three-universe count.",
        ("pool_set_co_occurrence",),
        {
            "sleeptalk_sets": co["sleeptalk_sets"],
            "paired": co["sleeptalk_with_haze_psychup_roar_whirlwind_batonpass"],
        },
    )

    # -- R26 -- carried, not measured ------------------------------------------
    records["R26"] = {
        "row": "R26",
        "candidate": "Trick-style item acquisition reaching White Herb",
        "verdict": "WITHDRAWN_BEFORE_THIS_PASS",
        "ledger_reason_status": "FALSE",
        "correction": (
            "Withdrawn and reclassified REACHABLE at G49 before this pass. Carried here "
            "with no verdict of its own because an inventory that silently drops a name is "
            "how a 'closed' row turns out to be a fourth category in disguise -- and "
            "because R6, R10 and R27 were still resting on the generation-time argument "
            "R26 died of, which is what this pass found."
        ),
        "demonstration": None,
        "instruments": [],
        "measurements": {"trick": moves["trick"], "knockoff": moves["knockoff"]},
    }

    # -- R27 -------------------------------------------------------------------
    row(
        "R27",
        "Quick Claw, King's Rock, Bright Powder, Lax Incense, Focus Band, Scope Lens, Berry "
        "Juice, the status berries, and every type-boosting item outside the 13",
        "UNREACHABLE_TRACED",
        "INCOMPLETE",
        f"The universe is {gen['distinct_items']} items over {gen['pokemon']:,} generated "
        f"Pokemon under a second seed scheme: {sorted(gen['items'])}. Every item the row "
        "names is absent from it, and each exists in the gen3 dex, so the closure is the "
        "generator's rather than the dex's. What it retires is stated positively: no "
        "priority randomness, no item-sourced flinch, no evasion or crit item, no "
        "HP-restoring berry, no item-sourced heal-on-damage.",
        ("generation", "dex", "pool_move"),
        {
            "distinct_items": gen["distinct_items"],
            "items": sorted(gen["items"]),
            "named_absent": {
                k: v for k, v in pool["dex_items"].items() if k != "leftovers"
            },
        },
        correction=(
            "⚠ Same missing runtime clause as R6 and R10, and it matters most here because "
            "this row is the one the ledger uses to retire a whole block of mechanics at "
            "once. A 13-item generation census bounds what a team STARTS with. It is "
            "closed in play as well, measured: `trick` "
            f"({moves['trick']} of {pool['species']}) swaps and `knockoff` "
            f"({moves['knockoff']}) removes, `thief`/`covet`/`recycle`/`switcheroo`/"
            "`bugbite` are each 0, and no gen3 mechanism creates an item -- so no "
            "fourteenth item can appear mid-battle."
        ),
    )

    return records


def source_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None)
    args = parser.parse_args()

    pool = census(default_showdown_root())
    verdicts = build_verdicts(pool)
    document = {
        "_README": (
            "C154. Re-adjudication of all 26 UNREACHABLE verdicts in "
            "reports/c138_known_gaps_ledger.md section 4, against the C153 rule 'trace "
            "the raise site to the caller that actually reaches it'. Regenerate with "
            "scripts/c154_unreachable_readjudication.py --write, from a checkout with a "
            "built pokemon-showdown; CI builds none, so nothing re-derives the POOL half "
            "against a live pool. The CITATION half is re-derived on every run of "
            "tests/test_unreachable_readjudication.py. NEVER edit this by hand to make a "
            "test pass: the pin compares the ledger against it, so a hand edit silently "
            "moves a verdict. It lives under tests/data/ and NOT under reports/ on "
            "purpose -- see the module docstring, it is keyed by counter names."
        ),
        "source_commit": source_commit(),
        "pool": pool,
        "verdicts": verdicts,
        "counts": correction_counts(verdicts),
    }
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.write is None:
        print(text, end="")
    else:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text, encoding="utf-8")
        print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
