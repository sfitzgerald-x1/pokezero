"""Codex's Phase D protocol constructor worlds as DepthNPricer input (gap 4, part A): depth-N prices on Showdown roots.

Design. ProtocolConstructor.prepare(request, index) folds a Showdown protocol prefix plus the literal own request (B2)
and returns (detached device_public_worlds.Root, phase_d_live_worlds.WorldSet of W in {2, 8, 16, 32} worlds, B2's own
legal mask). Its worlds are D3's world type (device_public_world_state.World: the engine BattleState plus the public
tracker carry), so DepthN's executor steps them unchanged; only the carry's pytree differs from D3's (from_b2 adds four
dict keys and leaves two members None). This module turns prepared roots into DepthN requests and pre-built worlds:

- prepare_roots: prepare() per protocol request, on the CPU device by default (its eager per-world array updates would
  otherwise dispatch one by one on a GPU), timed and synced per root, each result stamped with the constructor's
  binding. A raised refusal (phase_d_live_worlds.Refusal, or the errors ProtocolConstructor.construct treats as
  refusals) is a RootRefusal: the root gets no request. Every constructed world's public carry is completed (below)
  before its clock stops, so the live search's structure check, build, the pricer and every capture see one world set.
- Carry completion (complete_world / complete_public; build applies it again, a no-op on a completed world).
  phase_d_live_worlds.from_b2 builds the root's tracker carry from B2's public fold, not through the device public hooks,
  and leaves these where the device path (engine_device_public_hooks.initialize, then a hook per engine step) would hold
  other values on the same history (the peer order, the Spikes credit, Baton Pass arrivals and the peer's PP count read the
  request's own protocol lines: line_facts, kept on Prepared; the PP count also the root's own carry):
  - shown_types (fill_shown_types): from_b2 writes zeros, so the encoder's TYPE_1/TYPE_2 of every revealed Pokemon of
    the side other than the observer read 0 (<pad>), at the root and at every node and leaf. The hook rule
    (engine_device_public_hooks.install: species_types[clip(species)] where 0 < species < len(species_types), else 0) is
    applied to each (side, public slot) whose row is still (0, 0); a side's active whose engine types differ from its
    base types (a type change the state carries) takes those types instead (the color_change hook's write, the only
    device write of shown_types besides install's; no hook writes it for Transform). A non-zero row is never touched.
    Today that branch cannot fire: from_b2 refuses a root whose active is retyped (PUBLIC_LIVE_RETYPE_NOT_MATERIALIZED:
    Color Change, Forecast) or transformed (PUBLIC_TRANSFORM_NOT_MATERIALIZED), and the worlds' engine types are their
    base types. species_types is the executor's own table:
    engine_device_belief.compile_tables(metadata, catalog), which engine_device_observation.compile_tables builds the
    executor's observation (and belief) tables with, here from the constructor's metadata and catalog (the binding
    check makes them the pricer's); the tracker's species are engine ids (from_b2 remaps B2's into the same
    metadata['vocabulary']['species'] the hook's state species index).
  - negative_items (fill_negative_items): from_b2 leaves None, which turns off the Leftovers non-proc rule for the whole
    search; the device path's empty(B) is exact at a root (upkeep clears the evidence) except an active installed after
    the last upkeep (an end-of-turn faint replacement), whose install-time evidence is not recoverable here. With the
    evidence on, an in-tree upkeep can now rule Leftovers out (a Pokemon below 100% with no Leftovers heal and no item
    shown): the hard belief mask narrows and the opponent's belief columns (posterior weights, possible-item count and
    buckets, expected stats) change in the views after that upkeep, the depth-2 leaves, as on the device path. The root
    view and the root parity check (B2 against world 0) do not see it.
  - the items-removed credits (orient_credits): from_b2 stores them owner-oriented as [B] arrays, which the readout
    passes through unchanged, so the peer's view read the owner's own counts; they become per-side [B, 2] arrays
    (self = items_removed[side], opp = items_removed[1 - side], B2's project_credits), the owner's view unchanged.
  - the peer's request order (order_peer): the peer's view (the reply prior's, and every grid column's lane) orders its
    own Pokemon by own_order[peer]: self tokens 1-6, switch tokens 17-21 and switch lanes 4-8 (engine_device_observation
    .encode's targets), which map_actions turns back into engine switch lanes, so lane b means one Pokemon for the
    peer's logits, its legal mask and the grid. device_public_world_state._rebase leaves it [active, the rest ascending];
    Showdown's request order (LocalShowdownEnv.observe of that player) starts from the party order and swaps every
    switch-in into position 0 (switch, drag, faint replacement, Baton Pass recipient), the device install hook's own
    rule. order_peer rebuilds it from the world's physical party order (the materialized team's: the true team's original
    order in the true world, the sampled team's otherwise) and the peer's public switch-ins on the lines; unrevealed
    members never move. The engine state is untouched (switch lanes map to physical slots through own_order), so the
    engine stays consistent; the owner's own order (from_b2: the request's) is unchanged. A switch-in that does not join
    exactly one revealed member, or a last switch-in that is not the world's peer active, refuses the root
    (PeerOrderUnavailable: PEER_REQUEST_ORDER_UNAVAILABLE). Every prepare_roots caller gets this refusal as a RootRefusal
    of that kind, whose build reason is constructor-root:PEER_REQUEST_ORDER_UNAVAILABLE:<detail> (build's own refusal of
    it uses the same reason): the live search counts it constructor_refused; the D3 gate's R1-C
    (d3_depthn_gate.constructor_roots, prepare_roots per roster root) treats it as a root that did not fully construct
    (listed among the selection's refused roots, never selected).
  - the peer's OPP_HAZARD_CREDIT (fill_owner_spikes): from_b2 fills only hazards[owner, exact] and hazards[foe, shown]
    (phase_d_live_worlds.py:415), so hazards[owner, shown], the peer's view of the owner's Spikes losses, read 0. The
    device's shown half is the per-hit sum of (_percent(hp) - _percent(new)) / 100 (engine_device_public_port.spikes);
    B2 keeps only the exact sum, whose per-hit rounding is lost (300 max HP, 37 damage: 300 -> 263 shows 12%, 299 -> 262
    shows 11%), but the owner's own lines carry each hit exactly (a Spikes -damage line after the owner's exact HP on its
    switch-in line), so it is rebuilt hit by hit with the device's own compensated sum. Left 0 when a hit's previous HP
    is not on the lines.
  - Baton Pass arrivals (fill_baton): from_b2 writes last_moves -1 for an active whose last public action was a switch
    (phase_d_live_worlds.py:320); the device writes -2 after a Baton Pass arrival (engine_device_public_hooks.baton_copy,
    :569) and B2 encodes the Baton Pass sentinel (mons.arrived_by_baton). The carry does not keep B2's flag, but the lines
    carry the same fact by the env's and B2's host rule (line_facts' baton), so -1 becomes -2 for that active, both sides.
    That rule is the env's pending flag: a passer that faints to Pursuit before its recipient arrives leaves the flag
    set, so the env and the fill mark the next switch a Baton Pass arrival where the device (no baton_copy) may not;
    very rare.
  - roots whose opponent must recharge are REFUSED, OPPONENT_RECHARGE_UNREPRESENTED (the lead's decision): the GPU engine has
    no recharge or locked-move state, so a world would let the recharging opponent act on its recharge turn. The facts are
    line_facts' recharge for the peer side (B2's rule, below); prepare_roots refuses such a root before completing or
    pricing it (a RootRefusal, constructor-root:OPPONENT_RECHARGE_UNREPRESENTED:..., counted constructor_refused) and
    build refuses one it is handed with the same reason. The live actor then plays greedy. The owner's own recharge turn
    is single-legal (the request offers only the recharge pseudo-move) and is never sent.
  - a recharging active (fill_recharge): B2 holds "must recharge" beside its volatile bag and merges it in only when it
    encodes, so from_b2's copy of the bag lacked it and the encoder showed no volatile:mustrecharge (15 owner-view roots of
    the 160-battle replay: a Hyper Beam user's next decision). The lines carry B2's rule (-mustrecharge sets it; the side's
    |cant| recharge, arrival or faint clears it), so the bit is set on that side's active, both sides. In search: no
    device hook writes or clears the bit and the engine has no recharge turn, so at every node after the root it stays set
    on that Pokemon while it stays in (the install hook's bag reset clears it on a switch), whether or not it would still
    have to recharge there. With the refusal above the fill is inert for every priced root: the opponent's recharge refuses
    the root before completion, and the owner's recharge turn is never searched. It is kept (not removed) for the
    opponent-view parity harness (scripts/depthn_opponent_view_parity.py), which runs complete_carry on the worlds of every
    captured position, refused roots included, and compares their views with B2's and the env's.
  - the peer's count of the owner's PP (fill_peer_pressure): from_b2 copies B2's uses, which charge an owner move twice only
    against a PUBLIC Pressure (public_abilities.move_context); Gen 3 Pressure is silent, and the peer knows its own
    ability, so the device move hook (the receiving observer's own active ability) and the env charge twice whenever the
    peer's active has Pressure. Recounted from the root's own pp_used (B2's) plus one per owner foe-targeted declaration
    against a live, silently Pressured peer active of this world (its base ability); the same on every pass. The owner's
    count of the peer's PP stays B2's (the owner view's reference): from_b2's copy at the root; the device hook counts the
    owner's own Pressure in search. B2 also charges an owner move a [from] tag attributes to an unknown caller (e.g.
    [from]Magic Coat: public_moves' OTHER_FROM caller); line_facts and the device do not count those declarations, so
    neither does this recount: an existing B2-vs-device difference, not one this fill introduces.
  - a Trace holder's base ability (on the world STATE, both sides): the materializer writes an ACTIVE Trace holder's current
    copy into both ability and base_ability (local_showdown._apply_traced_ability_materialization_state ->
    engine_world._resolved_ability -> v5_hypothetical_state.map_spec), for the owner's own tracer as for the peer's, so that
    side's own ability column (the encoder reads base_ability) showed the copy and the engine would revert to the copy on
    switch-out. fix_trace_base_ability sets base_ability to Trace for every member the lines show announcing Trace (the
    subject of a [from] ability: Trace line: its own ability is Trace), the running copy unchanged; it is the rule that
    fires on real lines. fix_peer_base_ability (the carry's public original over its borrowed ability, peer only) never
    does for a tracer: B2's fold records the tracer's running copy and the DONOR's original, never the tracer's own
    ability (pokezero.belief._trace_copy_source; public_abilities.trace_copy).
  The peer's own ability and item belief columns otherwise need nothing: the encoder writes the observer's own Pokemon's
  from the world's engine state (engine_device_observation.encode: base_ability and item as belief:possible_ability / item
  categories, in own_order), as the env writes a player's own (showdown._encode_pokemon_tokens' self rule: the request's
  baseAbility and item, Gen 3 requests carrying no other ability); they follow the request order once own_order does.
  The completion is pure host numpy on the world's own leaves (no other leaf changes) and raises CarryLayoutError (a
  RuntimeError, never a per-root refusal) when a carry is not the engine_device_trackers.Trackers layout it was
  written against.
  Deliberately not completed (the carry and B2's public context do not determine the device's value; owner = the root's
  owner, peer = the other seat, whose view is the reply prior):
  - Peer-view hazard credit: completed from the lines (above) when every owner Spikes hit's previous HP is on them; left
    0 otherwise. Its precision against the env (the env sums the owner's exact HP, the device per-hit public percents, at
    most 0.007) is a documented redaction of the opponent-view parity, not a difference this layer can close.
  - Peer-view last damage: the peer's exact own losses (own.LAST_DAMAGE_TAKEN, foe.LAST_DAMAGE_DEALT,
    field.SELF_HAZARD_CREDIT: damage_exact_*, hazards[foe, exact]) are redacted to zero by design
    (device_public_teacher.REPLY_HISTORY_APPROXIMATION, device_public_worlds._redact_peer_private); and from_b2's
    shown-stream last_damage_taken[owner] / last_damage_dealt[foe] hold B2's exact owner-HP fractions where the device
    holds shown-percent fractions (a rounding difference in the peer view at the root; the first TURN in search
    overwrites both).
  - A Pokemon benched after a type change: the device carry keeps the changed types (nothing clears shown_types before
    the next install); B2 clears its overlay at the next arrival and the worlds' engine types are base types, so the
    fill writes the species' types, as B2's encoding shows.
  - Values the device path never writes but from_b2 copies from B2 (choice_locked, wrap_trap_turns, item_swapped,
    B2-only volatile bits, the items-removed credits): kept, as B2's encoding shows them. In search the three
    mon_features are not reset on switch-in (the install hook's reset list omits them); the volatile bits are.
- build: W worlds become K = W/2 blocks of two, one Request per block, packed into ONE frame (row = the root's rows[b],
  root-major), since the worlds are pre-built (no per-block recipe or re-materialization). Plans are synthesized:
  selected=True (the quarter selection is bypassed), Start(world index, the constructor's world seed, no fixtures,
  refusals); source_fixture_unknown_mass is NaN (no D3 source-fixture draw exists here; the refused-world share is the
  root's `refused_world_share`). `root_legal` is B2's own mask (pre 2 holds each world's own encoder mask to it, as the
  ladder's legality check). Actors default to 0/0 (one evaluator bank, the champion's (champion,)).
- The root prior (build's root_prior, ROOT_PRIORS): `root_logits` and `root_value` are the pricer's own evaluator
  (forward=, e.g. the champion) through the same jitted root forward on
  - 'b2' (the default, the live search): B2's own encoding of the literal root, as in Codex's ladder
    (phase_d_ladder_search.process_many forwards detached.view['_b2_root_inputs']): root_inputs with the first TOKENS
    (23) tokens of each token tensor replaced by B2's (splice; history_mask and the later tokens unchanged);
    `root_view_source` 'b2'. A root without B2 inputs takes world 0's own view ('world0'). B2 inputs that do not fit (a
    missing key, a shape, a lossy cast) refuse the root, ROOT_B2_INPUTS_UNUSABLE.
  - 'world0': world 0's own view, root_inputs itself ('world0'), whatever the root carries: the D3 gate's R1-C
    (d3_depthn_gate.constructor_roots), whose bind_constructor re-runs the champion on each Request's root_inputs and
    fails closed unless that gives build's root logits exactly.
- root_inputs (every Request's) stay world 0's own view under the pricer's executor (the v5 encoder), after the carry
  completion. The turn-1 input check (pre 3+) then holds every OTHER world of every block to world 0's view: a
  consistency check across the sampled worlds only. As a diagnostic (reported, never a gate), each root's `b2_view`
  compares the evaluator's view of root_inputs (its `view`; the 23-token v4 prefix for an evaluator without one) with
  B2's encoding, where the root carries it (from_b2 stores it): per key equal, and the largest numeric difference.
- Refusals: a world without a state (a constructor world refusal) is pre -1 in its block
  (`physical-materialization:<refusals>` in the world report), so that block refuses and the root refuses: DepthN's
  rule (depthn_backups._root), the constructor's all-W `fully_constructed` rule. `only_complete=True` refuses such
  roots before pricing instead (saves the device work; the frame then never shows them). A root whose world 0 is
  refused takes its root view from the lowest constructed world (it refuses anyway); one with no constructed world, an
  erroring root view, unusable B2 inputs or an unusable root forward is refused before pricing with a named reason.
- Bindings: `binding(source, metadata, catalog)` is a DepthN WorldBinding (source hash, ordered rows, the
  source/engine-namespace binding sha256, the engine metadata's canonical-JSON sha256, the encoder catalog's checkpoint,
  category vocabulary and masks). build requires every prepared root's stamp and the constructor's binding to equal the
  pricer side's (the run's D3 Tables source, engine metadata and catalog), and with a champion evaluator
  champion_evaluator.check_binding(evaluator, the constructor's catalog); the pricer checks the binding again on every
  call: fail closed.
- Programs: the worlds go in as DepthN's PrebuiltWorlds under the tag SOURCE, so every compiled-program key gains
  ('worlds', SOURCE, carry signature) after the evaluator's identity: constructor and D3 worlds never share a program,
  nor do two carry structures, nor two evaluators.
- price: `prior` = one packed pricing. `searched` = the full-width depth-2 pass (a DepthNPricer at depth 2, k 9, prior,
  same executor, evaluator and k1) then searched_maps on the host, then depth N with the maps, both passes on the SAME
  PrebuiltWorlds (prepare refuses a second attempt at an address, and none is needed). Nothing in it is D3-only. A batch
  without requests (every root refused) prices nothing and returns (None, None).
- reduce: per root, depthn_backups.backup(rule) and reduce_completed_q per block, then the K blocks by the equal-weight
  mean of their completed-Q (d3_search_latency.compose: every world weighted 1/W); any refused block refuses the root.
  reduce_completed_q's target range check (v5_contract._real, RANGE_EDGE) raises when a block's completed-Q rounds outside
  [-1, 1] (an action winning against every reply in both worlds, its reply prior summing to 1 + ulp): that refuses the
  root only, its cause the first refused block's if any, else 'reduction', the error in `reduction_error`; the batch's
  other roots still label. Any other ValueError there is malformed input (build validated it) and raises, as
  v2_search_head_to_head's _reduce_tolerating_rounding matches the same message.
  Two moves: `action_mean_q`, the argmax of that mean completed-Q over the legal actions (lower action on ties): the
  ladder's root operator (phase_d_ladder_search.reduce_q); and `action_policy_target`, the argmax of the improved
  target softmax(root logits + q) (D's training target, the latency tool's move).

Chance: DepthN samples each node's own keys; constructor states carry kernel_seed-derived keys, so prices are
reproducible but not bit-comparable to the ladder's R1/R7, which fold a domain seed into the keys.

Constructor API this module depends on (part B, the batch 2 and tail sync, must keep it): ProtocolConstructor.prepare's
(Root, WorldSet, legal) return and its `count`, `source`, `metadata`, `catalog` attributes; Refusal's `kind`/`detail`;
WorldSet's `root`, `worlds`, `seeds`, `requested_worlds`; World's fields; Root's `address`, `source_binding_sha256` and
`view['_b2_root_inputs']` (the root prior's inputs); Source's `source_hash`, `source_rows_sha256`, `binding_sha256`; the
constructor's `metadata` and `catalog` (the completion's tables). A changed carry structure compiles new programs under
its own signature, but the completion fails loudly (CarryLayoutError) until it is reviewed against it; a changed engine
metadata or source binding fails closed until both sides agree.
"""
from contextlib import nullcontext
from dataclasses import dataclass,replace
import hashlib
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from pokezero import v5_gumbel as original
from pokezero.dex import normalize_id
from . import champion_evaluator as evaluators,depthn_backups as backups,device_depth2_pricer as depth2,device_depthn_pricer as dn
from . import device_public_teacher_batch as batch,device_public_worlds as plans
from . import engine_device_negative_items as negative_items,engine_device_trackers as public_trackers

SOURCE='phase-d-protocol-constructor'
# What ProtocolConstructor.construct treats as a root refusal (phase_d_live_worlds.Refusal is a ValueError).
ROOT_ERRORS=(ValueError,KeyError,TypeError,IndexError,AttributeError)
ROOT_PRIORS=('b2','world0')   # build's root_prior: B2's literal-root encoding (else world 0's view), or world 0's view
UNNAMED='UNNAMED_CONSTRUCTOR_WORLD_REFUSAL'
RANGE_EDGE='finite in-range numerical value required'   # v5_contract._real's error: on reduce's path only RootImprovement.validate raises it
B2_TOKEN_KEYS=('numeric_features','categorical_ids','token_type_ids','attention_mask')   # B2's encode(): [23,F] / [23]
TOKENS=evaluators.TOKENS   # the champion's v4 prefix of the V2 encoding: what splice replaces with B2's tokens
TOKEN_KEYS={'numeric_features':2,'categorical_ids':2,'token_type_ids':1,'attention_mask':1}   # trailing dims from the token axis
CREDIT_KEYS=('self_items_removed_credit','opp_items_removed_credit')   # from_b2's extra field_features keys


# Bindings ---------------------------------------------------------------------------------------------------------------------------
def metadata_sha256(metadata):
    """sha256 of the engine metadata's canonical JSON (sorted keys, no spaces)."""
    return hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def encoder_identity(catalog):
    """The encoder catalog's identity: checkpoint sha256, category vocabulary sha256 and size, feature masks, source hash."""
    from pokezero.live_protocol_actor import MASK_NAMES,_vocab_sha
    return (('checkpoint_sha256',catalog.checkpoint_sha256),('category_vocab_sha256',_vocab_sha(catalog.vocabulary)),
        ('category_vocab_size',int(catalog.vocabulary.size)),('feature_masks',tuple((k,getattr(catalog.masks,k))for k in MASK_NAMES)),
        ('source_hash',catalog.source.source_hash))


def binding(source,metadata,catalog):
    """The WorldBinding of a bound public Source, the engine metadata and the encoder catalog. Pricer side: the run's D3
    Tables.source, its engine metadata (Prepared.metadata_json) and catalog; constructor side: constructor_binding."""
    if type(source)is not plans.Source:raise ValueError('original bound device public source required')
    return dn.WorldBinding(source.source_hash,source.source_rows_sha256,source.binding_sha256,metadata_sha256(metadata),encoder_identity(catalog))


def constructor_binding(constructor):
    """The binding a ProtocolConstructor builds its worlds under: its own source, engine metadata and B2 catalog."""
    return binding(constructor.source,constructor.metadata,constructor.catalog)


def _check_binding(built,expected,what):
    if type(expected)is not dn.WorldBinding or type(built)is not dn.WorldBinding or built!=expected:
        differ=[f for f in dn.WorldBinding.__dataclass_fields__ if getattr(built,f,None)!=getattr(expected,f,None)]
        raise ValueError(what+': '+', '.join(differ))


# Carry completion -------------------------------------------------------------------------------------------------------------------
class CarryLayoutError(RuntimeError):
    """A constructor world's carry is not the layout complete_world was written against (module docstring): never a refusal."""


@dataclass(frozen=True)
class CarryTables:
    """What the completion reads: species_types [S, 2] int32, engine species id -> engine type ids (the public hooks'
    belief_tables.species_types, engine_device_belief.Tables), held as host numpy; species_names, the engine species
    vocabulary normalized (metadata['vocabulary']['species'] through normalize_id), to join protocol switch-ins."""
    species_types:object
    species_names:tuple=()
    move_names:tuple=()         # metadata['vocabulary']['moves'] normalized
    foe_targeted:tuple=()       # per engine move id: Pressure doubles its PP (the hooks' belief_tables.public_foe_targeted)
    pressure:tuple=()           # per engine ability id: the ability is Pressure (the hooks' belief_tables.public_pressure)
    ability_names:tuple=()      # metadata['vocabulary']['abilities'] normalized


def _table(species_types,species_count=None,type_count=None,species_names=(),move_names=(),foe_targeted=(),pressure=(),ability_names=()):
    move_names=tuple(map(normalize_id,move_names));foe_targeted=tuple(map(bool,foe_targeted));pressure=tuple(map(bool,pressure))
    ability_names=tuple(map(normalize_id,ability_names))
    if ability_names and pressure and len(ability_names)!=len(pressure):
        raise CarryLayoutError(f'ability_names: {len(ability_names)} names for {len(pressure)} engine abilities')
    if len(move_names)!=len(foe_targeted):raise CarryLayoutError(f'foe_targeted: {len(foe_targeted)} flags for {len(move_names)} engine moves')
    species_names=tuple(map(normalize_id,species_names))
    if species_names and len(species_names)!=np.shape(species_types)[0]:
        raise CarryLayoutError(f'species_names: {len(species_names)} names for {np.shape(species_types)[0]} engine species')
    st=np.asarray(species_types)
    if st.ndim!=2 or st.shape[1]!=2 or not st.shape[0] or st.dtype!=np.int32:
        raise CarryLayoutError(f'species_types: an [S, 2] int32 table required, got {st.shape} {st.dtype}')
    if species_count is not None and st.shape[0]!=species_count:
        raise CarryLayoutError(f'species_types: {st.shape[0]} rows for {species_count} engine species')
    if np.any(st<0)or type_count is not None and np.any(st>=type_count):raise CarryLayoutError('species_types: a type id outside the engine namespace')
    st=st.copy();st.setflags(write=False)
    return CarryTables(st,species_names,move_names,foe_targeted,pressure,ability_names)


def carry_tables(metadata,catalog):
    """CarryTables from the engine metadata and encoder catalog through engine_device_belief.compile_tables, the function and
    inputs the executor's observation tables are built with (engine_device_observation.compile_tables(metadata, catalog)
    .belief, whose species_types the public hooks read); compiled on the CPU, only species_types kept."""
    from . import engine_device_belief
    try:context=jax.default_device(jax.devices('cpu')[0])
    except RuntimeError:context=nullcontext()   # no CPU backend in this process: the default device
    with context:belief=engine_device_belief.compile_tables(metadata,catalog)
    v=metadata['vocabulary']
    return _table(jax.device_get(belief.species_types),len(v['species']),len(v['types']),v['species'],v['moves'],
        np.asarray(jax.device_get(belief.public_foe_targeted)),np.asarray(jax.device_get(belief.public_pressure)),v['abilities'])


_CARRY_TABLES={}


def constructor_carry_tables(constructor):
    """carry_tables of the constructor's own metadata and catalog, once per constructor object (the ROOT_FORWARD_CACHE most
    recent kept; each entry holds its constructor, so a live entry's id is never reused)."""
    held=_CARRY_TABLES.get(id(constructor))
    if held is None or held[0]is not constructor:
        while len(_CARRY_TABLES)>=ROOT_FORWARD_CACHE:_CARRY_TABLES.pop(next(iter(_CARRY_TABLES)))
        held=_CARRY_TABLES[id(constructor)]=(constructor,carry_tables(constructor.metadata,constructor.catalog))
    return held[1]


def world_completer(constructor):
    """The completion prepare_roots and build apply to each world of this constructor: (world, owner, facts=None) -> world,
    facts the root's LineFacts (None: the fills only, no peer order or owner Spikes credit)."""
    tables=constructor_carry_tables(constructor)
    return lambda world,owner,facts=None,root=None:complete_world(world,tables,owner,facts=facts,root=root)


def _host(x):return np.asarray(jax.device_get(x))


def _leaf(x,shape,dtype,name):
    if tuple(np.shape(x))!=shape or np.dtype(getattr(x,'dtype',None))!=np.dtype(dtype):
        raise CarryLayoutError(f'{name}: {shape} {np.dtype(dtype)} required, got {tuple(np.shape(x))} {getattr(x,"dtype",None)}')


def _trackers(public,what='public carry'):
    if type(public)is not public_trackers.Trackers:
        raise CarryLayoutError(f'{what}: an engine_device_trackers.Trackers required, got {type(public).__name__}')
    b=tuple(np.shape(public.error))
    if len(b)!=1 or not b[0]:raise CarryLayoutError(f'{what}: error [B] required, got {b}')
    return b[0]


def fill_shown_types(state,public,tables):
    """The public carry with only shown_types filled (module docstring): per (side, public slot) whose row is (0, 0), the
    hook's species_types[species] for a known species (0 < species < S), or, for a side's active whose engine types differ
    from its base types, those types (the color_change hook's write); any other row unchanged. Host numpy; the same
    carry object when nothing changes. CarryLayoutError on another layout."""
    b=_trackers(public);st=tables.species_types if type(tables)is CarryTables else None
    if st is None:raise CarryLayoutError('CarryTables required')
    for name,value,shape in(('species',public.species,(b,2,6)),('physical_to_public',public.physical_to_public,(b,2,6)),
            ('shown_types',public.shown_types,(b,2,6,2)),('state.active',state.active,(b,2)),('state.types',state.types,(b,2,6,2)),
            ('state.base_types',state.base_types,(b,2,6,2))):_leaf(value,shape,np.int32,name)
    species,shown,mapping,active,types,base=(_host(x)for x in(public.species,public.shown_types,public.physical_to_public,state.active,state.types,state.base_types))
    n=st.shape[0];known=(species>0)&(species<n);values=st[np.clip(species,0,n-1)].copy();retyped=np.zeros(species.shape,bool)
    rows=np.arange(b)
    for side in(0,1):
        physical=active[:,side];p=np.clip(physical,0,5);slot=mapping[rows,side,p];at=np.clip(slot,0,5)
        current,original_types=types[rows,side,p],base[rows,side,p]
        hit=(physical>=0)&(physical<6)&(slot>=0)&(slot<6)&np.any(current!=original_types,axis=-1)
        values[rows[hit],side,at[hit]]=current[hit];retyped[rows[hit],side,at[hit]]=True
    fill=(known|retyped)&np.all(shown==0,axis=-1)
    if not fill.any():return public
    filled=np.where(fill[...,None],values,shown).astype(np.int32)
    return public if np.array_equal(filled,shown)else public._replace(shown_types=jnp.asarray(filled))


def fill_negative_items(public):
    """The public carry with negative_items = engine_device_negative_items.empty(B) where from_b2 left None (the device
    path's initial value; exact at a root, module docstring); an existing one is checked and kept."""
    b=_trackers(public);carry=public.negative_items
    if carry is None:return public._replace(negative_items=negative_items.empty(b))
    if type(carry)is not negative_items.State:raise CarryLayoutError(f'negative_items: an engine_device_negative_items.State required, got {type(carry).__name__}')
    _leaf(carry.percent,(b,2,6),np.int32,'negative_items.percent');_leaf(carry.known,(b,2,6),np.bool_,'negative_items.known')
    return public


def orient_credits(public,owner):
    """The public carry with from_b2's owner-oriented [B] items-removed credits as per-side [B, 2] arrays (module docstring):
    items_removed[:, owner] = self, [:, 1 - owner] = opp; self = items_removed, opp = items_removed[:, ::-1]. A carry
    without them, or with [B, 2] ones already, is unchanged."""
    b=_trackers(public);fields=public.field_features
    if owner not in(0,1):raise CarryLayoutError(f'owner 0 or 1 required, got {owner!r}')
    if fields is None:return public
    if not isinstance(fields,dict):raise CarryLayoutError(f'field_features: a dict required, got {type(fields).__name__}')
    present=[k in fields for k in CREDIT_KEYS]
    if not any(present):return public
    if not all(present):raise CarryLayoutError('field_features: both items-removed credits or neither required')
    values=[fields[k]for k in CREDIT_KEYS];shapes={tuple(np.shape(x))for x in values};kinds={np.dtype(getattr(x,'dtype',None)).kind for x in values}
    if len(shapes)!=1 or kinds-{'i','u'}:raise CarryLayoutError(f'field_features: integer credits of one shape required, got {shapes} {kinds}')
    shape=shapes.pop()
    if shape==(b,2):return public
    if shape!=(b,):raise CarryLayoutError(f'field_features: [B] or [B, 2] credits required, got {shape}')
    own,opp=(_host(x)for x in values);removed=np.zeros((b,2),own.dtype);removed[:,owner]=own;removed[:,1-owner]=opp
    return public._replace(field_features={**fields,CREDIT_KEYS[0]:jnp.asarray(removed),CREDIT_KEYS[1]:jnp.asarray(removed[:,::-1].copy())})


# The owner's protocol prefix: the peer's request order and the owner's shown Spikes losses ------------------------------------------
# A public switch-in (Baton Pass's included: its recipient arrives on a |switch| line). |replace| counts as one: it never occurs in
# Gen 3; in Gen 5+ it ends an Illusion without moving anyone, so this rule is Gen 3's only.
SWITCH_KINDS=('switch','drag','replace')


OPPONENT_RECHARGE='OPPONENT_RECHARGE_UNREPRESENTED'
OPPONENT_RECHARGE_DETAIL='the opponent\'s active must recharge (line_facts.recharge) and the engine has no recharge turn'


def _constructor_reason(kind,detail=''):
    """build's reason for a root refused in this layer: RootRefusal's own format (constructor-root:<kind>[:<detail>]), so the live
    search counts it constructor_refused (depthn_live_search.refusal_cause) whichever of prepare_roots or build refuses it."""
    return 'constructor-root:'+kind+(':'+detail if detail and detail!=kind else '')


class PeerOrderUnavailable(ValueError):
    """The peer's public switch history does not join this world's party: the root is refused (PEER_REQUEST_ORDER_UNAVAILABLE)."""


@dataclass(frozen=True)
class LineFacts:
    """What the completion reads from the owner's protocol prefix (public lines only; line_facts):
    peer_switches: the peer side's switch-ins in protocol order, each the normalized species of a |switch| / |drag| /
    |replace| line's details; owner_spikes: the owner side's Spikes hits in order, (hp before, hp after, max HP) from the
    owner's own exact HP lines, or None when one hit's previous HP is not on the lines (its shown credit is then left 0);
    baton: per side (p1, p2), whether its current active arrived by Baton Pass (the env's and B2's host rule: a |switch|
    after that side's |move| Baton Pass with no other move of the side between, or a switch line naming Baton Pass);
    owner_moves: the owner's PP-charged move declarations in order (a |move| line of the owner without a [from] tag, not
    Struggle, the actor not transformed), each (actor species, move, the peer's active species, whether that active was
    alive, whether its Pressure was public: revealed by an -ability line, a Trace copy until it next arrives); recharge: per
    side, whether its active must recharge (B2's public_active.project_line: set by -mustrecharge, cleared by that side's
    |cant| recharge, arrival or faint); tracers: per side, the normalized species of each member that announced Trace (the
    SUBJECT of an -ability line tagged [from] ability: Trace: Showdown's setAbility line
    |-ability|<tracer>|<copied>|<tracer's own ability>|[from] ability: Trace|[of] <donor>), sorted."""
    peer_switches:tuple
    owner_spikes:tuple|None
    baton:tuple=(False,False)
    owner_moves:tuple=()
    recharge:tuple=(False,False)   # per side (p1, p2): its active must recharge (B2's public_active rule, line_facts)
    tracers:tuple=((),())          # per side (p1, p2): the species that announced Trace on the lines (line_facts)


def _seat_name(field):
    seat,_,name=str(field).partition(':')
    return seat.strip()[:2],name.strip()


def _condition_hp(condition):
    """(hp, max HP or None) of a protocol condition ('250/300', '250/300 par', '0 fnt'); ValueError otherwise."""
    head=(str(condition).split()or[''])[0]
    if head=='0':return 0,None
    hp,maximum=head.split('/')
    return int(hp),int(maximum)


def line_facts(lines,owner):
    """LineFacts of the owner's protocol prefix `lines` (the request's own lines) for owner seat `owner` (0 or 1)."""
    if owner not in(0,1):raise ValueError(f'owner 0 or 1 required, got {owner!r}')
    own,peer=f'p{owner+1}',f'p{2-owner}';switches=[];spikes=[];hp={};complete=True
    species={};active={'p1':'','p2':''};alive={'p1':False,'p2':False};pending=set();baton={'p1':False,'p2':False}
    transformed=set();original_pressure=set();running_pressure=set();moves=[];recharge={'p1':False,'p2':False}
    tracers={'p1':set(),'p2':set()}
    for line in lines:
        parts=str(line).split('|')
        if len(parts)<3 or parts[0]!='':continue
        kind=parts[1];seat,name=_seat_name(parts[2]);tags=[x.strip().lower()for x in parts[4:]]
        if kind=='faint'and seat in alive:alive[seat]=recharge[seat]=False   # |faint|IDENT: three fields
        if kind=='-mustrecharge'and seat in recharge:recharge[seat]=True   # |-mustrecharge|IDENT
        if kind=='cant'and seat in recharge and len(parts)>3 and normalize_id(parts[3])=='recharge':recharge[seat]=False
        if len(parts)<4:continue
        if kind=='-ability'and seat in tracers and any(t.startswith('[from] ability: trace')for t in tags):   # the subject traced
            tracers[seat].add(species.get((seat,name))or normalize_id(name))
        if kind=='move'and seat in active:
            move=normalize_id(parts[3])
            if move=='batonpass':pending.add(seat)
            else:pending.discard(seat)
            if (seat==own and not any(t.startswith('[from]')for t in tags)and move!='struggle'and seat not in transformed
                    and active[own]):
                foe=active[peer];moves.append((active[own],move,foe,alive[peer],foe in original_pressure or foe in running_pressure))
        elif kind=='-transform'and seat in active:transformed.add(seat)
        elif kind=='-ability'and seat==peer and normalize_id(parts[3])=='pressure':
            target=species.get((seat,name),'')
            (running_pressure if any(t.startswith('[from] ability: trace')for t in tags)else original_pressure).add(target)
        if kind in SWITCH_KINDS and seat in active:
            arrived=normalize_id(parts[3].split(',')[0]);species[(seat,name)]=arrived;active[seat]=arrived;alive[seat]=True
            transformed.discard(seat);running_pressure.discard(arrived);recharge[seat]=False
            baton[seat]=kind=='switch'and(seat in pending or any('baton pass'in t for t in tags));pending.discard(seat)
        if kind in SWITCH_KINDS:
            if seat==peer:switches.append(normalize_id(parts[3].split(',')[0]))
            elif seat==own and len(parts)>4:
                try:hp[name]=_condition_hp(parts[4])
                except ValueError:hp.pop(name,None)
        elif kind in('-damage','-heal','-sethp')and seat==own:
            try:new,maximum=_condition_hp(parts[3])
            except ValueError:hp.pop(name,None);continue
            before=hp.get(name);maximum=maximum if maximum is not None else(before[1]if before else None)
            if kind=='-damage'and any(x.strip().lower()=='[from] spikes'for x in parts[4:]):
                if before is None or maximum is None:complete=False
                else:spikes.append((before[0],new,maximum))
            hp[name]=(new,maximum)
    return LineFacts(tuple(switches),tuple(spikes)if complete else None,(baton['p1'],baton['p2']),tuple(moves),(recharge['p1'],recharge['p2']),
        (tuple(sorted(tracers['p1'])),tuple(sorted(tracers['p2']))))


def _member(names,want,where):
    found=[p for p,n in enumerate(names)if n and n==want]
    if not found:found=[p for p,n in enumerate(names)if n and(want.startswith(n)or n.startswith(want))]   # a forme suffix on one side
    if len(found)!=1:raise PeerOrderUnavailable(f'{want}: {len(found)} members of {where} match a public switch-in')
    return found[0]


def order_peer(state,public,owner,facts,tables):
    """The public carry with own_order[peer] (the peer = 1 - owner) the peer's Showdown request order: the world's physical
    party order (the materialized team's, phase_d_protocol_world's start) with each public peer switch-in, in protocol order,
    swapped into position 0 (Showdown's switchIn; the device install hook's rule). Unrevealed members never move. Members
    join by normalized species (one per side: species clause); a switch-in matching no member or several, one not a
    revealed member of this world, or a final position 0 that is not the world's peer active raises PeerOrderUnavailable.
    Host numpy, B = 1; the same carry when the order is already this one."""
    b=_trackers(public);peer=1-owner
    if b!=1:raise CarryLayoutError(f'one world per carry required, got B={b}')
    for name,value,shape in(('own_order',public.own_order,(1,2,6)),('physical_to_public',public.physical_to_public,(1,2,6)),
            ('state.species',state.species,(1,2,6)),('state.active',state.active,(1,2))):_leaf(value,shape,np.int32,name)
    if type(facts)is not LineFacts:raise CarryLayoutError('LineFacts required')
    if not tables.species_names:raise CarryLayoutError('CarryTables.species_names required')
    if not facts.peer_switches:raise PeerOrderUnavailable('no public switch-in of the peer on the lines')
    species=_host(state.species)[0,peer];mapping=_host(public.physical_to_public)[0,peer];active=int(_host(state.active)[0,peer])
    n=len(tables.species_names);names=[tables.species_names[int(x)]if 0<int(x)<n else''for x in species]
    order=list(range(6))
    for want in facts.peer_switches:
        p=_member(names,want,'the peer\'s party')
        if mapping[p]<0:raise PeerOrderUnavailable(f'{want}: switched in publicly, not a revealed member of this world')
        at=order.index(p);order[0],order[at]=order[at],order[0]
    if order[0]!=active:raise PeerOrderUnavailable(f'the last public switch-in {names[order[0]]} is not the world\'s peer active {names[active] if 0<=active<6 else active}')
    current=_host(public.own_order)
    if np.array_equal(current[0,peer],order):return public
    current=current.copy();current[0,peer]=order
    return public._replace(own_order=jnp.asarray(current))


def fill_owner_spikes(public,owner,facts):
    """The public carry with hazards[owner, shown] (the peer's OPP_HAZARD_CREDIT) filled where from_b2 left it zero: the
    owner's Spikes hits on the lines, each (_percent(hp before) - _percent(hp after)) / 100 in float32 (the peer sees the
    owner's HP as Showdown's public percent), accumulated by the device's own compensated sum
    (engine_device_damage_history._add), as engine_device_public_port.spikes would have. Unchanged when the hits are not
    all on the lines (facts.owner_spikes None), when there is none, or when the half is already non-zero."""
    from .engine_device_damage_history import _add
    b=_trackers(public);port=public.public_port
    if port is None or facts is None or not facts.owner_spikes:return public
    _leaf(port.hazards,(b,2,2,2),np.float32,'public_port.hazards')
    hazards=_host(port.hazards)
    if np.any(hazards[:,owner,0]!=0):return public
    pair=jnp.zeros(2,jnp.float32)
    for before,after,maximum in facts.owner_spikes:
        drop=max(0,int(public_trackers._percent(before,maximum))-int(public_trackers._percent(after,maximum)))
        pair=_add(pair,jnp.asarray(drop,jnp.float32)/100)
    hazards=hazards.copy();hazards[:,owner,0]=np.asarray(pair)
    if not np.any(hazards[:,owner,0]):return public
    return public._replace(public_port=port._replace(hazards=jnp.asarray(hazards)))


def fill_baton(public,facts):
    """The public carry with last_moves -2 (the device's Baton Pass arrival sentinel, engine_device_public_hooks.baton_copy)
    for each side whose current active arrived by Baton Pass (facts.baton) and still reads -1 (from_b2's plain switch, its
    last public action a switch); every other value unchanged."""
    b=_trackers(public)
    if facts is None or not any(facts.baton):return public
    _leaf(public.last_moves,(b,2,6),np.int32,'last_moves');_leaf(public.active_slot,(b,2),np.int32,'active_slot')
    last=_host(public.last_moves);slots=_host(public.active_slot);new=last.copy()
    for side,arrived in enumerate(facts.baton):
        at=int(slots[0,side])
        if arrived and 0<=at<6 and new[0,side,at]==-1:new[0,side,at]=-2
    return public if np.array_equal(new,last)else public._replace(last_moves=jnp.asarray(new))


def fill_recharge(public,facts):
    """The public carry with the mustrecharge volatile set on each side's active that must recharge (facts.recharge). B2 keeps
    that fact in its own flag beside the volatile bag (public_active.ActiveState.recharge, merged into the bag only when it
    encodes: public_active.project, `bag=state.volatiles.at[...,RECHARGE_ID].set(state.recharge)`), so from_b2's copy of the
    bag (phase_d_live_worlds.py:311) drops it and the encoder showed no volatile:mustrecharge where B2 does. No device hook
    writes or clears the bit and the engine models no recharge turn, so at every in-search node after the root the bit stays
    set on that Pokemon while it stays in (the install hook's bag reset clears it on a switch)."""
    from .public_active import NAMES
    b=_trackers(public)
    if facts is None or not any(facts.recharge):return public
    _leaf(public.public_volatiles,(b,2,6,len(NAMES)),np.bool_,'public_volatiles');_leaf(public.active_slot,(b,2),np.int32,'active_slot')
    bag=_host(public.public_volatiles);slots=_host(public.active_slot);new=bag.copy();bit=NAMES.index('mustrecharge')
    for side,must in enumerate(facts.recharge):
        at=int(slots[0,side])
        if must and 0<=at<6:new[0,side,at,bit]=True
    return public if np.array_equal(new,bag)else public._replace(public_volatiles=jnp.asarray(new))


def _name_index(names,want,*,move=False):
    """The one index of `names` naming `want`: exact, else (a move) any Hidden Power for 'hiddenpower...', else (a species) a
    unique forme prefix; None otherwise."""
    found=[i for i,n in enumerate(names)if n and n==want]
    if not found and move and want.startswith('hiddenpower'):found=[i for i,n in enumerate(names)if n.startswith('hiddenpower')][:1]
    if not found and not move:found=[i for i,n in enumerate(names)if n and(want.startswith(n)or n.startswith(want))]
    return found[0]if len(found)==1 else None


def fill_peer_pressure(state,public,owner,facts,tables,root):
    """The public carry with pp_used[owner] (the peer's view of the owner's PP) charged as the peer counts it: each owner
    declaration (facts.owner_moves) against a live peer active whose Pressure was not public costs one more PP when this
    world gives that active Pressure (its base ability) and the move is foe-targeted (tables.foe_targeted), the device move
    hook's rule (the receiving observer knows its own active's ability; engine_device_public_hooks.move) and the env's
    (Foul Play counts its own Pressure). Gen 3 Pressure is silent, so B2's public rule (public_abilities.move_context:
    revealed Pressure only) charged one. The count starts from `root`'s pp_used (from_b2's, B2's own), so it is the same
    on every pass; unchanged without root or facts, or when a move or member does not join."""
    b=_trackers(public)
    if facts is None or root is None or not facts.owner_moves or not tables.pressure:return public
    peer=1-owner
    for name,value,shape in(('pp_used',public.pp_used,(1,2,6,4)),('observed_moves',public.observed_moves,(1,2,6,4)),
            ('root pp_used',root.pp_used,(1,2,6,4)),('state.base_ability',state.base_ability,(1,2,6)),('state.species',state.species,(1,2,6)),
            ('species',public.species,(1,2,6))):_leaf(value,shape,np.int32,name)
    n=len(tables.species_names);names=lambda ids:[tables.species_names[int(x)]if 0<int(x)<n else''for x in ids]
    peers=names(_host(state.species)[0,peer]);ability=_host(state.base_ability)[0,peer]
    owners=names(_host(public.species)[0,owner]);observed=_host(public.observed_moves)[0,owner]
    m=len(tables.move_names);target=_host(root.pp_used).copy()
    for actor,move,foe,alive,public_pressure in facts.owner_moves:
        if not alive or public_pressure or not foe:continue
        p=_name_index(peers,foe);a=int(ability[p])if p is not None else 0
        if p is None or not(0<a<len(tables.pressure)and tables.pressure[a]):continue
        engine=_name_index(tables.move_names,move,move=True)
        if engine is None or not tables.foe_targeted[engine]:continue
        slot=_name_index(owners,actor)
        if slot is None:continue
        column=_name_index([tables.move_names[int(x)]if 0<int(x)<m else''for x in observed[slot]],move,move=True)
        if column is not None:target[0,owner,slot,column]+=1
    current=_host(public.pp_used)
    if np.array_equal(current[0,owner],target[0,owner]):return public
    current=current.copy();current[0,owner]=target[0,owner]
    return public._replace(pp_used=jnp.asarray(current))


def fix_peer_base_ability(state,public,owner):
    """The world state with each revealed peer member's base_ability its public original ability where the materialized
    spec wrote the publicly borrowed one into it (engine_world._resolved_ability: a Trace holder's revealedAbility is the
    copied ability): the peer's own ability column (engine_device_observation.encode reads base_ability) then shows the
    set's ability, as the env and the Gen 3 request's baseAbility do, and the engine reverts to it on switch-out. Running
    abilities, the owner side and every other member are unchanged. It needs the member's public ORIGINAL ability, which
    B2's fold never records for a Trace holder (fix_trace_base_ability, from the lines, covers that case on both sides)."""
    b=_trackers(public);peer=1-owner
    if public.original_ability is None or public.borrowed_ability is None:return state
    for name,value,shape in(('original_ability',public.original_ability,(b,2,6)),('borrowed_ability',public.borrowed_ability,(b,2,6)),
            ('physical_to_public',public.physical_to_public,(b,2,6)),('state.base_ability',state.base_ability,(b,2,6))):_leaf(value,shape,np.int32,name)
    original,borrowed,mapping=(_host(x)[0,peer]for x in(public.original_ability,public.borrowed_ability,public.physical_to_public))
    base=_host(state.base_ability);new=base.copy()
    for p in range(6):
        slot=int(mapping[p])
        if 0<=slot<6 and borrowed[slot]and original[slot]and original[slot]!=borrowed[slot]and new[0,peer,p]==borrowed[slot]:
            new[0,peer,p]=original[slot]
    return state if np.array_equal(new,base)else state._replace(base_ability=jnp.asarray(new))


TRACE='trace'


def fix_trace_base_ability(state,owner,facts,tables):
    """The world state with each member that announced Trace on the lines (facts.tracers, both sides) holding Trace as its
    base_ability. The materializer writes a Trace holder's CURRENT copy into both abilities of the active row
    (local_showdown._apply_traced_ability_materialization_state sets revealedAbility to the copy, engine_world._resolved_ability
    takes it as the spec's ability, v5_hypothetical_state.map_spec defaults base_ability to it), and B2's fold never records a
    tracer's own ability (pokezero.belief._trace_copy_source; public_abilities.trace_copy writes the donor's original and the
    tracer's running override only), so fix_peer_base_ability's original-ability rule cannot fire for a tracer. The line is
    the proof: its subject's own ability is Trace. Each side's own ability column (engine_device_observation reads
    base_ability for own mons) then shows Trace, as the env and B2 do, and the engine reverts to Trace on switch-out. The
    running ability (the active's copy) is unchanged. Members join by species as order_peer does (PeerOrderUnavailable when
    a tracer does not join exactly one member); CarryLayoutError when the tables have no ability names or no Trace."""
    if facts is None or not any(facts.tracers):return state
    if not tables.species_names or not tables.ability_names:raise CarryLayoutError('CarryTables.species_names and ability_names required')
    if TRACE not in tables.ability_names:raise CarryLayoutError('Trace is outside the engine ability vocabulary')
    trace=tables.ability_names.index(TRACE)
    for name,value in(('state.species',state.species),('state.base_ability',state.base_ability)):_leaf(value,(1,2,6),np.int32,name)
    species=_host(state.species);base=_host(state.base_ability);new=base.copy();n=len(tables.species_names)
    for side,tracers in enumerate(facts.tracers):
        names=[tables.species_names[int(x)]if 0<int(x)<n else''for x in species[0,side]]
        for want in tracers:new[0,side,_member(names,want,f'side p{side+1}\'s party')]=trace
    return state if np.array_equal(new,base)else state._replace(base_ability=jnp.asarray(new))


def complete_public(state,public,tables,owner,*,facts=None,root=None):
    """A constructor world's public carry completed (module docstring): fill_shown_types, fill_negative_items,
    orient_credits, and with the root's LineFacts order_peer, fill_owner_spikes, fill_baton, fill_recharge and (with the root's
    own carry, `root`) fill_peer_pressure. Pure; the same carry when nothing changes. CarryLayoutError on another carry layout,
    PeerOrderUnavailable when the peer order cannot be joined."""
    _trackers(public)
    done=orient_credits(fill_negative_items(fill_shown_types(state,public,tables)),owner)
    if facts is not None:
        done=fill_recharge(fill_baton(fill_owner_spikes(order_peer(state,done,owner,facts,tables),owner,facts),facts),facts)
        done=fill_peer_pressure(state,done,owner,facts,tables,root)
    return done


def complete_carry(state,public,tables,owner,*,facts=None,root=None):
    """(state, public) of one constructor world completed: fix_peer_base_ability and (with the root's LineFacts)
    fix_trace_base_ability on the state first, then complete_public on
    the carry against the CORRECTED state (fill_peer_pressure reads the peer's base abilities: a Trace holder running a
    copied Pressure has Trace as its own ability, so earlier moves against it are not recharged, and a second pass, which
    sees the corrected state, gives the same carry)."""
    fixed=fix_trace_base_ability(fix_peer_base_ability(state,public,owner),owner,facts,tables)
    return fixed,complete_public(fixed,public,tables,owner,facts=facts,root=root)


def complete_world(world,tables,owner,*,facts=None,root=None):
    """A constructor world completed (complete_carry), `owner` the root's owner, `facts` its LineFacts (None: the fills
    only) and `root` the root's own public carry (the peer Pressure count's base); a refused world (no state) and an
    already completed one are returned as they are. Pure: the world's other leaves are the same objects."""
    if world.state is None:return world
    public=world.public
    _trackers(public,f'world {world.index} public carry')
    state,done=complete_carry(world.state,public,tables,owner,facts=facts,root=root)
    return world if done is public and state is world.state else replace(world,state=state,public=done)


# Prepare ----------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Prepared:
    """One root's ProtocolConstructor.prepare output, its synced wall seconds, the binding of the constructor that made it
    and the LineFacts of its request's lines (None for a request without lines)."""
    request_id:object
    detached:object
    made:object
    legal:tuple
    seconds:float
    binding:object
    facts:object=None


@dataclass(frozen=True)
class RootRefusal:
    """A root prepare() refused (no worlds): the refusal's kind and detail, and the constructor's binding."""
    request_id:object
    kind:str
    detail:str
    seconds:float
    binding:object


def _complete(complete,world,owner,facts,root=None):
    return complete(world,owner)if facts is None else complete(world,owner,facts=facts,root=root)


def _completed(made,complete,owner,facts=None,root=None):
    """The world set with every constructed world completed; the same object when no world changes."""
    worlds=tuple(w if w.state is None else _complete(complete,w,owner,facts,root)for w in made.worlds)
    return made if all(a is b for a,b in zip(worlds,made.worlds))else replace(made,worlds=worlds)


def prepare_roots(constructor,protocol_requests,*,start_index=0,device='cpu'):
    """prepare() per protocol request (index start_index + i) under `device` (default the CPU; None keeps the process
    default), each world's carry completed (world_completer, module docstring) and its arrays synced before its clock
    stops. Returns one Prepared or RootRefusal per request, each stamped with constructor_binding(constructor) (build
    refuses a stamp other than its constructor's). CarryLayoutError (the completion) is never a refusal."""
    stamp=constructor_binding(constructor)
    if device is None:context=nullcontext()
    else:
        try:context=jax.default_device(jax.devices(device)[0])
        except RuntimeError as error:raise ValueError(f'device {device!r} unavailable ({error}); pass device=None for the process default')from None
    out=[];complete=None
    with context:
        for i,request in enumerate(protocol_requests):
            rid=request.get('request_id')if isinstance(request,dict)else None;clock=time.perf_counter()
            try:result=constructor.prepare(request,start_index+i)
            except ROOT_ERRORS as error:
                out.append(RootRefusal(rid,str(getattr(error,'kind',type(error).__name__)),str(getattr(error,'detail',error)),time.perf_counter()-clock,stamp))
                continue
            # Outside the refusal net: a changed prepare() interface raises here instead of refusing every root.
            if type(result)is not tuple or len(result)!=3:
                raise TypeError(f'ProtocolConstructor.prepare must return (Root, WorldSet, legal mask), got {type(result).__name__}')
            detached,made,legal=result
            if complete is None:complete=world_completer(constructor)
            lines=request.get('lines')if isinstance(request,dict)else None
            facts=None if lines is None else line_facts(lines,detached.address.owner)
            if facts is not None and facts.recharge[1-detached.address.owner]:   # module docstring: never priced
                out.append(RootRefusal(rid,OPPONENT_RECHARGE,OPPONENT_RECHARGE_DETAIL,time.perf_counter()-clock,stamp));continue
            try:made=_completed(made,complete,detached.address.owner,facts,getattr(detached,'public',None))
            except PeerOrderUnavailable as error:
                out.append(RootRefusal(rid,'PEER_REQUEST_ORDER_UNAVAILABLE',str(error),time.perf_counter()-clock,stamp));continue
            jax.block_until_ready([(w.state,w.public)for w in made.worlds if w.state is not None])
            out.append(Prepared(rid,detached,made,tuple(map(bool,legal)),time.perf_counter()-clock,stamp,facts))
    return tuple(out)


# Build ------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Batch:
    """DepthN input for prepared roots: `requests` (one per block of two worlds, root-major), `worlds` (PrebuiltWorlds,
    pairs aligned with requests), `roots` (one dict per prepared root: status PRICED with its request `rows`, or REFUSED
    with a reason), `count` W and `blocks` K = W/2."""
    requests:tuple
    worlds:object
    roots:tuple
    count:int
    blocks:int


_ROOT_FORWARDS={}
ROOT_FORWARD_CACHE=8


def _root_forward(forward):
    """One jitted root forward per evaluator object, the ROOT_FORWARD_CACHE most recent kept (each entry holds its
    evaluator, so a live entry's id is never reused)."""
    held=_ROOT_FORWARDS.get(id(forward))
    if held is None or held[0]is not forward:
        while len(_ROOT_FORWARDS)>=ROOT_FORWARD_CACHE:_ROOT_FORWARDS.pop(next(iter(_ROOT_FORWARDS)))
        held=_ROOT_FORWARDS[id(forward)]=(forward,jax.jit(lambda p,x:forward(p,x)))
    return held[1]


def _v4_prefix(inputs):
    """The 23-token v4 prefix of V2 encoder inputs (champion_evaluator's slice), for an evaluator without its own view."""
    t=evaluators.TOKENS
    return dict(numeric_features=inputs['numeric_features'][...,:t,:],categorical_ids=inputs['categorical_ids'][...,:t,:],
        token_type_ids=inputs['token_type_ids'][...,:t],attention_mask=inputs['attention_mask'][...,:t])


def b2_root_inputs(detached):
    """B2's own encoding of the literal root, detached.view['_b2_root_inputs'], or None when the root carries none."""
    view=getattr(detached,'view',None)
    return view.get('_b2_root_inputs')if hasattr(view,'get')else None


def _bits(a):
    a=np.ascontiguousarray(np.asarray(a))
    return a.view({1:np.uint8,2:np.uint16,4:np.uint32,8:np.uint64}[a.itemsize])if a.dtype.kind=='f'else a


def _prefix_index(key,tokens=TOKENS):
    return(Ellipsis,slice(0,tokens))+((slice(None),)if TOKEN_KEYS[key]==2 else())


def splice(root_inputs,b2,tokens=TOKENS):
    """(X, lossy keys): copies of the V2 root inputs with the first `tokens` tokens of each token tensor (TOKEN_KEYS) replaced
    by B2's (cast to the root's dtype); history_mask and tokens >= `tokens` unchanged. A key whose B2 values do not survive
    the cast is listed. ValueError when the root lacks a token key or B2's tensor does not fit the root's prefix; KeyError
    when B2 lacks a token key."""
    missing=[k for k in TOKEN_KEYS if k not in root_inputs]
    if missing:raise ValueError('root inputs without '+', '.join(missing))
    out,lossy={},[]
    for key,value in root_inputs.items():
        a=np.array(value,copy=True)
        if key in TOKEN_KEYS:
            source=np.asarray(b2[key]);index=_prefix_index(key,tokens);target=a[index]
            if source.shape[0]!=tokens or source.shape!=target.shape[target.ndim-source.ndim:]:
                raise ValueError(f'{key}: B2 {source.shape} does not fit the root prefix {target.shape}')
            cast=source.astype(a.dtype);back=cast.astype(source.dtype)
            if not(back.shape==source.shape and np.array_equal(_bits(back),_bits(source))):lossy.append(key)
            a[index]=cast
        out[key]=a
    return out,lossy


def b2_view_check(forward,root_inputs,detached):
    """Diagnostic, never a gate: the evaluator's view of root_inputs against B2's own encoding of the literal root
    (detached.view['_b2_root_inputs']). None when the root carries no B2 encoding; else dict(equal, keys: per B2 token key
    equal, numeric_max_abs_diff), or dict(equal=None, error=...) when the two cannot be compared."""
    b2=b2_root_inputs(detached)
    if b2 is None:return None
    try:
        seen=evaluators.digest_view(forward)
        mine=seen(root_inputs)if seen is not None else _v4_prefix(root_inputs)
        keys={}
        for k in B2_TOKEN_KEYS:
            a=np.asarray(mine[k]);b=np.asarray(b2[k])
            keys[k]=bool(a.size==b.size and np.array_equal(a.reshape(b.shape),b))
        a=np.asarray(mine['numeric_features'],np.float64);b=np.asarray(b2['numeric_features'],np.float64)
        diff=float(np.max(np.abs(a.reshape(b.shape)-b)))if a.size==b.size and b.size else None
        return dict(equal=all(keys.values()),keys=keys,numeric_max_abs_diff=diff)
    except(KeyError,ValueError,TypeError,IndexError)as error:
        return dict(equal=None,error=f'{type(error).__name__}: {error}')


def build(constructor,prepared,pricer,params,*,binding,recipe,learner_actor=0,opponent_actor=0,only_complete=False,root_prior='b2'):
    """Requests and pre-built worlds for `prepared` (prepare_roots' output for this constructor), for `pricer` (its executor
    gives the root view, its evaluator the root prior: on B2's encoding, else world 0's view, for root_prior 'b2'; on world
    0's view for 'world0') under the pricer-side `binding`. Every world's carry is completed first (a no-op on prepare_roots' worlds). ValueError on a binding mismatch
    (the pricer's, the constructor's, or a prepared root stamped by another constructor), a champion evaluator not bound
    to the constructor's catalog, a malformed world set or a bad actor index; CarryLayoutError from the completion;
    per-root refusals are recorded, never raised (module docstring). A priced root records root_view_source ('b2' or
    'world0') and b2_view."""
    built=constructor_binding(constructor);_check_binding(built,binding,'constructor worlds are built under another binding than the pricer\'s')
    for item in prepared:
        _check_binding(getattr(item,'binding',None),built,'a prepared root comes from a constructor with another binding than this one\'s')
    if isinstance(pricer.forward,evaluators.ChampionEvaluator):evaluators.check_binding(pricer.forward,constructor.catalog)
    if type(recipe)is not original.Recipe:raise ValueError('a random recipe (lane order) required')
    if any(type(x)is not int or not 0<=x<len(params)for x in(learner_actor,opponent_actor)):raise ValueError('verified actor index required')
    if root_prior not in ROOT_PRIORS:raise ValueError(f'root_prior in {ROOT_PRIORS} required, got {root_prior!r}')
    W=constructor.count
    if type(W)is not int or W<2 or W%2:raise ValueError('an even declared world budget required')
    K=W//2;views=depth2._make_views(pricer.executor);forward=_root_forward(pricer.forward)
    requests=[];pairs=[];roots=[];complete=None
    for index,item in enumerate(prepared):
        base=dict(index=index,request_id=item.request_id,prepare_seconds=item.seconds,worlds=W,rows=())
        if type(item)is RootRefusal:
            roots.append(dict(base,status='REFUSED',reason=_constructor_reason(item.kind,item.detail),constructed=0,refused_worlds=None,
                refused_world_share=None));continue
        if type(item)is not Prepared:raise ValueError('prepare_roots output required')
        made,detached=item.made,item.detached
        if (made.requested_worlds!=W or len(made.worlds)!=W or len(made.seeds)!=W or made.root is not detached
                or tuple(w.index for w in made.worlds)!=tuple(range(W))):
            raise ValueError('a world set of the declared budget, worlds 0..W-1, for its own root required')
        if detached.source_binding_sha256!=binding.source_binding_sha256:raise ValueError('root source binding differs from the pricer\'s')
        if item.facts is not None and item.facts.recharge[1-detached.address.owner]:   # a Prepared made elsewhere: module docstring
            roots.append(dict(base,status='REFUSED',reason=_constructor_reason(OPPONENT_RECHARGE,OPPONENT_RECHARGE_DETAIL),constructed=0,
                refused_worlds=None,refused_world_share=None));continue
        if complete is None:complete=world_completer(constructor)
        # A world without a state always names its refusal, so it stays this root's refusal (the pricer refuses the frame otherwise).
        try:worlds=tuple(_complete(complete,w,detached.address.owner,item.facts,getattr(detached,'public',None))if w.state is not None else w if w.refusals else replace(w,refusals=(UNNAMED,))
            for w in made.worlds)
        except PeerOrderUnavailable as error:
            roots.append(dict(base,status='REFUSED',reason=_constructor_reason('PEER_REQUEST_ORDER_UNAVAILABLE',str(error)),constructed=0,refused_worlds=None,
                refused_world_share=None));continue
        refused=tuple(w.index for w in worlds if w.state is None);kinds=';'.join(sorted({r for w in worlds for r in w.refusals}))
        base.update(address=(detached.address.stream,detached.address.frame,detached.address.owner),constructed=W-len(refused),refused_worlds=refused,
            refused_world_share=len(refused)/W)
        first=next((w for w in worlds if w.state is not None),None)
        if first is None or only_complete and refused:
            roots.append(dict(base,status='REFUSED',reason='WORLD_CONSTRUCTOR_REFUSED:'+kinds));continue
        state,public=jax.device_get((first.state,first.public))
        own,_,inputs,_=views(state,public,detached.address.owner)
        if int(np.asarray(own.error).reshape(-1)[0])!=0:
            roots.append(dict(base,status='REFUSED',reason='ROOT_VIEW_ENCODER_ERROR'));continue
        root_inputs={k:np.asarray(v)for k,v in inputs.items()};legal=item.legal
        base['b2_view']=b2_view_check(pricer.forward,root_inputs,detached)
        if len(legal)!=9 or not any(legal):
            roots.append(dict(base,status='REFUSED',reason='ROOT_LEGAL_EMPTY'));continue
        # The root prior reads B2's encoding of the literal root (Codex's ladder), else world 0's view (module docstring).
        b2=b2_root_inputs(detached)if root_prior=='b2'else None;prior_inputs=root_inputs;base['root_view_source']='world0'
        if b2 is not None:
            base['root_view_source']='b2'
            try:prior_inputs,lossy=splice(root_inputs,b2)
            except(KeyError,ValueError,TypeError,IndexError)as error:
                roots.append(dict(base,status='REFUSED',reason=f'ROOT_B2_INPUTS_UNUSABLE: {type(error).__name__}: {error}'));continue
            if lossy:
                roots.append(dict(base,status='REFUSED',reason='ROOT_B2_INPUTS_UNUSABLE: lossy cast of '+','.join(lossy)));continue
        output=jax.device_get(forward(params[learner_actor],prior_inputs))
        logits=tuple(map(float,np.asarray(output.policy_logits,np.float32).reshape(-1)));value=float(np.asarray(output.value).reshape(-1)[0])
        if not(bool(np.asarray(output.valid).reshape(-1)[0])and all(np.isfinite(logits))and np.isfinite(value)and abs(value)<=1):
            roots.append(dict(base,status='REFUSED',reason='ROOT_FORWARD_REFUSED'));continue
        rows=[]
        for b in range(K):
            pair=worlds[2*b:2*b+2]
            starts=tuple(plans.Start(w.index,made.seeds[w.index],None,(),tuple(w.refusals))for w in pair)
            # No D3 source-fixture draw exists for a constructor world: its unknown source-fixture mass is not defined (NaN).
            p=plans.Plans(detached,True,starts,constructor.source.source_hash,constructor.source.source_rows_sha256,float('nan'))
            rows.append(len(requests));requests.append(batch.Request(p,logits,legal,root_inputs,learner_actor,opponent_actor));pairs.append(pair)
        roots.append(dict(base,status='PRICED',reason=None,rows=tuple(rows),root_logits=logits,root_legal=legal,root_value=value,root_view_world=first.index))
    return Batch(tuple(requests),dn.PrebuiltWorlds(SOURCE,built,tuple(pairs)),tuple(roots),W,K)


# Price and reduce -------------------------------------------------------------------------------------------------------------------
def price(pricer,batch,binding,recipe,params,*,pass_pricer=None):
    """The batch's frame on `pricer` (all K blocks of every root in one call), and the full-width pass frame in `searched`
    mode (else None). `searched` needs `pass_pricer`: a DepthNPricer at depth 2, k 9, `prior`, with the pricer's executor,
    evaluator and k1 widths. Both passes take batch.worlds unchanged. A batch without requests returns (None, None)
    without calling a pricer (reduce takes it)."""
    if pricer.selection=='prior':
        if pass_pricer is not None:raise ValueError('a pass pricer belongs to the searched mode')
    elif (pass_pricer is None or pass_pricer.depth!=2 or pass_pricer.selection!='prior'or(pass_pricer.k_own,pass_pricer.k_opp)!=((9,),(9,))
            or pass_pricer.executor is not pricer.executor or pass_pricer.evaluator_suffix!=pricer.evaluator_suffix
            or(pass_pricer.k1_own,pass_pricer.k1_opp)!=(pricer.k1_own,pricer.k1_opp)):
        raise ValueError('searched needs a full-width depth-2 prior pass pricer with the same executor, evaluator and k1 widths')
    if not batch.requests:return None,None
    if pricer.selection=='prior':return pricer.price(batch.requests,binding,recipe,params,worlds=batch.worlds),None
    raw=pass_pricer.price(batch.requests,binding,recipe,params,worlds=batch.worlds)
    own,opp,_=dn.searched_maps(dn.as_depth2_frame(raw),*pricer.searched_k)
    return pricer.price(batch.requests,binding,recipe,params,selection_maps={2:(own,opp)},worlds=batch.worlds),raw


def root_actions(completed_q,policy_target,legal):
    """(action_mean_q, action_policy_target): the argmax over the legal actions of the mean completed-Q (the ladder's root
    operator, phase_d_ladder_search.reduce_q) and of the improved target, each the lower action on ties."""
    acts=[a for a in range(9)if legal[a]]
    return max(acts,key=lambda a:(completed_q[a],-a)),max(acts,key=lambda a:(policy_target[a],-a))


def reduce(frame,batch,*,rule=backups.learner_max):
    """Per prepared root: status LABELED with the K-block composition (completed_q, policy_target, root_value and the
    moves `action_mean_q` and `action_policy_target`, module docstring), or REFUSED with a cause (the build reason, or the
    first refused block's depthn_backups.refusal_summary cause, 'materialization' for a constructor world refusal, else
    'reduction'; the target range check's ValueError (RANGE_EDGE) refuses that root, its text in `reduction_error`, never
    raised; any other ValueError raises).
    `block_causes` lists each block's cause. `frame` is None for a batch without requests."""
    from .d3_search_latency import compose,reduce_block
    if frame is None and batch.requests:raise ValueError('a frame is required for a batch with requests')
    causes=backups.refusal_summary(frame)['roots']if batch.requests else()
    results=[]
    for root in batch.roots:
        out=dict(index=root['index'],request_id=root['request_id'],status='REFUSED',cause=root['reason'],worlds=batch.count,blocks=batch.blocks,
            block_causes=None,completed_q=None,policy_target=None,root_value=None,action_mean_q=None,action_policy_target=None,reduction_error=None)
        if root['status']!='PRICED':results.append(out);continue
        logits,legal=root['root_logits'],root['root_legal'];out['block_causes']=tuple(causes[row]['cause']for row in root['rows'])
        prices=[backups.backup(frame,row,rule)for row in root['rows']]
        try:composed=compose([reduce_block(pair,logits,legal,original)for pair in prices],logits,legal,original)
        except ValueError as error:   # the rounding edge of the target range check refuses this root only; anything else is a bug
            if str(error)!=RANGE_EDGE:raise
            out.update(cause=next((c for c in out['block_causes']if c),'reduction'),reduction_error=f'{type(error).__name__}: {error}')
            results.append(out);continue
        out.update(status=composed['status'],completed_q=composed['completed_q'],policy_target=composed['policy_target'],root_value=composed['root_value'])
        if composed['status']=='LABELED':
            mean_q,target=root_actions(composed['completed_q'],composed['policy_target'],legal)
            out.update(cause=None,action_mean_q=mean_q,action_policy_target=target)
        else:out['cause']=next((c for c in out['block_causes']if c),'reduction')
        results.append(out)
    return tuple(results)
