const state = {
  catalog: null,
  variantById: new Map(),
  variantsBySpecies: new Map(),
  scenario: null,
  selection: { p1: { filter: "", species: "", variant: "" }, p2: { filter: "", species: "", variant: "" } },
  lastValidation: null,
  dirty: false,
};

const $ = (selector) => document.querySelector(selector);

const STATUS_OPTIONS = [
  ["", "Healthy"],
  ["brn", "Burn"],
  ["par", "Paralysis"],
  ["psn", "Poison"],
  ["tox", "Bad poison"],
  ["slp", "Sleep"],
  ["frz", "Freeze"],
];

const SIDE_CONDITIONS = [
  ["spikes", "Spikes layers", 3],
  ["reflect", "Reflect turns", 5],
  ["lightscreen", "Light Screen turns", 5],
  ["safeguard", "Safeguard turns", 5],
  ["mist", "Mist turns", 5],
];

const VOLATILE_DEFINITIONS = [
  { id: "confusion", label: "Confusion", turns: [1, 4], elapsed: [0, 5] },
  { id: "substitute", label: "Substitute", hp: true },
  { id: "leechseed", label: "Leech Seed" },
  { id: "taunt", label: "Taunt", turns: [1, 2] },
  { id: "encore", label: "Encore", turns: [1, 6], elapsed: [0, 6], move: true },
  { id: "disable", label: "Disable", turns: [1, 5], move: true },
  { id: "torment", label: "Torment" },
  { id: "attract", label: "Attract" },
  { id: "nightmare", label: "Nightmare" },
  { id: "curse", label: "Ghost Curse" },
  { id: "ingrain", label: "Ingrain" },
  { id: "focusenergy", label: "Focus Energy" },
  { id: "yawn", label: "Yawn", turns: [1, 1] },
  { id: "perishsong", label: "Perish Song", turns: [1, 3] },
  { id: "flashfire", label: "Flash Fire boost" },
  { id: "mudsport", label: "Mud Sport" },
  { id: "watersport", label: "Water Sport" },
];

function blankSide() {
  return {
    construction_mode: "source-composed",
    generated_team_seed: null,
    active_slot: 0,
    pokemon: [],
    side_conditions: {},
    active_volatiles: [],
  };
}

function blankScenario() {
  return {
    schema_version: "endgame-scenario-v1",
    scenario_id: "untitled-endgame",
    title: "Untitled endgame",
    description: "Synthetic fully revealed endgame scenario.",
    tags: [],
    format_id: "gen3customgame",
    source_format_id: "gen3randombattle",
    seed: 1,
    turn: 1,
    provenance: { randbat_source_hash: state.catalog?.source_hash || "", replay_proven: false },
    knowledge_mode: "fully_revealed",
    perspective: "p1",
    side_to_move: "p1",
    field: { weather: "", turns_remaining: 0, permanent: false },
    teams: { p1: blankSide(), p2: blankSide() },
    objective: { kind: "forced_win", expected_root_actions: [], principal_variation: [], max_plies: 6, verification: { status: "unverified", engine: null, artifact: null } },
    author_notes: "",
  };
}

function normalizeScenario(scenario) {
  scenario.turn = Math.max(1, Number(scenario.turn) || 1);
  scenario.field ||= { weather: "", turns_remaining: 0, permanent: false };
  scenario.field.weather ||= "";
  scenario.field.turns_remaining = Number(scenario.field.turns_remaining) || 0;
  scenario.field.permanent = Boolean(scenario.field.permanent);
  ["p1", "p2"].forEach((sideId) => {
    const side = scenario.teams[sideId];
    side.side_conditions ||= {};
    side.active_volatiles ||= [];
    side.pokemon.forEach((pokemon) => { pokemon.status ||= { id: "" }; });
  });
  return scenario;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: options.body ? { "Content-Type": "application/json" } : undefined,
    ...options,
  });
  const payload = await response.json().catch(() => ({ error: { message: "Server returned invalid JSON." } }));
  if (!response.ok) {
    const error = payload.error || {};
    throw new Error(`${error.path ? `${error.path}: ` : ""}${error.message || "Request failed."}`);
  }
  return payload;
}

function notice(message, kind = "") {
  const element = $("#notice");
  element.textContent = message;
  element.className = `notice ${kind}`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

function selectedSide(sideId) { return state.scenario.teams[sideId]; }

function variantPokemon(variant) {
  return {
    variant_id: variant.variant_id,
    species: variant.species,
    level: variant.level,
    ability: variant.ability,
    item: variant.item,
    moves: variant.moves.map((move) => ({ id: move.id, pp: move.max_pp, max_pp: move.max_pp })),
    current_hp: variant.max_hp,
    max_hp: variant.max_hp,
    nature: "",
    gender: null,
    evs: {},
    ivs: {},
    status: { id: "" },
  };
}

function invalidate() { state.lastValidation = null; state.dirty = true; }
function markClean() { state.dirty = false; }

function metadataRender() {
  const scenario = state.scenario;
  $("#scenario-id").value = scenario.scenario_id;
  $("#scenario-title").value = scenario.title;
  $("#scenario-tags").value = scenario.tags.join(", ");
  $("#perspective").value = scenario.perspective;
  $("#side-to-move").value = scenario.side_to_move;
  $("#scenario-seed").value = scenario.seed;
  $("#scenario-turn").value = scenario.turn;
  $("#scenario-description").value = scenario.description;
  $("#field-weather").value = scenario.field.weather;
  $("#field-weather-turns").value = scenario.field.turns_remaining || "";
  $("#field-weather-turns").disabled = !scenario.field.weather || scenario.field.permanent;
  $("#field-weather-permanent").checked = scenario.field.permanent;
  $("#field-weather-permanent").disabled = !scenario.field.weather || scenario.field.weather === "hail";
  $("#objective-kind").value = scenario.objective.kind;
  $("#objective-plies").value = scenario.objective.max_plies;
  $("#objective-status").value = scenario.objective.verification.status;
  $("#expected-actions").value = scenario.objective.expected_root_actions.join(", ");
  $("#principal-variation").value = scenario.objective.principal_variation.join("\n");
  $("#author-notes").value = scenario.author_notes;
  renderBadges();
}

function renderBadges() {
  const valid = state.lastValidation?.validation;
  $("#validation-badges").innerHTML = [
    `<span class="badge ${valid?.set_valid ? "good" : "muted"}">set valid ${valid?.set_valid ? "" : "pending"}</span>`,
    `<span class="badge ${valid?.state_consistent ? "good" : "muted"}">state consistent ${valid?.state_consistent ? "" : "pending"}</span>`,
    `<span class="badge ${state.scenario.provenance.replay_proven ? "good" : "muted"}">replay proven ${state.scenario.provenance.replay_proven ? "" : "no"}</span>`,
  ].join("");
}

function blankSelection() {
  return { p1: { filter: "", species: "", variant: "" }, p2: { filter: "", species: "", variant: "" } };
}

function speciesOptions(current = "", filter = "") {
  const query = filter.trim().toLowerCase();
  const matches = state.catalog.species.filter((species) => !query || species.name.toLowerCase().includes(query) || species.id.includes(query));
  if (current && !matches.some((species) => species.id === current)) {
    const selected = state.catalog.species.find((species) => species.id === current);
    if (selected) matches.unshift(selected);
  }
  return [`<option value="">Choose species...</option>`, ...matches.map((species) => `<option value="${escapeHtml(species.id)}" ${species.id === current ? "selected" : ""}>${escapeHtml(species.name)}</option>`)].join("");
}

function variantOptions(speciesId, current = "") {
  const variants = state.variantsBySpecies.get(speciesId) || [];
  return [`<option value="">Choose a legal set...</option>`, ...variants.map((variant) => `<option value="${escapeHtml(variant.variant_id)}" ${variant.variant_id === current ? "selected" : ""}>${escapeHtml(variant.role)} - ${escapeHtml(variant.moves.map((move) => move.name).join(" / "))}</option>`)].join("");
}

function pokemonCard(sideId, pokemon, index, activeSlot) {
  const fainted = pokemon.current_hp === 0;
  const variant = state.variantById.get(pokemon.variant_id);
  const variants = state.variantsBySpecies.get(variant ? normalizeId(variant.species) : "") || [];
  const moves = pokemon.moves.map((move, moveIndex) => `
    <label class="move"><span>${escapeHtml(move.id)}</span><input data-move-pp="${sideId}:${index}:${moveIndex}" type="number" min="0" max="${move.max_pp}" value="${move.pp}" aria-label="${escapeHtml(move.id)} remaining PP" /></label>`).join("");
  const status = pokemon.status || { id: "" };
  const statusOptions = STATUS_OPTIONS.map(([id, label]) => `<option value="${id}" ${status.id === id ? "selected" : ""}>${label}</option>`).join("");
  const genderOptions = [
    [null, "Unspecified"],
    ["M", "Male"],
    ["F", "Female"],
    ["N", "Genderless"],
  ].map(([id, label]) => `<option value="${id || ""}" ${pokemon.gender === id ? "selected" : ""}>${label}</option>`).join("");
  let statusCounter = `<span></span>`;
  if (status.id === "slp") {
    statusCounter = `<label>Sleep turns remaining <input data-status-counter="${sideId}:${index}:sleep_turns_remaining" type="number" min="1" max="4" value="${status.sleep_turns_remaining ?? 2}" /></label>`;
  } else if (status.id === "tox") {
    statusCounter = `<label>Toxic stage <input data-status-counter="${sideId}:${index}:toxic_stage" type="number" min="0" max="15" value="${status.toxic_stage ?? 1}" /></label>`;
  }
  return `
    <article class="pokemon ${index === activeSlot ? "active" : ""} ${fainted ? "fainted" : ""}">
      <div class="pokemon-head">
        <div><div class="pokemon-name">${escapeHtml(pokemon.species)}</div><div class="pokemon-meta">L${pokemon.level} | ${escapeHtml(pokemon.ability || "No ability")} | ${escapeHtml(pokemon.item || "No item")}</div></div>
        <div class="pokemon-actions">
          <button data-active="${sideId}:${index}" class="quiet" ${fainted ? "disabled" : ""}>${index === activeSlot ? "Active" : "Make active"}</button>
          <button data-move-roster="${sideId}:${index}:-1" class="quiet" ${index === 0 ? "disabled" : ""}>Up</button>
          <button data-move-roster="${sideId}:${index}:1" class="quiet" ${index === selectedSide(sideId).pokemon.length - 1 ? "disabled" : ""}>Down</button>
          <button data-remove="${sideId}:${index}" class="quiet">Remove</button>
        </div>
      </div>
      <div class="pokemon-grid">
        <label>Exact legal set <select data-variant="${sideId}:${index}">${variants.map((choice) => `<option value="${escapeHtml(choice.variant_id)}" ${choice.variant_id === pokemon.variant_id ? "selected" : ""}>${escapeHtml(choice.role)} - ${escapeHtml(choice.moves.map((move) => move.name).join(" / "))}</option>`).join("")}</select></label>
        <label>Current HP <div class="hp-row"><input data-hp-range="${sideId}:${index}" type="range" min="0" max="${pokemon.max_hp}" value="${pokemon.current_hp}" /><input data-hp="${sideId}:${index}" type="number" min="0" max="${pokemon.max_hp}" value="${pokemon.current_hp}" /></div></label>
        <div class="full status-row">
          <label>Gender <select data-gender="${sideId}:${index}">${genderOptions}</select></label>
          <label>Major status <select data-status="${sideId}:${index}">${statusOptions}</select></label>
          ${statusCounter}
        </div>
        <div class="full moves">${moves}</div>
      </div>
    </article>`;
}

function sideConditionControls(sideId, side) {
  return SIDE_CONDITIONS.map(([id, label, maximum]) => `
    <label>${label}<input data-side-condition="${sideId}:${id}" type="number" min="0" max="${maximum}" value="${side.side_conditions[id] || 0}" /></label>
  `).join("");
}

function volatileById(side, volatileId) {
  return side.active_volatiles.find((volatile) => volatile.id === volatileId);
}

function defaultVolatile(definition, active) {
  const result = { id: definition.id };
  if (definition.turns) result.turns_remaining = definition.turns[1];
  if (definition.elapsed) result.turns_elapsed = definition.elapsed[0];
  if (definition.hp) result.hp = Math.max(1, Math.floor((active?.max_hp || 4) / 4));
  if (definition.move) result.move_id = active?.moves.find((move) => move.pp > 0)?.id || active?.moves[0]?.id || "";
  return result;
}

function volatileControl(sideId, side, definition) {
  const volatile = volatileById(side, definition.id);
  const active = side.pokemon[side.active_slot];
  const params = [];
  if (volatile && definition.turns) {
    params.push(`<label>Turns left <input data-volatile-param="${sideId}:${definition.id}:turns_remaining" type="number" min="${definition.turns[0]}" max="${definition.turns[1]}" value="${volatile.turns_remaining}" /></label>`);
  }
  if (volatile && definition.elapsed) {
    params.push(`<label>Turns elapsed <input data-volatile-param="${sideId}:${definition.id}:turns_elapsed" type="number" min="${definition.elapsed[0]}" max="${definition.elapsed[1]}" value="${volatile.turns_elapsed}" /></label>`);
  }
  if (volatile && definition.hp) {
    params.push(`<label>Sub HP <input data-volatile-param="${sideId}:${definition.id}:hp" type="number" min="1" max="${Math.max(1, Math.floor((active?.max_hp || 4) / 4))}" value="${volatile.hp}" /></label>`);
  }
  if (volatile && definition.move) {
    params.push(`<label>Bound move <select data-volatile-param="${sideId}:${definition.id}:move_id">${(active?.moves || []).map((move) => `<option value="${escapeHtml(move.id)}" ${move.id === volatile.move_id ? "selected" : ""}>${escapeHtml(move.id)} (${move.pp} PP)</option>`).join("")}</select></label>`);
  }
  return `
    <div class="condition-control">
      <label class="check-label"><input data-volatile-toggle="${sideId}:${definition.id}" type="checkbox" ${volatile ? "checked" : ""} ${active ? "" : "disabled"} /> ${definition.label}</label>
      ${params.length ? `<div class="condition-params">${params.join("")}</div>` : ""}
    </div>`;
}

function teamPanel(sideId) {
  const side = selectedSide(sideId);
  const selection = state.selection[sideId];
  const cards = side.pokemon.length ? side.pokemon.map((pokemon, index) => pokemonCard(sideId, pokemon, index, side.active_slot)).join("") : `<div class="empty-roster">Choose a species and legal set to build this side.</div>`;
  return `
    <section class="team card" data-side="${sideId}">
      <div class="team-header"><div class="team-title"><strong>${sideId === "p1" ? "Player One" : "Player Two"}</strong><span>${escapeHtml(side.construction_mode)}${side.generated_team_seed !== null ? `, seed ${side.generated_team_seed}` : ""}</span></div><span>${side.pokemon.length}/6 Pokemon</span></div>
      <div class="team-controls">
        <label class="species-filter">Find species <input data-picker-filter="${sideId}" type="search" value="${escapeHtml(selection.filter || "")}" placeholder="Filter 220 species" /></label>
        <label>Species <select data-picker-species="${sideId}">${speciesOptions(selection.species, selection.filter)}</select></label>
        <label>Exact set <select data-picker-variant="${sideId}" ${selection.species ? "" : "disabled"}>${variantOptions(selection.species, selection.variant)}</select></label>
        <button data-add="${sideId}" ${!selection.variant || side.pokemon.length >= 6 ? "disabled" : ""}>Add</button>
        <button data-generate="${sideId}" class="quiet generate">Generate full randbats side</button>
      </div>
      <div class="condition-panel">
        <div class="condition-title">Side conditions <span>0 means absent; screens are turns remaining</span></div>
        <div class="side-condition-grid">${sideConditionControls(sideId, side)}</div>
      </div>
      <div class="condition-panel">
        <div class="condition-title">Active volatile effects <span>applied to the selected active Pokemon</span></div>
        <div class="volatile-grid">${VOLATILE_DEFINITIONS.map((definition) => volatileControl(sideId, side, definition)).join("")}</div>
      </div>
      <div class="roster">${cards}</div>
    </section>`;
}

function renderTeams() {
  $("#team-grid").innerHTML = [teamPanel("p1"), teamPanel("p2")].join("");
  bindTeamEvents();
}

function normalizeId(value) { return String(value || "").toLowerCase().replace(/[^a-z0-9]+/g, ""); }

function sourceComposePokemon(pokemon) {
  const variant = state.variantById.get(pokemon.variant_id);
  if (!variant) return pokemon;
  const canonical = variantPokemon(variant);
  const previousMoves = new Map(pokemon.moves.map((move) => [move.id, move]));
  canonical.moves.forEach((move) => {
    const prior = previousMoves.get(move.id);
    if (prior) move.pp = Math.min(move.max_pp, prior.pp);
  });
  // The source catalog does not prescribe EVs, IVs, nature, or gender. Preserve those generated
  // details so an edit changes roster provenance, not the visible battle stats by surprise.
  return {
    ...canonical,
    current_hp: Math.min(pokemon.current_hp, pokemon.max_hp),
    max_hp: pokemon.max_hp,
    nature: pokemon.nature,
    gender: pokemon.gender,
    evs: pokemon.evs,
    ivs: pokemon.ivs,
    status: pokemon.status || { id: "" },
  };
}

function markSourceComposed(side) {
  if (side.construction_mode === "generated") side.pokemon = side.pokemon.map(sourceComposePokemon);
  side.construction_mode = "source-composed";
  side.generated_team_seed = null;
}

function bindTeamEvents() {
  document.querySelectorAll("[data-picker-filter]").forEach((element) => element.addEventListener("input", () => {
    const side = element.dataset.pickerFilter;
    state.selection[side].filter = element.value;
    renderTeams();
  }));
  document.querySelectorAll("[data-picker-species]").forEach((element) => element.addEventListener("change", () => {
    const side = element.dataset.pickerSpecies;
    state.selection[side] = { ...state.selection[side], species: element.value, variant: "" };
    renderTeams();
  }));
  document.querySelectorAll("[data-picker-variant]").forEach((element) => element.addEventListener("change", () => {
    state.selection[element.dataset.pickerVariant].variant = element.value;
    renderTeams();
  }));
  document.querySelectorAll("[data-add]").forEach((element) => element.addEventListener("click", () => {
    const sideId = element.dataset.add;
    const variant = state.variantById.get(state.selection[sideId].variant);
    if (!variant) return;
    const side = selectedSide(sideId);
    markSourceComposed(side);
    side.pokemon.push(variantPokemon(variant));
    side.active_slot = Math.min(side.active_slot, side.pokemon.length - 1);
    invalidate(); renderTeams();
  }));
  document.querySelectorAll("[data-generate]").forEach((element) => element.addEventListener("click", async () => {
    const sideId = element.dataset.generate;
    try {
      notice("Generating a Showdown randbats team...");
      const seed = Math.floor(Math.random() * 2 ** 31);
      const result = await api("/api/teams/generate", { method: "POST", body: JSON.stringify({ seed }) });
      state.scenario.teams[sideId] = normalizeScenario({
        ...state.scenario,
        teams: { ...state.scenario.teams, [sideId]: result.side },
      }).teams[sideId];
      state.selection[sideId] = { filter: "", species: "", variant: "" };
      invalidate(); renderTeams(); notice(`Generated ${sideId} from seed ${result.seed}.`, "success");
    } catch (error) { notice(error.message, "error"); }
  }));
  document.querySelectorAll("[data-remove]").forEach((element) => element.addEventListener("click", () => {
    const [sideId, indexText] = element.dataset.remove.split(":"); const index = Number(indexText); const side = selectedSide(sideId);
    markSourceComposed(side); side.pokemon.splice(index, 1); side.active_slot = Math.max(0, Math.min(side.active_slot, side.pokemon.length - 1)); invalidate(); renderTeams();
  }));
  document.querySelectorAll("[data-move-roster]").forEach((element) => element.addEventListener("click", () => {
    const [sideId, indexText, deltaText] = element.dataset.moveRoster.split(":"); const index = Number(indexText); const delta = Number(deltaText); const side = selectedSide(sideId); const next = index + delta;
    if (next < 0 || next >= side.pokemon.length) return;
    markSourceComposed(side); [side.pokemon[index], side.pokemon[next]] = [side.pokemon[next], side.pokemon[index]];
    if (side.active_slot === index) side.active_slot = next; else if (side.active_slot === next) side.active_slot = index;
    invalidate(); renderTeams();
  }));
  document.querySelectorAll("[data-active]").forEach((element) => element.addEventListener("click", () => { const [sideId, index] = element.dataset.active.split(":"); selectedSide(sideId).active_slot = Number(index); invalidate(); renderTeams(); }));
  document.querySelectorAll("[data-hp], [data-hp-range]").forEach((element) => element.addEventListener("input", () => {
    const [sideId, indexText] = (element.dataset.hp || element.dataset.hpRange).split(":"); const pokemon = selectedSide(sideId).pokemon[Number(indexText)];
    pokemon.current_hp = Math.max(0, Math.min(pokemon.max_hp, Number(element.value) || 0)); invalidate(); renderTeams();
  }));
  document.querySelectorAll("[data-move-pp]").forEach((element) => element.addEventListener("input", () => {
    const [sideId, pokemonText, moveText] = element.dataset.movePp.split(":"); const move = selectedSide(sideId).pokemon[Number(pokemonText)].moves[Number(moveText)];
    move.pp = Math.max(0, Math.min(move.max_pp, Number(element.value) || 0)); invalidate(); renderTeams();
  }));
  document.querySelectorAll("[data-status]").forEach((element) => element.addEventListener("change", () => {
    const [sideId, indexText] = element.dataset.status.split(":");
    const pokemon = selectedSide(sideId).pokemon[Number(indexText)];
    pokemon.status = { id: element.value };
    if (element.value === "slp") pokemon.status.sleep_turns_remaining = 2;
    if (element.value === "tox") pokemon.status.toxic_stage = 1;
    invalidate(); renderTeams();
  }));
  document.querySelectorAll("[data-gender]").forEach((element) => element.addEventListener("change", () => {
    const [sideId, indexText] = element.dataset.gender.split(":");
    selectedSide(sideId).pokemon[Number(indexText)].gender = element.value || null;
    invalidate();
  }));
  document.querySelectorAll("[data-status-counter]").forEach((element) => element.addEventListener("input", () => {
    const [sideId, indexText, key] = element.dataset.statusCounter.split(":");
    selectedSide(sideId).pokemon[Number(indexText)].status[key] = Number(element.value);
    invalidate();
  }));
  document.querySelectorAll("[data-side-condition]").forEach((element) => element.addEventListener("input", () => {
    const [sideId, condition] = element.dataset.sideCondition.split(":");
    const value = Number(element.value) || 0;
    if (value > 0) selectedSide(sideId).side_conditions[condition] = value;
    else delete selectedSide(sideId).side_conditions[condition];
    invalidate();
  }));
  document.querySelectorAll("[data-volatile-toggle]").forEach((element) => element.addEventListener("change", () => {
    const [sideId, volatileId] = element.dataset.volatileToggle.split(":");
    const side = selectedSide(sideId);
    if (element.checked) {
      const definition = VOLATILE_DEFINITIONS.find((item) => item.id === volatileId);
      side.active_volatiles.push(defaultVolatile(definition, side.pokemon[side.active_slot]));
    } else {
      side.active_volatiles = side.active_volatiles.filter((item) => item.id !== volatileId);
    }
    invalidate(); renderTeams();
  }));
  document.querySelectorAll("[data-volatile-param]").forEach((element) => element.addEventListener("input", () => {
    const [sideId, volatileId, key] = element.dataset.volatileParam.split(":");
    const volatile = volatileById(selectedSide(sideId), volatileId);
    if (!volatile) return;
    volatile[key] = key === "move_id" ? element.value : Number(element.value);
    invalidate();
  }));
  document.querySelectorAll("[data-variant]").forEach((element) => element.addEventListener("change", () => {
    const [sideId, indexText] = element.dataset.variant.split(":"); const side = selectedSide(sideId); const replacement = state.variantById.get(element.value); if (!replacement) return;
    markSourceComposed(side); side.pokemon[Number(indexText)] = variantPokemon(replacement); invalidate(); renderTeams();
  }));
}

function bindMetadataEvents() {
  const fields = [
    ["#scenario-id", "scenario_id"], ["#scenario-title", "title"], ["#scenario-description", "description"], ["#scenario-seed", "seed"], ["#scenario-turn", "turn"], ["#perspective", "perspective"], ["#side-to-move", "side_to_move"], ["#author-notes", "author_notes"],
  ];
  fields.forEach(([selector, key]) => $(selector).addEventListener("input", (event) => { state.scenario[key] = ["seed", "turn"].includes(key) ? Number(event.target.value) || (key === "turn" ? 1 : 0) : event.target.value; invalidate(); renderBadges(); }));
  $("#scenario-tags").addEventListener("input", (event) => { state.scenario.tags = event.target.value.split(",").map((tag) => tag.trim()).filter(Boolean); invalidate(); renderBadges(); });
  $("#objective-kind").addEventListener("change", (event) => { state.scenario.objective.kind = event.target.value; invalidate(); });
  $("#objective-plies").addEventListener("input", (event) => { state.scenario.objective.max_plies = Math.max(1, Number(event.target.value) || 1); invalidate(); });
  $("#objective-status").addEventListener("change", (event) => { state.scenario.objective.verification.status = event.target.value; invalidate(); });
  $("#expected-actions").addEventListener("input", (event) => { state.scenario.objective.expected_root_actions = event.target.value.split(",").map((item) => item.trim()).filter(Boolean); invalidate(); });
  $("#principal-variation").addEventListener("input", (event) => { state.scenario.objective.principal_variation = event.target.value.split("\n").map((item) => item.trim()).filter(Boolean); invalidate(); });
  $("#field-weather").addEventListener("change", (event) => {
    state.scenario.field = event.target.value
      ? { weather: event.target.value, turns_remaining: 5, permanent: false }
      : { weather: "", turns_remaining: 0, permanent: false };
    invalidate(); metadataRender();
  });
  $("#field-weather-turns").addEventListener("input", (event) => {
    state.scenario.field.turns_remaining = Math.max(1, Math.min(5, Number(event.target.value) || 1));
    invalidate();
  });
  $("#field-weather-permanent").addEventListener("change", (event) => {
    state.scenario.field.permanent = event.target.checked;
    if (event.target.checked) state.scenario.field.turns_remaining = 5;
    invalidate(); metadataRender();
  });
}

function renderReadout(result) {
  const legal = result.legal_actions || {};
  $("#readout").innerHTML = ["p1", "p2"].map((side) => `<div><strong>${side} legal actions</strong><div class="legal-actions">${(legal[side] || []).map((action) => `<div class="action-line"><span>${escapeHtml(action.label)}</span><span>#${action.index}</span></div>`).join("")}</div></div>`).join("");
}

async function validateCurrent() {
  notice("Validating source, state, and Showdown materialization...");
  const result = await api("/api/validate", { method: "POST", body: JSON.stringify({ scenario: state.scenario }) });
  state.scenario = normalizeScenario(result.scenario); state.lastValidation = result; metadataRender(); renderTeams(); renderReadout(result); notice("Scenario is set-valid and state-consistent in Showdown.", "success");
  return result;
}

async function refreshScenarioList() {
  const result = await api("/api/scenarios"); const select = $("#scenario-select"); const current = select.value;
  select.innerHTML = `<option value="">Load a scenario...</option>${result.scenarios.map((scenario) => `<option value="${escapeHtml(scenario.slug)}">${escapeHtml(scenario.title || scenario.slug)}${scenario.invalid ? " (invalid)" : ""}</option>`).join("")}`;
  select.value = current;
}

function exportScenario() {
  const blob = new Blob([JSON.stringify(state.scenario, null, 2) + "\n"], { type: "application/json" });
  const url = URL.createObjectURL(blob); const anchor = document.createElement("a"); anchor.href = url; anchor.download = `${state.scenario.scenario_id || "endgame-scenario"}.json`; anchor.click(); URL.revokeObjectURL(url);
}

function bindCommands() {
  $("#new-button").addEventListener("click", () => { state.scenario = blankScenario(); state.selection = blankSelection(); invalidate(); markClean(); metadataRender(); renderTeams(); $("#readout").textContent = "Validate a scenario to inspect legal actions and materialized state."; $("#evaluation").innerHTML = ""; notice("Started a new scenario."); });
  $("#duplicate-button").addEventListener("click", () => {
    const sourceId = state.scenario.scenario_id.trim() || "endgame-scenario";
    state.scenario = structuredClone(state.scenario);
    state.scenario.scenario_id = `${sourceId}-copy`;
    state.scenario.title = state.scenario.title ? `${state.scenario.title} (copy)` : "Scenario copy";
    $("#slug-input").value = "";
    invalidate(); metadataRender(); renderTeams(); $("#evaluation").innerHTML = "";
    notice("Duplicated in memory. Choose a new filename before saving.", "success");
  });
  $("#validate-button").addEventListener("click", () => validateCurrent().catch((error) => notice(error.message, "error")));
  $("#save-button").addEventListener("click", async () => { try { const slug = $("#slug-input").value.trim(); if (!slug) throw new Error("Choose a lowercase scenario filename before saving."); const existing = [...$("#scenario-select").options].some((option) => option.value === slug); if (existing && !window.confirm(`Replace existing ${slug}.json?`)) return; notice("Validating and saving..."); const result = await api(`/api/scenarios/${encodeURIComponent(slug)}`, { method: "PUT", body: JSON.stringify({ scenario: state.scenario }) }); state.scenario = normalizeScenario(result.scenario); state.lastValidation = result; markClean(); metadataRender(); renderTeams(); renderReadout(result); await refreshScenarioList(); $("#scenario-select").value = slug; notice(`Saved ${slug}.json.`, "success"); } catch (error) { notice(error.message, "error"); } });
  $("#export-button").addEventListener("click", exportScenario);
  $("#load-button").addEventListener("click", async () => { try { const slug = $("#scenario-select").value; if (!slug) throw new Error("Choose a saved scenario first."); const result = await api(`/api/scenarios/${encodeURIComponent(slug)}`); state.scenario = normalizeScenario(result.scenario); $("#slug-input").value = slug; state.selection = blankSelection(); invalidate(); markClean(); metadataRender(); renderTeams(); $("#readout").textContent = "Loaded. Validate to inspect Showdown actions."; $("#evaluation").innerHTML = ""; notice(`Loaded ${slug}.`, "success"); } catch (error) { notice(error.message, "error"); } });
  $("#import-input").addEventListener("change", async (event) => { const [file] = event.target.files; if (!file) return; try { const imported = JSON.parse(await file.text()); state.scenario = normalizeScenario(imported); state.scenario.provenance ||= {}; state.scenario.provenance.randbat_source_hash ||= state.catalog.source_hash; state.selection = blankSelection(); invalidate(); metadataRender(); renderTeams(); notice("Imported JSON. Validate before saving.", "success"); } catch (error) { notice(`Import failed: ${error.message}`, "error"); } finally { event.target.value = ""; } });
  $("#evaluate-button").addEventListener("click", async () => { try { const checkpointPath = $("#checkpoint-path").value.trim(); if (!checkpointPath) throw new Error("Enter a checkpoint path first."); notice("Loading checkpoint and ranking root actions..."); const result = await api("/api/evaluate-root", { method: "POST", body: JSON.stringify({ scenario: state.scenario, checkpoint_path: checkpointPath }) }); $("#evaluation").innerHTML = result.actions.map((action) => `<div class="action-line ${action.expected ? "expected" : ""}"><span>#${action.rank} ${escapeHtml(action.label)}</span><span>${(action.probability * 100).toFixed(2)}%</span></div>`).join(""); notice("Root action ranking complete. This is a synthetic fully revealed scenario.", "success"); } catch (error) { notice(error.message, "error"); } });
}

async function boot() {
  try {
    state.catalog = await api("/api/catalog");
    state.catalog.species.forEach((species) => { state.variantsBySpecies.set(species.id, species.variants); species.variants.forEach((variant) => state.variantById.set(variant.variant_id, variant)); });
    $("#source-provenance").textContent = `Pinned Gen 3 randbats source ${state.catalog.source_hash} | ${state.catalog.species.length} species`;
    state.scenario = normalizeScenario(blankScenario());
    bindMetadataEvents(); bindCommands(); metadataRender(); renderTeams(); await refreshScenarioList(); markClean(); notice("Choose exact legal sets for each side, then validate the position.");
  } catch (error) { notice(`Unable to start the studio: ${error.message}`, "error"); }
}

window.addEventListener("beforeunload", (event) => {
  if (!state.dirty) return;
  event.preventDefault();
  event.returnValue = "";
});

boot();
