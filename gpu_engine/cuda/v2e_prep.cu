// D3 device world preparation: device_public_worlds.plan_starts (with _revealed) and
// device_public_world_state.materialize_host for every world of a root batch, one
// thread per world, bit-exact with the host path. The host supplies the once-per-source
// tables (gpu_engine/device_public_source_tables.py), the batched public readout and the
// CPython random.Random seeds; it maps the returned codes back to the original refusal
// strings and expands the compact records into the pricer's state tensors
// (gpu_engine/device_public_prep.py). Layout: v2e_prep.cuh.
#include <cmath>
#include <cstdint>
#include <string>
#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"
#include "v2e_mt19937.cuh"
#include "v2e_prep.cuh"

namespace ffi = xla::ffi;

namespace v2e {
namespace prep {

struct Inputs {
  const int32_t* misc;
  const int32_t* uni;
  const int32_t* var;
  const int32_t* species_to_universe;
  const int32_t* move_is_hp;
  const int32_t* owner;
  const int32_t* species;
  const int32_t* present;
  const int32_t* active;
  const uint32_t* belief;
  const int32_t* revealed_moves;
  const int32_t* ability;
  const int32_t* item;
  const int32_t* gender;
  const int32_t* level;
  const int32_t* error;
  const float* hp_fraction;
  const int32_t* hp_percent;
  const int32_t* status;
  const int32_t* boosts;
  const int32_t* types;
  const int32_t* borrowed;
  const int32_t* removed;
  const int32_t* pp_used;
  const int32_t* last_moves;
  const int32_t* volatiles;
  const int32_t* weather;
  const int32_t* turn;
  const int32_t* spikes;
  const int32_t* own_order;
  const int32_t* features;
  const int32_t* own_meanlook;
  const int32_t* fields;
  const int32_t* own_active;
  const int32_t* own_trap;
  const int32_t* own_flags;
  const uint32_t* seeds;
  const int32_t* count;
  int64_t flags;
  int32_t pre_error;
  int32_t pre_error_after;
  int32_t volatile_count;
  int64_t roots;
};

struct Outputs {
  int32_t* code;
  int32_t* mon;
  int32_t* side;
  int32_t* rev;
  int32_t* fix;
};

__device__ inline int popcount64(uint64_t x) { return __popcll(x); }
__device__ inline int popcount32(uint32_t x) { return __popc(x); }

// Index of the n-th (0-based) set bit of x; x has more than n set bits.
__device__ inline int nth_set_bit64(uint64_t x, int n) {
  for (int b = 0; b < 64; ++b) {
    if ((x >> b) & 1u) {
      if (n == 0) return b;
      --n;
    }
  }
  return -1;
}

__device__ inline int nth_set_bit_words(const uint32_t* words, int n) {
  for (int wi = 0; wi < kUniverseWords; ++wi) {
    const int c = popcount32(words[wi]);
    if (n < c) {
      for (int b = 0; b < 32; ++b) {
        if ((words[wi] >> b) & 1u) {
          if (n == 0) return wi * 32 + b;
          --n;
        }
      }
    }
    n -= c;
  }
  return -1;
}

__device__ inline bool feature_present(const Inputs& in, int f) {
  return (in.flags & F_MON_FEATURES) && (in.flags & (int64_t{1} << (F_FEATURE_BASE + f)));
}

// _mon_feature(view, name, slot): the value, or a refusal code ('public current feature
// unavailable: name'); the int32 features are always finite and correctly shaped.
__device__ inline int32_t feature_value(const Inputs& in, int64_t i, int f, int slot, int32_t& code) {
  if (!feature_present(in, f)) {
    code = in.misc[MISC_HEADER + in.volatile_count + f];
    return 0;
  }
  return in.features[(i * kFeatures + f) * kSlots + slot];
}

__global__ void prep_kernel(Inputs in, Outputs out) {
  const int64_t w = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (w >= 2 * in.roots) return;
  const int64_t i = w >> 1;
  const int world = static_cast<int>(w & 1);
  int32_t* code = out.code + w * C_COLUMNS;
  int32_t* mon = out.mon + w * kSlots * M_COLUMNS;
  int32_t* side = out.side + w * S_COLUMNS;
  int32_t* rev = out.rev + w * kSlots;
  int32_t* fix = out.fix + w * kSlots * FX_COLUMNS;
  // Empty-state defaults of the foe side; the host overlays the owner's own cells.
  for (int p = 0; p < kSlots; ++p) {
    int32_t* m = mon + p * M_COLUMNS;
    for (int c = 0; c < M_COLUMNS; ++c) m[c] = 0;
    m[M_HP_POWER] = 70;
    m[M_GENDER] = -1;
    m[M_TRAP_SOURCE] = -1;
    rev[p] = -1;
    for (int c = 0; c < FX_COLUMNS; ++c) fix[p * FX_COLUMNS + c] = -1;
  }
  for (int c = 0; c < S_COLUMNS; ++c) side[c] = 0;
  side[S_LAST_INDEX] = -1;
  code[C_PLAN] = 0;
  code[C_MATERIALIZE] = 0;
  if (i >= in.count[0]) {
    code[C_PLAN] = -1;  // padding root
    return;
  }
  const int32_t* misc = in.misc;
  const int32_t* uni = in.uni;
  const int32_t* var = in.var;
  const int S = misc[MISC_SPECIES], M = misc[MISC_MOVES], A = misc[MISC_ABILITIES], I = misc[MISC_ITEMS];
  const int U = misc[MISC_UNIVERSES], Vn = misc[MISC_VARIANTS], V = in.volatile_count;
  const int32_t struggle = misc[MISC_STRUGGLE], truant = misc[MISC_TRUANT];
  const int owner = in.owner[i];
  for (int p = 0; p < kSlots; ++p) side[S_OWN_TRAP0 + p] = in.own_trap[i * kSlots + p];

  // ---- plan_starts: _revealed(root, source) -----------------------------------------
  // Host-static _revealed failures: the error-array check precedes the per-root tracker error,
  // the other shape/dtype checks follow it.
  int32_t plan = in.pre_error;
  int uni_of[kSlots];
  uint64_t eligible[kSlots];
  int shown_gender[kSlots];
  uint32_t seen[kUniverseWords];
  for (int k = 0; k < kUniverseWords; ++k) seen[k] = 0u;
  for (int slot = 0; slot < kSlots; ++slot) uni_of[slot] = -1;
  const bool swapped_present = feature_present(in, FT_ITEM_SWAPPED);
  if (plan == 0 && in.error[i] != 0) plan = R_TRACKER_ERROR;
  if (plan == 0) plan = in.pre_error_after;
  for (int slot = 0; slot < kSlots && plan == 0; ++slot) {
    const int64_t cell = i * kSlots + slot;
    const int32_t sp = in.species[cell];
    const uint32_t m0 = in.belief[cell * 2], m1 = in.belief[cell * 2 + 1];
    if (!in.present[cell]) {
      if (sp != 0 || m0 != 0u || m1 != 0u) plan = R_UNREVEALED_EXPOSES;
      continue;
    }
    if (!(sp >= 1 && sp < S)) { plan = R_NS_SPECIES; break; }
    const int u = in.species_to_universe[sp];
    if (u < 0 || uni[u * U_COLUMNS + U_VAR_CNT] == 0) { plan = R_NO_SOURCE; break; }
    if (seen[u >> 5] & (1u << (u & 31))) { plan = R_DUPLICATE_SPECIES; break; }
    seen[u >> 5] |= 1u << (u & 31);
    const int count = uni[u * U_COLUMNS + U_VAR_CNT];
    const uint64_t raw = static_cast<uint64_t>(m0) | (static_cast<uint64_t>(m1) << 32);
    if (count > 64 || (count < 64 && (raw >> count) != 0u)) { plan = R_BITS_OUTSIDE; break; }
    if (raw == 0u) { plan = R_ZERO_MASS; break; }
    int shown[kMoveSlots];
    int shown_count = 0;
    bool bad = false;
    for (int k = 0; k < kMoveSlots; ++k) {
      const int32_t v = in.revealed_moves[cell * kMoveSlots + k];
      if (v == 0) continue;
      if (!(v >= 1 && v < M)) { bad = true; break; }
      shown[shown_count++] = v;
    }
    if (bad) { plan = R_NS_MOVES; break; }
    const int32_t known_ability = in.ability[cell];
    if (!(known_ability >= 0 && known_ability < A)) { plan = R_NS_ABILITIES; break; }
    int32_t known_item = in.item[cell];
    if (!(known_item >= 0 && known_item < I)) { plan = R_NS_ITEMS; break; }
    if (known_item != 0 && !swapped_present) { plan = R_ITEM_SWAP_UNAVAILABLE; break; }
    if (swapped_present && in.features[(i * kFeatures + FT_ITEM_SWAPPED) * kSlots + slot]) known_item = 0;
    const int off = uni[u * U_COLUMNS + U_VAR_OFF];
    const int32_t level = in.level[cell];
    uint64_t e = 0u;
    for (int index = 0; index < count; ++index) {
      if (!((raw >> index) & 1u)) continue;
      const int32_t* vr = var + static_cast<size_t>(off + index) * V_COLUMNS;  // matching columns: same in both cases
      bool ok = true;
      for (int k = 0; k < shown_count && ok; ++k) {
        const int32_t m = shown[k];
        if (m == struggle) continue;  // excluded from revealed_moves
        if (in.move_is_hp[m]) ok = vr[V_HAS_HP] != 0;
        else ok = vr[V_MOVE0] == m || vr[V_MOVE1] == m || vr[V_MOVE2] == m || vr[V_MOVE3] == m;
      }
      if (ok && known_ability != 0 && vr[V_ABILITY] != known_ability) ok = false;
      if (ok && known_item != 0 && vr[V_ITEM] != known_item) ok = false;
      if (ok && vr[V_LEVEL] != level) ok = false;
      if (ok) e |= uint64_t{1} << index;
    }
    if (e == 0u) { plan = R_NO_CONSISTENT; break; }
    const int32_t g = in.gender[cell];
    if (!(g >= -1 && g <= 2)) { plan = R_GENDER_NS; break; }
    uni_of[slot] = u;
    eligible[slot] = e;
    shown_gender[slot] = g;
  }

  // ---- plan_starts: the two shared source fixture draws (this world) -----------------
  Mt19937 rng;
  mt_seed_words(rng, in.seeds + (i * 2 + world) * kSeedWords, kSeedWords);
  int fix_u[kSlots], fix_k[kSlots], fix_g[kSlots], fix_slot[kSlots];
  int revealed_count = 0;
  uint32_t used[kUniverseWords];
  for (int k = 0; k < kUniverseWords; ++k) used[k] = 0u;
  if (plan == 0) {
    for (int slot = 0; slot < kSlots; ++slot) {
      if (uni_of[slot] < 0) continue;
      const int u = uni_of[slot];
      int g = shown_gender[slot];
      if (g < 0) {
        const int fixed = uni[u * U_COLUMNS + U_GENDER];
        g = fixed >= 0 ? fixed : (mt_randbelow(rng, 2u) == 0u ? 1 : 2);
      }
      const uint64_t e = eligible[slot];
      const uint32_t pick = mt_randbelow(rng, static_cast<uint32_t>(popcount64(e)));
      const int k = nth_set_bit64(e, static_cast<int>(pick));
      const int v = uni[u * U_COLUMNS + U_VAR_OFF] + k;
      if (var[(static_cast<size_t>(Vn) + v) * V_COLUMNS + V_FIXTURE_NONE]) { plan = R_REVEALED_FIXTURE; break; }
      fix_u[revealed_count] = u;
      fix_k[revealed_count] = k;
      fix_g[revealed_count] = g;
      fix_slot[revealed_count] = slot;
      rev[slot] = k;
      used[u >> 5] |= 1u << (u & 31);
      ++revealed_count;
    }
  }
  int fixture_count = revealed_count;
  if (plan == 0) {
    uint32_t avail[kUniverseWords];
    int candidates = 0;
    for (int k = 0; k < kUniverseWords; ++k) avail[k] = 0u;
    for (int u = 0; u < U; ++u) {
      if (uni[u * U_COLUMNS + U_VAR_CNT] > 0 && !(used[u >> 5] & (1u << (u & 31)))) {
        avail[u >> 5] |= 1u << (u & 31);
        ++candidates;
      }
    }
    for (int t = revealed_count; t < kSlots; ++t) {
      if (candidates == 0) { plan = R_HIDDEN_SOURCE; break; }
      const uint32_t pick = mt_randbelow(rng, static_cast<uint32_t>(candidates));
      const int u = nth_set_bit_words(avail, static_cast<int>(pick));
      avail[u >> 5] &= ~(1u << (u & 31));
      --candidates;
      const int k = static_cast<int>(mt_randbelow(rng, static_cast<uint32_t>(uni[u * U_COLUMNS + U_VAR_CNT])));
      const int fixed = uni[u * U_COLUMNS + U_GENDER];
      const int g = fixed >= 0 ? fixed : (mt_randbelow(rng, 2u) == 0u ? 1 : 2);
      fix_u[t] = u;
      fix_k[t] = k;
      fix_g[t] = g;
      fix_slot[t] = -1;
      fixture_count = t + 1;
    }
  }
  for (int p = 0; p < fixture_count; ++p) {
    fix[p * FX_COLUMNS + FX_UNIVERSE] = fix_u[p];
    fix[p * FX_COLUMNS + FX_VARIANT] = fix_k[p];
    fix[p * FX_COLUMNS + FX_GENDER] = fix_g[p];
    fix[p * FX_COLUMNS + FX_SLOT] = fix_slot[p];
  }
  code[C_PLAN] = plan;
  if (plan != 0) return;

  // ---- materialize_host --------------------------------------------------------------
  int32_t mat = 0;
  const int32_t* own_flags = in.own_flags + i * 3;
  if (own_flags[0] || own_flags[1] || own_flags[2]) mat = R_BOUNDARY;
  int active = -1;
  if (mat == 0) {
    int actives = 0;
    for (int slot = 0; slot < kSlots; ++slot) {
      if (!in.active[i * kSlots + slot]) continue;
      ++actives;
      for (int p = 0; p < revealed_count; ++p)
        if (fix_slot[p] == slot) active = p;
    }
    if (actives != 1 || active < 0) mat = R_ONE_ACTIVE;
  }
  if (mat == 0) side[S_ACTIVE] = active;
  // A real owner's Mean Look link is rebased through the PUBLIC current trap.
  for (int physical = 0; physical < kSlots && mat == 0; ++physical) {
    if (in.own_trap[i * kSlots + physical] < 0) continue;
    int slot = -1, hits = 0;
    for (int s = 0; s < kSlots; ++s)
      if (in.own_order[i * kSlots + s] == physical) { slot = s; ++hits; }
    if (!(in.flags & F_OWN_MEANLOOK) || hits != 1 || !in.own_meanlook[i * kSlots + slot]) { mat = R_OWNER_TRAP; break; }
    side[S_OWN_TRAP0 + physical] = active;
  }
  const int32_t own_active = in.own_active[i];
  const int32_t* records[kSlots];
  for (int p = 0; p < kSlots && mat == 0; ++p) {
    const bool revealed = fix_slot[p] >= 0;
    const int v = uni[fix_u[p] * U_COLUMNS + U_VAR_OFF] + fix_k[p];
    const int32_t* vr = var + (static_cast<size_t>(revealed ? Vn : 0) + v) * V_COLUMNS;
    records[p] = vr;
    if (vr[V_REFUSAL]) { mat = vr[V_REFUSAL]; break; }
    int32_t* m = mon + p * M_COLUMNS;
    m[M_SPECIES] = revealed ? in.species[i * kSlots + fix_slot[p]] : vr[V_SPECIES];
    m[M_LEVEL] = vr[V_LEVEL];
    for (int k = 0; k < 6; ++k) m[M_STAT0 + k] = vr[V_STAT0 + k];
    m[M_HP] = m[M_MAX_HP] = vr[V_STAT0];
    m[M_GENDER] = fix_g[p];
    m[M_ABILITY] = m[M_BASE_ABILITY] = vr[V_ABILITY];
    m[M_ITEM] = vr[V_ITEM];
    m[M_TYPE0] = m[M_BTYPE0] = vr[V_TYPE0];
    m[M_TYPE1] = m[M_BTYPE1] = vr[V_TYPE1];
    for (int k = 0; k < kMoveSlots; ++k) {
      m[M_MOVE0 + k] = vr[V_WMOVE0 + k];
      m[M_PP0 + k] = vr[V_PP0 + k];
    }
    m[M_HP_TYPE] = vr[V_HP_TYPE];
    m[M_HP_POWER] = vr[V_HP_POWER];
  }
  for (int p = 0; p < revealed_count && mat == 0; ++p) {
    const int slot = fix_slot[p];
    const int64_t cell = i * kSlots + slot;
    int32_t* m = mon + p * M_COLUMNS;
    const int32_t* vr = records[p];
    const float fraction = in.hp_fraction[cell];
    const int32_t shown = in.hp_percent[cell];
    int32_t status = in.status[cell];
    if (!isfinite(fraction) || !(shown >= 0 && shown <= 100) ||
        fabs(static_cast<double>(fraction) - static_cast<double>(shown) / 100.0) > 1e-6) { mat = R_HP_MALFORMED; break; }
    const int32_t maximum = m[M_MAX_HP];
    if (shown) {
      const long long rounded = llrint(static_cast<double>(shown * maximum) / 100.0);
      m[M_HP] = static_cast<int32_t>(rounded < 1 ? 1 : rounded);
    } else {
      m[M_HP] = 0;
      status = 0;
    }
    if (!(status == 0 || status == 1 || status == 2 || status == 3 || status == 4 || status == 6)) { mat = R_SLEEP; break; }
    m[M_STATUS] = status;
    for (int k = 0; k < kBoosts; ++k) m[M_BOOST0 + k] = in.boosts[cell * kBoosts + k];
    const int32_t t0 = in.types[cell * 2], t1 = in.types[cell * 2 + 1];
    if (t0 || t1) { m[M_TYPE0] = t0; m[M_TYPE1] = t1; }
    if ((in.flags & F_BORROWED) && in.borrowed[cell]) m[M_ABILITY] = in.borrowed[cell];
    if (!(in.flags & F_REMOVED)) { mat = R_ITEM_REMOVED_UNAVAILABLE; break; }
    if (in.removed[cell]) m[M_ITEM] = 0;
    else if (in.item[cell]) m[M_ITEM] = in.item[cell];
    if (!(in.flags & F_PP_USED)) { mat = R_PP_UNAVAILABLE; break; }
    const int nmoves = vr[V_WNMOVES];
    for (int j = 0; j < kMoveSlots; ++j) {
      const int32_t shown_move = in.revealed_moves[cell * kMoveSlots + j];
      if (!shown_move) continue;
      const int32_t count = in.pp_used[cell * kMoveSlots + j];
      int matching = -1, hits = 0;
      for (int k = 0; k < nmoves; ++k) {
        const int32_t id = m[M_MOVE0 + k];
        if (id == shown_move || (in.move_is_hp[shown_move] && in.move_is_hp[id])) { matching = k; ++hits; }
      }
      if (shown_move == struggle) continue;
      if (hits != 1 || count < 0) { mat = R_PP_JOIN; break; }
      const int32_t left = m[M_PP0 + matching] - count;
      m[M_PP0 + matching] = left < 0 ? 0 : left;
    }
    if (mat) break;
    if (status == 4) {
      const int32_t stage = feature_value(in, i, FT_TOXIC_STAGE, slot, mat);
      if (mat) break;
      m[M_STATUS_COUNTER] = stage < 15 ? stage : 15;
    }
    if (p == active) {
      const int32_t turns = feature_value(in, i, FT_TURNS_ACTIVE, slot, mat);
      if (mat) break;
      side[S_ACTIVE_TURNS] = turns - 1 > 0 ? turns - 1 : 0;
      m[M_PROTECT_COUNTER] = feature_value(in, i, FT_STALL_COUNTER, slot, mat);
      if (mat) break;
      if (feature_value(in, i, FT_CONFUSION_TURNS, slot, mat)) { mat = R_CLOCK_CONFUSION; break; }
      if (mat) break;
      if (feature_value(in, i, FT_ENCORE_TURNS, slot, mat)) { mat = R_CLOCK_ENCORE; break; }
      if (mat) break;
      if (feature_value(in, i, FT_SUB_PRESENT, slot, mat)) { mat = R_SUBSTITUTE; break; }
      if (mat) break;
      if (feature_value(in, i, FT_MEANLOOK_TRAP, slot, mat)) m[M_TRAP_SOURCE] = own_active;
      if (mat) break;
      if (!(in.flags & F_VOLATILES)) { mat = R_VOLATILE_BAG; break; }
      for (int kind = 0; kind < V; ++kind) {
        if (!in.volatiles[cell * V + kind]) continue;
        const int32_t entry = misc[MISC_HEADER + kind];
        if (entry == VOL_LEECH_SEED) m[M_VOLATILE] |= 4;
        else if (entry == VOL_INGRAIN) m[M_VOLATILE] |= 8;
        else if (entry == VOL_TRAPPED) m[M_TRAP_SOURCE] = own_active;
        else { mat = entry; break; }
      }
      if (mat) break;
      if (!(in.flags & F_LAST_MOVES)) { mat = R_LAST_MOVES_UNAVAILABLE; break; }
      const int32_t last = in.last_moves[cell];
      if (last > 0) {
        int match = -1, hits = 0;
        if (in.move_is_hp[last]) {
          for (int k = 0; k < nmoves; ++k)
            if (in.move_is_hp[m[M_MOVE0 + k]]) { match = k; ++hits; }
        } else {
          for (int k = 0; k < kMoveSlots; ++k)
            if (m[M_MOVE0 + k] == last) { match = k; ++hits; }
        }
        if (hits != 1) { mat = R_LAST_CALLER; break; }
        const int32_t move = m[M_MOVE0 + match];
        m[M_LAST_MOVE] = move;
        side[S_LAST_KIND] = 2;
        side[S_LAST_INDEX] = match;
        if (vr[V_CHOICEBAND] && m[M_ITEM]) m[M_CHOICE_MOVE] = move;
      } else if (last == -1 || last == -2) {
        side[S_LAST_KIND] = 1;
        side[S_LAST_INDEX] = active;
      }
      if (truant && m[M_ABILITY] == truant) { mat = R_TRUANT; break; }
    }
  }
  if (mat == 0 && mon[active * M_COLUMNS + M_HP] <= 0) mat = R_FAINT;
  if (mat == 0 && !(in.flags & F_FIELDS)) mat = R_FIELDS;
  if (mat == 0 && in.fields[i * 3 + 0]) mat = R_WISH;
  if (mat == 0) {
    const int32_t weather = in.weather[i], turn = in.turn[i];
    if (!(weather >= 0 && weather <= 4) || turn < 1) mat = R_WEATHER_NS;
    else {
      side[S_WEATHER] = weather;
      if (weather) {
        const int32_t remaining = in.fields[i * 3 + 1];
        const bool permanent = in.fields[i * 3 + 2] != 0;
        if (!permanent && !(remaining >= 1 && remaining <= 5)) mat = R_WEATHER_DURATION;
        else side[S_WEATHER_COUNTER] = permanent ? 0 : remaining;
      }
      if (mat == 0) {
        side[S_SPIKES0] = in.spikes[i * 2];
        side[S_SPIKES1] = in.spikes[i * 2 + 1];
        side[S_TURN] = turn - 1;
      }
    }
  }
  code[C_MATERIALIZE] = mat;
}

static ffi::Error cuda_error(const char* where) {
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return ffi::Error::Internal(std::string(where) + ": " + cudaGetErrorString(status));
  return ffi::Error::Success();
}

template <typename T>
static ffi::Error pointer(ffi::RemainingArgs& args, int index, size_t expected, const T*& out) {
  auto arg = args.get<ffi::AnyBuffer>(index);
  if (!arg.has_value()) return arg.error();
  if (arg->size_bytes() != expected * sizeof(T))
    return ffi::Error::InvalidArgument("prep: input " + std::to_string(index) + " has " + std::to_string(arg->size_bytes()) +
                                       " bytes, expected " + std::to_string(expected * sizeof(T)));
  out = static_cast<const T*>(arg->untyped_data());
  return ffi::Error::Success();
}

static ffi::Error PrepWorldsImpl(cudaStream_t stream, int64_t flags, int64_t pre_error, int64_t pre_error_after,
                                 int64_t volatile_count,
                                 int64_t universes, int64_t variants, int64_t species, int64_t moves,
                                 ffi::RemainingArgs args, ffi::Result<ffi::Buffer<ffi::S32>> code,
                                 ffi::Result<ffi::Buffer<ffi::S32>> mon, ffi::Result<ffi::Buffer<ffi::S32>> side,
                                 ffi::Result<ffi::Buffer<ffi::S32>> rev, ffi::Result<ffi::Buffer<ffi::S32>> fix) {
  if (args.size() != static_cast<size_t>(IN_COUNT_ARGS))
    return ffi::Error::InvalidArgument("prep: " + std::to_string(IN_COUNT_ARGS) + " inputs required");
  if (universes > kMaxUniverses) return ffi::Error::InvalidArgument("prep: too many universes");
  auto owner_arg = args.get<ffi::AnyBuffer>(IN_OWNER);
  if (!owner_arg.has_value()) return owner_arg.error();
  const size_t b = owner_arg->element_count();
  const size_t vn = static_cast<size_t>(variants), u = static_cast<size_t>(universes), v = static_cast<size_t>(volatile_count);
  Inputs in{};
  in.flags = flags;
  in.pre_error = static_cast<int32_t>(pre_error);
  in.pre_error_after = static_cast<int32_t>(pre_error_after);
  in.volatile_count = static_cast<int32_t>(volatile_count);
  in.roots = static_cast<int64_t>(b);
#define V2E_IN(index, member, count)                        \
  if (auto e = pointer(args, index, count, in.member); e.failure()) return e;
  V2E_IN(IN_MISC, misc, static_cast<size_t>(MISC_HEADER) + v + kFeatures)
  V2E_IN(IN_UNI, uni, u * U_COLUMNS)
  V2E_IN(IN_VAR, var, 2 * vn * V_COLUMNS)
  V2E_IN(IN_SPECIES_TO_UNIVERSE, species_to_universe, static_cast<size_t>(species))
  V2E_IN(IN_MOVE_IS_HP, move_is_hp, static_cast<size_t>(moves))
  V2E_IN(IN_OWNER, owner, b)
  V2E_IN(IN_SPECIES, species, b * kSlots)
  V2E_IN(IN_PRESENT, present, b * kSlots)
  V2E_IN(IN_ACTIVE, active, b * kSlots)
  V2E_IN(IN_BELIEF, belief, b * kSlots * 2)
  V2E_IN(IN_REVEALED_MOVES, revealed_moves, b * kSlots * kMoveSlots)
  V2E_IN(IN_ABILITY, ability, b * kSlots)
  V2E_IN(IN_ITEM, item, b * kSlots)
  V2E_IN(IN_GENDER, gender, b * kSlots)
  V2E_IN(IN_LEVEL, level, b * kSlots)
  V2E_IN(IN_ERROR, error, b)
  V2E_IN(IN_HP_FRACTION, hp_fraction, b * kSlots)
  V2E_IN(IN_HP_PERCENT, hp_percent, b * kSlots)
  V2E_IN(IN_STATUS, status, b * kSlots)
  V2E_IN(IN_BOOSTS, boosts, b * kSlots * kBoosts)
  V2E_IN(IN_TYPES, types, b * kSlots * 2)
  V2E_IN(IN_BORROWED, borrowed, b * kSlots)
  V2E_IN(IN_REMOVED, removed, b * kSlots)
  V2E_IN(IN_PP_USED, pp_used, b * kSlots * kMoveSlots)
  V2E_IN(IN_LAST_MOVES, last_moves, b * kSlots)
  V2E_IN(IN_VOLATILES, volatiles, b * kSlots * v)
  V2E_IN(IN_WEATHER, weather, b)
  V2E_IN(IN_TURN, turn, b)
  V2E_IN(IN_SPIKES, spikes, b * 2)
  V2E_IN(IN_OWN_ORDER, own_order, b * kSlots)
  V2E_IN(IN_FEATURES, features, b * kFeatures * kSlots)
  V2E_IN(IN_OWN_MEANLOOK, own_meanlook, b * kSlots)
  V2E_IN(IN_FIELDS, fields, b * 3)
  V2E_IN(IN_OWN_ACTIVE, own_active, b)
  V2E_IN(IN_OWN_TRAP, own_trap, b * kSlots)
  V2E_IN(IN_OWN_FLAGS, own_flags, b * 3)
  V2E_IN(IN_SEEDS, seeds, b * 2 * kSeedWords)
  V2E_IN(IN_COUNT, count, 1)
#undef V2E_IN
  if (code->element_count() != b * 2 * C_COLUMNS || mon->element_count() != b * 2 * kSlots * M_COLUMNS ||
      side->element_count() != b * 2 * S_COLUMNS || rev->element_count() != b * 2 * kSlots ||
      fix->element_count() != b * 2 * kSlots * FX_COLUMNS)
    return ffi::Error::InvalidArgument("prep: output record shapes differ from the layout");
  Outputs out{code->typed_data(), mon->typed_data(), side->typed_data(), rev->typed_data(), fix->typed_data()};
  if (b == 0) return ffi::Error::Success();
  const int threads = 128;
  const int64_t worlds = 2 * static_cast<int64_t>(b);
  prep_kernel<<<static_cast<unsigned>((worlds + threads - 1) / threads), threads, 0, stream>>>(in, out);
  return cuda_error("prep_kernel");
}

// ---------------------------------------------------------------------------
// MT19937 probe: random.Random(seed) draws for the CPython comparison.
//   spec[n,0] == 0: getrandbits(32) x draws; spec[n,0] == 1: randrange(spec[n,1]) x draws
//   (spec[n,1] is read as uint32).

__global__ void mt_probe_kernel(int64_t n, int64_t draws, const uint32_t* seeds, const int32_t* spec, uint32_t* out) {
  const int64_t t = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (t >= n) return;
  Mt19937 rng;
  mt_seed_words(rng, seeds + t * kSeedWords, kSeedWords);
  const int32_t mode = spec[2 * t];
  const uint32_t limit = static_cast<uint32_t>(spec[2 * t + 1]);
  for (int64_t d = 0; d < draws; ++d)
    out[t * draws + d] = mode == 0 ? mt_genrand_uint32(rng) : mt_randbelow(rng, limit);
}

static ffi::Error MtProbeImpl(cudaStream_t stream, int64_t draws, ffi::Buffer<ffi::U32> seeds, ffi::Buffer<ffi::S32> spec,
                              ffi::Result<ffi::Buffer<ffi::U32>> out) {
  const int64_t n = static_cast<int64_t>(spec.element_count() / 2);
  if (seeds.element_count() != static_cast<size_t>(n * kSeedWords) || out->element_count() != static_cast<size_t>(n * draws))
    return ffi::Error::InvalidArgument("mt probe: [n,8] seed words, [n,2] specs and [n,draws] outputs required");
  if (n == 0) return ffi::Error::Success();
  const int threads = 128;
  mt_probe_kernel<<<static_cast<unsigned>((n + threads - 1) / threads), threads, 0, stream>>>(
      n, draws, seeds.typed_data(), spec.typed_data(), out->typed_data());
  return cuda_error("mt_probe_kernel");
}

}  // namespace prep
}  // namespace v2e

XLA_FFI_DEFINE_HANDLER_SYMBOL(V2ePrepWorlds, v2e::prep::PrepWorldsImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("flags")
                                  .Attr<int64_t>("pre_error")
                                  .Attr<int64_t>("pre_error_after")
                                  .Attr<int64_t>("volatile_count")
                                  .Attr<int64_t>("universes")
                                  .Attr<int64_t>("variants")
                                  .Attr<int64_t>("species")
                                  .Attr<int64_t>("moves")
                                  .RemainingArgs()
                                  .Ret<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>(),
                              {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(V2eMtProbe, v2e::prep::MtProbeImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("draws")
                                  .Arg<ffi::Buffer<ffi::U32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::U32>>(),
                              {ffi::Traits::kCmdBufferCompatible});
