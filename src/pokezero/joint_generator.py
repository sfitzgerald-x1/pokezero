"""Exact finite-state team posterior; standalone until the integration gates pass.

Unrevealed bases are exchangeable only when legality, token multiplicity and
all sixteen set-context transition distributions agree. The DP still retains
the number selected from every such class. Hard evidence and integer damage
likelihoods are local factors; species selection and the backline are joint.
Exceeding the explicit work budget raises, never returns an approximation.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import reduce
from math import gcd, lcm
from time import perf_counter
import hashlib
import json
import weakref


# Finite-domain UNSAT check: every legal subset of at most five bases has
# an eligible extension, with either incoming Wobbuffet flag. This is a
# property of this exact standard-format catalog, not of arbitrary fixtures.
_COMPLETABLE_MODEL = "74d83850ba3cb47ced59caa136e2775d8dc6d7ee664e06dd8e3e80dce35d7aa9"


class JointBudgetExceeded(RuntimeError):
    pass


class ImpossibleJointEvidence(ValueError):
    pass


@dataclass(frozen=True)
class JointMember:
    species: str
    variant_index: int
    slot: int
    incoming_context: int


def _context_after(context, variant):
    moves = set(variant["moves"])
    ability = variant["ability"].lower().replace(" ", "")
    return (context | int("raindance" in moves or ability == "drizzle")
            | (int("sunnyday" in moves or ability == "drought") << 1)
            | (int(bool(moves & {"aromatherapy", "healbell"})) << 2)
            | (int("rapidspin" in moves) << 3))


def _pick(weighted, rng):
    total = sum(weight for _, weight in weighted)
    if total <= 0:
        raise ImpossibleJointEvidence("zero conditional branch mass")
    value = rng.randrange(total)
    for item, weight in weighted:
        if value < weight:
            return item
        value -= weight
    raise AssertionError("integer draw exhausted")


class JointGenerator:
    """Exact DP over legality-class occupancy and generation context.

    `evidence[species]` contains nonnegative integer likelihoods in variant
    order. Every named species is required somewhere on the team. A fixed slot
    is a fact about ORIGINAL generation order, not current party order.
    `initial_wobbuffet` is the incoming generator flag, not current own-team
    membership: the pinned generator does not set that flag for slot six.
    """

    def __init__(self, model, manifest, table, *, evidence=None, fixed_slots=None,
                 initial_wobbuffet=False, terminal_weights=(1, 1), max_states=100_000, max_seconds=10.0, deadline=None):
        from .joint_budget import check_deadline
        self.deadline = deadline
        check_deadline(deadline)
        # Snapshot caller-owned dictionaries before deriving and caching states.
        model, manifest = json.loads(json.dumps(model)), json.loads(json.dumps(manifest))
        check_deadline(deadline)
        self.model, self.manifest, self.table = model, manifest, table
        payload = {key: value for key, value in model.items() if key != "sha256"}
        digest = hashlib.sha256(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        if "sha256" in model and model["sha256"] != digest:
            raise ValueError("team model content hash mismatch")
        species = manifest["species"]
        if table.source_hash != manifest["source_hash"] or set(table.species) != set(species):
            raise ValueError("joint table/manifest source mismatch")
        manifest_digest = hashlib.sha256(json.dumps(manifest, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        if ((table.variant_manifest_sha256 is not None and table.variant_manifest_sha256 != manifest_digest)
                or (digest == _COMPLETABLE_MODEL and table.variant_manifest_sha256 != manifest_digest)):
            raise ValueError("joint physical manifest binding mismatch")
        forms = [form for base in model["bases"] for form in base["forms"]]
        if len(forms) != len(set(forms)) or set(forms) != set(species):
            raise ValueError("joint team-model species coverage mismatch")
        for name, variants in species.items():
            check_deadline(deadline)
            weights = table.species[name]
            if (not 0 < len(variants) <= 64 or tuple(v["variant_id"] for v in variants) != weights.variant_ids
                    or len(weights.contexts) != 16 or type(weights.denominator) is not int or weights.denominator <= 0
                    or any(len(row) != len(variants) or any(type(w) is not int or w < 0 for w in row)
                           or sum(row) != weights.denominator for row in weights.contexts)):
                raise ValueError("joint variant order/weight binding mismatch")
        self.certified_completion = (digest == _COMPLETABLE_MODEL and model.get("sha256") == digest
                                     and all(sum(row) == weights.denominator
                                             for weights in table.species.values() for row in weights.contexts))
        self.slots = model["slots"]
        if type(initial_wobbuffet) is not bool:
            raise ValueError("incoming generator flag must be boolean")
        self.initial_wobbuffet = initial_wobbuffet
        self.terminal_weights = tuple(terminal_weights)
        if len(self.terminal_weights) != 2 or any(type(w) is not int or w < 0 for w in self.terminal_weights):
            raise ValueError("invalid terminal likelihood")
        self.evidence = {species: tuple(weights) for species, weights in (evidence or {}).items()}
        self.fixed_species = dict(fixed_slots or {})
        self.max_states, self.max_seconds = max_states, max_seconds
        for slot, species in self.fixed_species.items():
            if not 0 <= slot < self.slots:
                raise ValueError("invalid fixed generation slot")
            self.evidence.setdefault(species, (1,) * len(manifest["species"][species]))
        if len(self.evidence) > self.slots:
            raise ImpossibleJointEvidence("more required species than team slots")
        if len(set(self.fixed_species.values())) != len(self.fixed_species):
            raise ImpossibleJointEvidence("one species fixed to multiple slots")
        for species, likelihood in self.evidence.items():
            if (len(likelihood) != len(manifest["species"][species]) or
                    any(type(w) is not int or w < 0 for w in likelihood)):
                raise ValueError("invalid variant likelihood")
        self.bases = model["bases"]
        required_bases = {}
        for bi, base in enumerate(self.bases):
            required = [form for form in base["forms"] if form in self.evidence]
            if len(required) > 1:
                raise ImpossibleJointEvidence("multiple formes of one base species")
            if required:
                required_bases[bi] = required[0]
        if sum(map(len, (b["forms"] for b in self.bases))) != len(manifest["species"]):
            raise ValueError("team model and variant manifest differ")
        if len(required_bases) != len(self.evidence):
            raise ImpossibleJointEvidence("unmodeled required species")
        self.required_bases = required_bases
        self.base_denominators = [lcm(*(table.species[s].denominator * len(b["forms"]) for s in b["forms"]))
                                  for b in self.bases]
        self.base_rows = [self._rows(bi) for bi in range(len(self.bases))]
        grouped = {}
        for bi, base in enumerate(self.bases):
            if initial_wobbuffet and base["name"] == "Wobbuffet":
                if bi in required_bases:
                    raise ImpossibleJointEvidence("incoming Wobbuffet flag excludes required species")
                continue
            profile = (tuple(base["types"]), tuple(base["weak"]), tuple(base["double_weak"]),
                       base["level100"], tuple(base["incompatible"]), base["tokens"])
            key = (profile, self.base_rows[bi], bi if bi in required_bases or base["name"] == "Wobbuffet" else None)
            grouped.setdefault(key, []).append(bi)
        self.groups = tuple(tuple(members) for members in grouped.values())
        self.base_group = {bi: gi for gi, members in enumerate(self.groups) for bi in members}
        self.group_rows = tuple(self.base_rows[members[0]] for members in self.groups)
        self.group_tokens = tuple(self.bases[members[0]]["tokens"] for members in self.groups)
        self.wobb_groups = frozenset(gi for gi, members in enumerate(self.groups) if self.bases[members[0]]["name"] == "Wobbuffet")
        self.required_low = sum(1 << (2 * self.base_group[bi]) for bi in required_bases)
        self.fixed = {slot: self.base_group[next(bi for bi, form in required_bases.items() if form == species)]
                      for slot, species in self.fixed_species.items()}
        self.fixed_group = {group: slot for slot, group in self.fixed.items()}
        self.required_contexts = tuple((1 << (2 * self.base_group[bi]),
                                        sum(1 << context for context, row in enumerate(self.base_rows[bi]) if row[1]))
                                       for bi in required_bases)
        self.future_contexts = tuple(sum(1 << later for later in range(16) if later & now == now) for now in range(16))
        self._configure_legality()
        max_tokens = sum(len(members) * self.group_tokens[gi] for gi, members in enumerate(self.groups))
        set_denominator = lcm(*(row[0] for rows in self.group_rows for row in rows))
        self.quantum = lcm(*range(1, max_tokens + 1)) * set_denominator
        self.powers = tuple(self.quantum ** i for i in range(self.slots + 1))
        self.cache, self.legality_cache, self.future_cache = {}, {}, {}
        self.nodes, self.edges, self.hits = 0, 0, 0
        self.started = None
        self._normalizer = None
        self.completion = _LegalityCompletion(self)

    def _configure_legality(self):
        self.low_bits = sum(1 << (2 * gi) for gi in range(len(self.groups)))
        self.all_groups = (1 << len(self.groups)) - 1
        caps = set()
        for field, limit in (("types", 2), ("weak", 3), ("double_weak", 1)):
            for kind in self.model["types"]:
                groups = tuple(gi for gi, members in enumerate(self.groups) if kind in self.bases[members[0]][field])
                if groups:
                    caps.add((groups, limit))
        levels = tuple(gi for gi, members in enumerate(self.groups) if self.bases[members[0]]["level100"])
        if levels:
            caps.add((levels, 1))
        self.caps = tuple((sum(1 << (2 * gi) for gi in groups), sum(1 << gi for gi in groups), limit)
                          for groups, limit in sorted(caps))
        self.incompatible = tuple(sum(1 << gj for gj, other in enumerate(self.groups)
                                      if self.bases[other[0]]["name"] in self.bases[members[0]]["incompatible"])
                                  for members in self.groups)

    def _variant_options(self, bi, context):
        base, denominator = self.bases[bi], self.base_denominators[bi]
        for species in base["forms"]:
            if bi in self.required_bases and species != self.required_bases[bi]:
                continue
            weights = self.table.species[species]
            likelihoods = self.evidence.get(species)
            scale = denominator // (weights.denominator * len(base["forms"]))
            for vi, (variant, weight) in enumerate(zip(self.manifest["species"][species], weights.contexts[context])):
                numerator = weight * scale * (likelihoods[vi] if likelihoods is not None else 1)
                if numerator:
                    yield (species, vi, _context_after(context, variant)), numerator

    def _rows(self, bi):
        rows = []
        for context in range(16):
            counts = Counter()
            for (_, _, after), numerator in self._variant_options(bi, context):
                counts[after] += numerator
            denominator = self.base_denominators[bi]
            common = reduce(gcd, counts.values(), denominator)
            rows.append((denominator // common, tuple((after, value // common) for after, value in sorted(counts.items()))))
        return tuple(rows)

    def _depth(self, code):
        return (code & self.low_bits).bit_count() + 2 * (code & (self.low_bits << 1)).bit_count()

    def eligible(self, code):
        """Retain *all* eligible prior tokens in the denominator, including evidence-excluded choices."""
        if code in self.legality_cache:
            return self.legality_cache[code]
        blocked = 0
        for digits, mask, limit in self.caps:
            count = (code & digits).bit_count() + 2 * (code & (digits << 1)).bit_count()
            if count >= limit:
                blocked |= mask
        for gi, mask in enumerate(self.incompatible):
            if mask and (code >> (2 * gi)) & 3:
                blocked |= mask
        choices = []
        total = 0
        remaining_mask = self.all_groups & ~blocked
        while remaining_mask:
            bit = remaining_mask & -remaining_mask
            gi = bit.bit_length() - 1
            remaining_mask ^= bit
            count = len(self.groups[gi]) - ((code >> (2 * gi)) & 3)
            if count:
                tokens = count * self.group_tokens[gi]
                total += tokens
                choices.append((gi, tokens))
        result = total, tuple(choices)
        if len(self.legality_cache) < 65_536:
            self.legality_cache[code] = result
        return result

    def _branches(self, code, context, depth):
        total, choices = self.eligible(code)
        if not total:
            return
        forced = self.fixed.get(depth)
        missing = self.required_low & ~code
        slots_left = self.slots - depth
        allowed = self._future_allowed(code)
        for gi, tokens in choices:
            if not (allowed & (1 << gi)):
                continue
            if forced is not None and gi != forced:
                continue
            if gi in self.fixed_group and self.fixed_group[gi] != depth:
                continue
            digit = 1 << (2 * gi)
            if missing.bit_count() == slots_left and not (missing & digit):
                continue
            denominator, outcomes = self.group_rows[gi][context]
            scale = self.quantum // (total * denominator)
            assert self.quantum % (total * denominator) == 0
            for after, numerator in outcomes:
                yield (gi, after, code + digit), tokens * numerator * scale

    def _future_allowed(self, code):
        if code in self.future_cache:
            return self.future_cache[code]
        projected = code | self.required_low
        blocked = 0
        for digits, mask, limit in self.caps:
            count = (projected & digits).bit_count() + 2 * (projected & (digits << 1)).bit_count()
            if count > limit:
                return 0
            if count == limit:
                blocked |= mask
        present, remaining = 0, projected
        while remaining:
            gi = ((remaining & -remaining).bit_length() - 1) // 2
            present |= 1 << gi
            remaining &= ~(3 << (2 * gi))
        for gi, mask in enumerate(self.incompatible):
            if present & (1 << gi):
                if present & mask:
                    return 0
                blocked |= mask
        # Missing required members are already in the projected counts;
        # drawing one consumes its reserved capacity rather than adding to it.
        missing_groups = sum(1 << gi for gi in range(len(self.groups))
                             if (self.required_low & ~code) & (1 << (2 * gi)))
        result = (self.all_groups & ~blocked) | missing_groups
        if len(self.future_cache) < 65_536:
            self.future_cache[code] = result
        return result

    def _charge(self):
        from .joint_budget import check_deadline
        check_deadline(self.deadline)
        self.nodes += 1
        if self.nodes > self.max_states or (self.nodes % 128 == 0 and perf_counter() - self.started > self.max_seconds):
            raise JointBudgetExceeded(f"joint DP budget exceeded: nodes={self.nodes}, seconds={perf_counter()-self.started:.6f}")

    def _mass(self, code, context, outgoing_wobb=False):
        from .joint_budget import check_deadline
        check_deadline(self.deadline)
        depth = self._depth(code)
        missing = (self.required_low & ~code).bit_count()
        if missing > self.slots - depth:
            return 0
        if depth == self.slots:
            return self.terminal_weights[int(outgoing_wobb)] if not missing else 0
        settled_terminal = outgoing_wobb or self.terminal_weights[0] == self.terminal_weights[1] or depth == self.slots - 1
        if not missing and settled_terminal:
            # With every evidence factor already consumed, set choices no
            # longer affect completion probability. Sum them out exactly;
            # still account for possible species-selection dead ends.
            if self.certified_completion:
                return self.powers[self.slots - depth] * self.terminal_weights[int(outgoing_wobb)]
            return self.completion._completion_mass(self.completion.project(code), depth) * self.terminal_weights[int(outgoing_wobb)]
        if any(not (code & digit) and not (support & self.future_contexts[context]) for digit, support in self.required_contexts):
            return 0
        key = (code << 5) | (context << 1) | outgoing_wobb
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self._charge()
        result = 0
        _, eligible = self.eligible(code)
        available_low = sum(1 << (2 * gi) for gi, _ in eligible)
        # Rejected species cannot become eligible later: every team cap
        # and incompatibility is monotone along the construction path.
        if not ((self.required_low & ~code) & ~available_low):
            for (gi, after, next_code), weight in self._branches(code, context, depth):
                self.edges += 1
                flag = outgoing_wobb or (gi in self.wobb_groups and depth < self.slots - 1)
                result += weight * self._mass(next_code, after, flag)
        self.cache[key] = result
        return result

    def _completion_mass(self, code, depth):
        from .joint_budget import check_deadline
        check_deadline(self.deadline)
        if depth == self.slots:
            return 1
        if code in self.completion_cache:
            self.hits += 1
            return self.completion_cache[code]
        self._charge()
        total, choices = self.eligible(code)
        if not total:
            result = 0
        elif depth == self.slots - 1:
            result = self.quantum
        else:
            result = 0
            scale = self.quantum // total
            for gi, tokens in choices:
                self.edges += 1
                result += tokens * scale * self._completion_mass(code + (1 << (2 * gi)), depth + 1)
        self.completion_cache[code] = result
        return result

    def normalizer(self):
        from .joint_budget import check_deadline
        check_deadline(self.deadline)
        if self._normalizer is None:
            self.started = perf_counter()
            value = self._mass(0, 0, self.initial_wobbuffet)
            if not value:
                raise ImpossibleJointEvidence("no generator team satisfies the evidence")
            self._normalizer = value
        return self._normalizer

    def draw(self, rng):
        self.normalizer()
        code, context = 0, 0
        flag = self.initial_wobbuffet
        selected, team = set(), []
        for slot in range(self.slots):
            choices = [(branch, weight * self._mass(branch[2], branch[1], flag or (branch[0] in self.wobb_groups and slot < self.slots - 1)))
                       for branch, weight in self._branches(code, context, slot)]
            (gi, after, next_code) = _pick(choices, rng)
            members = [bi for bi in self.groups[gi] if bi not in selected]
            bi = members[rng.randrange(len(members))]
            options = [(entry, weight) for entry, weight in self._variant_options(bi, context) if entry[2] == after]
            (species, vi, _) = _pick(options, rng)
            team.append(JointMember(species, vi, slot, context))
            selected.add(bi)
            code, context = next_code, after
            flag = flag or (gi in self.wobb_groups and slot < self.slots - 1)
        return tuple(team)

    def diagnostics(self):
        return dict(groups=len(self.groups), completion_groups=len(self.completion.groups), nodes=self.nodes,
                    certified_completion=self.certified_completion,
                    edges=self.edges + self.completion.edges, cache_hits=self.hits + self.completion.hits,
                    cached_states=len(self.cache), completion_states=len(self.completion.completion_cache),
                    legality_states=len(self.legality_cache) + len(self.completion.legality_cache),
                    elapsed_seconds=perf_counter() - self.started if self.started else 0,
                    quantum_bits=self.quantum.bit_length())


class _LegalityCompletion(JointGenerator):
    """Coarsen to legality-only classes after every evidence factor is consumed."""

    def __init__(self, parent):
        self.deadline = parent.deadline
        self.parent, self.model, self.bases = weakref.proxy(parent), parent.model, parent.bases
        self.slots, self.quantum = parent.slots, parent.quantum
        grouped = {}
        for bi in parent.base_group:
            base = self.bases[bi]
            key = (tuple(base["types"]), tuple(base["weak"]), tuple(base["double_weak"]),
                   base["level100"], tuple(base["incompatible"]), base["tokens"])
            grouped.setdefault(key, []).append(bi)
        self.groups = tuple(tuple(members) for members in grouped.values())
        group_for_base = {bi: gi for gi, members in enumerate(self.groups) for bi in members}
        self.parent_group = tuple(group_for_base[members[0]] for members in parent.groups)
        self.group_tokens = tuple(self.bases[members[0]]["tokens"] for members in self.groups)
        self._configure_legality()
        self.completion_cache, self.legality_cache = {}, {}
        self.hits, self.edges = 0, 0

    def _charge(self):
        self.parent._charge()

    def project(self, code):
        result = 0
        while code:
            gi = ((code & -code).bit_length() - 1) // 2
            count = (code >> (2 * gi)) & 3
            result += count << (2 * self.parent_group[gi])
            code &= ~(3 << (2 * gi))
        return result
