"""Exact joint DP checked against an independent, unaggregated tiny generator."""
from collections import Counter, defaultdict
from fractions import Fraction
import random
import unittest

from pokezero.generator_weights import GeneratorWeights, SpeciesWeights
from pokezero.joint_generator import JointGenerator, JointBudgetExceeded, ImpossibleJointEvidence


def fixture(slots=3, rich=False):
    profiles = [
        ("A", ["a"], "Fire", "Water", 1, ["E"]),
        ("B", ["b"], "Water", "Grass", 1, []),
        ("C", ["c"], "Normal", "Fighting", 1, []),
        ("D", ["d"], "Normal", "Fighting", 1, []),
        ("E", ["e"], "Grass", "Fire", 1, ["A"]),
        ("Deoxys", ["f", "g"], "Psychic", "Bug", 2, []),
        ("Wobbuffet", ["wobbuffet"], "Psychic", "Bug", 1, []),
    ]
    bases = [dict(name=name, forms=forms, types=[kind], weak=[weak], double_weak=[],
                  tokens=tokens, level100=False, incompatible=bad)
             for name, forms, kind, weak, tokens, bad in profiles]
    if rich:
        for base in bases:
            base['level100'] = base['name'] in ('C', 'Deoxys')
            base['double_weak'] = ['Water'] if base['name'] in ('B', 'D') else []
    variants = {}
    weights = {}
    for base in bases:
        for species in base["forms"]:
            count = 2 if species in ("a", "b") else 1
            variants[species] = [dict(variant_id=f"{species}-{i}", species=species, level=80,
                                      moves=["raindance" if species == "a" and i == 0 else "tackle"],
                                      ability="Swift Swim" if species == "b" and i == 0 else "Shell Armor",
                                      item="Leftovers") for i in range(count)]
            rows = tuple((1, 1) if species == "a" else
                         ((1, 0) if context & 1 else (0, 1)) if species == "b" else (1,)
                         for context in range(16))
            weights[species] = SpeciesWeights(2 if species == "a" else 1,
                                             tuple(v["variant_id"] for v in variants[species]), rows)
    if rich:
        variants['a'][1]['moves'] = ['sunnyday']
        variants['c'][0]['moves'] = ['healbell']
        variants['d'][0]['moves'] = ['rapidspin']
        weights['b'] = SpeciesWeights(1, weights['b'].variant_ids,
                                     tuple((1, 0) if context & 5 else (0, 1) for context in range(16)))
    model = dict(slots=slots, bases=bases, types=sorted({x for b in bases for field in ("types", "weak") for x in b[field]}))
    return model, dict(source_hash="fixture", species=variants), GeneratorWeights("fixture", "fixture", weights)


def brute(model, manifest, table, evidence=None, fixed=None, initial_wobbuffet=False, terminal_weights=(1, 1)):
    """Enumerate ordered individual bases/formes/variants without DP or classes."""
    evidence, fixed = dict(evidence or {}), dict(fixed or {})
    for species in fixed.values():
        evidence.setdefault(species, (1,) * len(manifest["species"][species]))
    outcomes = defaultdict(Fraction)

    def visit(chosen, team, context, probability):
        if len(team) == model["slots"]:
            if set(evidence) <= {species for species, _ in team}:
                flag = initial_wobbuffet or any(species == "wobbuffet" for species, _ in team[:-1])
                outcomes[team] += probability * terminal_weights[int(flag)]
            return
        types, weak, double = Counter(), Counter(), Counter()
        for index in chosen:
            base = model["bases"][index]
            types.update(base["types"]); weak.update(base["weak"]); double.update(base["double_weak"])
        eligible = []
        for index, base in enumerate(model["bases"]):
            if index in chosen or initial_wobbuffet and base["name"] == "Wobbuffet":
                continue
            if any(types[k] >= 2 for k in base["types"]) or any(weak[k] >= 3 for k in base["weak"]):
                continue
            if any(double[k] for k in base["double_weak"]):
                continue
            if base["level100"] and any(model["bases"][i]["level100"] for i in chosen):
                continue
            if any(model["bases"][i]["name"] in base["incompatible"] for i in chosen):
                continue
            eligible.append(index)
        total = sum(model["bases"][i]["tokens"] for i in eligible)
        for index in eligible:
            base = model["bases"][index]
            for species in base["forms"]:
                if len(team) in fixed and fixed[len(team)] != species:
                    continue
                for vi, weight in enumerate(table.species[species].contexts[context]):
                    likelihood = evidence[species][vi] if species in evidence else 1
                    if not weight or not likelihood:
                        continue
                    variant = manifest["species"][species][vi]
                    moves = variant['moves']
                    after = (context | int('raindance' in moves) | (int('sunnyday' in moves) << 1)
                             | (int('healbell' in moves or 'aromatherapy' in moves) << 2)
                             | (int('rapidspin' in moves) << 3))
                    step = Fraction(base["tokens"] * weight * likelihood,
                                    total * len(base["forms"]) * table.species[species].denominator)
                    visit(chosen + (index,), team + ((species, vi),), after, probability * step)
    visit((), (), 0, Fraction(1))
    return dict(outcomes)


class JointExactness(unittest.TestCase):
    def check_oracle(self, evidence=None, fixed=None, initial_wobbuffet=False):
        parts = fixture()
        exact = brute(*parts, evidence, fixed, initial_wobbuffet)
        joint = JointGenerator(*parts, evidence=evidence, fixed_slots=fixed, initial_wobbuffet=initial_wobbuffet)
        self.assertEqual(Fraction(joint.normalizer(), joint.quantum ** joint.slots), sum(exact.values()))
        # Check the first-member marginal, including class-member multiplicity
        # and variant conditioning, against the independent complete enumeration.
        actual = defaultdict(Fraction)
        for (gi, after, code), factor in joint._branches(0, 0, 0):
            mass = Fraction(factor * joint._mass(code, after), joint.normalizer() * len(joint.groups[gi]))
            for bi in joint.groups[gi]:
                choices = [(entry, weight) for entry, weight in joint._variant_options(bi, 0) if entry[2] == after]
                total = sum(w for _, w in choices)
                for (species, vi, _), weight in choices:
                    actual[species, vi] += mass * Fraction(weight, total)
        expected = defaultdict(Fraction)
        normalizer = sum(exact.values())
        for team, probability in exact.items():
            expected[team[0]] += probability / normalizer
        self.assertEqual({k: v for k, v in actual.items() if v}, dict(expected))
        rng = random.Random(72)
        for _ in range(100):
            draw = tuple((m.species, m.variant_index) for m in joint.draw(rng))
            self.assertIn(draw, exact)
        return joint

    def test_prior_classes_and_token_multiplicity(self):
        joint = self.check_oracle()
        self.assertTrue(any(len(group) == 2 for group in joint.groups))

    def test_weather_evidence_changes_hidden_team(self):
        joint = self.check_oracle({"b": (1, 0)})
        rng = random.Random(3)
        for _ in range(100):
            team = joint.draw(rng)
            species = [m.species for m in team]
            self.assertLess(species.index("a"), species.index("b"))
            self.assertEqual(team[species.index("a")].variant_index, 0)

    def test_integer_damage_likelihood(self):
        self.check_oracle({"a": (1, 7), "b": (2, 5)})

    def test_required_capacity_and_completion_projection(self):
        joint = self.check_oracle({"c": (2,), "d": (3,)}, {0: "c"})
        self.assertLess(len(joint.completion.groups), len(joint.groups))

    def test_partial_dead_ends_keep_exact_success_probability(self):
        model, manifest, table = fixture()
        model["bases"] = [b for b in model["bases"] if b["name"] in ("A", "B", "C", "E")]
        model["types"] = ["Fire", "Water", "Grass"]
        for base in model["bases"]:
            base.update(types=[{"A": "Fire", "B": "Water"}.get(base["name"], "Grass")],
                        weak=[], double_weak=[], incompatible={"A": ["C", "E"], "C": ["A"], "E": ["A"]}.get(base["name"], []))
        manifest["species"] = {s: v for s, v in manifest["species"].items() if s in ("a", "b", "c", "e")}
        table = GeneratorWeights("fixture", "fixture", {s: table.species[s] for s in manifest["species"]})
        exact = brute(model, manifest, table)
        joint = JointGenerator(model, manifest, table)
        self.assertEqual(sum(exact.values()), Fraction(2, 3))
        self.assertEqual(Fraction(joint.normalizer(), joint.quantum ** 3), Fraction(2, 3))
        self.assertFalse(joint.certified_completion)

    def test_inputs_are_snapshotted(self):
        model, manifest, table = fixture()
        likelihood = [1, 7]
        joint = JointGenerator(model, manifest, table, evidence={"a": likelihood})
        expected = joint.normalizer()
        model["bases"].clear()
        manifest["species"].clear()
        likelihood[:] = [0, 0]
        self.assertEqual(joint.normalizer(), expected)
        self.assertEqual(len(joint.draw(random.Random(2))), 3)

    def test_fixed_slots_keep_prior_denominator(self):
        self.check_oracle({"f": (1,)}, {0: "f", 2: "b"})

    def test_cross_team_wobbuffet_flag(self):
        self.check_oracle(initial_wobbuffet=True)
        with self.assertRaises(ImpossibleJointEvidence):
            JointGenerator(*fixture(), evidence={"wobbuffet": (1,)}, initial_wobbuffet=True)

    def test_type_caps_can_make_completion_impossible(self):
        model, manifest, table = fixture()
        for base in model["bases"]:
            base["types"] = ["Water"]
        model["types"] = sorted(set(model["types"]) | {"Water"})
        with self.assertRaises(ImpossibleJointEvidence):
            JointGenerator(model, manifest, table).normalizer()

    def test_zero_likelihood_never_falls_back(self):
        with self.assertRaises(ImpossibleJointEvidence):
            JointGenerator(*fixture(), evidence={"b": (0, 0)}).normalizer()

    def test_budget_never_returns_approximation(self):
        joint = JointGenerator(*fixture(), max_states=1)
        with self.assertRaises(JointBudgetExceeded):
            joint.normalizer()
        self.assertIsNone(joint._normalizer)


if __name__ == "__main__":
    unittest.main()
