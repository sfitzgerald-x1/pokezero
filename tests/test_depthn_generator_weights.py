"""Exact generator fixtures: these run Node, with no engine compilation."""

import json
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "gen3_generator_weights.cjs"


class ExactGenerator(unittest.TestCase):
    def node(self, body, *, showdown=False):
        prelude = f"const e = require({json.dumps(str(SCRIPT))});\n"
        prelude += """
            function signedEnumeration(species, complete=false) {
                const payload={schema:e.SCHEMA,showdown_commit:e.PIN,probability_model:e.PROBABILITY_MODEL,
                    context_keys:e.CONTEXT_KEYS,executed_sha256:e.EXECUTED_PIN,compiler_sha256:'b'.repeat(64),complete,species};
                return {...payload,sha256:e.sha256(JSON.stringify(payload))};
            }
        """
        if showdown:
            root = os.environ.get("POKEZERO_SHOWDOWN_ROOT")
            if root is None:
                from tests._showdown_root import showdown_root
                root = str(showdown_root())
            prelude += f"const g = e.loadGenerator({json.dumps(root)});\n"
        completed = subprocess.run(
            ["node", "-e", prelude + body], check=True, text=True,
            capture_output=True, cwd=REPO, timeout=60,
        )
        return json.loads(completed.stdout)

    def test_exact_togetic_fixture(self):
        result = self.node("""
            const r = e.enumerate(g, 'togetic');
            console.log(JSON.stringify({leaves:r.leaves, ...e.integerDistribution(r.outcomes)}));
        """, showdown=True)
        self.assertEqual(result["denominator"], "8")
        weights = {tuple(json.loads(key)[2]): int(value) for key, value in result["weights"].items()}
        self.assertEqual(weights, {
            ("charm", "seismictoss", "softboiled", "toxic"): 1,
            ("encore", "seismictoss", "softboiled", "toxic"): 1,
            ("flamethrower", "seismictoss", "softboiled", "toxic"): 1,
            ("protect", "seismictoss", "toxic", "wish"): 4,
            ("seismictoss", "softboiled", "thunderwave", "toxic"): 1,
        })
        # Includes every move permutation AND both shiny outcomes.
        self.assertEqual(result["leaves"], 1344)

    def test_variable_branch_depth_and_duplicate_mass(self):
        result = self.node("""
            const g = {randomSet() {
                const a = this.prng.sample(['x', 'x', 'y']);
                const item = a === 'x' ? (this.prng.randomChance(1, 3) ? 'a' : 'b') : 'a';
                const moves = ['one', 'two']; this.prng.shuffle(moves);
                return {species:'test',level:50,moves,ability:'same',item,evs:{},ivs:{}};
            }};
            const r = e.enumerate(g, 'test');
            console.log(JSON.stringify(e.integerDistribution(r.outcomes)));
        """)
        self.assertEqual(result["denominator"], "9")
        self.assertEqual({json.loads(k)[4]: int(v) for k, v in result["weights"].items()}, {"a": 5, "b": 4})

    def test_context_bits_and_spikes_threshold(self):
        result = self.node("""
            let error;try {e.contextIndex({spikes:2});}catch(e){error=e.message;}
            console.log(JSON.stringify({roundtrip:Array.from({length:16},(_,i)=>e.contextIndex(e.contextFromIndex(i))),
              one:e.contextIndex({spikes:1}), error}));
        """)
        self.assertEqual(result["roundtrip"], list(range(16)))
        self.assertEqual(result["one"], 0)
        self.assertIn("unreachable", result["error"])

    def test_hidden_power_identity_and_cosmetics(self):
        result = self.node("""
            const s={species:'Unown-B',speciesId:'unown',level:100,moves:['Hidden Power Bug'],ability:'Levitate',item:'Choice Band'};
            console.log(JSON.stringify([e.setKey(s),e.setKey({...s,species:'Unown-Z',role:'other',shiny:true}),
                e.setKey({...s,moves:['Hidden Power Fighting']})]));
        """)
        self.assertEqual(result[0], result[1])
        self.assertNotEqual(result[0], result[2])

    def test_leaf_budget_fails_closed(self):
        result = self.node("""
            try {e.enumerate(g, 'togetic', {}, {maxLeaves:1});} catch (error) {
                console.log(JSON.stringify(error.message));
            }
        """, showdown=True)
        self.assertIn("leaf budget exceeded", result)

    def test_mapping_sums_duplicates_without_reweighting_roles(self):
        result = self.node("""
            const v={species:'test',level:50,moves:['one'],ability:'same',item:'a'};
            const variants={source_hash:'source',species:{test:[{...v,variant_id:'first'},
                {...v,variant_id:'duplicate'},{...v,item:'never',variant_id:'zero'}]}};
            const enumeration=signedEnumeration({test:{0:{denominator:'1',weights:{[e.setKey(v)]:'1'}}}});
            console.log(JSON.stringify(e.mapToVariants(enumeration,variants,{diagnostic:true})));
        """)
        self.assertEqual(result["species"]["test"]["contexts"]["0"], ["1", "0", "0"])

    def test_mapping_fails_on_missing_generated_variant(self):
        result = self.node("""
            try {e.mapToVariants(signedEnumeration({test:{0:{denominator:'1',weights:{missing:'1'}}}}),
                {species:{test:[]}},{diagnostic:true});} catch(error) {console.log(JSON.stringify(error.message));}
        """)
        self.assertIn("generated set missing", result)

    def test_published_mapping_requires_all_contexts_and_valid_hash(self):
        result = self.node("""
            const errors=[];const enumeration=signedEnumeration({test:{0:{denominator:'1',weights:{x:'1'}}}});
            for(const table of [enumeration,{...enumeration,sha256:'bad'},signedEnumeration(enumeration.species,true)]) {
                try {e.mapToVariants(table,{species:{test:[]}});}catch(error){errors.push(error.message);}
            }
            console.log(JSON.stringify(errors));
        """)
        self.assertEqual(result, ["incomplete species coverage", "enumeration hash mismatch", "incomplete context coverage: test"])


class GeneratorTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pokezero.randbat import load_gen3_randbat_source_cached
        from tests._showdown_root import showdown_root

        cls.source = load_gen3_randbat_source_cached(showdown_root())

    def test_published_table_is_source_bound_and_complete(self):
        from pokezero.generator_weights import load_generator_weights

        table = load_generator_weights(self.source)
        self.assertEqual(len(table.species), 220)
        self.assertEqual(table.source_hash, self.source.metadata.source_hash)
        self.assertEqual(table.species["togetic"].denominator, 8)
        self.assertEqual(sorted(w for w in table.weights("togetic", team_context={}) if w), [1, 1, 1, 1, 4])
        for species in table.species.values():
            self.assertEqual(len(species.contexts), 16)
            self.assertLessEqual(len(species.variant_ids), 64)
            for weights in species.contexts:
                self.assertEqual(sum(weights), species.denominator)
        with self.assertRaises(TypeError):
            table.species["togetic"] = None
        with self.assertRaises(TypeError):
            table.weights("togetic")

    def test_context_is_explicit_and_reachable(self):
        from pokezero.generator_weights import context_index

        self.assertEqual(context_index(dict(rain=1, sun=True, statusCure=True, rapidSpin=1, spikes=1)), 15)
        for context in ({"spikes": 2}, {"rain": "1"}, {"rain": 0.1}, {"spikes": 1.0}):
            with self.assertRaises(ValueError):
                context_index(context)

    def test_source_reordering_rejected(self):
        from dataclasses import replace
        from pokezero.generator_weights import load_generator_weights

        universes = dict(self.source.universes)
        universes["togetic"] = replace(universes["togetic"], variants=tuple(reversed(universes["togetic"].variants)))
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            load_generator_weights(replace(self.source, universes=universes))

    def test_corrupt_and_incomplete_artifacts_rejected(self):
        from pokezero.generator_weights import digest, load_generator_weights

        raw = json.loads((REPO / "src/pokezero/data/gen3_generator_weights.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.json"
            raw["complete"] = False
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_generator_weights(self.source, path)
            del raw["sha256"]
            raw["sha256"] = digest(raw)
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                load_generator_weights(self.source, path)


class GeneratorStatistics(unittest.TestCase):
    def fixtures(self):
        from pokezero.generator_weights import CONTEXT_KEYS, EXECUTED_PIN, SCHEMA, SHOWDOWN_PIN, digest

        spec = importlib.util.spec_from_file_location("generator_chi_square", REPO / "scripts/gen3_generator_chi_square.py")
        gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate)
        variants = [dict(variant_id=f"test-{i}", species="Test", level=50,
                         moves=["tackle"], ability="Run Away", item=item)
                    for i, item in enumerate(("Leftovers", "Lum Berry"))]
        keys = [json.dumps(["test", 50, ["tackle"], "runaway", item], separators=(",", ":"))
                for item in ("leftovers", "lumberry")]
        manifest = dict(source_hash="fixture", species={"test": variants})
        raw = dict(schema=SCHEMA, showdown_commit=SHOWDOWN_PIN, probability_model="independent-uniform-api-choices",
                   executed_sha256=EXECUTED_PIN, compiler_sha256="b" * 64, complete=True, context_keys=list(CONTEXT_KEYS),
                   species={"test": {str(i): dict(denominator="2", weights=dict.fromkeys(keys, "1")) for i in range(16)}})
        raw["sha256"] = digest(raw)
        sample = dict(schema="gen3-generator-sample-v1", showdown_commit=SHOWDOWN_PIN, executed_sha256=EXECUTED_PIN,
                      seed="1,2,3,4", sets=1200, teams=200, counts={"test": {"0": dict.fromkeys(keys, 600)}})
        return gate, raw, sample, manifest

    def test_conditional_chi_square_and_coverage(self):
        gate, raw, sample, manifest = self.fixtures()
        result = gate.evaluate(raw, [sample], manifest, minimum_sets=1000)
        self.assertTrue(result["passed"])
        self.assertEqual(result["degrees_of_freedom"], 1)
        self.assertEqual(result["p"], 1.0)
        self.assertEqual(result["coverage_missing"], 0)
        self.assertEqual(result["enumeration_sha256"], raw["sha256"])

    def test_statistics_refuse_forged_provenance_and_missing_variant(self):
        from pokezero.generator_weights import digest

        gate, raw, sample, manifest = self.fixtures()
        raw["executed_sha256"] = sample["executed_sha256"] = "0" * 64
        del raw["sha256"]
        raw["sha256"] = digest(raw)
        with self.assertRaisesRegex(ValueError, "validation failed"):
            gate.evaluate(raw, [sample], manifest, minimum_sets=1000)
        gate, raw, sample, manifest = self.fixtures()
        manifest["species"]["test"].pop()
        with self.assertRaisesRegex(ValueError, "validation failed"):
            gate.evaluate(raw, [sample], manifest, minimum_sets=1000)

    def test_statistics_refuse_unsupported_set_and_small_sample(self):
        gate, raw, sample, manifest = self.fixtures()
        with self.assertRaisesRegex(ValueError, "need >="):
            gate.evaluate(raw, [sample], manifest)
        row = sample["counts"]["test"]["0"]
        key = next(iter(row))
        row["unlisted"] = row.pop(key)
        with self.assertRaisesRegex(ValueError, "outside exact support"):
            gate.evaluate(raw, [sample], manifest, minimum_sets=1000)

    def test_statistics_refuse_repeated_shards(self):
        gate, raw, sample, manifest = self.fixtures()
        with self.assertRaisesRegex(ValueError, "duplicate sample seed"):
            gate.evaluate(raw, [sample, sample], manifest, minimum_sets=1000)

    def test_rare_cells_pool_by_expected_counts(self):
        gate, _, _, _ = self.fixtures()
        self.assertEqual(gate.pooled_cells(dict(a=2, b=4, c=8, d=1), dict(a=1, b=3, c=10, d=1)), [[6.0, 4], [9.0, 11]])


if __name__ == "__main__":
    unittest.main()
