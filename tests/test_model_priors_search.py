"""Priors-in-the-loop gates for the encoded model search (engine swap).

`search_batched_multi_encoded` with `model_priors` (the default) prices the
root observation once for root priors and maps each priced branch's policy
output onto its child decision node's options. These tests pin the surface:

- batch=1 determinism holds with priors on (two identical runs, identical
  stats — priors are a deterministic function of the observations);
- the report carries the prior telemetry (`model_priors`, `root_priors`
  summing to 1 over the acting seat's arms, `prior_branches`,
  `prior_fallbacks`);
- `model_priors=False` restores uniform priors (kill switch for A/B);
- visit conservation holds in both modes.

Values-vs-exploration invariance is pinned Rust-side
(tree.rs `priors_reweight_exploration_not_values`); the mapping itself is
gated by tests/test_prior_action_mapping.py + scripts/prior_mapping_assert.py.
Random-weights artifact at the real v3 shape — never a strength claim.

## The opponent-priors half (`OpponentPriorsEncodedSearchTest`)

`use_opponent_priors` seeds the OPPONENT seat from the model's third output
head. Until this class existed, nothing anywhere ran that flag through a real
encoded search: `priors.rs`'s unit tests cover the gather/apply arithmetic off
hand-built `HeadPair`s, and `tests/test_opponent_priors_flag.py` covers the
call assembly, but the wiring BETWEEN them — the seat and head routing at the
`model.rs` boundary, which the `priors.rs` module header names as the surviving
residue — had no test at all. That residue is what this class is for.

Two properties carry the weight, and neither needs new crate telemetry:

* the ACTING seat's `root_priors` must equal the policy head softmaxed and
  gathered through the SELF map, recomputed here in torch off the same
  TorchScript artifact — an exact 6-decimal oracle, not a shape check;
* the OPPONENT seat's ROOT VISIT ORDER must agree with the OPPONENT head
  softmaxed and gathered through the OPPONENT map. There is no report field
  for the opponent's applied priors, so the visits ARE the observable, and
  they are a sharp one: PUCT's exploration term is proportional to the prior,
  so an arm priced from the wrong head, the wrong map or the wrong seat lands
  in the wrong place in the visit order. Measured at HEAD over 32
  (sims, batch, seed) combinations: 0 discordant pairs against the opponent
  head, 17-19 against the acting head.

Each test that leans on that second property first asserts its own
discriminating power, by checking that the wrong-head vector WOULD violate the
ordering it demands. A gate whose oracle happens to agree with its null is not
a gate.

WHAT THIS DOES NOT CLOSE. The opponent head's label space is stipulated here,
not derived: the test supplies `ctx["opponent_request_order"]` itself. Whether
`engine_search.opponent_request_order` names the slots a REAL checkpoint's
opponent head was trained against is the label-space half of the campaign's
gate, it needs that checkpoint, and it is not tested here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import numpy
except ModuleNotFoundError:  # pragma: no cover
    numpy = None

try:
    import pokezero_search
except ModuleNotFoundError:  # pragma: no cover
    pokezero_search = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_SAMPLE_DIR = Path(__file__).parent / "data" / "golden_corpus_sample"
SCRIPTS_DIR = REPO_ROOT / "scripts"
SEARCH_TEST_TRANSITION_TOKEN_BUDGET = 32

from pokezero.golden_corpus import (  # noqa: E402
    GOLDEN_CORPUS_SCHEMA_VERSION,
    load_golden_corpus,
)
from pokezero.golden_corpus_fold import iter_fold_records  # noqa: E402
from pokezero.observation import (  # noqa: E402
    OBSERVATION_SCHEMA_VERSION_V3,
)

_crate_ready = bool(
    pokezero_search is not None
    and getattr(pokezero_search, "MODEL_FEATURE_ENABLED", False)
    and hasattr(getattr(pokezero_search, "LeafEncoder", None), "self_action_map")
)


def _tables_json() -> str | None:
    local = REPO_ROOT / "corpus" / "encoder_tables.json"
    if local.exists():
        payload = json.loads(local.read_text(encoding="utf-8"))
        if payload.get("layout", {}).get("schema_version") == OBSERVATION_SCHEMA_VERSION_V3:
            payload["layout"]["default_feature_masks"][
                "transition_token_budget"
            ] = SEARCH_TEST_TRANSITION_TOKEN_BUDGET
            return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    try:
        from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT

        if not Path(DEFAULT_SHOWDOWN_ROOT).exists():
            return None
        sys.path.insert(0, str(SCRIPTS_DIR))
        from export_encoder_tables import build_tables  # noqa: E402

        payload = build_tables(
            str(DEFAULT_SHOWDOWN_ROOT),
            observation_schema_version=OBSERVATION_SCHEMA_VERSION_V3,
        )
        payload["layout"]["default_feature_masks"][
            "transition_token_budget"
        ] = SEARCH_TEST_TRANSITION_TOKEN_BUDGET
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    except Exception:  # pragma: no cover - environment-dependent
        return None


def _load_export_module():
    spec = importlib.util.spec_from_file_location(
        "export_model", REPO_ROOT / "scripts" / "export_model.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _EncodedSearchFixture:
    """Class fixture shared by the two gates below: one committed-sample world,
    one random-weights TorchScript artifact at the real v3 shape.

    A plain mixin rather than a `TestCase` base, so unittest does not collect it
    as a third (empty) test class. Both concrete classes carry their own skip
    decorators — decorators do not inherit.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tables_json = _tables_json()
        if cls.tables_json is None:
            # Name the fix. `corpus/` is gitignored, so a fresh checkout has no
            # tables artifact and this gate silently did not run; the artifact
            # is DERIVED, not committed, and `_tables_json` already regenerates
            # it from a Showdown checkout in-process. A skip that does not say
            # so reads like a missing fixture.
            raise unittest.SkipTest(
                "no encoder tables artifact and no Showdown checkout. Either set "
                "POKEZERO_SHOWDOWN_ROOT (the tables are then built in-process), or "
                "write corpus/encoder_tables.json with `python "
                "scripts/export_encoder_tables.py --showdown-root <root> "
                "--observation-schema v3 --out corpus/encoder_tables.json`"
            )
        try:
            from pokezero.dex import load_showdown_dex_cached
            from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT

            if not Path(DEFAULT_SHOWDOWN_ROOT).exists():
                raise unittest.SkipTest("no Showdown checkout (dex required)")
            dex = load_showdown_dex_cached(DEFAULT_SHOWDOWN_ROOT)
        except unittest.SkipTest:
            raise
        except Exception as error:  # pragma: no cover
            raise unittest.SkipTest(f"dex unavailable: {error}")
        from pokezero.env import BattleStartOverride
        from pokezero.engine_world import (
            EngineWorldUnsupported,
            battle_spec_from_payload,
        )
        from pokezero.poke_engine_adapter import build_poke_engine_state

        sys.path.insert(0, str(SCRIPTS_DIR))
        from golden_encoder_backends import row_inputs_from_decision_row  # noqa: E402

        corpus = load_golden_corpus(COMMITTED_SAMPLE_DIR)
        fold_states = {}
        for record in iter_fold_records(
            COMMITTED_SAMPLE_DIR, expected_schema_version=GOLDEN_CORPUS_SCHEMA_VERSION
        ):
            fold_states[int(record["array_row_index"])] = record["fold_state"]
        games = {game.record.battle_id: game for game in corpus.games}
        cls.position = None
        for index, row in enumerate(corpus.decision_rows):
            game = games[row.battle_id]
            packed = {
                slot: (game.record.true_teams.get(slot) or {}).get("packed")
                for slot in ("p1", "p2")
            }
            if not packed["p1"] or not packed["p2"]:
                continue
            try:
                world = battle_spec_from_payload(
                    row.public_materialization,
                    BattleStartOverride(player_teams=packed),
                    dex=dex,
                    approximate_sleep_turns=True,
                    approximate_substitute_health=True,
                )
                state = build_poke_engine_state(world.spec)
            except EngineWorldUnsupported:
                continue
            row_inputs = row_inputs_from_decision_row(row)
            # The committed sample predates V3, but this mechanics gate consumes only its
            # schema-independent public/belief inputs. Stamp the schema the live V3 policy
            # supplies so the native encoder exercises the current layout fail-closed.
            row_inputs["observation_schema_version"] = OBSERVATION_SCHEMA_VERSION_V3
            cls.position = {
                "state_str": state.to_string(),
                "row_inputs": json.dumps(row_inputs, sort_keys=True),
                "ctx": json.dumps(
                    {
                        "p1": list(world.party_species["p1"]),
                        "p2": list(world.party_species["p2"]),
                        "turn": int(row.public_materialization.get("turn") or 0),
                    }
                ),
                "fold_state": fold_states[index],
                "self_side": "side_one" if row.player_id == "p1" else "side_two",
            }
            break
        if cls.position is None:
            raise unittest.SkipTest("no committed-sample row could be driven")

        # Random-weights artifact at the REAL v3 shape (throughput/mechanics
        # only — never a strength claim).
        export = _load_export_module()
        from pokezero.neural_policy import (
            EntityTokenTransformerPolicy,
            TransformerPolicyConfig,
        )

        tables = json.loads(cls.tables_json)
        cls.layout = tables["layout"]
        config = TransformerPolicyConfig.compact_category(
            category_vocab=tuple(tables["vocab"]["tokens"]),
            category_oov_buckets=int(tables["vocab"]["oov_buckets"]),
            categorical_feature_count=int(tables["layout"]["categorical_feature_count"]),
            numeric_feature_count=int(tables["layout"]["numeric_feature_count"]),
            token_count=int(tables["layout"]["token_count"]),
            embedding_dim=32,
            transformer_layers=1,
            attention_heads=2,
            feedforward_dim=64,
            dropout=0.0,
            observation_schema_version=OBSERVATION_SCHEMA_VERSION_V3,
            transition_token_budget=SEARCH_TEST_TRANSITION_TOKEN_BUDGET,
        )
        torch.manual_seed(20260719)
        model = EntityTokenTransformerPolicy(config).eval()
        shim = export.build_exportable_module(model)
        cls.tmpdir = tempfile.TemporaryDirectory()
        # Retained: the opponent gate reloads this artifact in torch to
        # recompute both heads independently of the crate.
        artifact = Path(cls.tmpdir.name) / "priors_test_random.pt"
        cls.artifact = artifact
        export.export_torchscript(
            shim, export.make_random_inputs(config, export.TRACE_BATCH, seed=7), artifact
        )
        cls.native = pokezero_search.NativeLeafModel(
            str(artifact),
            device="cpu",
            window=1,
            tokens=int(tables["layout"]["token_count"]),
            categorical_features=int(tables["layout"]["categorical_feature_count"]),
            numeric_features=int(tables["layout"]["numeric_feature_count"]),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmpdir"):
            cls.tmpdir.cleanup()

    def _search(
        self,
        *,
        sims: int,
        batch: int,
        seed: int,
        model_priors: bool,
        early_stop_min_sims: int = 0,
        early_stop_side_one: bool = True,
        ctx_json: str | None = None,
        use_opponent_priors: bool | None = None,
    ) -> dict:
        fold = pokezero_search.FoldState.from_payload(self.position["fold_state"])
        args = [
            self.position["state_str"],
            sims,
            batch,
            self.tables_json,
            self.position["row_inputs"],
            self.position["ctx"] if ctx_json is None else ctx_json,
            fold,
            2,  # max_depth
            1.4,
            seed,
            True,
            model_priors,
            early_stop_min_sims,
            early_stop_side_one,
        ]
        # Appended only when asked for, so every pre-existing call in this file
        # keeps making the historical 14-positional call byte for byte.
        if use_opponent_priors is not None:
            args.append(use_opponent_priors)
        return json.loads(self.native.search_batched_multi_encoded(*args))

    # -- opponent-seat helpers (used only by OpponentPriorsEncodedSearchTest) --

    @property
    def _opponent_side(self) -> str:
        return "side_two" if self.position["self_side"] == "side_one" else "side_one"

    def _ctx_with_opponent_order(self, order: list[str] | None = None) -> str:
        """The position's ctx plus an explicit `opponent_request_order`.

        STIPULATED, not derived. The crate refuses the whole opponent seat
        without this key (#1194) and cannot compute it — it never sees the
        pre-root protocol lines. Production derives it in
        `engine_search.opponent_request_order`; whether that derivation agrees
        with a trained opponent head is the label-space half of the gate and is
        not what this file measures. What IS measured is that the order the
        caller supplies is the label space the opponent's arms are priced
        through, which `test_permuting_the_supplied_order_permutes...` pins by
        permuting it.
        """
        ctx = json.loads(self.position["ctx"])
        slot = "p2" if self.position["self_side"] == "side_one" else "p1"
        ctx["opponent_request_order"] = list(ctx[slot]) if order is None else list(order)
        return json.dumps(ctx)

    def _root_head_priors(self, ctx_json: str) -> dict[str, list]:
        """Both heads at the ROOT, recomputed in torch and gathered per seat.

        Independent of the crate's prior path end to end: the observation comes
        from the native encoder (the input the crate feeds its model), the
        forward is `torch.jit.load` on the same artifact, the softmax is
        torch's, and each seat's map is the crate's own PUBLIC map surface
        (`LeafEncoder.self_action_map` / `.opponent_action_map`) rather than the
        private one the search calls. The crate's masked-softmax parity with
        torch on this artifact is a separate gate
        (tests/test_crate_model_leafeval.py); here the two must agree because
        the search's root forward passes NO legal mask, so both sides are a
        plain softmax over the full action block.
        """
        state = self.position["state_str"]
        encoder = pokezero_search.LeafEncoder(
            self.tables_json, self.position["row_inputs"], ctx_json, state
        )
        fold = pokezero_search.FoldState.from_payload(self.position["fold_state"])
        turn = int(json.loads(self.position["ctx"]).get("turn") or 0)
        encoded = encoder.encode_leaf(state, fold, turn)

        tokens = int(self.layout["token_count"])
        categorical_features = int(self.layout["categorical_feature_count"])
        numeric_features = int(self.layout["numeric_feature_count"])

        def buf(name: str, dtype: str):
            return numpy.frombuffer(encoded[name], dtype=dtype)

        # Little-endian i4 / f8 / i2 / b1, the corpus's canonical dtypes
        # (encoder.rs `encoded_to_dict`). window == 1.
        inputs = (
            torch.from_numpy(buf("categorical_ids", "<i4").astype("int64")).reshape(
                1, 1, tokens, categorical_features
            ),
            torch.from_numpy(buf("numeric_features", "<f8").astype("float32")).reshape(
                1, 1, tokens, numeric_features
            ),
            torch.from_numpy(buf("token_type_ids", "<i2").astype("int64")).reshape(
                1, 1, tokens
            ),
            torch.from_numpy(buf("attention_mask", "?").copy()).reshape(1, 1, tokens),
            torch.ones((1, 1), dtype=torch.bool),
        )
        with torch.no_grad():
            policy_logits, _value, opponent_logits = torch.jit.load(str(self.artifact))(*inputs)

        def gathered(row: list[float], action_map: list) -> list[float]:
            slots = [slot for _display, slot in action_map]
            self.assertNotIn(
                None,
                slots,
                "every option must map for this oracle to be exact; an unmapped "
                "option makes the crate fall the whole node back to uniform",
            )
            picked = [row[slot] for slot in slots]
            total = sum(picked)
            return [value / total for value in picked]

        self_map = encoder.self_action_map(state)
        opponent_map = encoder.opponent_action_map(state)
        policy_row = torch.softmax(policy_logits, dim=-1)[0].tolist()
        opponent_row = torch.softmax(opponent_logits, dim=-1)[0].tolist()
        return {
            "acting": gathered(policy_row, self_map),
            "opponent": gathered(opponent_row, opponent_map),
            # The head-swap null: the ACTING head read through the OPPONENT's
            # map, i.e. exactly what a transposed `impl HeadSource` produces.
            "opponent_from_acting_head": gathered(policy_row, opponent_map),
            "opponent_arms": [display for display, _slot in opponent_map],
        }

    def _discordant_pairs(
        self, side_entries: list[dict], arms: list[str], priors: list[float], margin: float
    ) -> list[tuple]:
        """Ordered arm pairs whose visits contradict a strictly higher prior.

        `margin` keeps near-equal priors out of the comparison: PUCT's Q term
        can legitimately reorder two arms the prior barely separates, and a
        gate that forbids that would be flaky rather than strict.
        """
        visits = {entry["move"]: int(entry["visits"]) for entry in side_entries}
        self.assertEqual(sorted(visits), sorted(arms), "report arms must be the mapped arms")
        return [
            (arms[i], priors[i], visits[arms[i]], arms[j], priors[j], visits[arms[j]])
            for i in range(len(arms))
            for j in range(len(arms))
            if priors[i] > priors[j] + margin and visits[arms[i]] < visits[arms[j]]
        ]


@unittest.skipIf(numpy is None, "requires numpy")
@unittest.skipIf(torch is None, "requires torch")
@unittest.skipUnless(_crate_ready, "pokezero_search lacks the model feature or prior surfaces")
class ModelPriorsEncodedSearchTest(_EncodedSearchFixture, unittest.TestCase):
    """Encoded search with self-side model priors on a committed-sample world."""

    def test_priors_telemetry_and_batch1_determinism(self) -> None:
        first = self._search(sims=48, batch=1, seed=5, model_priors=True)
        second = self._search(sims=48, batch=1, seed=5, model_priors=True)
        self.assertEqual(first["side_one"], second["side_one"])
        self.assertEqual(first["side_two"], second["side_two"])
        self.assertEqual(first["chance_nodes"], second["chance_nodes"])
        self.assertTrue(first["model_priors"])
        root_priors = first["root_priors"]
        self.assertIsInstance(root_priors, list)
        self_arms = first[self.position["self_side"]]
        self.assertEqual(len(root_priors), len(self_arms))
        self.assertAlmostEqual(sum(root_priors), 1.0, places=3)
        self.assertEqual(first["prior_fallbacks"], 0)
        # Visit conservation in both modes.
        for side in ("side_one", "side_two"):
            self.assertEqual(sum(entry["visits"] for entry in first[side]), 48)

    def test_kill_switch_restores_uniform(self) -> None:
        report = self._search(sims=48, batch=1, seed=5, model_priors=False)
        self.assertFalse(report["model_priors"])
        self.assertIsNone(report["root_priors"])
        self.assertEqual(report["prior_branches"], 0)
        for side in ("side_one", "side_two"):
            self.assertEqual(sum(entry["visits"] for entry in report[side]), 48)

    def test_batched_priors_run(self) -> None:
        report = self._search(sims=64, batch=8, seed=11, model_priors=True)
        self.assertTrue(report["model_priors"])
        self.assertGreaterEqual(report["prior_branches"] + report["prior_fallbacks"], 0)
        for side in ("side_one", "side_two"):
            self.assertEqual(sum(entry["visits"] for entry in report[side]), 64)

    def test_safe_early_stop_preserves_target_side_argmax(self) -> None:
        stopped = self._search(
            sims=128,
            batch=8,
            seed=5,
            model_priors=True,
            early_stop_min_sims=16,
            early_stop_side_one=True,
        )
        full = self._search(sims=128, batch=8, seed=5, model_priors=True)

        self.assertTrue(stopped["early_stopped"])
        self.assertLess(stopped["iterations"], stopped["requested_iterations"])
        self.assertEqual(
            stopped["remaining_iterations"],
            stopped["requested_iterations"] - stopped["iterations"],
        )
        self.assertEqual(stopped["side_one"][0]["move"], full["side_one"][0]["move"])
        for side in ("side_one", "side_two"):
            self.assertEqual(
                sum(entry["visits"] for entry in stopped[side]),
                stopped["iterations"],
            )


@unittest.skipIf(numpy is None, "requires numpy")
@unittest.skipIf(torch is None, "requires torch")
@unittest.skipUnless(_crate_ready, "pokezero_search lacks the model feature or prior surfaces")
@unittest.skipUnless(
    _crate_ready and hasattr(pokezero_search.LeafEncoder, "opponent_action_map"),
    "wheel predates LeafEncoder.opponent_action_map",
)
class OpponentPriorsEncodedSearchTest(_EncodedSearchFixture, unittest.TestCase):
    """`use_opponent_priors=True` through a REAL encoded search — see the module
    docstring for what this closes and what it deliberately does not.

    MEASURED against the `model.rs` seat/head-routing boundary that
    `rust/pokezero-search/src/priors.rs`'s header names as its surviving
    residue: eight one-line mutations, one `--features model` wheel rebuilt per
    world, seven KILLED. The eighth (the ROOT's opponent option list replaced by
    the acting seat's) is EQUIVALENT on this fixture rather than missed — see
    that header for the structural reason and for what a fixture that could see
    it would need. Do not weaken the two oracle tests below without re-running
    that battery; five of the seven kills come from them.
    """

    # Priors this close together are not required to order the visits; see
    # `_discordant_pairs`. Both heads here separate their top arms by ~0.05+.
    ORDER_MARGIN = 0.02

    def _flag_on(self, **kwargs) -> dict:
        kwargs.setdefault("sims", 48)
        kwargs.setdefault("batch", 1)
        kwargs.setdefault("seed", 5)
        return self._search(model_priors=True, use_opponent_priors=True, **kwargs)

    def test_the_acting_seat_reads_the_policy_head_through_its_own_map(self) -> None:
        """`root_priors` is the policy head, exactly — under BOTH flag states.

        The value pin is the point. A shape/normalization check passes for a
        transposed `impl HeadSource for LeafBatchOutput`, which is one of the
        head swaps `priors.rs`'s header records as surviving; this does not,
        because the two heads are different tensors of the same width.
        """
        ctx_json = self._ctx_with_opponent_order()
        oracle = self._root_head_priors(ctx_json)
        on = self._flag_on(ctx_json=ctx_json)
        off = self._search(sims=48, batch=1, seed=5, model_priors=True, ctx_json=ctx_json)

        self.assertNotAlmostEqual(
            max(abs(a - b) for a, b in zip(oracle["acting"], oracle["opponent"])),
            0.0,
            places=3,
            msg="the two heads must differ or this pin cannot see a head swap",
        )
        for label, report in (("flag on", on), ("flag off", off)):
            self.assertEqual(
                [round(value, 6) for value in report["root_priors"]],
                [round(value, 6) for value in oracle["acting"]],
                f"{label}: root_priors must be the policy head gathered through the self map",
            )

    def test_the_opponent_head_orders_the_opponent_seats_own_visits(self) -> None:
        """The APPLIED end-to-end claim: the opponent head reaches the opponent's
        arms in the searched tree.

        The report has no opponent-prior field, so the visits are the observable.
        The null this must reject is the acting head read through the opponent's
        map — a transposed `HeadSource`, or an acting-head vector handed to the
        opponent resolve — and the first assertion below establishes that the
        null WOULD be rejected before the second one credits the truth.
        """
        ctx_json = self._ctx_with_opponent_order()
        oracle = self._root_head_priors(ctx_json)
        report = self._flag_on(ctx_json=ctx_json)
        arms = oracle["opponent_arms"]
        side = report[self._opponent_side]

        null = self._discordant_pairs(
            side, arms, oracle["opponent_from_acting_head"], self.ORDER_MARGIN
        )
        self.assertGreater(
            len(null),
            0,
            "this gate cannot discriminate on this position: the wrong head "
            "orders the opponent's visits just as well as the right one",
        )
        discordant = self._discordant_pairs(
            side, arms, oracle["opponent"], self.ORDER_MARGIN
        )
        self.assertEqual(
            discordant,
            [],
            "opponent arms are visited out of the opponent head's own prior order",
        )
        self.assertEqual(report["prior_fallbacks"], 0)
        self.assertGreater(report["prior_branches"], 0)

    def test_permuting_the_supplied_order_permutes_only_the_opponent_seat(self) -> None:
        """The opponent's label space is load-bearing AND confined to its seat.

        Swapping two entries of `ctx["opponent_request_order"]` swaps which
        action slot two of the opponent's SWITCH arms are priced from. The
        opponent's visit order must follow the permuted gather; the acting
        seat's `root_priors` must not move at all. A run that priced the
        opponent through the self map (or through the self seat's options)
        would be blind to this permutation.
        """
        ctx_json = self._ctx_with_opponent_order()
        order = json.loads(ctx_json)["opponent_request_order"]
        self.assertGreaterEqual(len(order), 3, "need a party to permute")
        permuted_order = list(order)
        permuted_order[1], permuted_order[2] = permuted_order[2], permuted_order[1]
        permuted_ctx = self._ctx_with_opponent_order(permuted_order)

        base_oracle = self._root_head_priors(ctx_json)
        permuted_oracle = self._root_head_priors(permuted_ctx)
        self.assertNotEqual(
            base_oracle["opponent"],
            permuted_oracle["opponent"],
            "the permutation must actually move the opponent's gathered priors",
        )
        self.assertEqual(
            base_oracle["acting"],
            permuted_oracle["acting"],
            "the opponent order channel must not touch the self seat's gather",
        )

        base = self._flag_on(ctx_json=ctx_json)
        permuted = self._flag_on(ctx_json=permuted_ctx)
        self.assertEqual(
            base["root_priors"],
            permuted["root_priors"],
            "the opponent order channel must not move the acting seat's priors",
        )
        self.assertEqual(
            self._discordant_pairs(
                permuted[self._opponent_side],
                permuted_oracle["opponent_arms"],
                permuted_oracle["opponent"],
                self.ORDER_MARGIN,
            ),
            [],
            "the permuted search must follow the PERMUTED opponent gather",
        )
        self.assertNotEqual(
            [(entry["move"], entry["visits"]) for entry in base[self._opponent_side]],
            [(entry["move"], entry["visits"]) for entry in permuted[self._opponent_side]],
            "the permutation must be visible in the opponent's visits",
        )

    def test_a_withheld_order_refuses_the_opponent_seat_and_changes_nothing(self) -> None:
        """#1194 fail-closed, end to end rather than at the map surface.

        With `ctx["opponent_request_order"]` absent the opponent map is
        all-`None`, so every opponent gather refuses. The flag-on search must
        then be IDENTICAL to the flag-off search — same visits on both seats,
        same acting priors, same applied-branch count — with the refusals
        visible in `prior_fallbacks` and nowhere else. A fail-OPEN
        reintroduction (the one-swap approximation `root_opponent_order` used
        to substitute) would show up here as a search that quietly diverged.
        """
        withheld = json.loads(self.position["ctx"])
        self.assertNotIn("opponent_request_order", withheld)

        on = self._flag_on()
        off = self._search(sims=48, batch=1, seed=5, model_priors=True)

        self.assertGreater(on["prior_fallbacks"], 0, "the refusals must be counted")
        self.assertEqual(off["prior_fallbacks"], 0)
        self.assertEqual(on["prior_branches"], off["prior_branches"])
        self.assertEqual(on["root_priors"], off["root_priors"])
        self.assertEqual(on["side_one"], off["side_one"])
        self.assertEqual(on["side_two"], off["side_two"])

    def test_the_order_channel_is_inert_while_the_flag_is_off(self) -> None:
        """Flag-off equivalence is the campaign's anchor: cell A's numbers were
        produced under a uniform opponent, and supplying the order must not
        move them. Byte-for-byte on both seats.
        """
        with_order = self._search(
            sims=48, batch=1, seed=5, model_priors=True, ctx_json=self._ctx_with_opponent_order()
        )
        without = self._search(sims=48, batch=1, seed=5, model_priors=True)
        for key in ("side_one", "side_two", "root_priors", "prior_branches", "prior_fallbacks"):
            self.assertEqual(with_order[key], without[key], key)

    def test_the_flag_changes_the_opponent_seat_and_not_the_acting_priors(self) -> None:
        """The kill switch, from the other direction: turning the flag ON with a
        usable order must move the opponent's visits (otherwise cells B and E
        would read 'opponent priors do not help' off a feature that never ran)
        while leaving the acting seat's priors untouched.
        """
        ctx_json = self._ctx_with_opponent_order()
        on = self._flag_on(ctx_json=ctx_json)
        off = self._search(sims=48, batch=1, seed=5, model_priors=True, ctx_json=ctx_json)

        self.assertEqual(on["root_priors"], off["root_priors"])
        self.assertNotEqual(
            [(entry["move"], entry["visits"]) for entry in on[self._opponent_side]],
            [(entry["move"], entry["visits"]) for entry in off[self._opponent_side]],
        )
        for side in ("side_one", "side_two"):
            self.assertEqual(sum(entry["visits"] for entry in on[side]), 48)


if __name__ == "__main__":
    unittest.main()
