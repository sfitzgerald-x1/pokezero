"""Bounded request-observer checks, not playing-strength measurements."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import unittest

from pokezero.policy_action_contract import PolicyActionContract, SCHEMA, public_binding, request_digest
from pokezero.policy_opponent_view import public_policy_lines
from pokezero.showdown import _legal_action_mask


ROOT = Path(__file__).resolve().parents[1]


def native_case(row):
    """Authored native flag fixture, not a full source-state parity claim."""
    sides = (ROOT / "rust/pokezero-search/src/test_fixtures/minimal.state").read_text().strip().split("/")
    types = {"Pikachu": ("ELECTRIC", "TYPELESS"), "Charizard": ("FIRE", "FLYING"),
             "Gengar": ("GHOST", "POISON"), "Metagross": ("STEEL", "PSYCHIC"),
             "Dugtrio": ("GROUND", "TYPELESS"), "Magneton": ("ELECTRIC", "STEEL"),
             "Gardevoir": ("PSYCHIC", "TYPELESS"), "Porygon2": ("NORMAL", "TYPELESS"),
             "Snorlax": ("NORMAL", "TYPELESS")}
    own_index = 0 if row["slot"] == "p1" else 1
    for index in (0, 1):
        party = sides[index].split("=")
        species, ability = ((row["own"], row["ownAbility"]) if index == own_index
                            else (row["foe"], row["ability"]))
        fields = party[0].split(",")
        fields[0] = species.upper()
        fields[2:4] = types[species]
        fields[8] = fields[9] = "".join(c for c in ability.upper() if c.isalnum())
        if index != own_index:
            fields[9] = "".join(c for c in row.get("baseAbility", ability).upper() if c.isalnum())
        fields[22:26] = ["TACKLE;false;56", "NONE;true;0", "NONE;true;0", "NONE;true;0"]
        party[0] = ",".join(fields)
        fields[0] = "SNORLAX"; fields[2:4] = types["Snorlax"]
        fields[8] = fields[9] = "IMMUNITY"
        party[1] = ",".join(fields)
        sides[index] = "=".join(party)
    return "/".join(sides)


def case():
    request = {"side": {"id": "p1", "pokemon": [
        {"ident": "p1: Pikachu", "active": True, "condition": "100/100"},
        {"ident": "p1: Snorlax", "active": False, "condition": "100/100"},
        {"ident": "p1: Charizard", "active": False, "condition": "100/100"}]},
        "active": [{"moves": [{"id": "tackle", "disabled": False},
                               {"id": "growl", "disabled": False}], "maybeTrapped": True}]}
    lines = ["|gametype|singles", "|gen|3", "|tier|[Gen 3] Random Battle", "|start",
             "|switch|p1a: Pikachu|Pikachu, L80|100/100",
             "|switch|p2a: Dugtrio|Dugtrio, L80|100/100", "|turn|1"]
    receipt = {"schema": SCHEMA, "format_id": "gen3randombattle", "seat": "p1",
               "public_binding": public_binding(lines), "request_sha256": request_digest(request),
               "policy_action_indices": [0, 1],
               "protocol_action_indices": [0, 1, 4, 5]}
    return dict(request=request, receipt=receipt, public_lines=lines, slot="p1",
                format_id="gen3randombattle", canonical_mask=_legal_action_mask(request))


class PrivateRequestObserverTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("POKEZERO_SHOWDOWN_ROOT"), "requires pinned Showdown and rebuilt native observer")
    def test_public_trace_trapping_abilities_match_source_and_missing_reveals_refuse(self):
        import pokezero_search
        rows = json.loads(subprocess.run([os.environ.get("POKEZERO_NODE_BINARY", "node"),
            str(ROOT / "tests/policy_request_observer_oracle.cjs"),
            os.environ["POKEZERO_SHOWDOWN_ROOT"], "--traced"],
            capture_output=True, text=True, check=True, timeout=30).stdout)["rows"]
        self.assertEqual(len(rows), 8)
        for row in rows:
            lines = list(public_policy_lines(row["lines"], hp_visibility={"p1": "exact", "p2": "exact"}))
            other = "p2" if row["slot"] == "p1" else "p1"
            reveal = (f'|-ability|{other}a: {row["foe"]}|{row["ability"]}|Trace|'
                      f'[from] ability: Trace|[of] {row["slot"]}a: {row["own"]}')
            self.assertIn(reveal, lines)
            self.assertTrue(row["request"]["active"][0]["maybeTrapped"])
            state = native_case(row)
            for root in (True, False):
                with self.subTest(tracer=row["foe"], ability=row["ability"], seat=row["slot"], root=root):
                    supplied = json.loads(pokezero_search.sampled_policy_request(state, row["slot"],
                        [row["own"], "Snorlax"], [row["own"], "Snorlax"], {"tackle": 56},
                        root=root, public_lines=lines, format_id="gen3randombattle", request_observer=True))
                    self.assertEqual(_legal_action_mask(supplied["request"]), _legal_action_mask(row["request"]))
                    self.assertEqual(supplied["request_observer"]["public_binding"], public_binding(lines))
                    omitted = [line for line in lines if line != reveal]
                    with self.assertRaisesRegex(ValueError, "changed trapping ability"):
                        pokezero_search.sampled_policy_request(state, row["slot"],
                            [row["own"], "Snorlax"], [row["own"], "Snorlax"], {"tackle": 56},
                            root=root, public_lines=omitted, format_id="gen3randombattle", request_observer=True)

    @unittest.skipUnless(os.environ.get("POKEZERO_SHOWDOWN_ROOT"), "requires pinned Showdown and rebuilt native observer")
    def test_native_projection_matches_source_flags_and_zero_switch_law(self):
        import pokezero_search
        from pokezero.policy_opponent_view import build_policy_opponent_view_from_native_bundle
        from test_policy_opponent_request import arguments
        rows = json.loads(subprocess.run([os.environ.get("POKEZERO_NODE_BINARY", "node"),
            str(ROOT / "tests/policy_request_observer_oracle.cjs"), os.environ["POKEZERO_SHOWDOWN_ROOT"]],
            capture_output=True, text=True, check=True, timeout=30).stdout)["rows"]
        for row in rows:
            lines = public_policy_lines(row["lines"], hp_visibility={"p1": "exact", "p2": "exact"})
            state = native_case(row)
            for root in (True, False):
                with self.subTest(foe=row["foe"], ability=row["ability"], own=row["own"], seat=row["slot"], root=root):
                    kwargs = dict(root=root, public_lines=lines, format_id="gen3randombattle", request_observer=True)
                    supplied = json.loads(pokezero_search.sampled_policy_request(state, row["slot"],
                        [row["own"], "Snorlax"], [row["own"], "Snorlax"], {"tackle": 56}, **kwargs))
                    actual = row["request"]["active"][0]
                    self.assertEqual(supplied["request"]["active"][0].get("maybeTrapped", False), actual.get("maybeTrapped", False))
                    self.assertEqual(supplied["request"]["active"][0].get("trapped", False), actual.get("trapped", False))
                    self.assertEqual(_legal_action_mask(supplied["request"]), _legal_action_mask(row["request"]))
                    view = build_policy_opponent_view_from_native_bundle(native_request_bundle=supplied, **arguments(lines, row["slot"]))
                    if actual.get("maybeTrapped"):
                        self.assertIsNotNone(view.action_contract)
                        self.assertEqual(view.action_contract.gather([1., 0., 0., 0., 0., 0., 0., 0., 0.]),
                                         tuple(1. if i == 0 else 0. for i in supplied["native_action_indices"]))
                        self.assertEqual(supplied["request_observer"]["public_binding"], public_binding(lines))
                        self.assertEqual(supplied["request_observer"]["request_sha256"], request_digest(supplied["request"]))
                    else:
                        self.assertNotIn("request_observer", supplied)
                    # Unobserved opposing item/stats/move details cannot alter
                    # the own request or the narrow server-private receipt.
                    hidden = state.split("/"); index = 1 if row["slot"] == "p1" else 0
                    fields = hidden[index].split(","); fields[10] = "LEFTOVERS"; fields[13] = "222"
                    fields[22] = "THUNDERBOLT;false;24"; hidden[index] = ",".join(fields)
                    changed = json.loads(pokezero_search.sampled_policy_request("/".join(hidden), row["slot"],
                        [row["own"], "Snorlax"], [row["own"], "Snorlax"], {"tackle": 56}, **kwargs))
                    self.assertEqual(supplied, changed)

    def test_certified_view_keeps_canonical_observation_and_conserves_full_row(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from pokezero.policy_opponent import policy_opponent_distribution
        from pokezero.policy_opponent_view import build_policy_opponent_view_from_native_bundle, PolicyOpponentViewError
        from test_policy_opponent import config
        from test_policy_opponent_request import arguments, bundle
        from test_policy_opponent_view import VOCAB
        from test_engine_world import _dex

        native = bundle()
        request = native["request"]
        request["active"][0]["trapped"] = False
        request["active"][0]["maybeTrapped"] = True
        args = arguments()
        lines = public_policy_lines(args["public_lines"], hp_visibility=args["hp_visibility"])
        native["request_observer"] = {"schema": SCHEMA, "format_id": args["format_id"],
            "seat": args["opponent_slot"], "public_binding": public_binding(lines),
            "request_sha256": request_digest(request), "protocol_action_indices": [0, 1, 4],
            "policy_action_indices": [0, 1]}
        cfg = config()
        for indices in ([0, 1, 4], [1, 0]):
            native["native_action_indices"] = indices
            view = build_policy_opponent_view_from_native_bundle(native_request_bundle=native, **args)
            observation = view.observation(category_vocab=VOCAB, dex=_dex())
            self.assertEqual(tuple(i for i, yes in enumerate(observation.legal_action_mask) if yes), (0, 1))
            for payload in (view.row_inputs(dex=_dex()), observation.metadata):
                self.assertNotIn("request_observer", json.dumps(payload))
                self.assertNotIn("protocol_action_indices", json.dumps(payload))
            with patch("pokezero.neural_policy.evaluate_transformer_action_priors",
                       return_value=(.25, .75, 0., 0., 0., 0., 0., 0., 0.)):
                actual = policy_opponent_distribution(view, native_action_indices=indices,
                    model=SimpleNamespace(config=cfg), result=SimpleNamespace(model_config=cfg),
                    category_vocab=VOCAB, dex=_dex())
            self.assertEqual(actual, tuple(.25 if i == 0 else .75 if i == 1 else 0. for i in indices))
            with patch("pokezero.neural_policy.evaluate_transformer_action_priors",
                       return_value=(.25, .65, 0., 0., .1, 0., 0., 0., 0.)):
                with self.assertRaises(PolicyOpponentViewError):
                    policy_opponent_distribution(view, native_action_indices=indices,
                        model=SimpleNamespace(config=cfg), result=SimpleNamespace(model_config=cfg),
                        category_vocab=VOCAB, dex=_dex())
            if 4 in indices:
                with self.assertRaises(PolicyOpponentViewError):
                    build_policy_opponent_view_from_native_bundle(
                        native_request_bundle={k: v for k, v in native.items() if k != "request_observer"}, **args)

    def test_exact_native_surfaces_and_zero_mass_conservation(self):
        for native in ([0, 1], [5, 1, 4, 0]):
            contract = PolicyActionContract.certify(**case(), native_indices=native)
            row = [.25, .75, 0, 0, 0, 0, 0, 0, 0]
            self.assertEqual(contract.gather(row), tuple(row[index] for index in native))
            for bad in ([.25, .65, 0, 0, .1, 0, 0, 0, 0], [float("nan")] * 9, [0] * 9):
                with self.assertRaises(ValueError):
                    contract.gather(bad)
        for native in ([0], [0, 1, 4], [0, 1, 2], [0, 1, 4, 5, 6], [0, 1, 1], [False, 1]):
            with self.assertRaises(ValueError):
                PolicyActionContract.certify(**case(), native_indices=native)

    def test_every_observer_exception_is_bound_and_unsupported_choices_refuse(self):
        mutations = (
            lambda c: c.update(format_id="gen3customgame"),
            lambda c: c.update(slot="p2"),
            lambda c: c["public_lines"].append("|turn|2"),
            lambda c: c["public_lines"].append("|switch|p2a: Dugtrio|Dugtrio, L80|100/100"),
            lambda c: c["request"].update(forceSwitch=[True]),
            lambda c: c["request"].update(wait=True),
            lambda c: c["request"]["active"][0].update(trapped=True),
            lambda c: c["request"]["active"][0].pop("maybeTrapped"),
            lambda c: c["request"]["active"][0].update(maybeLocked=True),
            lambda c: c["request"]["side"]["pokemon"][1].update(condition="0 fnt"),
            lambda c: c["request"]["side"]["pokemon"][1].update(active=True),
            lambda c: c["request"]["side"]["pokemon"][0].update(ident="p1: Raichu"),
            lambda c: c["receipt"].update(policy_action_indices=[0]),
            lambda c: c["receipt"].update(extra_private_truth="forbidden"),
            lambda c: c.update(canonical_mask=(True, True, False, False, True, True, False, False, False)),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                data = case(); mutate(data)
                with self.assertRaises(ValueError):
                    PolicyActionContract.certify(**data, native_indices=[0, 1])

        stale = case()
        stale["public_lines"].append("|-ability|p2a: Dugtrio|Arena Trap")
        stale["receipt"]["public_binding"] = public_binding(stale["public_lines"])
        stale["public_lines"][-1] = "|-ability|p2a: Dugtrio|Sand Veil"
        with self.assertRaisesRegex(ValueError, "public-entry binding"):
            PolicyActionContract.certify(**stale, native_indices=[0, 1])

    @unittest.skipUnless(os.environ.get("POKEZERO_SHOWDOWN_ROOT"), "requires pinned Showdown")
    def test_actual_source_projection_and_canonical_masks_in_both_seats(self):
        output = subprocess.run([os.environ.get("POKEZERO_NODE_BINARY", "node"),
            str(ROOT / "tests/policy_request_observer_oracle.cjs"), os.environ["POKEZERO_SHOWDOWN_ROOT"]],
            capture_output=True, text=True, check=True, timeout=30)
        result = json.loads(output.stdout)
        self.assertFalse(result["scientific_measurement"])
        self.assertEqual(len(result["rows"]), 20)
        paired = {}
        for row in result["rows"]:
            with self.subTest(foe=row["foe"], ability=row["ability"], own=row["own"], seat=row["slot"]):
                request = row["request"]
                mask = _legal_action_mask(request)
                maybe = request["active"][0].get("maybeTrapped", False)
                expected = ((row["foe"] == "Dugtrio" and row["own"] == "Pikachu")
                    or (row["ability"] == "Magnet Pull" and row["own"] == "Metagross"))
                self.assertEqual(maybe, expected)
                self.assertEqual([index for index, on in enumerate(mask) if on], [0] if maybe else [0, 4])
                lines = public_policy_lines(row["lines"], hp_visibility={"p1": "exact", "p2": "exact"})
                key = row["foe"], row["own"], row["slot"]
                if key in paired:
                    self.assertEqual(lines, paired[key], "alternate hidden ability changed public prefix")
                else:
                    paired[key] = lines
                if maybe:
                    receipt = {"schema": SCHEMA, "format_id": "gen3randombattle", "seat": row["slot"],
                        "public_binding": public_binding(lines), "request_sha256": request_digest(request),
                        "policy_action_indices": [0],
                        "protocol_action_indices": [0, 4]}
                    native = [0] if row["actual_trapped"] else [0, 4]
                    contract = PolicyActionContract.certify(request=request, receipt=receipt,
                        native_indices=native, canonical_mask=mask, public_lines=lines,
                        slot=row["slot"], format_id="gen3randombattle")
                    self.assertEqual(contract.gather([1., 0, 0, 0, 0, 0, 0, 0, 0]), tuple(1. if i == 0 else 0. for i in native))


if __name__ == "__main__":
    unittest.main()
