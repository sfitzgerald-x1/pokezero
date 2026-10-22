"""Public-only trap certificates; synthetic native mapping, not game fidelity."""
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.engine_search import EngineMctsPolicy, EngineMctsConfig
from pokezero.policy_opponent_view import public_policy_lines

try:
    import pokezero_search
except ImportError:
    pokezero_search = None

ROOT=Path(__file__).resolve().parents[1]


class PublicTrapTests(unittest.TestCase):
    def test_highlevel_transports_only_the_filtered_public_prefix_and_off_is_inert(self):
        policy=object.__new__(EngineMctsPolicy)
        policy._config=EngineMctsConfig(leaf_eval="model",model_path="fixture.pt",checkpoint_path="fixture.pt",
            tables_path="fixture.json",strict_fallbacks=True,policy_opponent=True,policy_opponent_seed=13)
        tokens=("species:snorlax",)
        policy._tables_json=json.dumps(dict(vocab=dict(tokens=tokens,index={tokens[0]:1},oov_buckets=16)))
        policy._policy_opponent_result=SimpleNamespace(model_config=SimpleNamespace(category_vocab=tokens,category_oov_buckets=16))
        policy._policy_opponent_model=object();policy._set_source=None;policy._dex=None
        policy._policy_opponent_inference_lock=None
        lines=("|start","|split|p1","|switch|p1a: Snorlax|Snorlax, L80|23/101",
            "|switch|p1a: Snorlax|Snorlax, L80|23/100")
        replay=SimpleNamespace(public_lines=lines,hp_visibility={"p1":"exact","p2":"exact"})
        context=SimpleNamespace(public_materialization_state=SimpleNamespace(replay=replay),
            battle_id="authored",seed=7,player_id="p1",decision_round_index=1,format_id="gen3randombattle")
        with patch("pokezero.policy_opponent.make_policy_opponent_callback",return_value=lambda raw:()) as callback:
            kwargs=policy._policy_opponent_native_kwargs(context,{"seed":9,"_policy_opponent_request_order":["snorlax"]})
        callback.assert_called_once()
        self.assertEqual(kwargs["policy_opponent_public_lines"],list(public_policy_lines(lines,hp_visibility=replay.hp_visibility)))
        self.assertFalse(any("split" in line or "101" in line for line in kwargs["policy_opponent_public_lines"]))
        policy._config=SimpleNamespace(policy_opponent=False)
        self.assertEqual(policy._policy_opponent_native_kwargs(None,{}),{})

    @unittest.skipUnless(pokezero_search is not None and os.environ.get("POKEZERO_SHOWDOWN_ROOT"),
                         "requires fresh native constructor and pinned Gen3 source")
    def test_real_normal_source_request_proof_matches_native_and_keeps_unknowns_refused(self):
        result=json.loads(subprocess.run([os.environ.get("POKEZERO_NODE_BINARY","node"),
            str(ROOT/"tests/policy_opponent_public_trap_oracle.cjs"),os.environ["POKEZERO_SHOWDOWN_ROOT"]],
            capture_output=True,text=True,check=True,timeout=30).stdout)
        self.assertFalse(result["scientific_measurement"])
        self.assertEqual([r["abilities"] for r in result["default_abilities"]],[{"0":"Shadow Tag"}]*2)
        for row in result["rows"]:
            self.assertTrue(row["trapped"])
            lines=list(public_policy_lines(row["lines"],hp_visibility={"p1":"exact","p2":"exact"}))
            self.assertFalse(any(line.startswith("|-ability|") for line in lines))
            own=row["acting_species"]
            names=[own]+[s for s in ("Snorlax","Gengar","Blissey") if s!=own][:2]
            base=(ROOT/"rust/pokezero-search/src/test_fixtures/minimal.state").read_text().strip().split("/")
            party=base[0].split("=")
            for i,name in enumerate(names):
                mon=party[0].split(",");mon[0]=name.upper();party[i]=",".join(mon)
            base[0]="=".join(party)
            mon=base[1].split(",");mon[0]="WOBBUFFET";mon[8]=mon[9]="SHADOWTAG";base[1]=",".join(mon)
            if row["slot"]=="p2":base[0],base[1]=base[1],base[0]
            for root in (True,False):
                def run(public=lines,format_id=row["format_id"]):
                    return json.loads(pokezero_search.sampled_policy_request("/".join(base),row["slot"],names,names,
                        {"ember":40,"tackle":56},root=root,public_lines=public,format_id=format_id))
                accepted=run()
                self.assertEqual(accepted["native_action_indices"],[0,1])
                self.assertTrue(accepted["request"]["active"][0]["trapped"])
                for public,format_id in (([],row["format_id"]),(lines,"gen3customgame"),
                    (lines+[f"|-endability|{row['holder']}a: Wobbuffet"],row["format_id"])):
                    with self.assertRaisesRegex(ValueError,"legal surfaces differ"):run(public,format_id)
                self.assertEqual(run(),accepted) # no sibling knowledge contamination


if __name__=="__main__":unittest.main()
