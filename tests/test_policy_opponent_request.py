"""Native side-only request → canonical opponent view integration gate."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
import os

from pokezero.policy_opponent_view import (
    PolicyOpponentViewError, build_policy_opponent_view_from_native_bundle,
)
from test_policy_opponent_view import LINES, request, VOCAB
from test_engine_world import _dex
from test_showdown import FakeSetSource
from pokezero.observation import ObservationFeatureMasks
from pokezero.showdown import V4_REPLAY_OBSERVATION_SPEC

try:
    import pokezero_search
except ImportError:
    pokezero_search = None


def arguments(lines=LINES, slot="p2"):
    return dict(public_lines=lines, hp_visibility={"p1": "exact", "p2": "exact"},
        opponent_slot=slot, battle_id="test", battle_seed=10, format_id="gen3randombattle",
        set_source=FakeSetSource(), spec=V4_REPLAY_OBSERVATION_SPEC,
        feature_masks=ObservationFeatureMasks(transition_token_budget=0))


def bundle():
    own = request()
    return {"request": own, "native_action_indices": [0, 1, 4], "self_move_states": {
        "snorlax": deepcopy(own["active"][0]["moves"]),
        "starmie": [{"id": "surf", "move": "Surf", "pp": 7, "maxpp": 24, "disabled": False}],
    }}


class PolicyOpponentRequestTest(unittest.TestCase):
    def test_complete_bench_pp_survives_canonical_materialization_and_cloning(self):
        supplied = bundle()
        view = build_policy_opponent_view_from_native_bundle(native_request_bundle=supplied, **arguments())
        self.assertEqual(view.native_action_indices, (0, 1, 4))
        before = view.row_inputs(dex=_dex())
        rows = before["public_materialization"]["sides"]["p2"]["pokemon"]
        self.assertEqual(rows[1]["moves"][0]["pp"], 7)
        supplied["self_move_states"]["starmie"][0]["pp"] = 99
        self.assertEqual(before, view.row_inputs(dex=_dex()))
        view.observation(category_vocab=VOCAB, dex=_dex()).validate(V4_REPLAY_OBSERVATION_SPEC)

    def test_missing_invalid_or_wrong_party_pp_refuses(self):
        mutations = (
            lambda b: b.pop("self_move_states"),
            lambda b: b.update(self_move_states=None),
            lambda b: b["self_move_states"].pop("starmie"),
            lambda b: b["self_move_states"].update(secret=[]),
            lambda b: b["self_move_states"]["starmie"][0].update(pp=-1),
            lambda b: b["self_move_states"]["starmie"][0].update(pp=True),
            lambda b: b["self_move_states"]["starmie"][0].update(pp=25),
            lambda b: b["self_move_states"]["starmie"][0].update(maxpp=0),
            lambda b: b["self_move_states"]["starmie"][0].update(id="earthquake"),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                supplied = bundle()
                mutate(supplied)
                with self.assertRaises(PolicyOpponentViewError):
                    build_policy_opponent_view_from_native_bundle(native_request_bundle=supplied, **arguments())

    def test_native_action_map_must_match_canonical_mask_exactly(self):
        for indices in ([], [0, 0, 4], [0, 1], [0, 1, 5], [True, 1, 4], [0, None, 4], [9]):
            with self.subTest(indices=indices), self.assertRaises(PolicyOpponentViewError):
                build_policy_opponent_view_from_native_bundle(
                    native_request_bundle={**bundle(), "native_action_indices": indices}, **arguments())

    @unittest.skipUnless(pokezero_search is not None and hasattr(pokezero_search, "sampled_policy_request"),
                         "requires freshly built native private-request constructor")
    def test_real_native_both_seats_and_private_truth_noninterference(self):
        state = (Path(__file__).resolve().parents[1] / "rust/pokezero-search/src/test_fixtures/minimal.state").read_text().strip()
        lines = ("|player|p1|One", "|player|p2|Two", "|start",
            "|switch|p1a: Charmander|Charmander, L100|100/100",
            "|switch|p2a: Squirtle|Squirtle, L100|100/100", "|turn|1")
        maximum = {"ember": 40, "tackle": 56, "watergun": 40}
        for slot, species in (("p1", "Charmander"), ("p2", "Squirtle")):
            with self.subTest(slot=slot):
                native = json.loads(pokezero_search.sampled_policy_request(state, slot, [species], [species], maximum))
                view = build_policy_opponent_view_from_native_bundle(native_request_bundle=native, **arguments(lines, slot))
                self.assertEqual(view.native_action_indices, (0, 1))
                self.assertEqual(view.materialization.self_move_states[species.casefold()][0]["pp"], 32)
                # Change the hidden other seat's Tackle to Thunderbolt without
                # altering the public transcript or the sampled acting side.
                one, two = state.split("/", 1)
                if slot == "p1":
                    two = two.replace("TACKLE;false;32", "THUNDERBOLT;false;32", 1)
                else:
                    one = one.replace("TACKLE;false;32", "THUNDERBOLT;false;32", 1)
                changed = json.loads(pokezero_search.sampled_policy_request(f"{one}/{two}", slot, [species], [species], maximum))
                self.assertEqual(native, changed)

    @unittest.skipUnless(pokezero_search is not None and os.environ.get("POKEZERO_SHOWDOWN_ROOT"),
                         "requires native constructor and pinned Showdown checkout")
    def test_gen3_transform_request_matches_live_private_pp_and_restores_on_switch(self):
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.engine_world import world_battle_spec
        from pokezero.env import BattleStartOverride
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
        from pokezero.poke_engine_adapter import build_poke_engine_state
        from pokezero.showdown_fixture import FixturePokemon, pack_team
        import poke_engine

        root = Path(os.environ["POKEZERO_SHOWDOWN_ROOT"])
        dex = load_showdown_dex_cached(root)
        maximum = {name: info.max_pp for name, info in dex.moves.items()}
        base = {name: info.pp for name, info in dex.moves.items()}
        ditto = FixturePokemon(species="Ditto", moves=("Transform",), ability="Limber", level=100)
        gengar = FixturePokemon(species="Gengar", moves=("Substitute", "Thunderbolt", "Explosion", "Ice Punch"),
                                ability="Levitate", level=74)
        bench = FixturePokemon(species="Swampert", moves=("Surf",), ability="Torrent", level=84)

        def choose(env, slot, kind, wanted):
            candidates = env.observe(slot).metadata["action_candidates"]
            return next(c["action_index"] for c in candidates if c.get("legal") and c["kind"] == kind
                        and wanted in str(c).lower())

        for slot in ("p1", "p2"):
            foe = "p2" if slot == "p1" else "p1"
            override = BattleStartOverride(player_teams={
                slot: pack_team((ditto, bench)), foe: pack_team((gengar, bench))})
            env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=root))
            try:
                env.reset_with_start_override(seed=99001, start_override=override)
                env.step({slot: choose(env, slot, "move", "transform"),
                          foe: choose(env, foe, "move", "thunderbolt")})
                materialization = env.public_materialization_state(slot)
                world = world_battle_spec(materialization, override, dex=dex, transformed_slots={slot: "Gengar"})
                native_state = build_poke_engine_state(world.spec, module=poke_engine)
                native = json.loads(pokezero_search.sampled_policy_request(native_state.to_string(), slot,
                    [dex.species_info(name).name for name in world.party_species[slot]],
                    ["Ditto", "Swampert"], maximum, base_pp=base))
                actual = materialization.self_request
                self.assertEqual(native["request"]["side"]["pokemon"][0]["ident"], f"{slot}: Ditto")
                # Gen 3 does NOT set maxpp=5. Slot 0 retains Ditto's PP Ups;
                # formerly empty slots use the copied move's unboosted base PP.
                for supplied, expected in zip(native["request"]["active"][0]["moves"], actual["active"][0]["moves"]):
                    self.assertEqual((supplied["id"], supplied["pp"], supplied["maxpp"]),
                                     (expected["id"], expected["pp"], expected["maxpp"]))
                copied_pp = [m["maxpp"] for m in native["request"]["active"][0]["moves"]]
                self.assertEqual(copied_pp, [16, 15, 5, 15])
                view = build_policy_opponent_view_from_native_bundle(native_request_bundle=native,
                    **arguments(tuple(event.raw_line for event in materialization.replay.public_events), slot))
                self.assertEqual(view.materialization.self_move_states["ditto"][0]["pp"], 5)
                self.assertEqual(view.materialization.self_move_states["ditto"][1]["maxpp"], 15)
                env.step({slot: choose(env, slot, "switch", "swampert"),
                          foe: choose(env, foe, "move", "thunderbolt")})
                materialization = env.public_materialization_state(slot)
                world = world_battle_spec(materialization, override, dex=dex)
                native_state = build_poke_engine_state(world.spec, module=poke_engine)
                native = json.loads(pokezero_search.sampled_policy_request(native_state.to_string(), slot,
                    [dex.species_info(name).name for name in world.party_species[slot]],
                    ["Swampert", "Ditto"], maximum, base_pp=base))
                self.assertEqual(native["self_move_states"]["ditto"][0]["id"], "transform")
                self.assertEqual(native["self_move_states"]["ditto"][0]["maxpp"], 16)
                # The standalone constructor does not apply branch charges:
                # the production bridge repairs original PP BEFORE this seam.
                # Native bank/charge regressions cover that repair separately.
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
