"""Authored Transform request identity fixtures; no games, model or native calls."""
import unittest

from pokezero.engine_world import _apply_transform, EngineWorldUnsupported
from pokezero.local_showdown import _request_active_moves
from pokezero.poke_engine_adapter import MoveSpec, PokemonSpec, SideSpec


def mon(species, moves):
    return PokemonSpec(species, 100, ("normal",), 100, 100, 80, 90, 100, 110, 120,
        tuple(MoveSpec(move, 8) for move in moves), ability="limber")


class KnownCopiedMovesTests(unittest.TestCase):
    def build(self, donor_moves, rows, actor="p1", *, struggle=False):
        base=mon("ditto", ["transform"])
        donor=mon("cloyster", donor_moves)
        other="p2" if actor=="p1" else "p1"
        sides={actor:SideSpec((base,), boosts={"attack":2}),other:SideSpec((donor,))}
        out=_apply_transform(sides,{actor:"cloyster"},self_player=actor,
            self_active_request_moves=rows,self_request_struggle_only=struggle)
        return base,sides,out[actor]

    def test_real_failure_shape_cannot_sample_away_or_reorder_known_self_moves(self):
        rows=[dict(id=move,pp=pp,disabled=disabled) for move,pp,disabled in
            (("explosion",5,False),("spikes",3,False),("toxic",0,True),("surf",1,False))]
        for actor in ("p1","p2"):
            for donor in (("explosion","spikes","surf","toxic"),
                    ("rapidspin","spikes","surf","toxic"),
                    ("explosion","rapidspin","spikes","surf"),
                    ("explosion","icebeam","spikes","surf")):
                with self.subTest(actor=actor,donor=donor):
                    base,original,side=self.build(donor,rows,actor)
                    copy=side.pokemon[0]
                    self.assertEqual([(m.id,m.pp,m.disabled) for m in copy.moves],
                        [(r["id"],r["pp"],r["disabled"]) for r in rows])
                    self.assertIs(copy.pre_transform,base)
                    self.assertEqual(base.moves[0].id,"transform")
                    self.assertEqual(original[actor].pokemon[0],base)
                    self.assertEqual(side.boosts,{"attack":2})
                    self.assertEqual(set(side.volatile_statuses),{"transformed","typechange"})

    def test_hidden_power_uses_retained_request_display_not_sampled_donor(self):
        rows=_request_active_moves({"active":[{"moves":[dict(id="hiddenpower",
            move="Hidden Power Ice 60",pp=2,maxpp=24,disabled=True)]}]})
        self.assertEqual(rows[0]["move"],"Hidden Power Ice 60")
        _,_,side=self.build(("hiddenpowerfire70",),rows)
        self.assertEqual(side.pokemon[0].moves[0],MoveSpec("hiddenpowerice60",2,True))
        self.assertEqual([m.id for m in side.pokemon[0].moves[1:]],["none"]*3)

    def test_unresolved_or_conflicting_hidden_power_does_not_guess(self):
        for row in (dict(id="hiddenpower",pp=5),dict(id="hiddenpowerice",pp=5),
                dict(id="hiddenpowerfire70",move="Hidden Power Ice 70",pp=5),
                dict(id="hiddenpower",move="Hidden Power Ice 99",pp=5)):
            with self.subTest(row=row),self.assertRaises(EngineWorldUnsupported):
                self.build(("hiddenpowerfire70",),[row])

    def test_opponent_transform_and_request_suppressed_struggle_keep_separate_law(self):
        base=mon("ditto",["transform"]);donor=mon("cloyster",["surf","toxic"])
        sides={"p1":SideSpec((donor,)),"p2":SideSpec((base,))}
        out=_apply_transform(sides,{"p2":"cloyster"},self_player="p1",
            self_active_request_moves=[dict(id="icebeam",pp=0,disabled=True)])
        self.assertEqual([(m.id,m.pp,m.disabled) for m in out["p2"].pokemon[0].moves],
            [("surf",5,False),("toxic",5,False)])
        _,_,side=self.build(("surf","toxic"),[],struggle=True)
        self.assertTrue(all(m.disabled for m in side.pokemon[0].moves))

    def test_malformed_rows_cannot_silently_become_a_different_moveset(self):
        for rows in ([dict(id="surf",pp=True)],[dict(id="surf",pp=-1)],
                [dict(id="surf",pp=1,disabled=1)],[dict(id="surf",pp=1)]*2,
                [dict(id="surf",pp=1)]*5):
            with self.subTest(rows=rows),self.assertRaises(EngineWorldUnsupported):
                self.build(("surf",),rows)


if __name__=="__main__":unittest.main()
