//! Authored source-adjudicated terminal replacement and absorption boundaries.
use poke_engine::choices::Choices;
use poke_engine::engine::abilities::Abilities;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::items::Items;
use poke_engine::engine::state::{MoveChoice,PokemonVolatileStatus};
use poke_engine::state::{pokemon_index_iter,PokemonIndex,PokemonMoveIndex,PokemonType,SideReference,State};

fn successors(state:&State,actions:[MoveChoice;2])->Vec<(f64,State)> {
    let before=state.serialize();let mut parent=state.clone();
    let branches=generate_instructions_from_move_pair(&mut parent,&actions[0],&actions[1],true);
    assert_eq!(parent.serialize(),before);let mut mass=0.;let mut out=Vec::new();
    for branch in branches {
        let p=branch.percentage as f64/100.;mass+=p;if p==0. {continue;}
        let mut next=state.clone();next.apply_instructions(&branch.instruction_list);
        let mut reverse=next.clone();reverse.reverse_instructions(&branch.instruction_list);
        assert_eq!(reverse.serialize(),before);out.push((p,next));
    }
    assert!((mass-1.).abs()<1e-6);out
}

#[test]
fn flash_fire_wisp_obeys_accuracy_and_protect_in_both_seats() {
    for holder in [SideReference::SideOne,SideReference::SideTwo] {for protect in [false,true] {
        let mut state=State::default();
        for side in [SideReference::SideOne,SideReference::SideTwo] {
            let p=state.get_side(&side).get_active();p.hp=300;p.maxhp=300;p.speed=200;
            p.ability=if side==holder {Abilities::FLASHFIRE} else {Abilities::NONE};
            p.item=Items::NONE;p.types=(PokemonType::NORMAL,PokemonType::TYPELESS);
            p.replace_move(PokemonMoveIndex::M0,if side!=holder {Choices::WILLOWISP}
                else if protect {Choices::PROTECT} else {Choices::SPLASH});
        }
        let branches=successors(&state,[MoveChoice::Move(PokemonMoveIndex::M0);2]);
        let activated:f64=branches.iter().filter(|(_,s)|s.get_side_immutable(&holder)
            .volatile_statuses.contains(&PokemonVolatileStatus::FLASHFIRE)).map(|(p,_)|*p).sum();
        assert!((activated-if protect {0.} else {0.75}).abs()<1e-6);
    }}
}

#[test]
fn terminal_explosion_never_requests_a_winners_replacement_after_voluntary_switch() {
    for switcher in [SideReference::SideOne,SideReference::SideTwo] {
        let mut state=State::default();
        for side in [SideReference::SideOne,SideReference::SideTwo] {
            let team=state.get_side(&side);
            for p in pokemon_index_iter() {team.pokemon[p].hp=0;}
            let p=team.get_active();p.hp=300;p.maxhp=300;p.speed=200;p.ability=Abilities::NONE;
            p.types=(PokemonType::NORMAL,PokemonType::TYPELESS);p.item=Items::NONE;
            p.replace_move(PokemonMoveIndex::M0,Choices::EXPLOSION);
            if side==switcher {team.pokemon[PokemonIndex::P1].hp=1;team.pokemon[PokemonIndex::P1].maxhp=300;}
        }
        let mut actions=[MoveChoice::Move(PokemonMoveIndex::M0);2];
        actions[if switcher==SideReference::SideOne {0} else {1}]=MoveChoice::Switch(PokemonIndex::P1);
        for (_,next) in successors(&state,actions) {
            assert_eq!(next.get_side_immutable(&switcher.get_other_side()).visible_alive_pkmn(),0);
            assert_eq!(next.get_side_immutable(&switcher).get_active_immutable().hp,0);
            assert!(!next.side_one.force_switch && !next.side_two.force_switch);
        }
    }
}
