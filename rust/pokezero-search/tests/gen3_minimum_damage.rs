//! Authored Gen3 nonimmune minimum damage and failure-boundary controls.
//! No scientific roster, action selection, or full-game qualification.
use poke_engine::choices::Choices;
use poke_engine::engine::abilities::Abilities;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::items::Items;
use poke_engine::engine::state::{MoveChoice,PokemonVolatileStatus};
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, PokemonStatus, PokemonType,
    SideReference, State};

fn fixture(side: SideReference, mv: Choices, double: bool) -> State {
    let mut state=State::default();
    for seat in [SideReference::SideOne,SideReference::SideTwo] {
        let team=state.get_side(&seat);
        for index in [PokemonIndex::P0,PokemonIndex::P1] {
            let p=&mut team.pokemon[index];
            p.hp=1000;p.maxhp=1000;p.level=100;
            p.speed=if seat==side {301} else {101};
            p.ability=Abilities::NONE;p.base_ability=Abilities::NONE;p.item=Items::NONE;
            p.types=(PokemonType::PSYCHIC,PokemonType::TYPELESS);
            p.attack=1;p.special_attack=1;p.defense=30000;p.special_defense=30000;
            for slot in [PokemonMoveIndex::M0,PokemonMoveIndex::M1,PokemonMoveIndex::M2,PokemonMoveIndex::M3] {
                p.replace_move(slot,Choices::SPLASH);p.moves[&slot].pp=9;
            }
        }
    }
    state.get_side(&side).get_active().replace_move(PokemonMoveIndex::M2,mv);
    let p=state.get_side(&side.get_other_side()).get_active();
    p.types=if mv==Choices::THUNDERBOLT {
        (PokemonType::GRASS,if double {PokemonType::DRAGON} else {PokemonType::TYPELESS})
    } else {(PokemonType::ROCK,if double {PokemonType::STEEL} else {PokemonType::TYPELESS})};
    state
}

fn successors(state: &State,side: SideReference) -> Vec<(f64,State)> {
    let before=state.serialize();let mut parent=state.clone();
    let own=MoveChoice::Move(PokemonMoveIndex::M2);let foe=MoveChoice::Move(PokemonMoveIndex::M1);
    let (a,b)=if side==SideReference::SideOne {(own,foe)} else {(foe,own)};
    let branches=generate_instructions_from_move_pair(&mut parent,&a,&b,true);
    assert_eq!(parent.serialize(),before);
    let mut out=Vec::new();let mut mass=0.0;
    for branch in branches {
        let p=branch.percentage as f64/100.0;mass+=p;if p==0.0 {continue;}
        let mut next=state.clone();next.apply_instructions(&branch.instruction_list);
        let mut reverse=next.clone();reverse.reverse_instructions(&branch.instruction_list);
        assert_eq!(reverse.serialize(),before,"apply/reverse drift");
        out.push((p,next));
    }
    assert!((mass-1.0).abs()<1e-6);out
}

#[test]
fn nonimmune_low_damage_never_disappears_in_either_seat_or_roll_profile() {
    for side in [SideReference::SideOne,SideReference::SideTwo] {
        for mv in [Choices::THUNDERBOLT,Choices::BODYSLAM,Choices::EXPLOSION] {
            for double in [false,true] {
                for (_,next) in successors(&fixture(side,mv,double),side) {
                    let hp=next.get_side_immutable(&side.get_other_side()).get_active_immutable().hp;
                    assert!(hp<1000,"nonimmune {:?}, double={} dealt zero",mv,double);
                }
            }
        }
    }
}

#[test]
fn true_type_and_ability_immunities_do_not_become_one_damage() {
    for side in [SideReference::SideOne,SideReference::SideTwo] {
        for mv in [Choices::THUNDERBOLT,Choices::BODYSLAM,Choices::EXPLOSION] {
            for mode in 0..3 {
                let mut state=fixture(side,mv,false);
                let foe=state.get_side(&side.get_other_side()).get_active();
                if mode==0 {foe.types=(if mv==Choices::THUNDERBOLT {PokemonType::GROUND} else {PokemonType::GHOST},PokemonType::TYPELESS);}
                else if mode==1 {foe.ability=Abilities::WONDERGUARD;}
                else {foe.ability=if mv==Choices::THUNDERBOLT {Abilities::VOLTABSORB} else if mv==Choices::EXPLOSION {Abilities::DAMP} else {continue};}
                for (_,next) in successors(&state,side) {
                    assert_eq!(next.get_side_immutable(&side.get_other_side()).get_active_immutable().hp,1000);
                    if mode==2 && mv==Choices::EXPLOSION {
                        assert_eq!(next.get_side_immutable(&side).get_active_immutable().hp,1000,"Damp suppresses self-KO too");
                    }
                }
            }
        }
    }
}

#[test]
fn protect_misses_and_failed_beforemove_prefixes_remain_zero_damage() {
    for side in [SideReference::SideOne,SideReference::SideTwo] {
        for mv in [Choices::THUNDERBOLT,Choices::BODYSLAM,Choices::EXPLOSION] {
            for mode in 0..3 {
                let mut state=fixture(side,mv,false);
                if mode==0 {state.get_side(&side.get_other_side()).get_active().replace_move(PokemonMoveIndex::M1,Choices::PROTECT);}
                if mode==1 {state.get_side(&side).get_active().moves[&PokemonMoveIndex::M2].choice.accuracy=0.0;}
                if mode==2 {let p=state.get_side(&side).get_active();p.status=PokemonStatus::SLEEP;p.rest_turns=2;}
                for (_,next) in successors(&state,side) {
                    assert_eq!(next.get_side_immutable(&side.get_other_side()).get_active_immutable().hp,1000,"{:?}/mode{}",mv,mode);
                }
            }
        }
    }
}

#[test]
fn endure_and_fixed_damage_keep_their_own_rules() {
    for side in [SideReference::SideOne,SideReference::SideTwo] {
        let mut state=fixture(side,Choices::THUNDERBOLT,false);
        let foe=state.get_side(&side.get_other_side()).get_active();foe.hp=1;
        foe.replace_move(PokemonMoveIndex::M1,Choices::ENDURE);
        for (_,next) in successors(&state,side) {
            assert_eq!(next.get_side_immutable(&side.get_other_side()).get_active_immutable().hp,1);
        }
        for mv in [Choices::NIGHTSHADE,Choices::COUNTER] {
            let mut state=fixture(side,mv,false);
            state.get_side(&side.get_other_side()).get_active().types=(PokemonType::PSYCHIC,PokemonType::TYPELESS);
            for (_,next) in successors(&state,side) {
                assert_eq!(next.get_side_immutable(&side.get_other_side()).get_active_immutable().hp,
                    if mv==Choices::NIGHTSHADE {900} else {1000});
            }
        }
    }
}

#[test]
fn minimum_is_per_hit_and_lethal_minimum_cancels_the_waiting_move() {
    for side in [SideReference::SideOne,SideReference::SideTwo] {
        let mut state=fixture(side,Choices::DOUBLEKICK,true);
        state.get_side(&side.get_other_side()).get_active().types=(PokemonType::POISON,PokemonType::FLYING);
        for (_,next) in successors(&state,side) {
            assert_eq!(next.get_side_immutable(&side.get_other_side()).get_active_immutable().hp,998,
                "each of the two successful hits has its own minimum");
        }
        let mut state=fixture(side,Choices::THUNDERBOLT,true);
        state.get_side(&side.get_other_side()).get_active().hp=1;
        for (_,next) in successors(&state,side) {
            let foe=next.get_side_immutable(&side.get_other_side());
            assert_eq!(foe.get_active_immutable().hp,0);
            assert!(foe.force_switch);
            assert_eq!(foe.get_active_immutable().moves[&PokemonMoveIndex::M1].pp,9,
                "a lethal minimum cancels the queued move before PP");
        }
    }
}

#[test]
fn semi_invulnerable_targets_stay_unhit_before_their_slower_release_and_crash_still_occurs() {
    for side in [SideReference::SideOne,SideReference::SideTwo] {
        for (volatile,release) in [(PokemonVolatileStatus::DIG,Choices::DIG),
            (PokemonVolatileStatus::DIVE,Choices::DIVE),(PokemonVolatileStatus::FLY,Choices::FLY),
            (PokemonVolatileStatus::BOUNCE,Choices::BOUNCE)] {
            let mut state=fixture(side,Choices::THUNDERBOLT,false);
            let foe=state.get_side(&side.get_other_side());
            foe.volatile_statuses.insert(volatile);
            foe.get_active().replace_move(PokemonMoveIndex::M1,release);
            for (_,next) in successors(&state,side) {
                assert_eq!(next.get_side_immutable(&side.get_other_side()).get_active_immutable().hp,1000,
                    "faster attacker must not hit {:?} before the slower release",volatile);
            }
        }
        let mut state=fixture(side,Choices::HIGHJUMPKICK,false);
        let foe=state.get_side(&side.get_other_side());
        foe.volatile_statuses.insert(PokemonVolatileStatus::DIG);
        foe.get_active().replace_move(PokemonMoveIndex::M1,Choices::DIG);
        let own=MoveChoice::Move(PokemonMoveIndex::M2);let foe=MoveChoice::Move(PokemonMoveIndex::M1);
        let (a,b)=if side==SideReference::SideOne {(own,foe)} else {(foe,own)};
        let branches=generate_instructions_from_move_pair(&mut state,&a,&b,true);
        for branch in branches {
            if branch.percentage==0.0 {continue;}
            let hits=branch.instruction_list.iter().filter(|i| matches!(i,
                poke_engine::instruction::Instruction::Damage(v) if v.side_ref==side)).count();
            assert_eq!(hits,2,"both crash and the guaranteed later Dig hit remain, without pinning crash magnitude");
        }
    }
}

#[test]
fn a_rounded_low_jump_kick_hit_is_not_a_crash_but_real_misses_still_crash() {
    for side in [SideReference::SideOne,SideReference::SideTwo] {
        let mut state=fixture(side,Choices::HIGHJUMPKICK,true);
        state.get_side(&side.get_other_side()).get_active().types=(PokemonType::POISON,PokemonType::FLYING);
        let mut hit=0.0;let mut missed=0.0;
        for (mass,next) in successors(&state,side) {
            let own=next.get_side_immutable(&side).get_active_immutable().hp;
            let foe=next.get_side_immutable(&side.get_other_side()).get_active_immutable().hp;
            if foe==999 {hit+=mass;assert_eq!(own,1000,"successful minimum damage is not a crash");}
            else {missed+=mass;assert_eq!(foe,1000);assert!(own<1000,"actual accuracy failure still crashes");}
        }
        // Pinned Gen3 High Jump Kick accuracy is90%; this checks the complete
        // native hit/miss distribution rather than a representative trace.
        assert!((hit-0.9).abs()<1e-6);assert!((missed-0.1).abs()<1e-6);
    }
}
