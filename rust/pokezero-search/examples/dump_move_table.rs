//! Dump the ENGINE'S OWN move table as JSON, so a reach census never hand-copies
//! accuracies. Read straight out of `poke_engine::choices::MOVES` — the same
//! `LazyLock` the renderer and the branch generator read — so the numbers cannot
//! drift from the engine the crate is linked against.
//!
//! `cargo run -p pokezero-search --example dump_move_table`
//!
//! Emitted per move: `accuracy`, `category`, `target`, and which effect slots are
//! occupied. The consumer decides the family; this only reports.

use poke_engine::choices::MOVES;

fn main() {
    let mut names: Vec<String> = MOVES.keys().map(|id| format!("{id:?}")).collect();
    names.sort();
    println!("{{");
    let mut first = true;
    for name in &names {
        let (id, choice) = MOVES
            .iter()
            .find(|(id, _)| &format!("{id:?}") == name)
            .expect("name came from the same map");
        if !first {
            println!(",");
        }
        first = false;
        print!(
            "  \"{name}\": {{\"move_id\": \"{id:?}\", \"accuracy\": {}, \"category\": \"{:?}\", \
             \"target\": \"{:?}\", \"base_power\": {}, \"has_status\": {}, \"has_volatile\": {}, \
             \"has_boost\": {}, \"has_side_condition\": {}, \"has_heal\": {}, \"has_drain\": {}, \
             \"has_crash\": {}, \"has_recoil\": {}, \"has_secondaries\": {}, \"volatile\": \"{}\", \
             \"boost\": \"{}\", \"flag_protect\": {}}}",
            choice.accuracy,
            choice.category,
            choice.target,
            choice.base_power,
            choice.status.is_some(),
            choice.volatile_status.is_some(),
            choice.boost.is_some(),
            choice.side_condition.is_some(),
            choice.heal.is_some(),
            choice.drain.is_some(),
            choice.crash.is_some(),
            choice.recoil.is_some(),
            choice.secondaries.is_some(),
            choice
                .volatile_status
                .as_ref()
                .map(|v| format!("{:?}:{:?}", v.target, v.volatile_status))
                .unwrap_or_default(),
            choice
                .boost
                .as_ref()
                .map(|b| format!("{:?}:{:?}", b.target, b.boosts))
                .unwrap_or_default(),
            choice.flags.protect,
        );
    }
    println!("\n}}");
}
