//! Gen3 randombattle PUBLIC Shadow Tag knowledge, never sampled ability truth.
//! Recomputed from battle start for each provider invocation, not cached by
//! species/node or inherited as the engine's root-only force_trapped latch.

#[derive(Default)]
struct Active {
    ident: String,
    species: String,
    entry: usize,
    alive: bool,
    shadow_tag: bool,
}

fn id(text: &str) -> String {
    text.chars().filter(|c| c.is_ascii_alphanumeric()).flat_map(char::to_lowercase).collect()
}

fn alive(condition: &str) -> bool {
    let hp = condition.split_whitespace().next().unwrap_or("");
    hp.split('/').next().and_then(|s| s.parse::<u32>().ok()).is_some_and(|n| n > 0)
}

/// The caller's normal-ability Gen3 randombattle contract and complete filtered
/// history are required. Custom/hackmons/contextless calls never get a default
/// ability certificate. Equality with the real native surface remains separate.
pub(crate) fn known_shadow_tag(
    lines: &[String], slot: &str, current_own_species: &str, format: Option<&str>,
) -> bool {
    if format != Some("gen3randombattle") || !matches!(slot, "p1" | "p2") { return false; }
    let own_seat = if slot == "p1" { 0 } else { 1 };
    let mut sides = [Active::default(), Active::default()];
    let (mut gen3, mut singles, mut tier, mut started) = (false, false, false, false);
    let mut turn = 0u32;
    for line in lines {
        if !line.starts_with('|') { return false; }
        let f: Vec<_> = line.split('|').collect();
        let event = f.get(1).copied().unwrap_or("");
        let field = |n| f.get(n).copied().unwrap_or("");
        if matches!(event, "request" | "split" | "win" | "tie") { return false; }
        match event {
            "gen" => { if started { return false; } gen3 = field(2) == "3"; }
            "gametype" => { if started { return false; } singles = field(2) == "singles"; }
            "tier" => { if started { return false; } tier = id(field(2)) == "gen3randombattle"; }
            "start" => {
                if started || !(gen3 && singles && tier) { return false; }
                started = true;
            }
            _ => {}
        }
        if !started { continue; }
        let seat = if field(2).starts_with("p1a: ") { Some(0) }
            else if field(2).starts_with("p2a: ") { Some(1) } else { None };
        match event {
            "turn" => {
                let Some(next) = field(2).parse::<u32>().ok() else { return false; };
                if next != turn + 1 { return false; }
                turn = next;
            }
            "switch" | "drag" => {
                let Some(seat) = seat else { return false; };
                let active = &mut sides[seat];
                active.entry += 1;
                active.ident = field(2).to_string();
                active.species = id(field(3).split(',').next().unwrap_or(""));
                active.alive = alive(field(4));
                // Only these validated Gen3 default-ability species. A later
                // invalidator cannot re-certify merely from the same species.
                active.shadow_tag = matches!(active.species.as_str(), "wobbuffet" | "wynaut");
            }
            "replace" | "detailschange" | "-formechange" | "-transform" => {
                if let Some(seat) = seat {
                    sides[seat].shadow_tag = false;
                    sides[seat].species.clear(); // no stale acting identity certificate
                }
                else { return false; }
            }
            "-endability" => {
                if let Some(seat) = seat { sides[seat].shadow_tag = false; }
                else { return false; }
            }
            "faint" => {
                let Some(seat) = seat else { return false; };
                if sides[seat].ident != field(2) { return false; }
                sides[seat].alive = false; sides[seat].shadow_tag = false;
            }
            "-damage" | "-sethp" => {
                // A multi-owner sethp/ambiguous HP event cannot certify life.
                if f.len() > 4 && !field(4).starts_with('[') { return false; }
                let Some(seat) = seat else { return false; };
                if sides[seat].ident != field(2) { return false; }
                if !alive(field(3)) { sides[seat].alive = false; sides[seat].shadow_tag = false; }
            }
            "-ability" => {
                let Some(seat) = seat else { return false; };
                if sides[seat].entry == 0 || sides[seat].ident != field(2) { return false; }
                // The holder is field2, never a different [of] source.
                sides[seat].shadow_tag = id(field(3)) == "shadowtag";
            }
            "move" => {
                let name = id(field(3));
                if name == "skillswap" { for active in &mut sides { active.shadow_tag = false; } }
                else if matches!(name.as_str(), "roleplay" | "transform") {
                    let Some(seat) = seat else { return false; };
                    sides[seat].shadow_tag = false;
                }
            }
            "-activate" if line.contains("ability:") || line.contains("Skill Swap") || line.contains("Role Play") => {
                // Conservative even on failed/ambiguous ability-changing work.
                for active in &mut sides { active.shadow_tag = false; }
            }
            _ => {}
        }
    }
    let own = &sides[own_seat]; let holder = &sides[1-own_seat];
    started && turn > 0 && own.entry > 0 && holder.entry > 0 && own.alive && holder.alive
        && own.species == id(current_own_species) && holder.shadow_tag
}

#[cfg(test)]
mod tests {
    use super::*;
    fn prefix() -> Vec<String> {
        ["|gametype|singles", "|gen|3", "|tier|[Gen 3] Random Battle", "|start",
         "|switch|p1a: Snorlax|Snorlax, L80|100/100", "|switch|p2a: Wobbuffet|Wobbuffet, L80|100/100", "|turn|1"]
            .map(String::from).to_vec()
    }
    fn known(lines: &[String]) -> bool { known_shadow_tag(lines,"p1","Snorlax",Some("gen3randombattle")) }
    #[test]
    fn requires_normal_gen3_complete_public_context_and_current_living_actives() {
        let p = prefix(); assert!(known(&p));
        for format in [None,Some("gen3customgame"),Some("gen4randombattle")] {
            assert!(!known_shadow_tag(&p,"p1","Snorlax",format));
        }
        assert!(!known_shadow_tag(&p,"p1","Gengar",Some("gen3randombattle")));
        for extra in ["|faint|p2a: Wobbuffet", "|-damage|p2a: Wobbuffet|0 fnt", "|faint|p1a: Snorlax",
            "|request|private", "|split|p1", "|turn|3", "|switch|p2a: Ditto|Ditto, L80|100/100"] {
            let mut q=p.clone();q.push(extra.into());assert!(!known(&q),"{extra}");
        }
        assert!(!known(&p[3..]));
        let mut changed=p.clone();changed.push("|replace|p1a: Snorlax|Ditto, L80|100/100".into());
        assert!(!known(&changed));
    }
    #[test]
    fn invalidation_is_not_undone_by_species_and_siblings_do_not_share_knowledge() {
        let p=prefix();
        for extra in ["|move|p1a: Snorlax|Skill Swap|p2a: Wobbuffet", "|move|p2a: Wobbuffet|Role Play|p1a: Snorlax",
            "|move|p2a: Wobbuffet|Transform|p1a: Snorlax", "|-transform|p2a: Wobbuffet|p1a: Snorlax",
            "|-endability|p2a: Wobbuffet", "|-ability|p2a: Wobbuffet|Levitate", "|-formechange|p2a: Wobbuffet|Ditto"] {
            let mut q=p.clone();q.push(extra.into());assert!(!known(&q),"{extra}");
            q.push("|turn|2".into());assert!(!known(&q));assert!(known(&p));
        }
        let mut q=p.clone();q.push("|-endability|p2a: Wobbuffet".into());
        q.push("|-ability|p1a: Snorlax|Shadow Tag|[of] p2a: Wobbuffet".into());assert!(!known(&q));
        q.push("|-ability|p2a: Wobbuffet|Shadow Tag|[from] ability: Trace".into());assert!(known(&q));
        q.push("|switch|p2a: Wobbuffet|Wobbuffet, L80|100/100".into());assert!(known(&q));
    }
}
