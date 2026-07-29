# Prediction, recorded BEFORE the 300-game differential (patch 34)

Base: origin/main @ 49d9e43 (post-#958). Engine 34 patches, fingerprint
5b29e611468d3baa930984d5b8557280835e72f1ce38d8dc3c6b183e15c344dc
(unpatched-33 comparison build was 9ecfacadc938c0da).
Population: seeds 1500000-1500299, strict matcher. Post-#958 baseline is 96 outside-limits.

## Row identities predicted to clear

Exactly two, both named by the c9 decomposition's `explosion_incapacitated_status_signature`
family and both replayed at base before the fix:

  * **1500074/57** — `|cant|p2a: Forretress|par`; Showdown heals both sides, engine's blocked
    branch had already killed Forretress with its own Explosion.
  * **1500188/33** — `|cant|p2a: Swalot|par`; Showdown heals p2 +19, engine's 25% blocked
    branch carried no components at all.

Predicted outside-limits: **96 -> 94**.

## Zero-change-elsewhere

The patch only relocates the self-destruct faint from `choice_before_move` to just after the
status-condition gate. It cannot affect:
  * any branch where the user is not using EXPLOSION/SELFDESTRUCT (no code path reached);
  * the firing branch of a self-destruct (faint still applied, still before damage) — pinned;
  * the DAMP case (guard preserved) — pinned.

So the 5 adjudicated `limit:roll_divergent_lethality` rows and the 3
`limit_not_established_keeps_label` rows must be untouched, as must the other 94.

## Full-frame check (learned from #958)

Report clearance against the FULL frame as well as outside-limits. The Encore fix cleared 16
full-frame against 12 outside-limits because four rows sat in the limit bucket; the same
under-count is possible here, since a self-destruct branch is exactly the shape that lands in
`limit:roll_divergent_lethality`. **If full-frame clearance exceeds 2, identity-diff the extra
rows rather than assuming they are noise.**
