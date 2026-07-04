"""Entry-hazard awareness probe for foundation checkpoints.

Entry hazards (Spikes in gen3) are the canonical delayed / positional interaction: setting Spikes
pays off cumulatively over future opponent switch-ins (front-loaded value), and Rapid Spin's value
is reactive — it only matters when hazards are on your OWN side. A searchless value head under a
sparse ±1 reward tends to under-credit both, and self-play under-explores them (Spikes and spinners
are individually rare in randbats, and the joint state is rarer still). This probe measures, from a
checkpoint alone, whether the policy has learned either interaction:

  - spikes signals   — over states where Spikes is a legal move: P(Spikes) bucketed by turn (does it
                       front-load?) and by opponent-side layers already down (does it stop when
                       stacked?), plus how often Spikes is the argmax choice.
  - rapid-spin signal — a CONTROLLED counterfactual: over states where Rapid Spin is legal, inject
                       self-side Spikes layers 0->3 holding all else fixed and measure dP(Rapid Spin).
                       A model that learned hazard clearing raises P(Rapid Spin) with more self-side
                       layers; a flat response means it ignores the hazard feature.

  - value response (dV) — the credit-assignment decomposition: inject Spikes layers 0->3 on the
                       SELF side (bad for us) and separately on the OPPONENT side (good for us),
                       holding all else fixed, and measure the VALUE HEAD's response. This separates
                       the two failure modes behind the flat Spikes policy:
                         value blind  (dV ~ 0)            -> credit assignment failed; fix = dense
                                                             shaping / value work, search inherits
                                                             the blindness at 1-ply.
                         value aware, policy flat         -> exploration/equilibrium failure; fix =
                                                             coverage curriculum (and search can
                                                             exploit the payoff immediately).

Summary scalars (near 0 => no hazard awareness; growth over milestones => it is emerging):
  spikes_early_tilt        mean P(Spikes | turn<=5) - mean P(Spikes | turn>=11)   (>0: front-loads)
  spikes_layer_sensitivity mean P(Spikes | 0 opp layers) - mean P(Spikes | >=2)   (>0: stops when stacked)
  spin_hazard_response     mean P(Rapid Spin | self-spikes=3) - (| self-spikes=0)  (>0: spins to clear)
  value_self_hazard_response  mean V(self-spikes=3) - mean V(self-spikes=0)        (<0 expected)
  value_opp_hazard_response   mean V(opp-spikes=3) - mean V(opp-spikes=0)          (>0 expected)
  (judge dV magnitudes against value_spread — the std of V over the same states)

Usage:
    python scripts/hazard_probe.py \
      --checkpoint checkpoints/pokezero-belief-gen3-1-5m.pt=belief-1.5M \
      --showdown-root /path/to/pokemon-showdown --games 300 --out evals/hazard_signals.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from pokezero.actions import MOVE_ACTION_COUNT
from pokezero.checkpoint_factors import build_corpus, choice_label
from pokezero.neural_policy import (
    evaluate_transformer_action_priors,
    evaluate_transformer_observation_value,
)
from pokezero.online_client import build_agent
from pokezero.showdown import observation_from_player_state


def _is_move(state, idx: int, needle: str) -> bool:
    if not state.legal_action_mask[idx]:
        return False
    label = choice_label(state, idx).lower()
    return label.startswith("move:") and needle in label


def _with_self_spikes(state, layers: int):
    """Counterfactual copy of `state` with exactly `layers` Spikes on the player's own side; all
    other side conditions preserved. Feeds NUMERIC_SELF_HAZARDS = layers/3 to the encoder."""
    counts = dict(state.self_side_condition_counts or {})
    conds = set(state.self_side_conditions or ())
    if layers <= 0:
        counts.pop("spikes", None)
        conds.discard("spikes")
    else:
        counts["spikes"] = layers
        conds.add("spikes")
    return replace(state, self_side_condition_counts=counts, self_side_conditions=tuple(sorted(conds)))


def _with_opp_spikes(state, layers: int):
    """Counterfactual copy of `state` with exactly `layers` Spikes on the OPPONENT side."""
    counts = dict(state.opponent_side_condition_counts or {})
    conds = set(state.opponent_side_conditions or ())
    if layers <= 0:
        counts.pop("spikes", None)
        conds.discard("spikes")
    else:
        counts["spikes"] = layers
        conds.add("spikes")
    return replace(
        state, opponent_side_condition_counts=counts, opponent_side_conditions=tuple(sorted(conds))
    )


def _mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


def _std(xs):
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return round((sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5, 4)


def probe_checkpoint(label: str, checkpoint: str, showdown_root: str, corpus, value_states: int = 1000) -> dict:
    agent = build_agent(checkpoint, showdown_root, our_name="hazard", deterministic=True)

    def priors(state):
        obs = observation_from_player_state(
            state, category_vocab=agent.vocab, spec=agent.spec, dex=agent.dex
        )
        return evaluate_transformer_action_priors(
            model=agent.policy.model, result=agent.policy.result, observations=[obs]
        )

    # --- Spikes: P(Spikes) by turn + by opponent-side layers already down ---
    spikes_by_turn: dict[str, list] = {}
    spikes_by_opp_layers: dict[int, list] = {}
    spikes_states = 0
    spikes_argmax = 0

    def turn_bucket(t):
        return "1-2" if t <= 2 else "3-5" if t <= 5 else "6-10" if t <= 10 else "11-20" if t <= 20 else "21+"

    # --- Rapid Spin: controlled self-side-spikes counterfactual ---
    spin_states = 0
    spin_p = {0: [], 1: [], 2: [], 3: []}
    spin_argmax_at3 = 0

    for entry in corpus:
        st = entry.state
        probs = None
        for idx in range(MOVE_ACTION_COUNT):
            if _is_move(st, idx, "spikes") and "rapid" not in choice_label(st, idx).lower():
                probs = priors(st) if probs is None else probs
                p = probs[idx]
                spikes_by_turn.setdefault(turn_bucket(entry.turn), []).append(p)
                layers = int((st.opponent_side_condition_counts or {}).get("spikes", 0))
                spikes_by_opp_layers.setdefault(layers, []).append(p)
                spikes_states += 1
                legal = [i for i, m in enumerate(st.legal_action_mask) if m]
                spikes_argmax += (max(legal, key=lambda i: probs[i]) == idx)
                break
        for idx in range(MOVE_ACTION_COUNT):
            if _is_move(st, idx, "rapid") and "spin" in choice_label(st, idx).lower():
                spin_states += 1
                for k in (0, 1, 2, 3):
                    cf = _with_self_spikes(st, k)
                    obs = observation_from_player_state(cf, category_vocab=agent.vocab, spec=agent.spec, dex=agent.dex)
                    all_p = evaluate_transformer_action_priors(
                        model=agent.policy.model, result=agent.policy.result, observations=[obs]
                    )
                    spin_p[k].append(all_p[idx])
                    if k == 3:
                        legal = [i for i, m in enumerate(st.legal_action_mask) if m]
                        spin_argmax_at3 += (max(legal, key=lambda i: all_p[i]) == idx)
                break

    # --- dV: value-head response to injected hazards (credit-assignment decomposition) ---
    def value_of(state) -> float:
        obs = observation_from_player_state(
            state, category_vocab=agent.vocab, spec=agent.spec, dex=agent.dex
        )
        return evaluate_transformer_observation_value(
            model=agent.policy.model, result=agent.policy.result, observations=[obs]
        )

    v_self = {0: [], 1: [], 2: [], 3: []}
    v_opp = {0: [], 1: [], 2: [], 3: []}
    v_base = []
    for entry in corpus[:value_states]:
        st = entry.state
        v_base.append(value_of(st))
        for k in (0, 1, 2, 3):
            v_self[k].append(value_of(_with_self_spikes(st, k)))
            v_opp[k].append(value_of(_with_opp_spikes(st, k)))

    early = [p for b in ("1-2", "3-5") for p in spikes_by_turn.get(b, [])]
    late = [p for b in ("11-20", "21+") for p in spikes_by_turn.get(b, [])]
    fresh = spikes_by_opp_layers.get(0, [])
    stacked = [p for k, v in spikes_by_opp_layers.items() if k >= 2 for p in v]
    spin0, spin3 = spin_p[0], spin_p[3]

    return {
        "label": label,
        "checkpoint": checkpoint,
        "spikes_legal_states": spikes_states,
        "spikes_p_by_turn": {k: _mean(v) for k, v in sorted(spikes_by_turn.items())},
        "spikes_p_by_opp_layers": {k: _mean(v) for k, v in sorted(spikes_by_opp_layers.items())},
        "spikes_argmax_rate": round(spikes_argmax / spikes_states, 4) if spikes_states else None,
        "rapid_spin_legal_states": spin_states,
        "rapid_spin_p_by_self_spikes": {k: _mean(spin_p[k]) for k in (0, 1, 2, 3)},
        "rapid_spin_argmax_rate_at_3": round(spin_argmax_at3 / spin_states, 4) if spin_states else None,
        # summary scalars — track these over milestones
        "spikes_early_tilt": (round(_mean(early) - _mean(late), 4) if early and late else None),
        "spikes_layer_sensitivity": (round(_mean(fresh) - _mean(stacked), 4) if fresh and stacked else None),
        "spin_hazard_response": (round(_mean(spin3) - _mean(spin0), 4) if spin0 and spin3 else None),
        # dV decomposition — see module docstring
        "value_states": len(v_base),
        "value_spread": _std(v_base),
        "value_by_self_spikes": {k: _mean(v_self[k]) for k in (0, 1, 2, 3)},
        "value_by_opp_spikes": {k: _mean(v_opp[k]) for k in (0, 1, 2, 3)},
        "value_self_hazard_response": (
            round(_mean(v_self[3]) - _mean(v_self[0]), 4) if v_self[0] and v_self[3] else None
        ),
        "value_opp_hazard_response": (
            round(_mean(v_opp[3]) - _mean(v_opp[0]), 4) if v_opp[0] and v_opp[3] else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="PATH[=LABEL]")
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--games", type=int, default=300, help="driver games for the state corpus")
    parser.add_argument("--max-states", type=int, default=8000)
    parser.add_argument(
        "--value-states",
        type=int,
        default=1000,
        help="corpus states sampled for the dV hazard-injection value probe (9 value evals each)",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print(f"[hazard] building corpus ({args.games} games)…", file=sys.stderr)
    corpus = build_corpus(args.showdown_root, num_games=args.games, max_states=args.max_states)
    print(f"[hazard] corpus: {len(corpus)} states", file=sys.stderr)

    rows = []
    for spec in args.checkpoint:
        path, _, label = spec.partition("=")
        label = label or Path(path).stem
        print(f"[hazard] probing {label}…", file=sys.stderr)
        row = probe_checkpoint(label, path, args.showdown_root, corpus, value_states=args.value_states)
        rows.append(row)
        print(f"  spikes: n={row['spikes_legal_states']} argmax={row['spikes_argmax_rate']} "
              f"early_tilt={row['spikes_early_tilt']} layer_sens={row['spikes_layer_sensitivity']}", file=sys.stderr)
        print(f"  rapid-spin: n={row['rapid_spin_legal_states']} P_by_self_spikes={row['rapid_spin_p_by_self_spikes']} "
              f"hazard_response={row['spin_hazard_response']}", file=sys.stderr)
        print(f"  dV: n={row['value_states']} spread={row['value_spread']} "
              f"self_response={row['value_self_hazard_response']} (expect <0) "
              f"opp_response={row['value_opp_hazard_response']} (expect >0)", file=sys.stderr)

    payload = {"corpus_games": args.games, "corpus_states": len(corpus), "checkpoints": rows}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[hazard] wrote {out}", file=sys.stderr)
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
