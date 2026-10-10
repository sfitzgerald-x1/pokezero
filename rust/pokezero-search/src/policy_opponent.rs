//! Fixed learned-policy opponent, distinct from opponent-prior PUCT.
//!
//! The provider must infer the policy head from the opponent's own valid
//! observation. This module deliberately cannot substitute the auxiliary
//! opponent-action head, infer observations by swapping seats, or fall back to
//! uniform/adversarial play. Distributions are bound to a tree node's ordered
//! legal surface. One sampler is owned by one search, never reused across trees.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::tree::DecisionNode;
use crate::{select, MoveStats};

/// A normalized distribution in native legal-option order, not action-slot
/// order. f64 normalization keeps tiny positive mapped masses representable.
#[derive(Clone, Debug)]
pub(crate) struct ActionDistribution(Vec<f64>);

impl ActionDistribution {
    pub(crate) fn new(weights: &[f32], option_count: usize) -> PyResult<Self> {
        if weights.len() != option_count || option_count == 0 {
            return Err(PyValueError::new_err(
                "policy opponent: legal distribution arity mismatch",
            ));
        }
        if weights.iter().any(|p| !p.is_finite() || *p < 0.0) {
            return Err(PyValueError::new_err(
                "policy opponent: invalid legal probability",
            ));
        }
        let total: f64 = weights.iter().map(|p| f64::from(*p)).sum();
        if total <= 0.0 {
            return Err(PyValueError::new_err(
                "policy opponent: zero legal probability mass",
            ));
        }
        Ok(Self(
            weights.iter().map(|p| f64::from(*p) / total).collect(),
        ))
    }

    /// Strict sparse option mapping. Refuse an unmappable option instead of
    /// dropping it and silently sampling a smaller game. A single mapped arm
    /// is certain even if its global softmax underflowed to zero.
    pub(crate) fn from_policy_row(row: &[f32], map: &[Option<usize>]) -> PyResult<Self> {
        let mut seen = std::collections::HashSet::new();
        let mut weights = Vec::with_capacity(map.len());
        for slot in map {
            let slot = slot
                .ok_or_else(|| PyValueError::new_err("policy opponent: unmapped legal option"))?;
            if !seen.insert(slot) {
                return Err(PyValueError::new_err(
                    "policy opponent: duplicate action-slot mapping",
                ));
            }
            let weight = *row.get(slot).ok_or_else(|| {
                PyValueError::new_err("policy opponent: action slot out of range")
            })?;
            if !weight.is_finite() || weight < 0.0 {
                return Err(PyValueError::new_err(
                    "policy opponent: invalid mapped probability",
                ));
            }
            weights.push(weight);
        }
        if weights.len() == 1 {
            weights[0] = 1.0;
        }
        Self::new(&weights, map.len())
    }

    /// A raw-policy leaf must be deterministic. Do not turn a stochastic
    /// callback into native-order argmax or consume a policy-sampling RNG.
    pub(crate) fn certain_action(&self) -> PyResult<usize> {
        let mut chosen = None;
        for (index, weight) in self.0.iter().enumerate() {
            if *weight != 0.0 {
                if *weight != 1.0 || chosen.is_some() {
                    return Err(PyValueError::new_err("raw policy terminal: callback must return a one-hot legal action"));
                }
                chosen = Some(index);
            }
        }
        chosen.ok_or_else(|| PyValueError::new_err("raw policy terminal: no certain legal action"))
    }

    fn sample(&self, rng: &mut StdRng) -> usize {
        let draw = rng.random::<f64>();
        let mut cumulative = 0.0;
        let mut last_positive = 0;
        for (index, p) in self.0.iter().enumerate() {
            if *p > 0.0 {
                last_positive = index;
            }
            cumulative += p;
            if draw < cumulative {
                return index;
            }
        }
        // Only floating-point summation residue can reach here. Never select
        // a trailing zero-probability option to cover that residue.
        last_positive
    }
}

struct BoundDistribution {
    ordered_arms: Vec<String>,
    distribution: ActionDistribution,
}

pub(crate) struct PolicyOpponent {
    pub(crate) opponent_side_one: bool,
    rng: StdRng,
    nodes: HashMap<usize, BoundDistribution>,
    pub(crate) samples: u64,
}

impl PolicyOpponent {
    /// The separate seed is explicit: opponent draws must not consume the
    /// environment/chance RNG or alter its trace merely by requesting a policy.
    pub(crate) fn new(opponent_side_one: bool, policy_seed: u64) -> Self {
        Self {
            opponent_side_one,
            rng: StdRng::seed_from_u64(policy_seed),
            nodes: HashMap::new(),
            samples: 0,
        }
    }

    pub(crate) fn select_joint<F>(
        &mut self,
        node_id: usize,
        node: &DecisionNode,
        c_puct: f32,
        fpu_reduction: Option<f32>,
        mut provide: F,
    ) -> PyResult<(usize, usize)>
    where
        F: FnMut() -> PyResult<ActionDistribution>,
    {
        let opponent = if self.opponent_side_one {
            &node.s1_stats
        } else {
            &node.s2_stats
        };
        let ordered_arms: Vec<String> = opponent.iter().map(|s| s.display.clone()).collect();
        if !self.nodes.contains_key(&node_id) {
            // Arity alone cannot certify forced play: hidden trapping can
            // suppress legal switches and leave one move. Always cross the
            // side-only request boundary, even for singleton/WAIT nodes.
            // The Python provider may skip inference AFTER certification.
            let distribution = provide()?;
            if distribution.0.len() != opponent.len() || opponent.is_empty() {
                return Err(PyValueError::new_err(
                    "policy opponent: node distribution arity mismatch",
                ));
            }
            self.nodes.insert(
                node_id,
                BoundDistribution {
                    ordered_arms: ordered_arms.clone(),
                    distribution,
                },
            );
        }
        let bound = &self.nodes[&node_id];
        if bound.ordered_arms != ordered_arms {
            return Err(PyValueError::new_err(
                "policy opponent: cached legal surface changed",
            ));
        }
        let sampled = bound.distribution.sample(&mut self.rng);
        self.samples += 1;
        // Opponent visits, Q estimates, virtual loss, FPU and PUCT priors have
        // NO influence on this draw. Only the subject still searches.
        if self.opponent_side_one {
            Ok((
                sampled,
                subject_select(&node.s2_stats, node, c_puct, false, fpu_reduction),
            ))
        } else {
            Ok((
                subject_select(&node.s1_stats, node, c_puct, true, fpu_reduction),
                sampled,
            ))
        }
    }
}

fn subject_select(
    stats: &[MoveStats],
    node: &DecisionNode,
    c: f32,
    side_one: bool,
    fpu: Option<f32>,
) -> usize {
    select(stats, node.visits, c, side_one, fpu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use poke_engine::engine::state::MoveChoice;

    fn node() -> DecisionNode {
        fn stats() -> Vec<MoveStats> {
            ["stay", "switch"]
                .iter()
                .map(|name| MoveStats {
                    display: (*name).into(),
                    prior: 0.5,
                    visits: 0,
                    total_value: 0.0,
                })
                .collect()
        }
        DecisionNode {
            visits: 0,
            depth: 0,
            s1_options: vec![MoveChoice::None; 2],
            s2_options: vec![MoveChoice::None; 2],
            s1_stats: stats(),
            s2_stats: stats(),
            children: HashMap::new(),
        }
    }

    #[test]
    fn sparse_mapping_preserves_native_identity_and_every_legal_action() {
        let row = [0.01, 0.0, 0.0, 0.09, 0.0, 0.0, 0.9];
        let d = ActionDistribution::from_policy_row(&row, &[Some(6), Some(0), Some(3)]).unwrap();
        let mut rng = StdRng::seed_from_u64(12);
        let mut counts = [0; 3];
        for _ in 0..100_000 {
            counts[d.sample(&mut rng)] += 1;
        }
        for (count, expected) in counts.iter().zip([90_000, 1_000, 9_000]) {
            assert!((*count as i64 - expected).abs() < 600, "{counts:?}");
        }
    }

    #[test]
    fn invalid_or_unmapped_distributions_refuse_without_fallback() {
        Python::initialize();
        for weights in [
            vec![],
            vec![0.0, 0.0],
            vec![f32::NAN, 1.0],
            vec![-0.1, 1.1],
            vec![f32::INFINITY, 1.0],
        ] {
            assert!(ActionDistribution::new(&weights, 2).is_err());
        }
        for map in [
            vec![Some(0), None],
            vec![Some(0), Some(5)],
            vec![Some(0), Some(0)],
        ] {
            assert!(ActionDistribution::from_policy_row(&[0.4, 0.6], &map).is_err());
        }
        assert_eq!(
            ActionDistribution::from_policy_row(&[0.0], &[Some(0)])
                .unwrap()
                .0,
            vec![1.0]
        );
        assert!(ActionDistribution::from_policy_row(&[0.0], &[None]).is_err());
    }

    #[test]
    fn tiny_legal_mass_is_normalized_and_zero_arms_never_sampled() {
        let d = ActionDistribution::new(&[1e-35, 3e-35, 0.0], 3).unwrap();
        let mut rng = StdRng::seed_from_u64(4);
        let mut counts = [0; 3];
        for _ in 0..10_000 {
            counts[d.sample(&mut rng)] += 1;
        }
        assert_eq!(counts[2], 0);
        assert!((counts[0] as i64 - 2500).abs() < 150);
    }

    #[test]
    fn policy_draws_ignore_opponent_search_stats_and_mirror_seats() {
        let mut a = node();
        let mut b = node();
        b.s1_stats[0].visits = 99_000;
        b.s1_stats[0].total_value = 99_000.0;
        b.s1_stats[0].prior = 0.0;
        b.s1_stats[1].prior = 1.0;
        let mut s1 = PolicyOpponent::new(true, 37);
        let mut s2 = PolicyOpponent::new(false, 37);
        for _ in 0..200 {
            let x = s1
                .select_joint(0, &b, 1.4, Some(0.3), || {
                    ActionDistribution::new(&[0.9, 0.1], 2)
                })
                .unwrap();
            let y = s2
                .select_joint(0, &a, 1.4, None, || ActionDistribution::new(&[0.9, 0.1], 2))
                .unwrap();
            assert_eq!(x.0, y.1);
            a.visits += 1;
        }
        assert_eq!(s1.samples, 200);
        assert_eq!(s2.samples, 200);
    }

    #[test]
    fn missing_provider_or_changed_cached_surface_is_an_error() {
        Python::initialize();
        let mut n = node();
        let mut sampler = PolicyOpponent::new(false, 1);
        assert!(sampler
            .select_joint(0, &n, 1.4, None, || Err(PyValueError::new_err(
                "missing observation"
            )))
            .is_err());
        assert_eq!(sampler.samples, 0);
        sampler
            .select_joint(0, &n, 1.4, None, || ActionDistribution::new(&[0.9, 0.1], 2))
            .unwrap();
        n.s2_stats.swap(0, 1);
        assert!(sampler
            .select_joint(0, &n, 1.4, None, || unreachable!())
            .is_err());
        assert_eq!(sampler.samples, 1);
    }

    #[test]
    fn single_choice_still_requires_certification_before_sampling() {
        Python::initialize();
        let mut n = node();
        n.s2_stats.truncate(1);
        n.s2_options.truncate(1);
        let mut sampler = PolicyOpponent::new(false, 31);
        assert!(sampler
            .select_joint(0, &n, 1.4, None, || {
                Err(PyValueError::new_err("uncertified singleton"))
            })
            .is_err());
        assert_eq!(sampler.samples, 0);
        let mut certifications = 0;
        assert_eq!(
            sampler
                .select_joint(0, &n, 1.4, None, || {
                    certifications += 1;
                    ActionDistribution::new(&[1.0], 1)
                })
                .unwrap()
                .1,
            0
        );
        assert_eq!(certifications, 1);
    }

    #[test]
    fn cached_nodes_infer_once_and_do_not_consume_chance_randomness() {
        let n = node();
        let mut sampler = PolicyOpponent::new(false, 31);
        let mut chance_a = StdRng::seed_from_u64(81);
        let mut chance_b = StdRng::seed_from_u64(81);
        let mut forwards = 0;
        for _ in 0..100 {
            sampler
                .select_joint(0, &n, 1.4, None, || {
                    forwards += 1;
                    ActionDistribution::new(&[0.9, 0.1], 2)
                })
                .unwrap();
            assert_eq!(chance_a.random::<u64>(), chance_b.random::<u64>());
        }
        assert_eq!(forwards, 1);
        assert_eq!(sampler.samples, 100);
    }

    /// Real subject PUCT with fixed-opponent samples and explicit payoff
    /// backup. The risky response wins on stay, loses on switch; the robust
    /// response returns .6 for either reply. Expected-response selection must
    /// prefer risky at 90/10 but robust at 50/50, unlike worst-case search.
    fn matrix_response(stay_probability: f32, subject_side_one: bool) -> usize {
        let mut n = node();
        let mut sampler = PolicyOpponent::new(!subject_side_one, 81);
        for _ in 0..30_000 {
            let (i, j) = sampler
                .select_joint(0, &n, 1.4, None, || {
                    ActionDistribution::new(&[stay_probability, 1.0 - stay_probability], 2)
                })
                .unwrap();
            let (own, opp) = if subject_side_one { (i, j) } else { (j, i) };
            let own_value = if own == 1 {
                0.6
            } else if opp == 0 {
                1.0
            } else {
                0.0
            };
            let v = if subject_side_one {
                own_value
            } else {
                1.0 - own_value
            };
            n.visits += 1;
            n.s1_stats[i].visits += 1;
            n.s1_stats[i].total_value += v;
            n.s2_stats[j].visits += 1;
            n.s2_stats[j].total_value += v;
        }
        let stats = if subject_side_one {
            n.s1_stats
        } else {
            n.s2_stats
        };
        usize::from(stats[1].visits > stats[0].visits)
    }

    #[test]
    fn expected_response_changes_with_switch_probability_in_both_seats() {
        for seat in [true, false] {
            assert_eq!(matrix_response(0.9, seat), 0);
            assert_eq!(matrix_response(0.5, seat), 1);
        }
    }
}
