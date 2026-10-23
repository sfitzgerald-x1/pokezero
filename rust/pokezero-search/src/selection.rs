//! Opt-in instantaneous positive-advantage sampling, NOT cumulative regret
//! matching and not a claim of equilibrium convergence.
//!
//! For each selecting seat: w(a)=max(Q(a)-V,0)+c*P(a)/sqrt(max(N,1)).
//! Q for an unvisited arm and V at an unvisited node are 0.5. V is the
//! visit-weighted mean of THAT seat's arms, including provisional virtual
//! losses. Priors are read, never rewritten. If all weights vanish (e.g. c=0
//! and equal Q), sample from the original prior, or uniformly if it is zero.
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::MoveStats;

pub(crate) struct PositiveAdvantageSampler {
    rng: StdRng,
    pub draws: u64,
}

impl PositiveAdvantageSampler {
    pub fn new(seed: u64) -> Self {
        // A separate stream: enabling selection does not consume chance draws;
        // disabled callers do not even construct or advance this RNG.
        Self { rng: StdRng::seed_from_u64(seed ^ 0x5245_4752_4554_0001), draws: 0 }
    }

    pub fn select(&mut self, stats: &[MoveStats], c: f32, side_one: bool) -> usize {
        let weights = weights(stats, c, side_one);
        let total: f64 = weights.iter().sum();
        let mut draw = self.rng.random::<f64>() * total;
        self.draws += 1;
        for (index, weight) in weights.iter().enumerate() {
            if draw < *weight { return index; }
            draw -= weight;
        }
        // Rounding can leave a tiny positive residue. Never select an arm
        // whose weight is zero in that case.
        weights.iter().rposition(|weight| *weight > 0.0).expect("nonempty legal options")
    }
}

fn weights(stats: &[MoveStats], c: f32, side_one: bool) -> Vec<f64> {
    assert!(!stats.is_empty());
    debug_assert!(c.is_finite() && c >= 0.0);
    let n: u64 = stats.iter().map(|stat| u64::from(stat.visits)).sum();
    let reflect = |q: f64| if side_one { q } else { 1.0 - q };
    let baseline = if n == 0 { 0.5 } else {
        reflect(stats.iter().map(|stat| f64::from(stat.total_value)).sum::<f64>() / n as f64)
    };
    let exploration = f64::from(c) / (n.max(1) as f64).sqrt();
    let mut result: Vec<f64> = stats.iter().map(|stat| {
        let q = reflect(if stat.visits == 0 { 0.5 } else {
            f64::from(stat.total_value) / f64::from(stat.visits)
        });
        (q - baseline).max(0.0) + exploration * f64::from(stat.prior)
    }).collect();
    if result.iter().sum::<f64>() == 0.0 {
        result = stats.iter().map(|stat| f64::from(stat.prior)).collect();
        if result.iter().sum::<f64>() == 0.0 { result.fill(1.0); }
    }
    debug_assert!(result.iter().all(|w| w.is_finite() && *w >= 0.0));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arms() -> Vec<MoveStats> {
        vec![MoveStats { display: "a".into(), prior: 0.25, visits: 4, total_value: 3.0 },
             MoveStats { display: "b".into(), prior: 0.75, visits: 4, total_value: 1.0 }]
    }

    #[test]
    fn weights_use_each_seats_frame_and_do_not_rewrite_priors() {
        let stats = arms();
        assert_eq!(weights(&stats, 0.0, true), vec![0.25, 0.0]);
        assert_eq!(weights(&stats, 0.0, false), vec![0.0, 0.25]);
        assert_eq!(stats.iter().map(|s| s.prior).collect::<Vec<_>>(), vec![0.25, 0.75]);
    }

    #[test]
    fn unvisited_and_zero_weight_initialization_are_explicit() {
        let mut stats = arms();
        for stat in &mut stats { stat.visits = 0; stat.total_value = 0.0; }
        assert_eq!(weights(&stats, 2.0, true), vec![0.5, 1.5]);
        assert_eq!(weights(&stats, 0.0, true), vec![0.25, 0.75]);
        // Q and V must use the same precision: f32(3/5) widened to f64
        // creates false positive advantages against an f64 baseline.
        for stat in &mut stats { stat.visits = 5; stat.total_value = 3.0; }
        assert_eq!(weights(&stats, 0.0, true), vec![0.25, 0.75]);
        assert_eq!(weights(&stats, 0.0, false), vec![0.25, 0.75]);
        for stat in &mut stats { stat.prior = 0.0; }
        assert_eq!(weights(&stats, 0.0, false), vec![1.0, 1.0]);
    }

    #[test]
    fn virtual_losses_use_the_seats_own_baseline() {
        let mut one = arms();
        one[0].visits += 1; // provisional zero in side-one frame
        let mut two = arms();
        two[0].visits += 1;
        two[0].total_value += 1.0; // provisional loss for side two
        let a = weights(&one, 0.0, true);
        let b = weights(&two, 0.0, false);
        assert!((a[0] - (0.6 - 4.0 / 9.0)).abs() < 1e-6);
        assert!((b[1] - (0.75 - 4.0 / 9.0)).abs() < 1e-6);
    }

    #[test]
    fn separate_stream_is_seeded_reproducible_and_actually_samples() {
        let mut a = PositiveAdvantageSampler::new(42);
        let mut b = PositiveAdvantageSampler::new(42);
        let stats = arms();
        let x: Vec<_> = (0..100).map(|_| a.select(&stats, 2.0, true)).collect();
        let y: Vec<_> = (0..100).map(|_| b.select(&stats, 2.0, true)).collect();
        assert_eq!(x, y);
        assert!(x.contains(&0) && x.contains(&1));
        assert_eq!(a.draws, 100);
    }
}
