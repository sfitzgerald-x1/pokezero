//! Carry per-branch diagnostics ACROSS the world-abort error boundary.
//!
//! THE DEFECT THIS MODULE EXISTS TO FIX. `model.rs` accumulates lossy sub-case
//! counts branch by branch and emits them in the search report -- but on any
//! attribution-unsafe branch (and on any other mid-search error, and on a
//! contained poke-engine panic) it returns `Err` BEFORE the report string is
//! built. `engine_search.py` catches that error, salvages exactly one number
//! (`attribution_unsafe_renders`), and everything else the world observed before
//! it died is discarded. Aborts are the MAJORITY of the fallback residue, so the
//! sub-case counters only ever described the clean-completion subset -- precisely
//! the subset that does not need diagnosing.
//!
//! It already cost a concrete result: #1158 added a production counter for the
//! Protect renderer marker in order to distinguish "the fix fires but worlds die
//! at their NEXT unsafe branch" from "the fix never fires", and because of this
//! defect BOTH readings are zero. See the `protect_marker_rendered` block in
//! `events.rs`, which states the limitation and is the best statement of why this
//! matters.
//!
//! HOW THE COUNTS TRAVEL: as a Python attribute on the aborting exception object,
//! not as text inside its message. The message becomes the
//! `world_failure_reasons` KEY, whose bytes are a measurement contract compared
//! across eras (and which `_bounded_reason_detail` truncates at 512 chars), so
//! encoding numbers into it would corrupt the key space and alias distinct
//! reasons together. Attaching a payload leaves `str(error)` byte-identical.
//!
//! WHY A PAYLOAD RATHER THAN A SIDE CHANNEL. A thread-local drained by a second
//! native call would work, but it makes double-counting a LIVE hazard: any path
//! that fails to drain (or fails before the per-search reset) leaks one world's
//! counts into the next world's abort. Binding the counts to the specific
//! exception object makes that unrepresentable -- a payload cannot outlive the
//! error it is attached to, and the clean path attaches nothing at all.
//!
//! Deliberately NOT behind the `model` feature, for the same reason
//! `panic_guard` is not: gating it would make the tests for the one piece of
//! code whose job is to keep telemetry alive require libtorch, so nobody could
//! run them. `model.rs` (feature-gated, untestable without a GPU build) holds
//! only the wiring; the ledger and the transport live here and are unit-tested.

use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The attribute name the aborting exception carries, read by
/// `EngineMctsPolicy._absorb_aborted_lossy_subcases`.
///
/// NAMESPACED on purpose. The Python side reads this off an arbitrary caught
/// exception, so a generic name (`lossy_subcases`, matching the report key)
/// would let some unrelated third-party exception that happens to carry that
/// attribute inject counts. A rename on either side silently zeroes the abort
/// arm, so both spellings are asserted against each other by a test in
/// `tests/test_engine_search.py`.
pub const ABORT_PAYLOAD_ATTR: &str = "pokezero_lossy_subcases";

/// Sub-case counts for ONE native search invocation, reported on both exits.
///
/// WHAT IS COUNTED (this rationale moved here from `model.rs` with the accumulator).
/// Renders that were COUNTED rather than refused, per sub-case. Without it the
/// usable-ambiguity class is invisible in aggregate: before the split it showed up as
/// `world_failure_reasons["...:ambiguous"]` because it refused, and after the split it
/// would show up nowhere at all. An invisible class is how this campaign spent two eras
/// unable to say what had changed.
///
/// One owner for both exits is the point. The clean path renders
/// [`Self::json_object`] into the search report and the abort path attaches
/// [`Self::attach_to`] to the error; the two are the same numbers from the same
/// map, so a world cannot report a count twice and cannot report it in only one
/// of the two shapes.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LossySubcaseLedger {
    counts: BTreeMap<String, usize>,
}

impl LossySubcaseLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Count one render that was KEPT-but-lossy, by sub-case slug.
    pub fn record(&mut self, subcase: &str) {
        *self.counts.entry(subcase.to_string()).or_insert(0) += 1;
    }

    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// The exact JSON object the search report's `lossy_subcases` field carries.
    ///
    /// Byte-pinned by a test: this replaced an inline `format!` in `model.rs`, and
    /// the field is parsed by `engine_search.py` and aggregated across eras, so a
    /// change in shape here is a change in a measurement contract.
    pub fn json_object(&self) -> String {
        format!(
            "{{{}}}",
            self.counts
                .iter()
                .map(|(name, count)| format!("\"{name}\":{count}"))
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    /// Attach the counts to an aborting error and return it, message untouched.
    ///
    /// ALWAYS attaches, even when empty. Key-absent and value-zero are
    /// deliberately distinguishable on the Python side -- a missing key reads as
    /// a genuine zero -- so the attribute's PRESENCE must be a property of the
    /// abort path, not of whether anything happened to be counted.
    ///
    /// Infallible by construction. A `PyErr` raised out of `setattr` is
    /// swallowed and the original error returned unchanged, and no `unwrap` or
    /// `expect` appears below: this runs on the world-abort path of a campaign
    /// worker, where a pyo3 panic escapes `except Exception` and kills the
    /// process. Losing telemetry is a bad outcome; losing the shard is a worse
    /// one.
    pub fn attach_to(&self, error: PyErr) -> PyErr {
        Python::attach(|py| {
            let payload = PyDict::new(py);
            for (name, count) in &self.counts {
                if payload.set_item(name.as_str(), *count).is_err() {
                    return error;
                }
            }
            let value = error.value(py);
            if value.setattr(ABORT_PAYLOAD_ATTR, payload).is_err() {
                return error;
            }
            error
        })
    }
}

#[cfg(test)]
mod abort_payload_tests {
    use super::{LossySubcaseLedger, ABORT_PAYLOAD_ATTR};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use std::collections::BTreeMap;

    fn read_payload(error: &PyErr) -> Option<BTreeMap<String, usize>> {
        Python::attach(|py| {
            let value = error.value(py);
            if !value.hasattr(ABORT_PAYLOAD_ATTR).unwrap_or(false) {
                return None;
            }
            let attr = value.getattr(ABORT_PAYLOAD_ATTR).ok()?;
            let dict = attr.downcast::<PyDict>().ok()?;
            let mut out = BTreeMap::new();
            for (key, count) in dict.iter() {
                out.insert(
                    key.extract::<String>().expect("keys are str"),
                    count.extract::<usize>().expect("counts are int"),
                );
            }
            Some(out)
        })
    }

    /// THE POINT OF THE MODULE: a world that dies still reports what it saw.
    #[test]
    fn an_aborting_error_carries_the_counts_accumulated_before_it_died() {
        Python::initialize();
        let mut ledger = LossySubcaseLedger::new();
        ledger.record("sleeptalk_called_unidentified:protect_marker_rendered");
        ledger.record("sleeptalk_called_unidentified:ambiguous");
        ledger.record("sleeptalk_called_unidentified:ambiguous");

        let error = ledger.attach_to(PyValueError::new_err(
            "attribution-unsafe renderer branch rejected before tree/model fold: \
             sleeptalk_called_unidentified",
        ));

        let payload = read_payload(&error).expect("the abort must carry a payload");
        assert_eq!(
            payload.get("sleeptalk_called_unidentified:protect_marker_rendered"),
            Some(&1),
            "the marker rendered before the world aborted and must still be visible: \
             {payload:?}"
        );
        assert_eq!(
            payload.get("sleeptalk_called_unidentified:ambiguous"),
            Some(&2)
        );
        assert_eq!(payload.len(), 2, "no invented keys: {payload:?}");
    }

    /// The measurement contract. The message becomes the `world_failure_reasons`
    /// key and is compared across eras; attaching a payload must not perturb one
    /// byte of it, and must not change the exception TYPE either (a
    /// BaseException subclass would slip past `except Exception` and kill the
    /// shard, the exact bug `panic_guard` exists to prevent).
    #[test]
    fn attaching_leaves_the_reason_key_and_the_exception_type_untouched() {
        Python::initialize();
        let message = "attribution-unsafe renderer branch rejected before tree/model fold: \
                       sleeptalk_called_unidentified";
        let mut ledger = LossySubcaseLedger::new();
        ledger.record("sleeptalk_called_unidentified:ambiguous");

        let error = ledger.attach_to(PyValueError::new_err(message));

        Python::attach(|py| {
            assert_eq!(
                error.value(py).to_string(),
                message,
                "the reason key's bytes are a cross-era measurement contract"
            );
            assert!(
                error.is_instance_of::<PyValueError>(py),
                "must stay catchable by `except Exception`"
            );
        });
    }

    /// Key-absent and value-zero must stay distinguishable, because a missing
    /// key otherwise reads as a genuine zero. So an abort that observed nothing
    /// still carries the key, holding an EMPTY object.
    #[test]
    fn an_abort_that_observed_nothing_still_carries_an_empty_payload() {
        Python::initialize();
        let ledger = LossySubcaseLedger::new();
        assert!(ledger.is_empty());

        let error = ledger.attach_to(PyValueError::new_err("battle is already over at the root"));

        let payload = read_payload(&error)
            .expect("the KEY must be present even with nothing counted, or absence reads as zero");
        assert!(payload.is_empty(), "{payload:?}");
    }

    /// A world both accumulates AND aborts. The clean path reports
    /// `json_object()`, the abort path reports the payload; they are one map, so
    /// each observation is reported exactly once whichever exit is taken.
    #[test]
    fn a_world_that_accumulates_and_aborts_reports_each_observation_exactly_once() {
        Python::initialize();
        let mut ledger = LossySubcaseLedger::new();
        ledger.record("attract_immobilization_source_unknown");
        ledger.record("attract_immobilization_source_unknown");
        ledger.record("attract_immobilization_source_unknown");

        let error = ledger.attach_to(PyValueError::new_err("aborted"));
        let payload = read_payload(&error).expect("payload");

        assert_eq!(
            payload.get("attract_immobilization_source_unknown"),
            Some(&3)
        );
        // Three records, not six: attaching must READ the ledger, never fold the
        // report's numbers back into it.
        assert_eq!(
            ledger.json_object(),
            "{\"attract_immobilization_source_unknown\":3}"
        );
        // And attaching does not mutate the ledger, so a second exit (the
        // early-stop full-budget replay re-enters the same accounting) cannot
        // observe inflated numbers.
        let again = ledger.attach_to(PyValueError::new_err("aborted"));
        assert_eq!(
            read_payload(&again)
                .expect("payload")
                .get("attract_immobilization_source_unknown"),
            Some(&3)
        );
    }

    /// The report field's bytes, pinned. `engine_search.py` parses this and the
    /// shard aggregate is compared across eras.
    #[test]
    fn the_report_object_is_sorted_compact_json() {
        let mut ledger = LossySubcaseLedger::new();
        ledger.record("zeta");
        ledger.record("alpha");
        ledger.record("alpha");
        assert_eq!(ledger.json_object(), "{\"alpha\":2,\"zeta\":1}");
        assert_eq!(LossySubcaseLedger::new().json_object(), "{}");
    }

    /// THE WIRING PIN, and the reason it is a source-text assertion.
    ///
    /// `model.rs` is behind the `model` feature, so it needs libtorch and is not
    /// compiled by `cargo test` at all -- an integration test cannot reach it.
    /// Without this pin every test above passes while the search path never
    /// attaches anything, which is precisely the failure this campaign has hit
    /// before (an entire renderer arm deleted with 418/418 still green).
    /// `include_str!` is compile-time and path-relative, so it works whatever
    /// features are selected and wherever the crate is built.
    #[test]
    fn the_search_path_records_into_the_ledger_and_attaches_it_on_abort() {
        let model_rs = include_str!("model.rs");
        assert!(
            model_rs.contains("lossy_subcases.record("),
            "model.rs no longer records sub-cases into the ledger, so nothing is \
             accumulated to carry across the abort"
        );
        assert!(
            model_rs.contains("lossy_subcases.json_object()"),
            "model.rs no longer renders the ledger into the search report, so the \
             CLEAN path lost its counts"
        );
        assert!(
            model_rs.contains("lossy_subcases.attach_to(error)"),
            "model.rs no longer attaches the ledger to an aborting error, so every \
             aborted world silently discards what it observed -- the defect this \
             module was added to fix"
        );
        // And the counts must not be smuggled into the message, which is the
        // `world_failure_reasons` key.
        assert!(
            !model_rs.contains("json_object()}"),
            "sub-case counts must never be interpolated into an error message: that \
             string is the world_failure_reasons key and its bytes are a contract"
        );
        // ORDERING, which is what makes the panic path work: the ledger must be
        // constructed OUTSIDE `py.detach` and therefore outside
        // `catch_native_panic`, or a contained poke-engine panic unwinds the frame
        // that owns it and the counts die with the world. Compared by position so
        // reformatting between the two lines cannot break the pin.
        let ledger_at = model_rs
            .find("LossySubcaseLedger::new()")
            .expect("model.rs must construct the ledger");
        let detach_at = model_rs
            .find("let result = py.detach(")
            .expect("model.rs must bind the detach result so the Err arm can be mapped");
        assert!(
            ledger_at < detach_at,
            "the ledger is constructed inside the detached/panic-guarded region, so an \
             unwind destroys it and a panicking world reports nothing"
        );
    }

    /// A CONTAINED PANIC must carry the counts too, not just a returned `Err`.
    ///
    /// `catch_native_panic` unwinds the search frame and substitutes an error of its
    /// own, so the ledger cannot live in that frame. This drives the real
    /// composition: accumulate, panic, contain, attach. Measured 3 panics over ~80
    /// shards in the 2026-07-31 probe, so this arm is not hypothetical.
    #[test]
    fn a_contained_poke_engine_panic_still_reports_what_the_world_observed() {
        Python::initialize();
        let mut ledger = LossySubcaseLedger::new();
        let result: PyResult<String> = crate::panic_guard::catch_native_panic(|| {
            ledger.record("sleeptalk_called_unidentified:protect_marker_rendered");
            panic!("Invalid rest_turns value: 32")
        });
        let error = ledger.attach_to(result.expect_err("the panic must surface as Err"));

        let payload = read_payload(&error).expect("a panicking world must still report");
        assert_eq!(
            payload.get("sleeptalk_called_unidentified:protect_marker_rendered"),
            Some(&1),
            "{payload:?}"
        );
        // And the panic detail still reaches world_failure_reasons untouched.
        Python::attach(|py| {
            let message = error.value(py).to_string();
            assert!(
                message.contains("Invalid rest_turns value: 32"),
                "{message}"
            );
        });
    }
}
