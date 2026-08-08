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
    ///
    /// KEYS ARE NOT JSON-ESCAPED, carried over verbatim from the `format!` this
    /// replaced. Safe today only because `assert_subcase_vocabulary` bounds the slug
    /// alphabet upstream to registered tokens joined by `:` and `+` -- nothing that
    /// needs escaping can reach here. Note the ASYMMETRY between the two exits, so a
    /// future vocabulary change does not surprise anyone: [`Self::attach_to`] builds a
    /// real `PyDict` and is unaffected, so a key that broke this would corrupt the
    /// CLEAN path's report JSON while the ABORT path kept working.
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
    /// a genuine zero -- so wherever this is CALLED, the attribute's presence is a
    /// property of the call and not of whether anything happened to be counted.
    ///
    /// SCOPE, corrected by review: this says nothing about exits that never reach a
    /// call. `model.rs` has six argument-validation and root-parse `return Err`s
    /// ahead of the ledger's construction, and those attach nothing. None of them
    /// can have observed a sub-case (no branch has been rendered yet), and Python
    /// treats a missing attribute exactly like an empty payload, so the counter is
    /// unaffected -- but the invariant is "a search ran and aborted", not "any error
    /// left the pyfunction".
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

/// Run a native search under the panic guard with a ledger, attaching the counts to
/// EVERY failure it can produce.
///
/// THIS EXISTS TO MAKE ONE BUG UNREPRESENTABLE, and the bug is subtle enough that it
/// survived a source-text pin written specifically to catch it. The attach must happen
/// OUTSIDE [`crate::panic_guard::catch_native_panic`]: inside it, a contained
/// poke-engine panic unwinds the search frame and the guard substitutes an error of its
/// own AFTER the attach ran, so every panicking world silently reports nothing --
/// measured at 3 panics / ~80 shards in the 2026-07-31 probe.
///
/// The original wiring spelled that ordering out in `model.rs`, which `cargo test`
/// never compiles (the `model` cargo feature needs libtorch), so the only available
/// guard was an assertion over `include_str!("model.rs")`. Review demonstrated the
/// regression compiling and passing that pin: binding the inner call and mapping it
/// there keeps the attach textually after `py.detach(` -- which is all a position
/// comparison can see -- while moving it inside the guard. Owning both halves here
/// replaces a pin that can be evaded with an ordering the type system enforces, in a
/// module that IS compiled and IS unit-tested (see
/// `a_contained_poke_engine_panic_still_reports_what_the_world_observed`).
pub fn guarded_search_with_ledger<T>(
    f: impl FnOnce(&mut LossySubcaseLedger) -> PyResult<T>,
) -> PyResult<T> {
    let mut ledger = LossySubcaseLedger::new();
    // The ledger is owned by THIS frame, not the guarded one, so an unwind cannot
    // destroy it. Order is load-bearing: contain first, attach second.
    let result = crate::panic_guard::catch_native_panic(|| f(&mut ledger));
    result.map_err(|error| ledger.attach_to(error))
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
        // THE CONTAIN-THEN-ATTACH ORDERING IS NOT PINNED HERE, ON PURPOSE. It used to
        // be, as `find("let result = py.detach(") < rfind("attach_to(error)")`, and that
        // assertion is EVADABLE: binding the inner call and mapping it there
        //
        //     catch_native_panic(|| {
        //         let inner = multiply_batched_encoded_core(.., &mut lossy_subcases);
        //         inner.map_err(|e| lossy_subcases.attach_to(e))
        //     })
        //
        // compiles, keeps the attach textually after `py.detach(` -- all a position
        // comparison can see -- and moves it INSIDE the guard, so every contained engine
        // panic drops its counts. Rather than chase that with a cleverer regex, the
        // ordering moved into `guarded_search_with_ledger`, where the type system holds
        // it and `a_contained_poke_engine_panic_still_reports_what_the_world_observed`
        // exercises it in a module `cargo test` actually compiles. What is left here is
        // the wiring: that model.rs goes through that helper at all.
        assert!(
            model_rs.contains("abort_telemetry::guarded_search_with_ledger("),
            "model.rs no longer runs the search through `guarded_search_with_ledger`, \
             so nothing attaches the ledger to an aborting error and every aborted \
             world silently discards what it observed -- the defect this module was \
             added to fix. Reintroducing a hand-rolled `catch_native_panic` + `map_err` \
             here also reintroduces the ordering hazard the helper exists to remove."
        );
        assert!(
            !model_rs.contains("panic_guard::catch_native_panic"),
            "model.rs calls the panic guard directly again. The guard and the attach \
             must stay in one place: separated, their ORDER decides whether a panicking \
             world reports anything, and that order is invisible from model.rs"
        );
        // AND NOT ITS OWN LEDGER. This one line is the difference between the abort arm
        // working and being 100% dead, and every other assertion in this test is blind
        // to it:
        //
        //     guarded_search_with_ledger(|_unused| {
        //         let mut lossy_subcases = LossySubcaseLedger::new();
        //         multiply_batched_encoded_core(.., &mut lossy_subcases)
        //     })
        //
        // compiles, keeps the binding NAME so every `contains`/ordering check above
        // still holds, and leaves the report rendering correct counts on the clean path
        // -- while the helper's ledger is never written, so every aborted world attaches
        // `{}`. That is exactly the pre-PR state, invisible from outside. Found by
        // review, which also showed it green against the first commit of this branch:
        // the earlier "throwaway ledger" mutant only ever tripped a spelling check
        // because it RENAMED the binding, so this class was never actually killed.
        //
        // Now that `guarded_search_with_ledger` owns construction, model.rs has no
        // legitimate reason to build one: measured 0 occurrences here, 1 under the
        // mutant.
        assert!(
            !model_rs.contains("LossySubcaseLedger::new()"),
            "model.rs constructs its own ledger. The one the search records into must be \
             the one `guarded_search_with_ledger` attaches, or the abort arm silently \
             carries an empty payload for every world -- the pre-PR state, with the \
             clean path's report still correct so nothing looks wrong"
        );

        // ORDERING WITHIN THE SEARCH LOOP, which the helper cannot enforce. Positions
        // are compared rather than exact layout matched, so reformatting between the
        // anchors cannot break the pin.
        //
        // `record` BEFORE the attribution-unsafe gate. Swapping the record loop
        // with the `reject_attribution_unsafe` block restores the ORIGINAL DEFECT for
        // the aborting branch -- the primary target class -- with everything else here
        // still green, because the branch returns before it can record.
        let record_at = model_rs
            .find("lossy_subcases.record(")
            .expect("model.rs must record sub-cases into the ledger");
        let gate_at = model_rs
            .find("seam.reject_attribution_unsafe(")
            .expect("model.rs must still gate attribution-unsafe branches");
        assert!(
            record_at < gate_at,
            "sub-cases are recorded AFTER the attribution-unsafe gate, so the branch \
             that aborts the world records nothing -- exactly the defect this module \
             was added to fix, restored"
        );

        // THE COUNTS MUST NEVER REACH THE ERROR MESSAGE, pinned as a PROPERTY rather
        // than as one forbidden spelling. `!contains("json_object()}")` caught exactly
        // the `format!("..{}..", ..json_object())` shape and sailed past
        // `format!("{} [lossy={}]", raw, ..json_object())`, which corrupts the
        // `world_failure_reasons` key space -- the one contract this change calls
        // inviolable. So: exactly one call site, and it is the report block.
        let occurrences = model_rs.matches("json_object()").count();
        assert_eq!(
            occurrences, 1,
            "`json_object()` must have exactly ONE call site in model.rs (the search \
             report). A second one is how sub-case counts reach a string that is not \
             the report -- and the reason key's bytes are a cross-era measurement \
             contract that `_bounded_reason_detail` truncates at 512 chars."
        );
        let json_at = model_rs
            .find("json_object()")
            .expect("checked non-empty above");
        // Anchored on the report FORMAT STRING that declares the field, not on
        // `let extra = format!(` -- model.rs has two of those and `find` would take the
        // wrong one. This is the same literal `test_the_crate_and_python_agree_on_the_
        // lossy_subcases_key` pins from the Python side.
        let field_at = model_rs
            .find(r#"\"lossy_subcases\":{}"#)
            .expect("model.rs must still declare the lossy_subcases report field");
        let report_end = model_rs[field_at..]
            .find("\n    );")
            .map(|offset| field_at + offset)
            .expect("the report format! must terminate");
        assert!(
            field_at < json_at && json_at < report_end,
            "the sole `json_object()` call is outside the argument list of the search \
             report that declares `lossy_subcases`, so the counts are being rendered \
             into some OTHER string"
        );
    }

    /// A CONTAINED PANIC must carry the counts too, not just a returned `Err`.
    ///
    /// THE ORDERING TEST, and the reason `guarded_search_with_ledger` exists. It drives
    /// the HELPER, not a hand-composed `catch_native_panic` + `attach_to`, so that
    /// swapping contain and attach fails HERE -- in a module `cargo test` compiles --
    /// rather than in `model.rs`, which it does not. Measured 3 panics over ~80 shards
    /// in the 2026-07-31 probe, so this arm is not hypothetical.
    #[test]
    fn a_contained_poke_engine_panic_still_reports_what_the_world_observed() {
        Python::initialize();
        let result: PyResult<String> = super::guarded_search_with_ledger(|ledger| {
            ledger.record("sleeptalk_called_unidentified:protect_marker_rendered");
            panic!("Invalid rest_turns value: 32")
        });
        let error = result.expect_err("the panic must surface as Err");

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
            assert!(
                message.contains("world was aborted"),
                "panic containment must still be in force: {message}"
            );
        });
    }

    /// The helper's other two exits, so the panic arm is not its only driver.
    ///
    /// A returned `Err` must be attached -- that is the ordinary abort, ~92% of the
    /// residue -- and a success must pass through with NOTHING attached, because the
    /// clean path carries its counts in the report and attaching there is how the two
    /// channels would begin double-counting.
    ///
    /// AND THE MESSAGE MUST SURVIVE THE HELPER BYTE FOR BYTE. `attach_to` has its own
    /// purity pin one layer down, and the exactly-one-`json_object()` pin covers
    /// `model.rs` only -- which left `guarded_search_with_ledger` exempt from both, in a
    /// module that is now the single most natural place for a future author to "add
    /// context" to a reason key. Restating the error here as
    /// `format!("{} [lossy_subcases={}]", ..)` passed the entire suite. That is the
    /// blocking N2b finding re-opened one stack frame up; it does not stop being
    /// blocking by moving file. Found by review.
    #[test]
    fn the_guard_attaches_on_a_returned_error_and_reports_through_the_ledger_on_success() {
        Python::initialize();

        const REASON: &str = "attribution-unsafe renderer branch rejected before tree/model fold: \
             sleeptalk_called_unidentified";
        let aborted: PyResult<String> = super::guarded_search_with_ledger(|ledger| {
            ledger.record("sleeptalk_called_unidentified:ambiguous");
            Err(PyValueError::new_err(REASON))
        });
        let error = aborted.expect_err("the search returned Err");
        let payload = read_payload(&error).expect("an ordinary abort must carry counts");
        assert_eq!(
            payload.get("sleeptalk_called_unidentified:ambiguous"),
            Some(&1)
        );
        Python::attach(|py| {
            assert_eq!(
                error.value(py).to_string(),
                REASON,
                "the helper altered the reason key. Those bytes become the \
                 `world_failure_reasons` key, are compared across eras, and are \
                 truncated at 512 chars by `_bounded_reason_detail` -- the counts ride \
                 on an ATTRIBUTE precisely so the message never has to change"
            );
            assert!(
                error.is_instance_of::<PyValueError>(py),
                "the helper changed the exception type; a BaseException subclass slips \
                 past `except Exception` and kills the shard"
            );
        });

        let clean: PyResult<String> = super::guarded_search_with_ledger(|ledger| {
            ledger.record("sleeptalk_called_unidentified:ambiguous");
            Ok(ledger.json_object())
        });
        assert_eq!(
            clean.expect("the search succeeded"),
            "{\"sleeptalk_called_unidentified:ambiguous\":1}",
            "the clean path must still report through the ledger it accumulated into"
        );
    }
}
