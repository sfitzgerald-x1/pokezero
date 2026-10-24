//! Structured origin evidence attached BEFORE a branch frame unwinds.
//! Python's shared quarantine writer owns persistence and run context.
use pyo3::exceptions::PyBaseException;
use pyo3::prelude::*;
use poke_engine::state::State;
use serde_json::json;

fn capture_loss(message: String) -> PyErr {
    let error = PyBaseException::new_err(message);
    Python::attach(|py| {
        // A structured signal, not a message substring that a scientific
        // contract failure could accidentally imitate.
        error.value(py).setattr("pokezero_capture_failed", true)
            .expect("fresh built-in BaseException supports exception attributes");
    });
    error
}

/// Keep origin creation outside the panicking frame, and materialize it only
/// on error. A more local attached witness always wins over this backstop.
pub(crate) fn guarded<T>(origin: impl FnOnce() -> serde_json::Value,
                        operation: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    crate::panic_guard::catch_native_panic(operation).map_err(|error| {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(origin)) {
            Ok(witness) => attach(error, witness),
            Err(_) => capture_loss(format!(
                "capture.failed constructing native origin; original={error}"
            )),
        }
    })
}

pub(crate) fn attach(error: PyErr, witness: serde_json::Value) -> PyErr {
    Python::attach(|py| {
        // Never replace a more local origin with a less specific outer frame.
        if error.value(py).hasattr("pokezero_failure_witness").unwrap_or(false) {
            return error;
        }
        if let Err(capture_error) = error.value(py).setattr("pokezero_failure_witness", witness.to_string()) {
            return capture_loss(format!(
                "capture.failed attaching native origin; original={error}; capture={capture_error}"
            ));
        }
        error
    })
}

pub(crate) fn attach_events(error: PyErr, events: &[serde_json::Value]) -> PyErr {
    if events.is_empty() { return error; }
    Python::attach(|py| {
        match error.value(py).setattr("pokezero_refusal_events", serde_json::to_string(events).unwrap()) {
            Ok(()) => error,
            Err(failure) => capture_loss(format!(
                "capture.failed attaching prior events; original={error}; capture={failure}")),
        }
    })
}

pub(crate) fn branch_origin(leaf: &State, seam: &crate::tree::BranchSeam<'_>, codes: &[String]) -> serde_json::Value {
    let mut before = leaf.clone();
    before.reverse_instructions(&seam.instructions.to_vec());
    json!({
        "schema": "pokezero.native-failure-origin.v1",
        "reason_codes": codes,
        "engine_state": before.serialize(), "post_state": leaf.serialize(),
        "instruction_list": seam.instructions.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>(),
        "offending_instruction": {"availability": "unavailable", "reason": "branch-level contract failure; no unique instruction identified"},
        "s1": format!("{:?}", seam.s1), "s2": format!("{:?}", seam.s2),
        "branch_index": seam.branch_index, "chance_node": seam.chance, "depth": seam.depth,
    })
}

pub(crate) fn branch_error(error: PyErr, leaf: &State, seam: &crate::tree::BranchSeam<'_>, code: &str) -> PyErr {
    attach(error, branch_origin(leaf, seam, &[code.to_string()]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::PyValueError;
    #[test]
    fn lost_origin_is_not_an_ordinary_catch_and_continue_error() {
        Python::initialize();
        let result: PyResult<()> = guarded(|| panic!("authored origin loss"),
            || Err(PyValueError::new_err("original failure")));
        Python::attach(|py| {
            let error = result.unwrap_err();
            assert!(error.is_instance_of::<PyBaseException>(py));
            assert!(!error.is_instance_of::<PyValueError>(py));
            assert!(error.to_string().contains("capture.failed"));
            assert!(error.to_string().contains("original failure"));
            assert!(error.value(py).getattr("pokezero_capture_failed").unwrap().extract::<bool>().unwrap());
        });
    }
    #[test]
    fn native_origin_survives_without_changing_error_message_or_more_local_evidence() {
        Python::initialize();
        let error = attach(PyValueError::new_err("authored contract"), json!({"reason_codes":["crate.authored"], "engine_state":"serialized"}));
        let error = attach(error, json!({"reason_codes":["outer.generic"]}));
        Python::attach(|py| {
            assert_eq!(error.value(py).to_string(), "authored contract");
            let witness: String = error.value(py).getattr("pokezero_failure_witness").unwrap().extract().unwrap();
            assert!(witness.contains("crate.authored"));
            assert!(witness.contains("serialized"));
            assert!(!witness.contains("outer.generic"));
        });
    }
}
