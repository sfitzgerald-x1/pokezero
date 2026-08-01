//! Contain poke-engine's panics at the FFI boundary.
//!
//! Deliberately NOT behind the `model` feature. The guard is not
//! model-specific, and gating it would make its tests require libtorch --
//! so the one piece of code whose job is to keep a shard alive would be the
//! piece nobody can compile or test without a GPU build.

use pyo3::exceptions::PyValueError;

/// Run `f`, converting a Rust panic into a catchable Python error.
///
/// poke-engine validates aggressively and panics on a state it considers
/// impossible. Across the pyo3 boundary that becomes `PanicException`, which
/// derives from `BaseException` and therefore slips past the caller's
/// `except Exception` handler whose whole job is to count a bad world and keep
/// the others. One such panic then kills the shard process rather than the
/// world, and the paired driver -- correctly -- refuses to write a partial
/// shard, so the cell silently loses pairs.
///
/// The panic payload is preserved in the message so the reason still lands in
/// `world_failure_reasons` and stays visible in fallback telemetry. Containing
/// it must not hide it.
///
/// `lib.rs::parse_state` applies the same remedy to deserialization; this is
/// the search path.
pub fn catch_native_panic<T>(
    f: impl FnOnce() -> pyo3::PyResult<T>,
) -> pyo3::PyResult<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => Err(PyValueError::new_err(panic_detail(payload.as_ref()))),
    }
}

/// Render a panic payload as the message the world-failure reason will carry.
///
/// Split out so the preservation of the original text is testable without a
/// Python interpreter: containing a panic must not hide WHICH panic it was, or
/// `world_failure_reasons` degrades into an undifferentiated count.
pub fn panic_detail(payload: &(dyn std::any::Any + Send)) -> String {
    let detail = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&'static str>().copied())
        .unwrap_or("unknown panic payload");
    format!("native search panicked and the world was aborted: {detail}")
}

#[cfg(test)]
mod panic_containment_tests {
    use super::panic_detail;

    /// A `panic!("literal")` payload is a `&'static str`.
    #[test]
    fn static_str_payload_is_preserved() {
        let payload = std::panic::catch_unwind(|| panic!("Invalid rest_turns value: 32"))
            .expect_err("must panic");
        let detail = panic_detail(payload.as_ref());
        assert!(detail.contains("Invalid rest_turns value: 32"), "{detail}");
        assert!(detail.contains("world was aborted"), "{detail}");
    }

    /// A formatted `panic!("{}", ..)` payload is a `String`; the other arm.
    #[test]
    fn string_payload_is_preserved() {
        let value = 32;
        let payload = std::panic::catch_unwind(|| panic!("Invalid rest_turns value: {value}"))
            .expect_err("must panic");
        let detail = panic_detail(payload.as_ref());
        assert!(detail.contains("Invalid rest_turns value: 32"), "{detail}");
    }

    /// An unrecognised payload must still produce a message rather than panic
    /// inside the handler.
    #[test]
    fn unknown_payload_still_reports() {
        let payload =
            std::panic::catch_unwind(|| std::panic::panic_any(7u8)).expect_err("must panic");
        let detail = panic_detail(payload.as_ref());
        assert!(detail.contains("unknown panic payload"), "{detail}");
    }
}
