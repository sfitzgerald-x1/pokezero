//! JSON boundary for the explicitly built conditional continuation oracle.
//! No sampling, action selection, scoring, model, or ordinary-search caller.
use poke_engine::engine::conditioned_resume::{resume, resume_joint, ResumeCapsule};
use poke_engine::state::PokemonIndex;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyInt, PyList, PyTuple};
use serde_json::{json, Value};

const FIELDS: [&str; 15] = [
    "version",
    "pending_residual",
    "forced_switch",
    "baton_pass",
    "queued_actor",
    "queued_actions",
    "queued_stall",
    "residual_draws",
    "phazing_draws",
    "secondary_draws",
    "sleep_talk_draws",
    "replacement_count",
    "turn",
    "keys",
    "schema",
];

fn capsule_from_json(text: &str) -> Result<ResumeCapsule, String> {
    let value: Value = serde_json::from_str(text).map_err(|e| format!("CAPSULE_JSON: {e}"))?;
    let object = value.as_object().ok_or("CAPSULE_OBJECT_REQUIRED")?;
    if object.len() != FIELDS.len() || object.keys().any(|k| !FIELDS.contains(&k.as_str())) {
        return Err("CAPSULE_FIELDS_MISMATCH".into());
    }
    if value["version"] != json!(1) || value["schema"] != json!("pokezero.suspended-turn") {
        return Err("CAPSULE_SCHEMA_MISMATCH".into());
    }
    macro_rules! field {
        ($name:literal) => {
            serde_json::from_value(value[$name].clone())
                .map_err(|e| format!("CAPSULE_FIELD {}: {e}", $name))?
        };
    }
    Ok(ResumeCapsule {
        pending_residual: field!("pending_residual"),
        forced_switch: field!("forced_switch"),
        baton_pass: field!("baton_pass"),
        queued_actor: field!("queued_actor"),
        queued_actions: field!("queued_actions"),
        queued_stall: field!("queued_stall"),
        residual_draws: field!("residual_draws"),
        phazing_draws: field!("phazing_draws"),
        secondary_draws: field!("secondary_draws"),
        sleep_talk_draws: field!("sleep_talk_draws"),
        replacement_count: field!("replacement_count"),
        turn: field!("turn"),
        keys: field!("keys"),
    })
}

fn capsule_json(c: &ResumeCapsule) -> Value {
    json!({"version":1,"schema":"pokezero.suspended-turn",
        "pending_residual":c.pending_residual,"forced_switch":c.forced_switch,
        "baton_pass":c.baton_pass,"queued_actor":c.queued_actor,
        "queued_actions":c.queued_actions,"queued_stall":c.queued_stall,
        "residual_draws":c.residual_draws,"phazing_draws":c.phazing_draws,
        "secondary_draws":c.secondary_draws,"sleep_talk_draws":c.sleep_talk_draws,
        "replacement_count":c.replacement_count,"turn":c.turn,"keys":c.keys})
}

/// Explicit replacement index avoids species-name or duplicate-move aliases.
/// New ordinary draws are deliberately absent: the capsule owns those draws.
/// Only replacement-entry ties are fresh, and the current domain proves them
/// inactive. Unsupported mechanics refuse before native generation.
#[pyfunction]
pub fn conditioned_resume(
    state_str: &str,
    capsule: &str,
    replacement: u8,
    entry_ties: [u8; 2],
) -> PyResult<String> {
    let replacement = match replacement {
        0 => PokemonIndex::P0,
        1 => PokemonIndex::P1,
        2 => PokemonIndex::P2,
        3 => PokemonIndex::P3,
        4 => PokemonIndex::P4,
        5 => PokemonIndex::P5,
        _ => return Err(PyValueError::new_err("INVALID_REPLACEMENT_INDEX")),
    };
    let capsule = capsule_from_json(capsule).map_err(PyValueError::new_err)?;
    crate::panic_guard::catch_native_panic(|| {
        let state = crate::parse_state(state_str)?;
        let branches =
            resume(&state, &capsule, replacement, entry_ties).map_err(PyValueError::new_err)?;
        let result: Vec<_> = branches
            .into_iter()
            .map(|b| {
                json!({"percentage":b.percentage,"state":b.state.serialize(),
                "capsule":capsule_json(&b.capsule),
                "instructions":format!("{:?}", b.instructions.instruction_list)})
            })
            .collect();
        Ok(json!({"version":1,"branches":result}).to_string())
    })
}

#[pyfunction]
pub fn conditioned_resume_joint(
    state_str: &str,
    capsule: &str,
    replacements: &Bound<'_, PyAny>,
    entry_ties: &Bound<'_, PyAny>,
) -> PyResult<String> {
    fn exact_pair(raw: &Bound<'_, PyAny>, optional: bool) -> PyResult<[Option<u8>; 2]> {
        if !(raw.is_instance_of::<PyTuple>() || raw.is_instance_of::<PyList>()) || raw.len()? != 2 {
            return Err(PyValueError::new_err("EXACT_TWO_COORDINATES_REQUIRED"));
        }
        let coordinate = |index| -> PyResult<Option<u8>> {
            let value = raw.get_item(index)?;
            if optional && value.is_none() { return Ok(None); }
            if !value.is_exact_instance_of::<PyInt>() {
                return Err(PyValueError::new_err("EXACT_INTEGER_COORDINATE_REQUIRED"));
            }
            Ok(Some(value.extract::<u8>()?))
        };
        Ok([coordinate(0)?, coordinate(1)?])
    }
    let replacements = exact_pair(replacements, true)?;
    let ties = exact_pair(entry_ties, false)?;
    let entry_ties = [ties[0].unwrap(), ties[1].unwrap()];
    let convert = |value: Option<u8>| -> PyResult<Option<PokemonIndex>> {
        value.map(|index| match index {
            0 => Ok(PokemonIndex::P0), 1 => Ok(PokemonIndex::P1),
            2 => Ok(PokemonIndex::P2), 3 => Ok(PokemonIndex::P3),
            4 => Ok(PokemonIndex::P4), 5 => Ok(PokemonIndex::P5),
            _ => Err(PyValueError::new_err("INVALID_REPLACEMENT_INDEX")),
        }).transpose()
    };
    let choices = [convert(replacements[0])?, convert(replacements[1])?];
    let capsule = capsule_from_json(capsule).map_err(PyValueError::new_err)?;
    crate::panic_guard::catch_native_panic(|| {
        let state = crate::parse_state(state_str)?;
        let branches = resume_joint(&state, &capsule, choices, entry_ties)
            .map_err(PyValueError::new_err)?;
        let result: Vec<_> = branches.into_iter().map(|b| {
            json!({"percentage":b.percentage,"state":b.state.serialize(),
                "capsule":capsule_json(&b.capsule),
                "instructions":format!("{:?}", b.instructions.instruction_list)})
        }).collect();
        Ok(json!({"version":1,"branches":result}).to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn capsule() -> ResumeCapsule {
        ResumeCapsule {
            pending_residual: true,
            forced_switch: [true, false],
            baton_pass: [true, false],
            queued_actor: 1,
            queued_actions: [0, 0],
            queued_stall: [0, 0],
            residual_draws: [[0; 12]; 2],
            phazing_draws: [0; 2],
            secondary_draws: [0; 2],
            sleep_talk_draws: [0; 2],
            replacement_count: 0,
            turn: 4,
            keys: [123, 456],
        }
    }
    #[test]
    fn complete_capsule_roundtrips_and_unknown_or_missing_fields_refuse() {
        let c = capsule();
        let value = capsule_json(&c);
        assert_eq!(capsule_from_json(&value.to_string()).unwrap(), c);
        let mut missing = value.clone();
        missing.as_object_mut().unwrap().remove("phazing_draws");
        assert!(capsule_from_json(&missing.to_string()).is_err());
        let mut extra = value.clone();
        extra["new_draws"] = serde_json::to_value([[0; 12]; 2]).unwrap();
        assert!(capsule_from_json(&extra.to_string()).is_err());
        let mut bad = value;
        bad["secondary_draws"] = json!([false, 10]);
        assert!(capsule_from_json(&bad.to_string()).is_err());
    }
}
