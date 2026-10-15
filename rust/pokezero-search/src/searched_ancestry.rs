//! Private actual-path retention shared by prospective exports and the old
//! excluded fixture callback. No engine labels, replay or backup is performed.
use std::collections::{BTreeMap, BTreeSet};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub(crate) fn capture_expired(fixed_work: bool, deadline_expired: bool) -> bool {
    !fixed_work && deadline_expired
}

pub(crate) fn sample_priority(seed: u64, worker: usize, root: &str, ordinal: usize) -> u64 {
    let bytes = Sha256::digest(serde_json::to_vec(&json!([
        "visited-model-sample.v1", seed, worker, root, ordinal])).expect("finite sampling identity"));
    u64::from_be_bytes(bytes[..8].try_into().unwrap())
}

#[derive(Default)]
pub(crate) struct SearchedLines {
    pub(crate) root: String,
    pub(crate) steps: BTreeMap<(usize, usize), Value>,
    pub(crate) bytes: usize,
}

impl SearchedLines {
    pub(crate) fn path(&self, key: (usize, usize)) -> PyResult<Vec<Value>> {
        let mut cursor = Some(key);
        let mut reverse = Vec::new();
        let mut seen = BTreeSet::new();
        while let Some(key) = cursor {
            if reverse.len() >= 64 || !seen.insert(key) {
                return Err(PyValueError::new_err("searched line: ancestry cap/cycle"));
            }
            let step = self.steps.get(&key)
                .ok_or_else(|| PyValueError::new_err("searched line: missing actual ancestor"))?;
            let actual: (usize, usize) = serde_json::from_value(step["key"].clone())
                .map_err(|_| PyValueError::new_err("searched line: invalid branch identity"))?;
            if actual != key { return Err(PyValueError::new_err("searched line: branch key drift")); }
            cursor = serde_json::from_value(step["parent"].clone())
                .map_err(|_| PyValueError::new_err("searched line: invalid parent identity"))?;
            reverse.push(step.clone());
        }
        reverse.reverse();
        let mut expected = self.root.as_str();
        for step in &reverse {
            if step["before"].as_str() != Some(expected) {
                return Err(PyValueError::new_err("searched line: actual ancestry discontinuity"));
            }
            expected = step["after"].as_str()
                .ok_or_else(|| PyValueError::new_err("searched line: missing actual successor"))?;
        }
        Ok(reverse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn lines() -> SearchedLines {
        let mut lines = SearchedLines { root: "root".into(), ..Default::default() };
        lines.steps.insert((0, 0), json!({"key":[0,0],"parent":null,"before":"root","after":"a"}));
        lines.steps.insert((1, 0), json!({"key":[1,0],"parent":[0,0],"before":"a","after":"b"}));
        lines
    }
    #[test]
    fn actual_path_preserves_every_ancestor() {
        let path = lines().path((1, 0)).unwrap();
        assert_eq!(path.len(), 2);
        assert_eq!(path[0]["before"], "root");
        assert_eq!(path[1]["after"], "b");
    }
    #[test]
    fn missing_cycle_key_and_continuity_refuse() {
        let mut bank = lines();
        assert!(bank.path((9, 9)).is_err());
        bank.steps.get_mut(&(0, 0)).unwrap()["parent"] = json!([1,0]);
        assert!(bank.path((1, 0)).is_err());
        let mut bank = lines();
        bank.steps.get_mut(&(1, 0)).unwrap()["key"] = json!([2,0]);
        assert!(bank.path((1, 0)).is_err());
        let mut bank = lines();
        bank.steps.get_mut(&(1, 0)).unwrap()["before"] = json!("substitute");
        assert!(bank.path((1, 0)).is_err());
    }
    #[test]
    fn priority_matches_frozen_python_law() {
        assert_eq!(sample_priority(17, 0, "aa", 3), 8939513739264610312);
    }
    #[test]
    fn fixed_capture_never_adds_a_decision_deadline() {
        assert!(!capture_expired(true, true));
        assert!(!capture_expired(true, false));
        assert!(capture_expired(false, true));
        assert!(!capture_expired(false, false));
    }
}
