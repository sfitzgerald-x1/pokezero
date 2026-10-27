//! Strict JSON boundary for the opt-in ordinary fixed-chance oracle.
use poke_engine::engine::conditioned_step::{requirements, step, BankOrigin, StepAddress, StepChance};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{json,Value};

fn fields(value: &Value, expected: &[&str]) -> Result<(),String> {
    let o=value.as_object().ok_or("STEP_OBJECT_REQUIRED")?;
    if o.len()!=expected.len() || o.keys().any(|k| !expected.contains(&k.as_str())) {
        return Err("STEP_FIELDS_MISMATCH".into());
    }
    Ok(())
}

fn request(text: &str) -> Result<([u8;2],StepChance,StepAddress),String> {
    let v:Value=serde_json::from_str(text).map_err(|e| format!("STEP_JSON: {e}"))?;
    fields(&v,&["version","schema","actions","chance","address"])?;
    if v["version"]!=json!(1) || v["schema"]!=json!("pokezero.ordinary-step") {
        return Err("STEP_SCHEMA_MISMATCH".into());
    }
    fields(&v["chance"],&["draws","phazing","secondary","sleep_talk","entry_ties","initial_tie"])?;
    fields(&v["address"],&["turn","keys","next_keys"])?;
    macro_rules! field { ($v:expr,$name:literal) => {
        serde_json::from_value($v[$name].clone()).map_err(|e| format!("STEP_FIELD {}: {e}",$name))?
    }; }
    Ok((field!(v,"actions"),StepChance {draws:field!(v["chance"],"draws"),
        phazing:field!(v["chance"],"phazing"),secondary:field!(v["chance"],"secondary"),
        sleep_talk:field!(v["chance"],"sleep_talk"),entry_ties:field!(v["chance"],"entry_ties"),
        initial_tie:field!(v["chance"],"initial_tie")},
        StepAddress {turn:field!(v["address"],"turn"),keys:field!(v["address"],"keys"),
            next_keys:field!(v["address"],"next_keys")}))
}

#[pyfunction]
pub fn conditioned_step_requirements(state_str: &str, actions: &str) -> PyResult<String> {
    let actions:[u8;2]=serde_json::from_str(actions).map_err(|e| PyValueError::new_err(format!("STEP_ACTIONS: {e}")))?;
    crate::panic_guard::catch_native_panic(|| {
        let state=crate::parse_state(state_str)?;
        let cuts=requirements(&state,actions).map_err(PyValueError::new_err)?;
        let rows:Vec<_>=cuts.into_iter().map(|c| json!({"coordinate":c.coordinate,"kind":c.kind,"argument":c.argument})).collect();
        Ok(json!({"version":1,"schema":"pokezero.native-step-consumers","requirements":rows}).to_string())
    })
}

#[pyfunction]
pub fn conditioned_step(state_str: &str, input: &str) -> PyResult<String> {
    let (actions,chance,address)=request(input).map_err(PyValueError::new_err)?;
    crate::panic_guard::catch_native_panic(|| {
        let state=crate::parse_state(state_str)?;
        let branches=step(&state,actions,&chance,&address).map_err(PyValueError::new_err)?;
        let rows:Vec<_>=branches.into_iter().map(|b| {
            let bank:Vec<_>=b.bank.into_iter().map(|c| match c.origin {
                BankOrigin::Copy(coordinate) => json!({"value":c.value,"copy":coordinate}),
                BankOrigin::Constant(value) => json!({"value":c.value,"constant":value}),
            }).collect();
            json!({"percentage":b.percentage,"state":b.state.serialize(),
                "capsule":crate::conditioned_resume::capsule_json(&b.capsule),"bank":bank,
                "instructions":format!("{:?}",b.instructions.instruction_list)})
        }).collect();
        Ok(json!({"version":1,"schema":"pokezero.ordinary-step-result","branches":rows}).to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strict_schema_rejects_boolean_integer_aliases_missing_and_extra_fields() {
        let v=json!({"version":1,"schema":"pokezero.ordinary-step","actions":[0,3],
            "chance":{"draws":vec![vec![0u8;12];2],"phazing":[0,59],"secondary":[0,99],
                "sleep_talk":[0,59],"entry_ties":[0,1],"initial_tie":0},
            "address":{"turn":7,"keys":[1,2],"next_keys":[3,4]}});
        assert!(request(&v.to_string()).is_ok());
        for path in ["actions","chance","address"] {
            let mut bad=v.clone();
            match path {
                "actions" => bad["actions"][0]=json!(true),
                "chance" => bad["chance"]["draws"][0][0]=json!(false),
                _ => bad["address"]["turn"]=json!(7.0),
            }
            assert!(request(&bad.to_string()).is_err());
        }
        let mut bad=v.clone();bad["chance"]["new_draw"]=json!(0);
        assert!(request(&bad.to_string()).is_err());
        let mut bad=v;bad["address"].as_object_mut().unwrap().remove("next_keys");
        assert!(request(&bad.to_string()).is_err());
    }
}
