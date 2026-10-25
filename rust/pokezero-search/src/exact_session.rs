//! Bounded one-ply transport. Root parsed once; no success-path leaf JSON,
//! post-state serialization, summaries, or instruction debug formatting.
//! Keep P2's rendered-line metadata law, not incumbent typed-status metadata.
use crate::encoder::{encoded_to_dict, EncodedArrays};
use crate::events::{move_choice_from_str, render_branch_events, EventContext};
use crate::fold::{FoldStateInner, PyFoldState};
use crate::leaf::PyLeafEncoder;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::instruction::StateInstructions;
use poke_engine::pokemon::PokemonName;
use poke_engine::state::{SideReference, State};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::{json, Value};

fn err(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hp_attestation_ignores_padding_and_refuses_empty_or_negative_teams() {
        Python::initialize();
        let mut state=State::default();
        assert!(terminal(&state).is_err()); // NONE padding has HP100.
        state.side_one.pokemon.pkmn[0].id=PokemonName::SNORLAX;
        state.side_two.pokemon.pkmn[0].id=PokemonName::BLISSEY;
        assert_eq!(terminal(&state).unwrap(),None);
        state.side_one.pokemon.pkmn[0].hp=0;
        assert_eq!(terminal(&state).unwrap(),Some(-1.));
        state.side_two.pokemon.pkmn[0].hp=0;
        assert_eq!(terminal(&state).unwrap(),Some(0.));
        state.side_one.pokemon.pkmn[0].hp=100;
        assert_eq!(terminal(&state).unwrap(),Some(1.));
        state.side_two.pokemon.pkmn[0].hp=-1;
        assert!(terminal(&state).is_err());
    }
}

fn terminal(state: &State) -> PyResult<Option<f64>> {
    // Exactly the old post_state_summary -> outcomes HP attestation: NONE
    // padding is not a party member, and empty/negative-HP teams refuse.
    let mut exhausted = [true, true];
    for (index, side) in [&state.side_one, &state.side_two].iter().enumerate() {
        let mut count = 0;
        for p in side
            .pokemon
            .into_iter()
            .filter(|p| p.id != PokemonName::NONE)
        {
            count += 1;
            if p.hp < 0 {
                return Err(err("native outcome lacks complete nonnegative team HP"));
            }
            exhausted[index] &= p.hp == 0;
        }
        if count == 0 {
            return Err(err("native outcome lacks complete nonnegative team HP"));
        }
    }
    Ok(match exhausted {
        [true, true] => Some(0.0),
        [true, false] => Some(-1.0),
        [false, true] => Some(1.0),
        _ => None,
    })
}

struct Branch {
    instructions: StateInstructions,
    lines: Vec<String>,
    turn: i64,
    terminal: Option<f64>,
}
struct Cell {
    row: String,
    column: String,
    branches: Vec<Branch>,
}

/// Handles are valid only within this immutable-root session. Python owns the
/// world ordinal and session identity; native bounds-checks every cell/branch.
#[pyclass(module = "pokezero_search")]
pub struct ExactSession {
    root: State,
    fold: FoldStateInner,
    context: EventContext,
    turn: i64,
    max_branches: usize,
    max_cells: usize,
    retained_branches: usize,
    cells: Vec<Cell>,
    encoder: Py<PyLeafEncoder>,
}

impl ExactSession {
    fn origin(&self, cell: usize, branch: usize, code: &str) -> Value {
        let c = &self.cells[cell];
        let b = &c.branches[branch];
        json!({"schema":"pokezero.native-failure-origin.v1", "reason_codes":[code],
            "engine_state":self.root.serialize(), "row_action":c.row, "column_action":c.column,
            "cell_index":cell, "branch_index":branch, "branch_count":c.branches.len(),
            "instruction_list":b.instructions.instruction_list.iter().map(|i|format!("{i:?}")).collect::<Vec<_>>(),
            "lines":b.lines, "turn":b.turn,
            "offending_instruction":{"availability":"unavailable","reason":"branch-level contract; no unique instruction cursor"}})
    }

    fn encode_rows(
        &self,
        encoder: &PyLeafEncoder,
        addresses: &[(usize, usize)],
        workers: usize,
    ) -> PyResult<EncodedArrays> {
        let chunk_size = addresses.len().div_ceil(workers.min(addresses.len()));
        let groups = std::thread::scope(|scope| {
            let handles: Vec<_> = addresses
                .chunks(chunk_size)
                .enumerate()
                .map(|(chunk_index, chunk)| {
                    scope.spawn(move || {
                        let mut state = self.root.clone();
                        let mut output = Vec::new();
                        for (offset, &(cell, index)) in chunk.iter().enumerate() {
                            let branch = &self.cells[cell].branches[index];
                            let row = crate::failure_capture::guarded(
                                || {
                                    let mut v = self.origin(cell, index, "crate.exact_leaf_encode");
                                    v["batch_row"] = json!(chunk_index * chunk_size + offset);
                                    v
                                },
                                || {
                                    // Each sibling begins with precisely the root fold.
                                    let mut fold = self.fold.clone();
                                    fold.advance_in_place(&branch.lines)?;
                                    let (order, meta) = encoder.branch_context(Some(&branch.lines));
                                    state.apply_instructions(&branch.instructions.instruction_list);
                                    let result = encoder.ctx.encode_leaf(
                                        &state,
                                        &fold,
                                        branch.turn,
                                        order.as_deref(),
                                        meta.as_ref(),
                                        false,
                                    );
                                    state.reverse_instructions(
                                        &branch.instructions.instruction_list,
                                    );
                                    result
                                },
                            )?;
                            output.push(row);
                        }
                        Ok::<_, PyErr>(output)
                    })
                })
                .collect();
            let mut groups = Vec::new();
            let mut failure = None;
            // Every worker joins. Deterministic first input-order error; no
            // prefix is returned and no surviving worker mutates shared state.
            for handle in handles {
                match handle.join() {
                    Ok(Ok(rows)) => groups.push(rows),
                    Ok(Err(e)) => {
                        if failure.is_none() {
                            failure = Some(e)
                        }
                    }
                    Err(_) => {
                        if failure.is_none() {
                            failure = Some(err("exact encode worker panic; no partial batch"))
                        }
                    }
                }
            }
            match failure {
                Some(e) => Err(e),
                None => Ok(groups),
            }
        })?;
        let mut packed = EncodedArrays {
            categorical: Vec::new(),
            numeric: Vec::new(),
            token_types: Vec::new(),
            attention: Vec::new(),
            legal: Vec::new(),
        };
        let mut widths = None;
        for row in groups.into_iter().flatten() {
            let shape = (
                row.categorical.len(),
                row.numeric.len(),
                row.token_types.len(),
                row.attention.len(),
                row.legal.len(),
            );
            if widths.is_some_and(|w| w != shape) {
                return Err(err("exact row shapes differ"));
            }
            widths = Some(shape);
            packed.categorical.extend(row.categorical);
            packed.numeric.extend(row.numeric);
            packed.token_types.extend(row.token_types);
            packed.attention.extend(row.attention);
            packed.legal.extend(row.legal);
        }
        Ok(packed)
    }
}

#[pymethods]
impl ExactSession {
    #[new]
    fn new(
        state_str: &str,
        fold: &PyFoldState,
        context_json: &str,
        turn: i64,
        max_branches: usize,
        max_cells: usize,
        encoder: Py<PyLeafEncoder>,
    ) -> PyResult<Self> {
        if max_branches == 0 || max_cells == 0 || max_branches > 2_000_000 || max_cells > 10_000 {
            return Err(err(
                "exact session requires bounded positive branch/cell limits",
            ));
        }
        Ok(Self {
            root: crate::parse_state(state_str)?,
            fold: fold.inner().clone(),
            context: EventContext::from_json(context_json).map_err(err)?,
            turn,
            max_branches,
            max_cells,
            retained_branches: 0,
            cells: Vec::new(),
            encoder,
        })
    }

    fn root_terminal(&self) -> PyResult<Option<f64>> {
        terminal(&self.root)
    }

    fn root_actions(&self) -> (Vec<String>, Vec<String>) {
        let (one, two) = self.root.root_get_all_options();
        (
            one.iter()
                .map(|m| crate::move_display(&self.root.side_one, m))
                .collect(),
            two.iter()
                .map(|m| crate::move_display(&self.root.side_two, m))
                .collect(),
        )
    }

    /// Full strict fan, in engine order. Only scalar mass/terminal/address
    /// data crosses Python. Fold/encode is deferred to bounded model batches.
    fn enumerate_cell(
        &mut self,
        row: &str,
        column: &str,
    ) -> PyResult<(usize, Vec<(f64, Option<f64>)>, Vec<String>)> {
        if self.cells.len() >= self.max_cells {
            return Err(err("exact session cell cap"));
        }
        let mut state = self.root.clone();
        let one = move_choice_from_str(row, &state, SideReference::SideOne)?;
        let two = move_choice_from_str(column, &state, SideReference::SideTwo)?;
        let generated = crate::failure_capture::guarded(
            || {
                json!({
                    "schema":"pokezero.native-failure-origin.v1","reason_codes":["crate.transition"],
                    "engine_state":self.root.serialize(),"row_action":row,"column_action":column,
                    "instruction_list":{"availability":"unavailable","reason":"generator failed before returning branches"}
                })
            },
            || {
                Ok(generate_instructions_from_move_pair(
                    &mut state, &one, &two, true,
                ))
            },
        )?;
        if generated.is_empty() || generated.len() > self.max_branches - self.retained_branches {
            return Err(err("exact session empty fan or retained branch cap"));
        }
        let mut branches = Vec::new();
        let mut report = Vec::new();
        let mut subcases = Vec::new();
        let branch_count = generated.len();
        for (index, instructions) in generated.into_iter().enumerate() {
            let (rendered, value) = crate::failure_capture::guarded(
                || {
                    json!({
                        "schema":"pokezero.native-failure-origin.v1","reason_codes":["crate.exact_transition_branch"],
                        "engine_state":self.root.serialize(),"row_action":row,"column_action":column,"branch_index":index,
                        "branch_count":branch_count,
                        "instruction_list":instructions.instruction_list.iter().map(|i|format!("{i:?}")).collect::<Vec<_>>()
                    })
                },
                || {
                    let rendered = render_branch_events(
                        &mut state,
                        &one,
                        &two,
                        &instructions.instruction_list,
                        true,
                        &self.context,
                    );
                    if !rendered.lossy.is_empty() || !rendered.attribution_unsafe.is_empty() {
                        let codes: Vec<_> = rendered
                            .lossy
                            .iter()
                            .chain(&rendered.attribution_unsafe)
                            .chain(&rendered.lossy_subcases)
                            .map(|r| format!("renderer.{r}"))
                            .collect();
                        return Err(crate::failure_capture::attach(
                            err("native event rendering is not lossless for P2"),
                            json!({
                                "schema":"pokezero.native-failure-origin.v1","reason_codes":codes,
                                "engine_state":self.root.serialize(),"row_action":row,"column_action":column,"branch_index":index,
                                "instruction_list":instructions.instruction_list.iter().map(|i|format!("{i:?}")).collect::<Vec<_>>(),
                                "events":rendered.lines,"lossy":rendered.lossy,"lossy_subcases":rendered.lossy_subcases,
                                "attribution_unsafe_reasons":rendered.attribution_unsafe,
                                "offending_instruction":{"availability":"unavailable","reason":"branch-level refusal"}
                            }),
                        ));
                    }
                    state.apply_instructions(&instructions.instruction_list);
                    let value = terminal(&state);
                    state.reverse_instructions(&instructions.instruction_list);
                    Ok((rendered, value?))
                },
            )?;
            report.push((f64::from(instructions.percentage) / 100.0, value));
            subcases.extend(rendered.lossy_subcases);
            branches.push(Branch {
                instructions,
                lines: rendered.lines,
                turn: self.turn + i64::from(rendered.turn_completed),
                terminal: value,
            });
        }
        let index = self.cells.len();
        self.retained_branches += branches.len();
        self.cells.push(Cell {
            row: row.to_string(),
            column: column.to_string(),
            branches,
        });
        Ok((index, report, subcases))
    }

    fn encode_packed(
        &self,
        py: Python<'_>,
        addresses: Vec<(usize, usize)>,
        workers: usize,
    ) -> PyResult<Py<PyDict>> {
        if addresses.is_empty() || addresses.len() > 4096 || workers == 0 || workers > 32 {
            return Err(err(
                "exact encoding requires 1..4096 rows and 1..32 workers",
            ));
        }
        for &(cell, index) in &addresses {
            let branch = self
                .cells
                .get(cell)
                .and_then(|c| c.branches.get(index))
                .ok_or_else(|| err("exact batch address outside session"))?;
            if branch.terminal.is_some() {
                return Err(err("terminal branch is not a model leaf"));
            }
        }
        let borrowed = self.encoder.borrow(py);
        let encoder: &PyLeafEncoder = &borrowed;
        let packed = py.detach(|| self.encode_rows(encoder, &addresses, workers))?;
        let out = encoded_to_dict(py, &packed)?;
        out.bind(py).set_item("batch_rows", addresses.len())?;
        Ok(out)
    }

    /// Failure-only reconstruction. No scientific replay, action, or scoring.
    fn reconstruct(&self, cell: usize, branch: usize) -> PyResult<String> {
        let b = self
            .cells
            .get(cell)
            .and_then(|c| c.branches.get(branch))
            .ok_or_else(|| err("reconstruction address outside session"))?;
        let mut witness = self.origin(cell, branch, "crate.exact_diagnostic");
        let mut state = self.root.clone();
        state.apply_instructions(&b.instructions.instruction_list);
        witness["post_state"] = json!(state.serialize());
        Ok(witness.to_string())
    }
}
