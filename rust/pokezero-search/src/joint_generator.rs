//! Integer joint-team recurrence. No floating-point probability calculations.
use num_bigint::BigUint;
use num_traits::{One, Zero};
use pyo3::{exceptions::PyValueError, prelude::*};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, String>;
fn integer(v: &Value) -> Result<BigUint> {
    BigUint::parse_bytes(v.as_str().ok_or("expected decimal string")?.as_bytes(), 10)
        .ok_or("invalid integer".into())
}
fn small(v: &Value) -> Result<usize> {
    v.as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or("expected small integer".into())
}
fn array(v: &Value) -> Result<&Vec<Value>> {
    v.as_array().ok_or("expected array".into())
}
fn err(s: String) -> PyErr {
    PyValueError::new_err(s)
}
fn members(mut code: u64) -> Vec<usize> {
    let mut result = Vec::with_capacity(6);
    while code != 0 {
        result.push(((code & 255) - 1) as usize);
        code >>= 8;
    }
    result
}
fn append(code: u64, group: usize) -> u64 {
    let mut values = members(code);
    values.push(group);
    values.sort_unstable();
    values
        .iter()
        .enumerate()
        .fold(0, |acc, (i, g)| acc | ((*g as u64 + 1) << (8 * i)))
}

#[derive(Default)]
struct Legality {
    signature: String,
    masks: Vec<u64>,
    limits: [u64; 3],
    incompatible: Vec<u128>,
    blocked: HashMap<u64, LegalState>,
    hits: u64,
}
#[derive(Clone, Copy)]
struct LegalState {
    blocked: u128,
    valid: bool,
}
impl Legality {
    fn state(&mut self, code: u64) -> LegalState {
        if let Some(&value) = self.blocked.get(&code) {
            self.hits += 1;
            return value;
        }
        let mut bits = [0u64; 3];
        let mut incompatible = 0;
        let mut present = 0;
        for class in members(code) {
            let carry = bits[0] & self.masks[class];
            bits[0] ^= self.masks[class];
            let carry2 = bits[1] & carry;
            bits[1] ^= carry;
            bits[2] ^= carry2;
            incompatible |= self.incompatible[class];
            present |= 1u128 << class;
        }
        let saturated = ((bits[0] | bits[1] | bits[2]) & self.limits[0])
            | ((bits[1] | bits[2]) & self.limits[1])
            | (((bits[0] & bits[1]) | bits[2]) & self.limits[2]);
        let mut blocked = incompatible;
        for (i, mask) in self.masks.iter().enumerate() {
            if mask & saturated != 0 {
                blocked |= 1u128 << i;
            }
        }
        let over = ((bits[1] | bits[2]) & self.limits[0])
            | (((bits[0] & bits[1]) | bits[2]) & self.limits[1])
            | (bits[2] & self.limits[2]);
        let state = LegalState {
            blocked,
            valid: over == 0 && incompatible & present == 0,
        };
        if self.blocked.len() < 131_072 {
            self.blocked.insert(code, state);
        }
        state
    }
}

#[pyclass]
#[derive(Default)]
pub struct JointLegalityCache {
    inner: Arc<Mutex<Legality>>,
}
#[pymethods]
impl JointLegalityCache {
    #[new]
    fn new() -> Self {
        Self::default()
    }
    fn stats(&self) -> PyResult<(usize, u64)> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyValueError::new_err("legality cache poisoned"))?;
        Ok((guard.blocked.len(), guard.hits))
    }
}

#[derive(Clone)]
struct Group {
    size: usize,
    tokens: usize,
    class: usize,
    required: u8,
    fixed: Option<usize>,
    wobb: bool,
    // Each factor is Q * row probability/likelihood, before species token / total.
    rows: Vec<Vec<(u8, BigUint)>>,
    support: u16,
    upper: BigUint,
}
#[derive(Clone)]
struct Mass {
    lower: BigUint,
    upper: BigUint,
}
impl Mass {
    fn exact(value: BigUint) -> Self {
        Self {
            lower: value.clone(),
            upper: value,
        }
    }
}
#[derive(Clone)]
struct Branch {
    group: usize,
    context: u8,
    code: u64,
    flag: bool,
    factor: BigUint,
}

#[pyclass]
pub struct NativeJointGenerator {
    groups: Vec<Group>,
    slots: usize,
    required: u8,
    fixed: Vec<Option<usize>>,
    powers: Vec<BigUint>,
    terminal: [BigUint; 2],
    completable: bool,
    initial_flag: bool,
    legality: Arc<Mutex<Legality>>,
    cache: HashMap<u64, Mass>,
    nodes: usize,
    edges: u64,
    max_states: usize,
    deadline: Instant,
    cut: bool,
    solved: Option<Mass>,
}

impl NativeJointGenerator {
    fn from_json(payload: &str, shared: Arc<Mutex<Legality>>) -> Result<Self> {
        let v: Value = serde_json::from_str(payload).map_err(|e| e.to_string())?;
        let quantum = integer(&v["quantum"])?;
        if quantum.is_zero() {
            return Err("zero quantum".into());
        }
        let slots = small(&v["slots"])?;
        if !(1..=6).contains(&slots) {
            return Err("invalid slots".into());
        }
        let spec = &v["legality"];
        let signature = spec.to_string();
        let mut legal = shared.lock().map_err(|_| "legality cache poisoned")?;
        if legal.signature.is_empty() {
            let classes = array(&spec["masks"])?;
            if classes.len() > 128 {
                return Err("too many legality classes".into());
            }
            legal.masks = classes
                .iter()
                .map(|x| x.as_u64().ok_or("invalid cap mask".into()))
                .collect::<Result<_>>()?;
            let limits = array(&spec["limits"])?;
            if limits.len() != 3 {
                return Err("invalid cap limits".into());
            }
            legal.limits = [
                small(&limits[0])? as u64,
                small(&limits[1])? as u64,
                small(&limits[2])? as u64,
            ];
            legal.incompatible = array(&spec["incompatible"])?
                .iter()
                .map(|x| {
                    let mut mask = 0;
                    for j in array(x)? {
                        let k = small(j)?;
                        if k >= classes.len() {
                            return Err("invalid incompatible class".into());
                        }
                        mask |= 1u128 << k;
                    }
                    Ok(mask)
                })
                .collect::<Result<_>>()?;
            if legal.incompatible.len() != classes.len() {
                return Err("legality shape".into());
            }
            legal.signature = signature;
        } else if legal.signature != signature {
            return Err("legality cache catalog mismatch".into());
        }
        let classes = legal.masks.len();
        drop(legal);
        let mut required = 0;
        let mut groups = Vec::new();
        let mut denominators = HashSet::new();
        for g in array(&v["groups"])? {
            let class = small(&g["class"])?;
            if class >= classes {
                return Err("invalid group class".into());
            }
            let req = small(&g["required"])?;
            if req > 63 || (req != 0 && (req.count_ones() != 1 || required & req as u8 != 0)) {
                return Err("invalid required mask".into());
            }
            required |= req as u8;
            let rows = array(&g["rows"])?;
            if rows.len() != 16 {
                return Err("invalid context rows".into());
            }
            let mut converted = Vec::new();
            let mut support = 0;
            for (context, row) in rows.iter().enumerate() {
                let den = integer(&row[0])?;
                if den.is_zero() || &quantum % &den != BigUint::zero() {
                    return Err("inexact row denominator".into());
                }
                denominators.insert(den.clone());
                let mut outcomes = Vec::new();
                for item in array(&row[1])? {
                    let after = small(&item[0])?;
                    if after >= 16 || after & context != context {
                        return Err("invalid context transition".into());
                    }
                    let numerator = integer(&item[1])?;
                    if !numerator.is_zero() {
                        outcomes.push((after as u8, (&quantum / &den) * numerator));
                        support |= 1 << context;
                    }
                }
                converted.push(outcomes);
            }
            let size = small(&g["size"])?;
            let tokens = small(&g["tokens"])?;
            if size == 0 || size > 255 || !(1..=2).contains(&tokens) || (req != 0 && size != 1) {
                return Err("invalid group population".into());
            }
            let upper = integer(&g["upper"])?;
            for row in &converted {
                let sum: BigUint = row.iter().map(|(_, factor)| factor).sum();
                if (req == 0 && sum != quantum) || (req != 0 && sum > &quantum * &upper) {
                    return Err("invalid likelihood upper bound or unnormalized prior".into());
                }
            }
            groups.push(Group {
                size,
                tokens,
                class,
                required: req as u8,
                fixed: if g["fixed"].is_null() {
                    None
                } else {
                    Some(small(&g["fixed"])?)
                },
                wobb: g["wobb"].as_bool().ok_or("invalid wobb flag")?,
                rows: converted,
                support,
                upper,
            });
        }
        if groups.len() > 254 || required.count_ones() as usize > slots {
            return Err("too many groups/required members".into());
        }
        let mut fixed = vec![None; slots];
        for (i, g) in groups.iter().enumerate() {
            if let Some(slot) = g.fixed {
                if slot >= slots || fixed[slot].is_some() || g.required == 0 {
                    return Err("invalid fixed slot".into());
                }
                fixed[slot] = Some(i);
            }
        }
        let mut max_total = 0usize;
        for g in &groups {
            max_total = max_total
                .checked_add(g.size.checked_mul(g.tokens).ok_or("population overflow")?)
                .ok_or("population overflow")?;
        }
        if max_total > 512 {
            return Err("native token capacity exceeded".into());
        }
        for den in denominators {
            let base = &quantum / den;
            for n in 1..=max_total {
                if &base % n != BigUint::zero() {
                    return Err("quantum not divisible by token denominator".into());
                }
            }
        }
        let mut powers = vec![BigUint::one()];
        for _ in 0..slots {
            powers.push(powers.last().unwrap() * &quantum);
        }
        let terminal = [integer(&v["terminal"][0])?, integer(&v["terminal"][1])?];
        let initial_flag = v["initial_flag"].as_bool().ok_or("invalid initial flag")?;
        let completable = v["completable"]
            .as_bool()
            .ok_or("invalid completion flag")?;
        if completable {
            // Bind the finite completion proof to the native population and
            // legality data actually consumed, not a caller's model label.
            let mut populations = BTreeMap::new();
            for g in &groups {
                *populations.entry((g.class, g.tokens)).or_insert(0usize) += g.size;
            }
            let populations: Vec<_> = populations
                .into_iter()
                .map(|((c, t), n)| json!([c, t, n]))
                .collect();
            let certificate = json!({"legality":spec,"populations":populations,"slots":slots});
            let hash = format!("{:x}", Sha256::digest(certificate.to_string().as_bytes()));
            let expected = if initial_flag {
                "548feb08410d7b66b251014218e50a5608084a21afa2fe94bb23fa0fddfe1e57"
            } else {
                "88add4e1ddd5c9bbd8541f68b2c3180230e4e686fb24797e54dc99e998945a61"
            };
            if hash != expected {
                return Err("native completion certificate catalog mismatch".into());
            }
        }
        Ok(Self {
            groups,
            slots,
            required,
            fixed,
            powers,
            terminal,
            completable,
            initial_flag,
            legality: shared,
            cache: HashMap::new(),
            nodes: 0,
            edges: 0,
            max_states: 0,
            deadline: Instant::now(),
            cut: false,
            solved: None,
        })
    }
    fn selected(&self, code: u64) -> (Vec<usize>, u8, u64) {
        let chosen = members(code);
        let mut found = 0;
        let mut legal_code = 0;
        for &g in &chosen {
            found |= self.groups[g].required;
            legal_code = append(legal_code, self.groups[g].class);
        }
        (chosen, found, legal_code)
    }
    fn choices(&self, code: u64) -> Result<(usize, Vec<(usize, usize)>)> {
        let (chosen, _, legal_code) = self.selected(code);
        let blocked = self
            .legality
            .lock()
            .map_err(|_| "legality cache poisoned")?
            .state(legal_code)
            .blocked;
        let mut total = 0;
        let mut choices = Vec::new();
        for (i, g) in self.groups.iter().enumerate() {
            if blocked & (1u128 << g.class) != 0 {
                continue;
            }
            let count = chosen.iter().filter(|&&v| v == i).count();
            if count < g.size {
                let tokens = (g.size - count) * g.tokens;
                total += tokens;
                choices.push((i, tokens));
            }
        }
        Ok((total, choices))
    }
    fn branches(&self, code: u64, context: u8, flag: bool) -> Result<Vec<Branch>> {
        let (chosen, found, _) = self.selected(code);
        let depth = chosen.len();
        let missing = self.required & !found;
        let (total, choices) = self.choices(code)?;
        if total == 0 {
            return Ok(Vec::new());
        }
        // Reserve all missing species: keep the prior denominator unchanged.
        let mut projected = chosen.clone();
        for (i, g) in self.groups.iter().enumerate() {
            if g.required & missing != 0 {
                projected.push(i);
            }
        }
        let mut reservation = 0;
        let mut allowed = vec![false; self.groups.len()];
        for &g in &projected {
            reservation = append(reservation, self.groups[g].class);
        }
        let reserved = self
            .legality
            .lock()
            .map_err(|_| "legality cache poisoned")?
            .state(reservation);
        if !reserved.valid {
            return Ok(Vec::new());
        }
        let blocked = reserved.blocked;
        for (i, g) in self.groups.iter().enumerate() {
            allowed[i] = g.required & missing != 0 || blocked & (1u128 << g.class) == 0;
        }
        let mut result = Vec::new();
        for (i, tokens) in choices {
            let g = &self.groups[i];
            if !allowed[i]
                || self.fixed[depth].is_some_and(|v| v != i)
                || g.fixed.is_some_and(|s| s != depth)
            {
                continue;
            }
            if missing.count_ones() as usize == self.slots - depth && g.required & missing == 0 {
                continue;
            }
            let next = append(code, i);
            let after_flag = flag || (g.wobb && depth + 1 < self.slots);
            for (after, factor) in &g.rows[context as usize] {
                result.push(Branch {
                    group: i,
                    context: *after,
                    code: next,
                    flag: after_flag,
                    factor: (factor / total) * tokens,
                });
            }
        }
        Ok(result)
    }
    fn upper(&self, missing: u8, depth: usize, flag: bool) -> BigUint {
        let mut upper = &self.powers[self.slots - depth]
            * if flag {
                &self.terminal[1]
            } else {
                self.terminal.iter().max().unwrap()
            };
        for g in &self.groups {
            if g.required & missing != 0 {
                upper *= &g.upper;
            }
        }
        upper
    }
    fn mass(&mut self, code: u64, context: u8, flag: bool) -> Result<Mass> {
        let (chosen, found, _) = self.selected(code);
        let depth = chosen.len();
        let missing = self.required & !found;
        if missing.count_ones() as usize > self.slots - depth {
            return Ok(Mass::exact(BigUint::zero()));
        }
        if depth == self.slots {
            return Ok(Mass::exact(if missing == 0 {
                self.terminal[flag as usize].clone()
            } else {
                BigUint::zero()
            }));
        }
        if missing == 0
            && self.completable
            && (flag || self.terminal[0] == self.terminal[1] || depth + 1 == self.slots)
        {
            return Ok(Mass::exact(
                &self.powers[self.slots - depth] * &self.terminal[flag as usize],
            ));
        }
        for g in &self.groups {
            if g.required & missing != 0
                && !(0..16).any(|later| {
                    later & context as usize == context as usize && g.support & (1 << later) != 0
                })
            {
                return Ok(Mass::exact(BigUint::zero()));
            }
        }
        let key = code | ((context as u64) << 48) | ((flag as u64) << 52);
        if let Some(value) = self.cache.get(&key) {
            return Ok(value.clone());
        }
        if self.cut
            || self.nodes >= self.max_states
            || (self.nodes % 128 == 0 && Instant::now() >= self.deadline)
        {
            self.cut = true;
            return Ok(Mass {
                lower: BigUint::zero(),
                upper: self.upper(missing, depth, flag),
            });
        }
        self.nodes += 1;
        let (_, eligible) = self.choices(code)?;
        let available = eligible
            .iter()
            .fold(0, |m, (g, _)| m | self.groups[*g].required);
        if missing & !available != 0 {
            return Ok(Mass::exact(BigUint::zero()));
        }
        let mut result = Mass::exact(BigUint::zero());
        for branch in self.branches(code, context, flag)? {
            self.edges += 1;
            let mass = self.mass(branch.code, branch.context, branch.flag)?;
            result.lower += &branch.factor * &mass.lower;
            result.upper += &branch.factor * &mass.upper;
        }
        self.cache.insert(key, result.clone());
        Ok(result)
    }
    fn draw_one(&mut self, rng: &mut StdRng) -> Result<Vec<(usize, u8)>> {
        let mut code = 0;
        let mut context = 0;
        let mut flag = self.initial_flag;
        let mut path = Vec::new();
        for _ in 0..self.slots {
            let branches = self.branches(code, context, flag)?;
            let mut weighted = Vec::new();
            let mut total = BigUint::zero();
            for b in branches {
                let weight = &b.factor * self.mass(b.code, b.context, b.flag)?.lower;
                total += &weight;
                weighted.push((b, weight));
            }
            if total.is_zero() {
                return Err("zero completed branch mass".into());
            }
            let mut value = random_below(&total, rng);
            let mut selected = None;
            for (b, w) in weighted {
                if value < w {
                    selected = Some(b);
                    break;
                }
                value -= w;
            }
            let b = selected.ok_or("integer draw exhausted")?;
            path.push((b.group, b.context));
            code = b.code;
            context = b.context;
            flag = b.flag;
        }
        Ok(path)
    }
}
fn random_below(n: &BigUint, rng: &mut StdRng) -> BigUint {
    let bits = n.bits();
    let mut bytes = vec![0; (bits.div_ceil(8)) as usize];
    loop {
        rng.fill_bytes(&mut bytes);
        if bits % 8 != 0 {
            *bytes.last_mut().unwrap() &= (1u8 << (bits % 8)) - 1;
        }
        let x = BigUint::from_bytes_le(&bytes);
        if &x < n {
            return x;
        }
    }
}
#[pymethods]
impl NativeJointGenerator {
    #[new]
    fn new(payload: &str, cache: &JointLegalityCache) -> PyResult<Self> {
        Self::from_json(payload, cache.inner.clone()).map_err(err)
    }
    fn solve(&mut self, seconds: f64, max_states: usize) -> PyResult<String> {
        if self.solved.is_some() {
            return Err(PyValueError::new_err(
                "posterior already solved; create a new evidence state",
            ));
        }
        if !seconds.is_finite() || seconds <= 0.0 || seconds > 3600.0 || max_states == 0 {
            return Err(PyValueError::new_err("invalid budget"));
        }
        self.deadline = Instant::now() + Duration::from_secs_f64(seconds);
        self.max_states = max_states;
        self.cut = false;
        let value = self.mass(0, 0, self.initial_flag).map_err(err)?;
        self.solved = Some(value.clone());
        Ok(json!({"lower":value.lower.to_string(),"upper":value.upper.to_string(),"exact":value.lower==value.upper,"nodes":self.nodes,"edges":self.edges,"cut":self.cut}).to_string())
    }
    fn draw_paths(&mut self, count: usize, seed: u64) -> PyResult<Vec<Vec<(usize, u8)>>> {
        if self.solved.as_ref().is_none_or(|m| m.lower.is_zero()) {
            return Err(PyValueError::new_err(
                "solve with positive lower mass first",
            ));
        }
        let mass = self.solved.as_ref().unwrap();
        if (&mass.upper - &mass.lower) * 100u32 > mass.upper {
            return Err(PyValueError::new_err("uncertified draw: TV exceeds .01"));
        }
        // Freeze the completed subdistribution: a draw must not expand the frontier.
        self.cut = true;
        let mut rng = StdRng::seed_from_u64(seed);
        (0..count)
            .map(|_| self.draw_one(&mut rng).map_err(err))
            .collect()
    }
    fn branch_masses(&mut self, code: u64, context: u8, flag: bool) -> PyResult<String> {
        if self.solved.is_none() {
            return Err(PyValueError::new_err(
                "solve before inspecting branch masses",
            ));
        }
        if (0..8).any(|i| code >> (8 * i) != 0 && (code >> (8 * i)) & 255 == 0) {
            return Err(PyValueError::new_err("invalid packed state"));
        }
        self.cut = true;
        if context >= 16
            || members(code).len() >= self.slots
            || members(code).iter().any(|&g| g >= self.groups.len())
        {
            return Err(PyValueError::new_err("invalid state"));
        }
        let (selected, found, legal_code) = self.selected(code);
        if selected.len() + (self.required & !found).count_ones() as usize > self.slots
            || selected
                .iter()
                .any(|&g| selected.iter().filter(|&&j| j == g).count() > self.groups[g].size)
            || flag != (self.initial_flag || selected.iter().any(|&g| self.groups[g].wobb))
            || !self
                .legality
                .lock()
                .map_err(|_| PyValueError::new_err("legality cache poisoned"))?
                .state(legal_code)
                .valid
        {
            return Err(PyValueError::new_err("unreachable packed state"));
        }
        let mut rows = Vec::new();
        for b in self.branches(code, context, flag).map_err(err)? {
            let m = self.mass(b.code, b.context, b.flag).map_err(err)?;
            rows.push(json!([
                b.group,
                b.context,
                b.code,
                b.flag,
                (&b.factor * m.lower).to_string(),
                (&b.factor * m.upper).to_string()
            ]));
        }
        Ok(Value::Array(rows).to_string())
    }
    fn revealed_context_masses(&mut self) -> PyResult<String> {
        if self.solved.as_ref().is_none_or(|m| m.lower.is_zero()) {
            return Err(PyValueError::new_err("positive solved mass required"));
        }
        let mass = self.solved.as_ref().unwrap();
        if (&mass.upper - &mass.lower) * 100u32 > mass.upper {
            return Err(PyValueError::new_err(
                "uncertified features: TV exceeds .01",
            ));
        }
        self.cut = true;
        let mut forward = HashMap::from([((self.initial_flag as u64) << 52, BigUint::one())]);
        let mut counts: BTreeMap<(usize, u8, u8), BigUint> = BTreeMap::new();
        for _ in 0..self.slots {
            let mut next = HashMap::<u64, BigUint>::new();
            for (key, prefix) in forward {
                let code = key & ((1u64 << 48) - 1);
                let context = ((key >> 48) & 15) as u8;
                let flag = (key >> 52) != 0;
                let (_, found, _) = self.selected(code);
                if found == self.required {
                    continue;
                }
                for branch in self.branches(code, context, flag).map_err(err)? {
                    let suffix = self
                        .mass(branch.code, branch.context, branch.flag)
                        .map_err(err)?
                        .lower;
                    if suffix.is_zero() {
                        continue;
                    }
                    let step = &prefix * &branch.factor;
                    if self.groups[branch.group].required != 0 {
                        *counts
                            .entry((branch.group, context, branch.context))
                            .or_default() += &step * &suffix;
                    }
                    let key = branch.code
                        | ((branch.context as u64) << 48)
                        | ((branch.flag as u64) << 52);
                    *next.entry(key).or_default() += step;
                }
            }
            forward = next;
        }
        let rows: Vec<_> = counts
            .into_iter()
            .map(|((g, before, after), n)| json!([g, before, after, n.to_string()]))
            .collect();
        Ok(Value::Array(rows).to_string())
    }
}
