//! Small CPU-only boundary-encoder benchmark, not a playing-strength benchmark.
//! Run with: cargo run --release --example bench_boundary_encoder -- TABLES ROWS [REPEATS]
//! ROWS is a committed golden-corpus rows.jsonl. Only sanctioned row inputs are used.
use std::{env, fs, hint::black_box, time::Instant};

use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;
use pokezero_search::encoder::{encode_row, Tables};
use serde_json::{json, Value};

fn main() {
    let args: Vec<String> = env::args().collect();
    assert!((3..=4).contains(&args.len()), "TABLES ROWS [REPEATS]");
    let tables = Tables::from_json(&fs::read_to_string(&args[1]).unwrap()).unwrap();
    let records: Vec<Value> = fs::read_to_string(&args[2])
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let schema = &records[0]["observation"]["schema_version"];
    let rows: Vec<String> = records
        .iter()
        .filter(|r| r["record_type"] == "decision")
        .map(|r| {
            assert_eq!(&r["observation"]["schema_version"], schema);
            json!({
                "battle_id": r["battle_id"], "battle_seed": r["battle_seed"],
                "format_id": r["format_id"], "player_id": r["player_id"],
                "observation_schema_version": r["observation"]["schema_version"],
                "observation_metadata": r["observation_metadata"],
                "public_materialization": r["public_materialization"],
            })
            .to_string()
        })
        .collect();
    assert!(!rows.is_empty());
    let mut hasher = Blake2bVar::new(32).unwrap();
    for row in &rows {
        let out = encode_row(&tables, row).unwrap();
        for v in out.categorical {
            hasher.update(&v.to_le_bytes());
        }
        for v in out.numeric {
            hasher.update(&v.to_bits().to_le_bytes());
        }
        for v in out.token_types {
            hasher.update(&v.to_le_bytes());
        }
        hasher.update(&out.attention);
        hasher.update(&out.legal);
    }
    let mut digest = [0; 32];
    hasher.finalize_variable(&mut digest).unwrap();
    let repeats: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(2000);
    assert!(repeats > 0);
    for _ in 0..100 {
        for row in &rows {
            black_box(encode_row(&tables, row).unwrap());
        }
    }
    let start = Instant::now();
    for _ in 0..repeats {
        for row in &rows {
            black_box(encode_row(&tables, row).unwrap());
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "{}",
        json!({"rows": rows.len(), "repeats": repeats,
        "encodes": repeats * rows.len(), "seconds": seconds,
        "microseconds_per_encode": seconds * 1e6 / (repeats * rows.len()) as f64,
        "output_blake2b_256": digest.iter().map(|b| format!("{b:02x}")).collect::<String>(),
        "scope": "boundary encode including JSON parse; no fold, tree, or neural inference"})
    );
}
