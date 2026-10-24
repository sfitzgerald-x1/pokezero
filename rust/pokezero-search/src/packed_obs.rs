//! Exact decoder for the leaf encoder's canonical little-endian byte rows.
//!
//! This is an additive transport seam, not an encoder or inference change.
//! It performs the same i32→i64, f64→f32 and i16→i64 conversions as the old
//! Python scalar-list / PyO3 Vec path, preserving row-major order.

#[derive(Debug, PartialEq)]
pub(crate) struct PackedObservation {
    pub categorical: Vec<i64>,
    pub numeric: Vec<f32>,
    pub token_types: Vec<i64>,
    pub attention: Vec<bool>,
    pub history: Vec<bool>,
}

#[derive(Clone, Copy)]
pub(crate) struct PackedShape {
    pub batch: usize,
    pub window: usize,
    pub tokens: usize,
    pub categorical_features: usize,
    pub numeric_features: usize,
}

fn product(parts: &[usize]) -> Result<usize, String> {
    parts.iter().try_fold(1usize, |total, &part| {
        if part == 0 {
            return Err("packed shape dimensions must be positive".into());
        }
        total
            .checked_mul(part)
            .ok_or_else(|| "packed shape overflow".into())
    })
}

fn require_width(name: &str, bytes: &[u8], count: usize, width: usize) -> Result<(), String> {
    let expected = count
        .checked_mul(width)
        .ok_or("packed byte width overflow")?;
    if bytes.len() != expected {
        return Err(format!(
            "{name} byte length {} differs from expected {expected}",
            bytes.len()
        ));
    }
    Ok(())
}

pub(crate) fn decode(
    shape: PackedShape,
    byte_order: &str,
    categorical: &[u8],
    numeric: &[u8],
    token_types: &[u8],
    attention: &[u8],
    history: &[u8],
) -> Result<PackedObservation, String> {
    if byte_order != "little" {
        return Err("canonical packed rows require explicit little byte order".into());
    }
    let tokens = product(&[shape.batch, shape.window, shape.tokens])?;
    let cats = product(&[tokens, shape.categorical_features])?;
    let nums = product(&[tokens, shape.numeric_features])?;
    let histories = product(&[shape.batch, shape.window])?;
    // Check every width and boolean BEFORE allocating or publishing arrays.
    require_width("categorical_ids/i32", categorical, cats, 4)?;
    require_width("numeric_features/f64", numeric, nums, 8)?;
    require_width("token_type_ids/i16", token_types, tokens, 2)?;
    require_width("attention_mask/bool", attention, tokens, 1)?;
    require_width("history_mask/bool", history, histories, 1)?;
    if attention.iter().chain(history).any(|&value| value > 1) {
        return Err("packed masks require only boolean bytes 0 or 1".into());
    }
    let numeric: Result<Vec<f32>, String> = numeric
        .chunks_exact(8)
        .map(|chunk| {
            let value = f64::from_le_bytes(chunk.try_into().expect("checked width"));
            let converted = value as f32;
            if !value.is_finite() || !converted.is_finite() {
                return Err("packed numeric row must be finite and representable as f32".into());
            }
            Ok(converted)
        })
        .collect();
    Ok(PackedObservation {
        categorical: categorical
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("checked width")) as i64)
            .collect(),
        numeric: numeric?,
        token_types: token_types
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes(chunk.try_into().expect("checked width")) as i64)
            .collect(),
        attention: attention.iter().map(|&value| value == 1).collect(),
        history: history.iter().map(|&value| value == 1).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> PackedShape {
        PackedShape {
            batch: 2,
            window: 1,
            tokens: 2,
            categorical_features: 1,
            numeric_features: 1,
        }
    }
    fn inputs() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        (
            [-2147483648i32, 3, 5, 2147483647]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect(),
            [-0.0f64, 1.0 / 3.0, -2.5, 1.0e-40]
                .into_iter()
                .flat_map(f64::to_le_bytes)
                .collect(),
            [-32768i16, 7, 9, 32767]
                .into_iter()
                .flat_map(i16::to_le_bytes)
                .collect(),
            vec![1, 0, 1, 1],
            vec![1, 0],
        )
    }

    #[test]
    fn preserves_five_effective_arrays_and_conversion_bits() {
        let (c, n, t, a, h) = inputs();
        let decoded = decode(shape(), "little", &c, &n, &t, &a, &h).unwrap();
        assert_eq!(decoded.categorical, vec![-2147483648, 3, 5, 2147483647]);
        assert_eq!(
            decoded
                .numeric
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            [-0.0f64, 1.0 / 3.0, -2.5, 1.0e-40].map(|x| (x as f32).to_bits())
        );
        assert_eq!(decoded.token_types, vec![-32768, 7, 9, 32767]);
        assert_eq!(decoded.attention, vec![true, false, true, true]);
        assert_eq!(decoded.history, vec![true, false]);
    }

    #[test]
    fn refuses_every_wrong_width_byte_order_or_nonboolean() {
        let original = inputs();
        for column in 0..5 {
            let (mut c, mut n, mut t, mut a, mut h) = original.clone();
            [&mut c, &mut n, &mut t, &mut a, &mut h][column].pop();
            assert!(decode(shape(), "little", &c, &n, &t, &a, &h).is_err());
        }
        let (c, n, t, mut a, mut h) = original;
        assert!(decode(shape(), "big", &c, &n, &t, &a, &h).is_err());
        a[0] = 2;
        assert!(decode(shape(), "little", &c, &n, &t, &a, &h).is_err());
        a[0] = 1;
        h[0] = 255;
        assert!(decode(shape(), "little", &c, &n, &t, &a, &h).is_err());
    }

    #[test]
    fn refuses_invalid_shapes_nonfinite_and_f32_overflow() {
        let (c, mut n, t, a, h) = inputs();
        for invalid in [0, usize::MAX] {
            let bad = PackedShape {
                batch: invalid,
                ..shape()
            };
            assert!(decode(bad, "little", &c, &n, &t, &a, &h).is_err());
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
            n[..8].copy_from_slice(&value.to_le_bytes());
            assert!(decode(shape(), "little", &c, &n, &t, &a, &h).is_err());
        }
    }
}
