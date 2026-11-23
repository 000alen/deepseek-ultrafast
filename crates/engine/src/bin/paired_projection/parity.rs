//! Host-side verification for the fixed two-request diagnostic.
pub const ROWS: usize = 8;
pub const LANE_ROWS: usize = ROWS / 2;

pub fn compare(
    label: &str,
    expected: &[f32],
    actual: &[f32],
    experts: usize,
    tolerance: f32,
) -> Result<[f32; 2], String> {
    if experts == 0
        || expected.is_empty()
        || expected.len() != actual.len()
        || !expected.len().is_multiple_of(ROWS * experts)
    {
        return Err(format!("{label}: incomplete projection matrix"));
    }
    let mut max_error = [0f32; 2];
    for (i, (&want, &got)) in expected.iter().zip(actual).enumerate() {
        let row = i / experts % ROWS;
        let lane = row / LANE_ROWS;
        let error = (got - want).abs();
        if !want.is_finite() || !got.is_finite() || error > tolerance * (1. + want.abs()) {
            return Err(format!(
                "{label}: layer {} lane {lane} row {} expert {} got {got} expected {want}",
                i / experts / ROWS,
                row % LANE_ROWS,
                i % experts
            ));
        }
        max_error[lane] = max_error[lane].max(error);
    }
    Ok(max_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_second_lane_alias_and_unwritten_output_after_the_first_layer() {
        let expected: Vec<_> = (0..32).map(|i| i as f32).collect();
        let mut actual = expected.clone();
        actual.copy_within(16..24, 24);
        assert!(
            compare("one-hot", &expected, &actual, 2, 0.)
                .unwrap_err()
                .contains("layer 1 lane 1 row 0 expert 0")
        );
        actual = expected.clone();
        actual[30] = f32::NAN;
        assert!(compare("dense", &expected, &actual, 2, 1e-4).is_err());
        assert!(compare("dense", &expected, &actual[..31], 2, 1e-4).is_err());
    }

    #[test]
    fn attributes_accepted_dense_error_to_the_correct_lane() {
        let expected = [1.; 32];
        let mut actual = expected;
        actual[25] += 1e-4;
        let errors = compare("dense", &expected, &actual, 2, 1e-4).unwrap();
        assert_eq!(errors, [0., actual[25] - 1.]);
        assert!(compare("one-hot", &expected, &actual, 2, 0.).is_err());
    }
}
