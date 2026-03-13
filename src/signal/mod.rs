pub mod filters;
pub mod fft;
pub mod peaks;
pub mod rate;
pub mod hrv;
pub mod bp;
pub mod resp;

pub use hrv::estimate_hrv;
pub use rate::estimate_rate;

/// Calculates the arithmetic mean of a signal.
///
/// # Arguments
/// * `signal` - The input data slice.
///
/// # Returns
/// A tuple of `(mean_value, confidence_score)`.
pub fn calculate_average(signal: &[f32]) -> (f32, f32) {
    if signal.is_empty() { return (0.0, 0.0); }
    let sum: f32 = signal.iter().sum();
    (sum / signal.len() as f32, 1.0)
}

/// Linearly interpolates values for given target x-coordinates.
///
/// # Arguments
/// * `x` - Known x-coordinates (must be strictly increasing).
/// * `y` - Known y-coordinates corresponding to `x`.
/// * `x_new` - Target x-coordinates to interpolate at.
///
/// # Returns
/// Interpolated y-values corresponding to `x_new`.
pub fn interp_linear_1d(x: &[f32], y: &[f32], x_new: &[f32]) -> Vec<f32> {
    let mut result = Vec::with_capacity(x_new.len());

    if x.is_empty() || y.is_empty() || x.len() != y.len() {
        return vec![f32::NAN; x_new.len()];
    }
    if x.len() == 1 {
        return vec![y[0]; x_new.len()];
    }

    let mut idx = 0;
    for &xn in x_new {
        // Clamp to edges if out of bounds
        if xn <= x[0] {
            result.push(y[0]);
            continue;
        }
        if xn >= x[x.len() - 1] {
            result.push(y[y.len() - 1]);
            continue;
        }

        // Advance idx to find the bounding segment
        while idx < x.len() - 1 && xn > x[idx + 1] {
            idx += 1;
        }

        let x0 = x[idx];
        let x1 = x[idx + 1];
        let y0 = y[idx];
        let y1 = y[idx + 1];

        let t = (xn - x0) / (x1 - x0);
        result.push(y0 + t * (y1 - y0));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interp_linear_1d_normal() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![10.0, 20.0, 10.0];
        let x_new = vec![0.5, 1.5];
        let res = interp_linear_1d(&x, &y, &x_new);
        assert_eq!(res, vec![15.0, 15.0]);
    }

    #[test]
    fn test_interp_linear_1d_clamping() {
        let x = vec![0.0, 1.0];
        let y = vec![10.0, 20.0];
        let x_new = vec![-1.0, 2.0];
        let res = interp_linear_1d(&x, &y, &x_new);
        assert_eq!(res, vec![10.0, 20.0]);
    }

    #[test]
    fn test_interp_linear_1d_exact_points() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![10.0, 20.0, 30.0];
        let x_new = vec![0.0, 1.0, 2.0];
        let res = interp_linear_1d(&x, &y, &x_new);
        assert_eq!(res, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_interp_linear_1d_empty() {
        let res = interp_linear_1d(&[], &[], &[1.0, 2.0]);
        assert_eq!(res.len(), 2);
        assert!(res[0].is_nan());
        assert!(res[1].is_nan());
    }

    #[test]
    fn test_interp_linear_1d_single_element() {
        let res = interp_linear_1d(&[5.0], &[42.0], &[0.0, 10.0]);
        assert_eq!(res, vec![42.0, 42.0]);
    }
}