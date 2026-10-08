use crate::data::Record;



/// Simple returns: (p_t - p_{t-1}) / p_{t-1}.
/// Output has `records.len() - 1` items; item k is the return from day k to day k+1.
pub fn simple_returns(records: &[Record]) -> Vec<f64> {

    // each window is a pair of consecutive days [yesterday, today]
    records.windows(2)
    .map(|w| (w[1].close - w[0].close) / w[0].close)
    .collect()

}

/// Log returns: ln(p_t / p_{t-1}), computed as ln(p_t) - ln(p_{t-1}).
/// Additive over time, which is why volatility is computed on these.
pub fn log_returns(records: &[Record]) -> Vec<f64> {
    // log return formula: ln(a/b) = ln(a) - ln(b)
    records.windows(2)
    .map(|w| w[1].close.ln() - w[0].close.ln())
    .collect()
}

/// Simple moving average, naive version: re-sum every window, O(n·w).
/// Output has `len - w + 1` items; returns empty if `w == 0` or `w > len`.
pub fn sma_naive(prices: &[f64], w: usize) -> Vec<f64> {
    if w > prices.len() || w == 0 {
        return vec![]
    }

    // mean of each window = window sum / w
    prices.windows(w).map(|p| p.iter().sum::<f64>()/w as f64).collect()
}

/// Simple moving average, sliding-sum version: O(n).
/// Each step adds the new price and subtracts the one leaving the window.
/// Same output as `sma_naive`.
pub fn sma_sliding(prices: &[f64], w:usize) -> Vec<f64> {
    if w > prices.len() || w == 0 {
        return vec![];
    }

    // holds window sums first; divided by w at the end
    let mut sma: Vec<f64> = Vec::with_capacity(prices.len()-w + 1);

    // sum of the first window
    sma.push(prices[0..w].iter().sum::<f64>());

    // sliding window: new sum = previous sum + entering price - leaving price
    for i in w..prices.len() {
        sma.push(sma[i-w]+prices[i]-prices[i-w]);
    }

    // turn sums into means
    sma.iter_mut().for_each(|i| *i = *i /w as f64);

    sma
}

/// Rolling sample standard deviation, two-pass version: O(n·w).
/// Pass 1 computes the window mean, pass 2 sums squared deviations.
/// Slow but numerically reliable; used as the reference for `rolling_std_welford`.
/// Output has `len - w + 1` items; returns empty if `w < 2` (w - 1 would be 0) or `w > len`.
pub fn rolling_std_two_pass(xs: &[f64], w: usize) -> Vec<f64> {
    if w < 2 || w > xs.len() {
        return vec![];
    }


    xs.windows(w).map(|vs|{
        // pass 1: window mean
        let average = vs.iter().sum::<f64>() / w as f64;
        // pass 2: sum of squared deviations (ss)
        let ss = vs.iter().map(|v| (v-average).powi(2)).sum::<f64>();
        // sample std: divide by w - 1 (Bessel's correction), then sqrt
        (ss/ (w as f64 - 1.0)).sqrt()

    }).collect()
    

}

/// Rolling sample standard deviation, Welford version: O(n), single pass.
/// Keeps only `mean` and `m2` (sum of squared deviations) and updates them
/// as one value enters and one leaves the window. Never forms sum(x^2),
/// so it avoids catastrophic cancellation on large values.
/// Same output as `rolling_std_two_pass`.
pub fn rolling_std_welford(xs: &[f64], w: usize) -> Vec<f64> {
    if w < 2 || w > xs.len() {
        return vec![];
    }

    let mut vol = Vec::<f64>::with_capacity(xs.len()-w+1);

    // precompute 1/w: multiplication is cheaper than division in the loop
    let inv_w = 1.0 / w as f64;
    // first window: initialise mean and m2 the two-pass way
    let mut mean = xs[0..w].iter().sum::<f64>() / w as f64;
    let mut m2 = xs[0..w].iter().map(|v| (v-mean).powi(2)).sum::<f64>();

    vol.push((m2/ (w as f64 - 1.0)).sqrt());

    // slide the window: x_new enters, x_old leaves
    for i in w..xs.len() {
        let x_new = xs[i];
        let x_old = xs[i-w];
        let mean_new = mean + (x_new - x_old) * inv_w;
        // m2 update needs both the old mean and the new mean
        m2 = m2 + (x_new - x_old) *((x_new - mean_new) + (x_old - mean));
        // overwrite the old mean only after m2 has used it
        mean = mean_new;
        // rounding drift can make m2 slightly negative; clamp to 0 to avoid sqrt(-x) = NaN
        vol.push((m2.max(0.0) / (w as f64 - 1.0)).sqrt());
    }

    vol
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::load_prices;

    // float comparison: fail and print both values unless |a - b| < tol
    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "got {a}, expected {b}, tol {tol}");
    }

    // compare two sequences element by element; lengths must match too
    fn assert_all_close(got: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected.iter()) {
            assert_close(*g, *e, tol);
        }
    }

    // load AAPL closes and compute log returns (751 items)
    fn real_log_returns() -> Vec<f64> {
        let (records, _) = load_prices("./data/aapl.csv").unwrap();
        log_returns(&records)
    }

    // case 1: [1,2,3,4,5], w=3; every window looks like the hand-computed [1,2,3], std = 1
    #[test]
    fn hand_computed() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let w = 3;

        let got_two_passed = rolling_std_two_pass(&xs, w);
        let got_welford = rolling_std_welford(&xs, w);

        assert_all_close(&got_two_passed, &[1.0, 1.0, 1.0], 1e-12);
        assert_all_close(&got_welford, &[1.0, 1.0, 1.0], 1e-12);
    }

    // case 2: window equals full length, a single value sqrt(2.5)
    #[test]
    fn full_window() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let expected = [2.5_f64.sqrt()];

        assert_all_close(&rolling_std_two_pass(&xs, 5), &expected, 1e-12);
        assert_all_close(&rolling_std_welford(&xs, 5), &expected, 1e-12);
    }

    // case 3: w=0, w=1, w=len+1 all return empty without panicking
    #[test]
    fn edge_cases() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        for w in [0, 1, xs.len() + 1] {
            assert!(rolling_std_two_pass(&xs, w).is_empty(), "two_pass w={w}");
            assert!(rolling_std_welford(&xs, w).is_empty(), "welford w={w}");
        }
    }

    // case 4: 10 distinct values followed by 10 identical values.
    // windows fully inside the constant part must give 0, never NaN (negative m2 drift)
    #[test]
    fn constant_is_zero_not_nan() {
        let mut xs = vec![3.7, 1.2, 9.9, 0.4, 5.5, 2.8, 7.1, 6.3, 0.9, 4.6];
        xs.extend([2.5; 10]);
        let w = 5;

        for got in [rolling_std_two_pass(&xs, w), rolling_std_welford(&xs, w)] {
            // the last 6 windows lie fully in the constant part (10 values, w=5: 10 - 5 + 1 = 6)
            for v in &got[got.len() - 6..] {
                assert!(!v.is_nan(), "got NaN");
                assert_close(*v, 0.0, 1e-6);
            }
        }
    }

    // case 5: 1e8 + [1,2,3,4,5], w=3; the one-pass formula cancels catastrophically, both versions must not
    #[test]
    fn cancellation_resistant() {
        let xs: Vec<f64> = [1.0, 2.0, 3.0, 4.0, 5.0].iter().map(|x| 1e8 + x).collect();

        assert_all_close(&rolling_std_two_pass(&xs, 3), &[1.0, 1.0, 1.0], 1e-6);
        assert_all_close(&rolling_std_welford(&xs, 3), &[1.0, 1.0, 1.0], 1e-6);
    }

    // case 6: real log returns, w=20; Welford vs two-pass element by element, max error < 1e-12
    #[test]
    fn welford_matches_two_pass() {
        let rets = real_log_returns();
        let tp = rolling_std_two_pass(&rets, 20);
        let wf = rolling_std_welford(&rets, 20);

        assert_all_close(&wf, &tp, 1e-12);
    }

    // case 7: acceptance values on real data (precomputed in Python), tol 1e-9
    #[test]
    fn real_data_values() {
        let rets = real_log_returns();
        assert_eq!(rets.len(), 751);

        for got in [rolling_std_two_pass(&rets, 20), rolling_std_welford(&rets, 20)] {
            assert_eq!(got.len(), 732);
            assert_close(got[0], 0.012661993373938, 1e-9);
            assert_close(got[1], 0.014210924238281, 1e-9);
            assert_close(got[2], 0.013820002180840, 1e-9);
            assert_close(got[731], 0.006162521727874, 1e-9);
        }
    }
}