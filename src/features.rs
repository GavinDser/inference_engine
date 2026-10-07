use crate::data::Record;



pub fn simple_returns(records: &[Record]) -> Vec<f64> {

    // function for calculating simple returns
    records.windows(2)
    .map(|w| (w[1].close - w[0].close) / w[0].close)
    .collect()

}

pub fn log_returns(records: &[Record]) -> Vec<f64> {
    // log return formula: ln(a/b)
    records.windows(2)
    .map(|w| w[1].close.ln() - w[0].close.ln())
    .collect()
}

// simple moving average, divide by w
pub fn sma_naive(prices: &[f64], w: usize) -> Vec<f64> {
    if w > prices.len() || w == 0 {
        return vec![]
    }
    
    prices.windows(w).map(|p| p.iter().sum::<f64>()/w as f64).collect()
}

pub fn sma_sliding(prices: &[f64], w:usize) -> Vec<f64> {
    if w > prices.len() || w == 0 {
        return vec![];
    }

    let mut sma: Vec<f64> = Vec::with_capacity(prices.len()-w + 1);

    sma.push(prices[0..w].iter().sum::<f64>());

    //slidng window
    for i in w..prices.len() {
        sma.push(sma[i-w]+prices[i]-prices[i-w]);
    }

    sma.iter_mut().for_each(|i| *i = *i /w as f64);

    sma
}

pub fn rolling_std(xs: &[f64], w: usize) -> Vec<f64> {

}