mod data;
mod features;



use data::load_prices;
use std::{error::Error, println};

use features::{simple_returns, log_returns, sma_sliding, sma_naive};

use crate::features::{rolling_std_two_pass, rolling_std_welford};


fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "./data/aapl.csv";

    let (records, skipped) = load_prices(file_path)?;
    
    println!("Number of Data: {}\nskipped: {}\n", records.len(), skipped);


    let returns = simple_returns(&records);
    println!("Simple Returns\nitem: {:.6?}\nLength: {}\n",&returns[..3], returns.len());

    let returns = log_returns(&records);
    println!("Log Returns\nitem: {:.6?}\nLength: {}\n",&returns[..3], returns.len());

    let prices :Vec<f64> = records.iter().map(|w| w.close).collect();

    let sma = sma_sliding(&prices, 5);

    let naive_sma = sma_naive(&prices, 5);

    println!("{:.6?}\nlength:{}\nlast: {:6?}" ,&sma[0..5],sma.len(), sma.last());
    println!("{:.6?}\nlength:{}\nlast: {:6?}" ,&naive_sma[0..5],naive_sma.len(), naive_sma.last());

    let diff = naive_sma.iter().zip(sma.iter())
    .map(|(a,b)| (a-b).abs())
    .fold(0.0_f64, f64::max);

    println!("{:e}, ok = {}\n", diff, diff < 1e-9);


    let vol_tp = rolling_std_welford(&returns, 20);


    println!("{:.6?}\nlength:{}" ,&vol_tp[0..3],vol_tp.len());

    Ok(())

}


