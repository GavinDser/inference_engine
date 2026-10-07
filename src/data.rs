
use serde::{Deserialize};
use csv::{ReaderBuilder};
use std::error::Error;

#[derive(Debug, Deserialize)]
pub struct Record {
    #[serde(rename = "Date")]
    pub date: String,
    #[serde(rename = "Close")]
    pub close: f64,
}




pub fn load_prices(path: &str) -> Result<(Vec<Record>, usize), Box<dyn Error>>{
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;

    let mut records: Vec<Record> = Vec::new();
    let mut skipped = 0;

    for result in rdr.deserialize::<Record>() {
        match result {
            Ok(record) => records.push(record),
            Err(e) => match e.kind() {
                csv::ErrorKind::Deserialize {..} => skipped += 1,
                _ => return Err(e.into())
            }
        }
    }

    Ok((records, skipped))
}