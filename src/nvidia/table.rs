//! `tu102_ops.csv` ingest: the measured rate table keyed by `row_id`. Sweep
//! rows share a `row_id`; the peak (first row on ties) is kept, mirroring
//! `project.py::load_rates`.

use crate::nvidia::csvio::Table;

pub struct RateRow {
    pub value: String,
    pub pipe: String,
}

pub struct Rates(pub Vec<(String, RateRow)>);

impl Rates {
    pub fn get(&self, row_id: &str) -> Option<&RateRow> {
        self.0.iter().find(|(id, _)| id == row_id).map(|(_, r)| r)
    }

    pub fn parse(csv_text: &str) -> Result<Rates, String> {
        let t = Table::parse(csv_text);
        let (id_col, value_col, pipe_col) = (t.col("row_id"), t.col("value"), t.col("pipe"));
        let mut rates: Vec<(String, RateRow)> = Vec::new();
        for row in &t.rows {
            let id = &row[id_col];
            if let Some((_, existing)) = rates.iter_mut().find(|(k, _)| k == id) {
                // keep the peak; on a tie the first-seen row stays
                let old: f64 = parse_value(&existing.value, id)?;
                let new: f64 = parse_value(&row[value_col], id)?;
                if old >= new {
                    continue;
                }
                existing.value = row[value_col].clone();
                existing.pipe = row[pipe_col].clone();
            } else {
                rates.push((
                    id.clone(),
                    RateRow { value: row[value_col].clone(), pipe: row[pipe_col].clone() },
                ));
            }
        }
        Ok(Rates(rates))
    }
}

fn parse_value(s: &str, row_id: &str) -> Result<f64, String> {
    s.parse()
        .map_err(|_| format!("row {}: value {:?} is not a number", row_id, s))
}
