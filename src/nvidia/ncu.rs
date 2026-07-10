//! `ncu --csv` metric-export parsing: the measured side of the
//! predicted-vs-measured close (achieved occupancy, cycle counts, stall
//! attribution).
//!
//! PENDING REAL-DATA VALIDATION: no raw ncu export is checked in anywhere
//! under reference/ (the one ncu run on record, data/ncu-atomics-20260610/,
//! kept only a hand-transcribed table in its NOTES.md). The parser is built
//! against a fixture in the documented `--csv` shape (header row naming
//! "Kernel Name"/"Metric Name"/"Metric Unit"/"Metric Value", quoted fields,
//! thousands separators in values) carrying that table's real numbers, and
//! must be re-validated against a live `ncu --csv` capture when the GPU lane
//! runs the measured close.

use crate::nvidia::csvio::parse_records;

/// The ncu metric name for achieved occupancy.
pub const ACHIEVED_OCCUPANCY: &str = "sm__warps_active.avg.pct_of_peak_sustained_active";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetricRow {
    /// launch ID ("" when the export carries none)
    pub launch: String,
    pub kernel: String,
    pub metric: String,
    pub unit: String,
    pub value: String,
}

/// Parse an `ncu --csv` export. Columns are located by name so reordering or
/// extra columns across ncu versions do not break the parse; preamble lines
/// before the header (`==PROF==` chatter) are skipped.
pub fn parse_ncu_csv(text: &str) -> Result<Vec<MetricRow>, String> {
    let header_at = text
        .lines()
        .position(|l| l.contains("Metric Name") && l.contains("Metric Value"))
        .ok_or("no ncu csv header (a line naming \"Metric Name\" and \"Metric Value\")")?;
    let body: String = text.lines().skip(header_at).collect::<Vec<_>>().join("\n");
    let mut records = parse_records(&body);
    if records.is_empty() {
        return Err("empty ncu csv".into());
    }
    let header = records.remove(0);
    let col = |name: &str| header.iter().position(|h| h == name);
    let kernel = col("Kernel Name").ok_or("no \"Kernel Name\" column")?;
    let metric = col("Metric Name").ok_or("no \"Metric Name\" column")?;
    let value = col("Metric Value").ok_or("no \"Metric Value\" column")?;
    let unit = col("Metric Unit");
    let launch = col("ID");
    let mut out = Vec::new();
    for r in records {
        if r.len() <= value.max(metric).max(kernel) {
            continue;
        }
        out.push(MetricRow {
            launch: launch.and_then(|i| r.get(i)).cloned().unwrap_or_default(),
            kernel: r[kernel].clone(),
            metric: r[metric].clone(),
            unit: unit.and_then(|i| r.get(i)).cloned().unwrap_or_default(),
            value: r[value].clone(),
        });
    }
    Ok(out)
}

/// A metric value as a number: ncu writes thousands separators
/// (`"45,526.70"`).
pub fn metric_value(row: &MetricRow) -> Option<f64> {
    row.value.replace(',', "").parse().ok()
}

/// Achieved occupancy per (launch, kernel), in export order.
pub fn achieved_occupancy(rows: &[MetricRow]) -> Vec<(String, String, f64)> {
    rows.iter()
        .filter(|r| r.metric == ACHIEVED_OCCUPANCY)
        .filter_map(|r| metric_value(r).map(|v| (r.launch.clone(), r.kernel.clone(), v)))
        .collect()
}
