//! The tensor pipe in the census projection: HMMA/IMMA route to `pipe:tensor`
//! (measured `tensor.*.tput` rows first, the shared Turing default when
//! unmeasured) and LDSM is costed on the shared-memory lane by matrix count —
//! closing the alu@2.0 default that projected tensor kernels ~4x optimistic.

use calx_mill::nvidia::projection::{project, Census, MemClass, TENSOR_HMMA_PER_SM_CLK};
use calx_mill::nvidia::table::Rates;

fn fixture(rel: &str) -> String {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {}", path, e))
}

fn table() -> Rates {
    Rates::parse(&fixture("tu102/table/tu102_ops.csv")).expect("table parses")
}

fn empty_table() -> Rates {
    Rates::parse("row_id,value,pipe\n").expect("empty table parses")
}

#[test]
fn tensor_pipe_uses_measured_row() {
    let mut c = Census::default();
    c.add("HMMA.1688.F32", 1000);
    let r = project(&c, 1.0, MemClass::None, &table(), 5.82);
    // the measured peak row (Rates keeps the sweep peak: w8_f16acc 0.499987)
    let (name, cycles) = &r.per_resource[0];
    assert_eq!(name, "pipe:tensor");
    assert!((cycles - 1000.0 / 0.499987).abs() < 1e-6);
    assert!(r.defaulted.is_empty()); // measured, not defaulted
}

#[test]
fn tensor_pipe_defaults_to_turing_rate_without_a_table() {
    let mut c = Census::default();
    c.add("HMMA.1688.F32", 1000);
    let r = project(&c, 1.0, MemClass::None, &empty_table(), 5.82);
    let (name, cycles) = &r.per_resource[0];
    assert_eq!(name, "pipe:tensor");
    // the shared default, NOT the old alu@2.0 (which read 4x optimistic)
    assert!((cycles - 1000.0 / TENSOR_HMMA_PER_SM_CLK).abs() < 1e-9);
    assert_eq!(r.defaulted, vec!["HMMA.1688.F32".to_string()]);
}

#[test]
fn unmeasured_tensor_shape_costs_at_base_row_but_is_flagged() {
    let mut c = Census::default();
    c.add("HMMA.16816.F32", 100); // Ampere-shaped: no measured row on this table
    let r = project(&c, 1.0, MemClass::None, &table(), 5.82);
    let (name, cycles) = &r.per_resource[0];
    assert_eq!(name, "pipe:tensor");
    assert!((cycles - 100.0 / 0.499987).abs() < 1e-6); // costed at the 1688 row
    assert_eq!(r.defaulted, vec!["HMMA.16816.F32".to_string()]); // ...but flagged
}

#[test]
fn imma_routes_to_tensor_at_its_measured_rate() {
    let mut c = Census::default();
    c.add("IMMA.8816.S8", 1000);
    let r = project(&c, 1.0, MemClass::None, &table(), 5.82);
    let (name, cycles) = &r.per_resource[0];
    assert_eq!(name, "pipe:tensor");
    assert!((cycles - 1000.0 / 0.999914).abs() < 1e-6);
}

#[test]
fn ldsm_is_smem_traffic_not_dram() {
    let mut c = Census::default();
    c.add("LDSM.16.M88.4", 100);
    let r = project(&c, 1.0, MemClass::Dram, &table(), 5.82);
    // x4 = 512 B/warp = 16 B/thread against the 64 B/clk/SM ceiling -> 8 cyc/inst;
    // the model rate 1/8 = 0.125 reproduces the measured tensor.ldsm.tput 0.125007.
    let smem = r.per_resource.iter().find(|(k, _)| k == "smem").expect("smem lane");
    assert!((smem.1 - 800.0).abs() < 1e-9);
    assert!((0.125f64 - 0.125007).abs() / 0.125007 < 1e-3);
    // and no DRAM bytes were charged for it (no `mem` demand at all)
    assert!(r.per_resource.iter().all(|(k, _)| k != "mem"));
}

#[test]
fn ldsm_count_suffix_scales_the_lane() {
    let mut c1 = Census::default();
    c1.add("LDSM.16.M88.1", 100);
    let r1 = project(&c1, 1.0, MemClass::None, &table(), 5.82);
    let s1 = r1.per_resource.iter().find(|(k, _)| k == "smem").unwrap().1;
    assert!((s1 - 200.0).abs() < 1e-9); // x1 -> 2 cyc/inst
    let mut c4 = Census::default();
    c4.add("LDSM.16.M88.4", 100);
    let r4 = project(&c4, 1.0, MemClass::None, &table(), 5.82);
    let s4 = r4.per_resource.iter().find(|(k, _)| k == "smem").unwrap().1;
    assert!((s4 - 4.0 * s1).abs() < 1e-9);
}
