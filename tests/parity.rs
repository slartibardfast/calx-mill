//! Differential tests: the NVIDIA adapter against goldens generated from the
//! reference Python tools (see tests/fixtures/README.md for provenance).

use calx_mill::nvidia::pattern::Pattern;
use calx_mill::nvidia::projection::{project, report, Census, MemClass};
use calx_mill::nvidia::table::{RateRow, Rates};

fn fixture(rel: &str) -> String {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {}", path, e))
}

/// The fixed rates test_project.py pins so its expectations are exact.
fn fixed_rates() -> Rates {
    let row = |id: &str, value: &str, pipe: &str| {
        (id.to_string(), RateRow { value: value.into(), pipe: pipe.into() })
    };
    Rates(vec![
        row("alu.ffma.tput", "2.0", "fma"),
        row("alu.lop3.tput", "2.0", "alu"),
        row("alu.idp4a.tput", "2.0", "fma"),
        row("sfu.mufu.ex2.tput", "0.5", "own_xu"),
    ])
}

fn census(items: &[(&str, u64)]) -> Census {
    let mut c = Census::default();
    for (op, n) in items {
        c.add(op, *n);
    }
    c
}

// The six test_project.py cases. The Python asserts within 1e-6 relative; the
// Rust mirrors the Python's float operations in order, so equality is exact.
#[test]
fn pure_ffma_is_fma_bound() {
    // 128 ops at 2.0/clk -> 64 cycles
    let r = project(&census(&[("FFMA", 128)]), 1.0, MemClass::None, &fixed_rates(), 5.82);
    assert_eq!(r.ppm_cycles, 64.0);
    assert_eq!(r.ppm_bound, "pipe:fma");
}

#[test]
fn even_mix_splits_pipes_and_add_sums_them() {
    // each pipe 32 cycles; issue = 128/4 = 32; PPM = 32 (any), ADD = 64
    let r = project(
        &census(&[("FFMA", 64), ("LOP3", 64)]),
        1.0,
        MemClass::None,
        &fixed_rates(),
        5.82,
    );
    assert_eq!(r.ppm_cycles, 32.0);
    assert_eq!(r.add_cycles, 64.0);
}

#[test]
fn issue_cap_floors_a_four_pipe_spread() {
    // 208 total insts at 1 warp -> issue floor 52 even if pipes say 32 each
    let r = project(
        &census(&[("FFMA", 64), ("LOP3", 64), ("MUFU", 16), ("IDP", 64)]),
        1.0,
        MemClass::None,
        &fixed_rates(),
        5.82,
    );
    let issue = r
        .per_resource
        .iter()
        .find(|(k, _)| k == "issue")
        .map(|(_, v)| *v)
        .expect("issue demand present");
    assert_eq!(issue, (64.0 + 64.0 + 16.0 + 64.0) / 4.0);
}

#[test]
fn dram_traffic_binds_on_the_byte_budget() {
    // 16 LDG.E.U8 per warp = 16*32 B at 5.82 B/clk/SM -> mem = 512/5.82
    let r = project(
        &census(&[("LDG.E.U8", 16), ("FFMA", 8)]),
        1.0,
        MemClass::Dram,
        &fixed_rates(),
        5.82,
    );
    assert_eq!(r.ppm_cycles, 16.0 * 32.0 / 5.82);
    assert_eq!(r.ppm_bound, "mem");
}

#[test]
fn smem_streams_at_the_measured_ceiling() {
    // 64 LDS.32 per warp at 0.5 inst/clk -> 128 cycles
    let r = project(&census(&[("LDS.32", 64)]), 1.0, MemClass::None, &fixed_rates(), 5.82);
    let smem = r
        .per_resource
        .iter()
        .find(|(k, _)| k == "smem")
        .map(|(_, v)| *v)
        .expect("smem demand present");
    assert_eq!(smem, 64.0 / 0.5);
}

#[test]
fn warps_scale_the_projection_linearly() {
    let c = census(&[("FFMA", 128)]);
    let r1 = project(&c, 1.0, MemClass::None, &fixed_rates(), 5.82);
    let r8 = project(&c, 8.0, MemClass::None, &fixed_rates(), 5.82);
    assert_eq!(r8.ppm_cycles, 8.0 * r1.ppm_cycles);
}

// project.py stdout on real censuses and the real measured table, byte for byte.
fn project_golden(census_file: &str, kernel: &str, mem_class: MemClass, golden: &str) {
    let rates = Rates::parse(&fixture("tu102/table/tu102_ops.csv")).expect("table parses");
    let c = Census::from_census_csv(&fixture(census_file), &Pattern::new(kernel));
    assert!(!c.is_empty(), "census matched no ops");
    let r = project(&c, 8.0, mem_class, &rates, 5.82);
    assert_eq!(report(&r), fixture(golden));
}

#[test]
fn ffma_anchor_report_matches_python() {
    project_golden(
        "goldens/census_full_anchors.csv",
        "ffma_anchor",
        MemClass::None,
        "goldens/project_ffma_anchor.txt",
    );
}

#[test]
fn stream_anchor_report_matches_python() {
    project_golden(
        "goldens/census_full_anchors.csv",
        "stream_anchor",
        MemClass::Dram,
        "goldens/project_stream_anchor.txt",
    );
}

#[test]
fn fa_mini_dp4a_report_matches_python() {
    project_golden(
        "goldens/census_full_fa_mini.csv",
        "fa_mini_kernelILi1",
        MemClass::L1,
        "goldens/project_fa_mini_dp4a.txt",
    );
}
