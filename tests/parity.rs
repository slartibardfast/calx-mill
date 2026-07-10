//! Differential tests: the NVIDIA adapter against goldens generated from the
//! reference Python tools (see tests/fixtures/README.md for provenance).

use calx_mill::nvidia::check::{census_match, check, GateOpts};
use calx_mill::nvidia::mktable::mk_table;
use calx_mill::nvidia::pattern::Pattern;
use calx_mill::nvidia::projection::{project, report, Census, MemClass};
use calx_mill::nvidia::sass::{census_csv, census_per_kernel};
use calx_mill::nvidia::table::{RateRow, Rates};
use calx_mill::nvidia::verify::verify_projection;

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

// sass_census.py CSV output, byte for byte, over the fixture disassemblies.
fn census_golden(sass_file: &str, full: bool, golden: &str) {
    let sass = fixture(sass_file);
    let counts = census_per_kernel(&sass, &Pattern::new(".*"), full);
    assert_eq!(census_csv(&counts), fixture(golden));
}

#[test]
fn census_full_matches_python_on_every_fixture() {
    census_golden("tu102/bench/proj/anchors.sass", true, "goldens/census_full_anchors.csv");
    census_golden("tu102/bench/proj/fa_mini.sass", true, "goldens/census_full_fa_mini.csv");
    census_golden("tu102/bench/proj/inject.sass", true, "goldens/census_full_inject.csv");
    census_golden("tu102/bench/alu/alu.sass", true, "goldens/census_full_alu.csv");
}

#[test]
fn census_base_mnemonics_match_python() {
    census_golden("tu102/bench/proj/anchors.sass", false, "goldens/census_base_anchors.csv");
}

// check_sass.py stdout and exit code over the fixture disassemblies. The
// basename is the .bin the golden was generated from (the exemption list and
// the Python key on it).
fn check_golden(basename: &str, sass_file: Option<&str>, golden: &str, want_exit: i32) {
    let sass = sass_file.map(fixture).unwrap_or_default();
    let (out, exit) = check(basename, &sass, GateOpts::default());
    assert_eq!(out, fixture(golden));
    assert_eq!(exit, want_exit);
}

#[test]
fn purity_gate_matches_python_on_anchors() {
    check_golden(
        "anchors.bin",
        Some("tu102/bench/proj/anchors.sass"),
        "goldens/check_sass_anchors.txt",
        0,
    );
}

#[test]
fn purity_gate_matches_python_on_inject() {
    check_golden(
        "inject.bin",
        Some("tu102/bench/proj/inject.sass"),
        "goldens/check_sass_inject.txt",
        0,
    );
}

#[test]
fn purity_gate_matches_python_on_the_template_kernels() {
    check_golden("alu.bin", Some("tu102/bench/alu/alu.sass"), "goldens/check_sass_alu.txt", 0);
}

#[test]
fn purity_gate_matches_python_on_the_mix_kernels() {
    // pipes.bin carries mix_kernel/pure_kernel instantiations and one genuine
    // FAIL row: the exit code must be 1
    check_golden(
        "pipes.bin",
        Some("tu102/bench/alu/pipes.sass"),
        "goldens/check_sass_pipes.txt",
        1,
    );
}

#[test]
fn exempt_binaries_short_circuit() {
    check_golden("fa_mini.bin", None, "goldens/check_sass_fa_mini.txt", 0);
}

#[test]
fn census_match_matches_python() {
    let a = format!(
        "{}/tests/fixtures/tu102/bench/proj/anchors.sass:ffma_anchor",
        env!("CARGO_MANIFEST_DIR")
    );
    let b = format!(
        "{}/tests/fixtures/tu102/bench/proj/anchors.sass:capmix_anchor",
        env!("CARGO_MANIFEST_DIR")
    );
    let (out, exit) = census_match(&a, &b, 10.0);
    assert_eq!(out, fixture("goldens/census_match.txt"));
    assert_eq!(exit, 1);
}

// mk_table.py regenerates the committed table byte for byte from the
// checked-in measurement CSVs; so must the port.
#[test]
fn mk_table_reproduces_the_committed_table_byte_for_byte() {
    let dir = format!(
        "{}/tests/fixtures/tu102/data/results/t5820-2xrtx6000",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut results = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("results dir exists") {
        let entry = entry.expect("dir entry reads");
        let name = entry.file_name().into_string().expect("utf-8 file name");
        let content = std::fs::read_to_string(entry.path()).expect("file reads");
        results.push((name, content));
    }
    let priors = fixture("tu102/table/priors_t4.csv");
    let na = fixture("tu102/table/na_sm75.csv");
    let (out, n_rows) = mk_table(&results, &priors, Some(&na));
    assert_eq!(n_rows, 264);
    assert_eq!(out, fixture("tu102/table/tu102_ops.csv"));
}

// verify_projection.py's whole gate table (measured medians x hot-loop
// censuses x PPM), byte for byte, including its 6 gate failures on the
// fixture SASS.
#[test]
fn verify_projection_matches_python() {
    let root = format!("{}/tests/fixtures/tu102", env!("CARGO_MANIFEST_DIR"));
    let (out, exit) = verify_projection(std::path::Path::new(&root)).expect("tree reads");
    assert_eq!(out, fixture("goldens/verify_projection.txt"));
    assert_eq!(exit, 1);
}
