//! Parser-loss surfaces at the process level: the telemetry strict mode and the
//! census uniform-datapath drop report. The in-module unit tests cover the
//! counting itself; these pin the CLI behaviour.

use std::process::{Command, Output};

fn calx(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_calx-mill"))
        .args(args)
        .output()
        .expect("spawn calx-mill")
}

fn write_scratch(name: &str, content: &str) -> String {
    let path = std::env::temp_dir().join(format!("calx-parsers-{}-{}", std::process::id(), name));
    std::fs::write(&path, content).expect("write scratch fixture");
    path.to_string_lossy().into_owned()
}

const TRUNCATED_TELE: &str = "\
op_index\top_kind\tlane\tgt_start_ns\tgt_end_ns\tcycles\tbytes\tops
6\tMMVQ_Q4_0\tmem\t100\t200\t75282\t0\t0
7\tGEMV_F16\tmem\t100\t200
";

#[test]
fn telemetry_reports_skipped_rows_and_strict_fails() {
    let path = write_scratch("truncated.tele", TRUNCATED_TELE);
    let lax = calx(&["telemetry", &path]);
    assert_eq!(lax.status.code(), Some(0)); // default: reported, not fatal
    assert!(String::from_utf8_lossy(&lax.stderr).contains("1 malformed rows skipped"));
    let strict = calx(&["telemetry", &path, "--strict"]);
    assert_eq!(strict.status.code(), Some(1)); // a lossy parse is a failed run
    let _ = std::fs::remove_file(&path);
}

const UP_PREDICATED_SASS: &str = "\
\tFunction : test_kernel
\t/*0000*/    FFMA R0, R1, R2, R3 ;
\t/*0010*/    @UP0 LDS R4, [R5] ;
\t/*0020*/    @!UP1 STS [R5], R4 ;
\t/*0030*/    @P0 FADD R6, R7, R8 ;
";

#[test]
fn census_reports_uniform_datapath_drops_on_stderr_only() {
    let path = write_scratch("up.sass", UP_PREDICATED_SASS);
    let o = calx(&["census", &path]);
    assert_eq!(o.status.code(), Some(0));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("2 uniform-datapath predicated instructions dropped"));
    // stdout (the golden-pinned CSV) carries the counted ops, no report line:
    // FFMA and the @P0 FADD count; the two @UP* lines cost zero.
    let out = String::from_utf8_lossy(&o.stdout);
    assert!(out.contains("FFMA") && out.contains("FADD"));
    assert!(!out.contains("LDS") && !out.contains("dropped"));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn unsupported_kernel_pattern_is_an_error_not_a_silent_miss() {
    let path = write_scratch("plain.sass", UP_PREDICATED_SASS);
    let o = calx(&["census", &path, "--kernel", "[0-9]"]);
    assert_eq!(o.status.code(), Some(2)); // an error, not "no kernels matched"
    assert!(String::from_utf8_lossy(&o.stderr).contains("unsupported regex construct"));
    let _ = std::fs::remove_file(&path);
}
