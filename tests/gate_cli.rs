//! The `gate` PROCESS contract (host call/0036): exit codes, required flags, the
//! registry-computed fit, and the A4 compose cap, pinned against the spawned
//! binary — the level a gate manifest actually consumes.

use std::process::{Command, Output};

fn gate(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_calx-mill"))
        .arg("gate")
        .args(args)
        .output()
        .expect("spawn calx-mill")
}

fn code(o: &Output) -> i32 {
    o.status.code().expect("exit code")
}

fn registry() -> String {
    format!("{}/tests/fixtures/registry/anchors.csv", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn exit_codes_pairwise_distinct() {
    let certified = gate(&[
        "--value", "1.0", "--anchor", "1.0", "--tol", "0.1", "--fit-override", "at-anchor",
    ]);
    let provisional = gate(&[
        "--value", "1.0", "--anchor", "1.0", "--tol", "0.1", "--fit-override", "in-domain",
    ]);
    let refused = gate(&[
        "--value", "9.0", "--anchor", "1.0", "--tol", "0.1", "--fit-override", "at-anchor",
    ]);
    let codes = [code(&certified), code(&provisional), code(&refused)];
    assert_eq!(codes, [0, 3, 2]);
    assert!(String::from_utf8_lossy(&certified.stdout).contains("CERTIFIED"));
    assert!(String::from_utf8_lossy(&provisional.stdout).contains("PROVISIONAL"));
    assert!(String::from_utf8_lossy(&refused.stdout).contains("REFUSED"));
}

#[test]
fn zero_arg_gate_is_usage_error() {
    let o = gate(&[]);
    assert_eq!(code(&o), 4);
    assert!(String::from_utf8_lossy(&o.stderr).contains("usage"));
}

#[test]
fn unknown_gate_flag_is_usage_error() {
    // Pre-contract, an unknown flag became a silent boolean and its value fell
    // into ignored positionals; both paths must be usage errors now.
    let o = gate(&["--frobnicate", "1", "--value", "1", "--anchor", "1", "--tol", "1"]);
    assert_eq!(code(&o), 4);
    let o = gate(&["stray", "--value", "1", "--anchor", "1", "--tol", "1"]);
    assert_eq!(code(&o), 4);
}

#[test]
fn retired_self_declared_fit_is_unknown() {
    // The old --fit honor-system flag is retired; it must not silently adjudicate.
    let o = gate(&["--value", "1", "--anchor", "1", "--tol", "1", "--fit", "in-domain"]);
    assert_eq!(code(&o), 4);
}

#[test]
fn tolerance_forms_are_exclusive_and_signed() {
    let both = gate(&[
        "--value", "1", "--anchor", "1", "--tol", "0.1", "--tol-rel", "10",
        "--fit-override", "at-anchor",
    ]);
    assert_eq!(code(&both), 4);
    let neg = gate(&["--value", "1", "--anchor", "1", "--tol", "-0.1", "--fit-override", "at-anchor"]);
    assert_eq!(code(&neg), 4);
    // --tol-rel resolves against |measured|: 10% of 68.19 covers 68.19 -> 63.
    let rel = gate(&[
        "--value", "63", "--anchor", "68.19", "--tol-rel", "10", "--fit-override", "at-anchor",
    ]);
    assert_eq!(code(&rel), 0);
}

#[test]
fn manual_mode_requires_fit_override() {
    let o = gate(&["--value", "1", "--anchor", "1", "--tol", "0.1"]);
    assert_eq!(code(&o), 4);
}

#[test]
fn registry_fit_is_computed() {
    let reg = registry();
    // at the anchor's own query point -> CERTIFIED.
    let at = gate(&[
        "--registry", &reg, "--anchor-id", "mmvq-deep-us", "--value", "60.2",
        "--at", "ctx=4096,batch=1",
    ]);
    assert_eq!(code(&at), 0);
    assert!(String::from_utf8_lossy(&at.stdout).contains("fit: at-anchor (computed"));
    // in the validated domain, a different query -> PROVISIONAL.
    let indom = gate(&[
        "--registry", &reg, "--anchor-id", "mmvq-deep-us", "--value", "60.2",
        "--at", "ctx=2048,batch=1",
    ]);
    assert_eq!(code(&indom), 3);
    // outside the recorded range -> REFUSED, no honesty required.
    let out = gate(&[
        "--registry", &reg, "--anchor-id", "mmvq-deep-us", "--value", "60.2",
        "--at", "ctx=100000,batch=1",
    ]);
    assert_eq!(code(&out), 2);
    // an axis the anchor never covered -> REFUSED.
    let unknown = gate(&[
        "--registry", &reg, "--anchor-id", "mmvq-deep-us", "--value", "60.2",
        "--at", "ctx=4096,batch=1,gpus=2",
    ]);
    assert_eq!(code(&unknown), 2);
}

// The recorded 10^5 case (call/0035): the op-precision claim 1e-7 against the
// measured full-stack KL anchor 2.00e-2 refuses mechanically.
#[test]
fn registry_refuses_the_op_precision_launder() {
    let o = gate(&[
        "--registry", &registry(), "--anchor-id", "fullstack-kl-naive-f16",
        "--value", "1e-7", "--at", "limbs=1,stack=full",
    ]);
    assert_eq!(code(&o), 2);
    assert!(String::from_utf8_lossy(&o.stdout).contains("does not reproduce"));
}

#[test]
fn fit_override_is_loud() {
    let o = gate(&[
        "--registry", &registry(), "--anchor-id", "mmvq-deep-us", "--value", "60.2",
        "--at", "ctx=100000,batch=1", "--fit-override", "in-domain",
    ]);
    assert_eq!(code(&o), 3); // overridden into PROVISIONAL...
    let out = String::from_utf8_lossy(&o.stdout);
    assert!(out.contains("OPERATOR OVERRIDE")); // ...but never silently
    assert!(out.contains("[fit-override]"));
}

#[test]
fn unit_mismatch_refuses() {
    let o = gate(&[
        "--registry", &registry(), "--anchor-id", "mmvq-deep-us", "--value", "60.2",
        "--at", "ctx=4096,batch=1", "--units", "tok/s",
    ]);
    assert_eq!(code(&o), 2);
    assert!(String::from_utf8_lossy(&o.stdout).contains("units"));
}

#[test]
fn registry_conflicts_with_manual_anchor() {
    let o = gate(&[
        "--registry", &registry(), "--anchor-id", "mmvq-deep-us", "--value", "60.2",
        "--at", "ctx=4096,batch=1", "--tol", "5",
    ]);
    assert_eq!(code(&o), 4);
}

#[test]
fn two_certified_compose_to_provisional() {
    let o = gate(&["--compose", "certified,certified"]);
    assert_eq!(code(&o), 3);
    assert!(String::from_utf8_lossy(&o.stdout).contains("COMPOSED"));
    // the meet with a refused input is refused.
    let o = gate(&["--compose", "certified,refused"]);
    assert_eq!(code(&o), 2);
}

#[test]
fn composite_with_own_anchor_stays_certified() {
    let o = gate(&[
        "--compose", "certified,certified",
        "--value", "1.0", "--anchor", "1.0", "--tol", "0.1", "--fit-override", "at-anchor",
    ]);
    assert_eq!(code(&o), 0);
    // a composite anchor that is merely in-domain does NOT lift the cap.
    let o = gate(&[
        "--compose", "certified,certified",
        "--value", "1.0", "--anchor", "1.0", "--tol", "0.1", "--fit-override", "in-domain",
    ]);
    assert_eq!(code(&o), 3);
}
