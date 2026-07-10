//! `check_sass.py` port: the purity gate (each bench kernel's timed loop
//! contains exactly the intended SASS op) and the census-match mode (two
//! kernels' hot-loop pipe-class shares within tolerance). Table order is the
//! Python dict order; the first-substring-wins matchers bind on it.

use crate::nvidia::pattern::Pattern;
use crate::nvidia::pyfmt::repr_f64;
use crate::nvidia::sass::{hot_loop, loop_mix, parse_functions, sass_text, Instr};

/// loop-control instructions every timed loop legitimately contains
const CONTROL: &[&str] = &["IADD3", "ISETP", "BRA", "NOP"];

/// binaries the purity gate cannot meaningfully bind, each with its reason
const EXEMPT_BINARIES: &[(&str, &str)] = &[
    ("launch.bin", "empty/identity kernels by design; the rows are host-side dispatch"),
    ("marshal.bin", "argument-marshalling probe; kernels empty by design"),
    (
        "l2bw.bin",
        "loop-structure parse quirk; the row self-validates via the \
         cg-vs-default policy contrast (382 vs 1110 GB/s)",
    ),
    ("fa_mini.bin", "composite kernels; gated by census-match mode instead"),
    ("nccl_pcie.bin", "host-side NCCL/cudaMemcpy timing; no timed device loops"),
    ("icache.bin", "loop bodies sized to exceed L0 BY DESIGN (the measurement)"),
];

pub struct Expect {
    pub primary: &'static [&'static str],
    pub companions: &'static [&'static str],
    pub min: Option<u32>,
}

const fn exp(primary: &'static [&'static str]) -> Option<Expect> {
    Some(Expect { primary, companions: &[], min: None })
}

const fn exp_c(
    primary: &'static [&'static str],
    companions: &'static [&'static str],
) -> Option<Expect> {
    Some(Expect { primary, companions, min: None })
}

const fn exp_m(
    primary: &'static [&'static str],
    companions: &'static [&'static str],
    min: u32,
) -> Option<Expect> {
    Some(Expect { primary, companions, min: Some(min) })
}

/// Op struct name (in the mangled kernel symbol) -> expectation. `None` is a
/// presence-only sequence (no purity gate).
const EXPECT: &[(&str, Option<Expect>)] = &[
    ("OpFFMA", exp(&["FFMA"])),
    ("OpFADD", exp(&["FADD"])),
    ("OpFMUL", exp(&["FMUL"])),
    ("OpIADD3", exp(&["IADD3", "LOP3"])),
    ("OpIMAD", exp(&["IMAD"])),
    ("OpLOP3", exp(&["LOP3"])),
    ("OpSHF", exp(&["SHF"])),
    ("OpSEL", exp_c(&["SEL"], &["ISETP"])),
    ("OpISETPSEL", exp(&["ISETP", "SEL"])),
    ("OpFSETPSEL", exp(&["FSETP", "SEL", "FSEL"])),
    ("OpFSEL", exp_c(&["SEL", "FSEL"], &["FSETP"])),
    ("OpPOPC", exp(&["POPC"])),
    ("OpFLO", exp(&["FLO"])),
    ("OpPRMT", exp(&["PRMT"])),
    ("OpIDP4A_S8", exp(&["IDP"])),
    ("OpIDP4A_U8", exp(&["IDP"])),
    ("OpHFMA2", exp(&["HFMA2"])),
    ("OpDADD", exp(&["DADD"])),
    ("OpDFMA", exp(&["DFMA"])),
    ("OpIDIV_U32", None),
];

/// Bespoke (non-template) bench kernels, matched by function-name substring,
/// first entry wins - order copied from the Python.
const EXPECT_FN: &[(&str, Option<Expect>)] = &[
    ("smem_chase_kernel", exp(&["LDS"])),
    ("smem_conflict_kernel", exp(&["LDS"])),
    (
        "smem_bw_kernel",
        exp_c(&["LDS"], &["FADD", "FFMA", "IMAD", "LOP3", "LEA", "SHF", "MOV"]),
    ),
    ("l1_chase_kernel", exp(&["LDG"])),
    ("pchase_kernel", exp(&["LDG"])),
    ("peer_chase_kernel", exp(&["LDG"])),
    ("rt_initiator", None),
    ("rt_responder", None),
    ("vis_initiator", None),
    ("vis_responder", None),
    ("peer_atom_chase", exp_m(&["ATOMG"], &[], 8)),
    ("stream_writer", None),
    (
        "local_read_bw",
        exp_m(&["LDG"], &["FADD", "IMAD", "LEA", "SHF", "MOV", "LOP3", "SEL"], 4),
    ),
    ("peer_ring_init", None),
    (
        "peer_read_bw_kernel",
        exp_m(&["LDG"], &["FADD", "IMAD", "LEA", "SHF", "MOV", "LOP3"], 4),
    ),
    ("peer_write_bw_kernel", exp_m(&["STG"], &["IMAD", "LEA", "SHF", "MOV", "LOP3"], 1)),
    (
        "peer_burst_kernel",
        exp_m(
            &["STG"],
            &["MEMBAR", "CCTL", "ERRBAR", "IMAD", "LEA", "SHF", "MOV", "LOP3", "SEL"],
            0,
        ),
    ),
    ("policy_chase_kernel", exp(&["LDG"])),
    ("policy_ring_init", None),
    ("carveout_chase_kernel", exp(&["LDG"])),
    ("carveout_ring_init", None),
    ("pchase_ring_init", None),
    ("tlb_chase_kernel", exp(&["LDG"])),
    ("tlb_ring_init", None),
    ("tex_chase_kernel", exp(&["LDG"])),
    ("tex_ring_init", None),
    ("icache_kernel", exp(&["FFMA"])),
    ("hmma_f16_tput_kernel", exp(&["HMMA"])),
    ("hmma_f32_tput_kernel", exp(&["HMMA"])),
    ("hmma_f16_lat_kernel", exp(&["HMMA"])),
    ("imma_s8_tput_kernel", exp(&["IMMA"])),
    ("atom_shared_lat_kernel", exp_m(&["ATOMS"], &[], 16)),
    ("atom_global_lat_kernel", exp_m(&["ATOMG"], &[], 16)),
    ("atom_cas_lat_kernel", exp_m(&["ATOMG"], &[], 16)),
    ("const_chase_div_kernel", exp(&["LDC"])),
    ("const_chase_kernel", exp(&["ULDC"])),
    (
        "dram_read_kernel",
        exp_m(&["LDG"], &["FADD", "IMAD", "LEA", "SHF", "MOV", "LOP3"], 4),
    ),
    ("dram_write_kernel", exp_m(&["STG"], &["IMAD", "LEA", "SHF", "MOV", "LOP3"], 1)),
    (
        "dram_copy_kernel",
        exp_m(&["LDG", "STG"], &["IMAD", "LEA", "SHF", "MOV", "LOP3"], 2),
    ),
    ("l2bw_kernel", None),
    (
        "l1bw_kernel",
        exp_c(&["LDG"], &["FADD", "FFMA", "IMAD", "LOP3", "LEA", "SHF", "MOV"]),
    ),
    (
        "stride_kernel",
        exp_m(&["LDG"], &["IMAD", "LEA", "SHF", "LOP3", "MOV", "IADD3"], 12),
    ),
    ("cvt_f2f", exp(&["F2F", "HADD2"])),
    ("cvt_i2f", exp(&["F2I", "I2F"])),
    ("cvt_derived_lat_kernel", exp_m(&["F2F", "HADD2", "F2I", "I2F"], &["FADD"], 32)),
    ("f2f_pair", exp(&["F2F", "HADD2"])),
    ("i2f_pair", exp(&["F2I", "I2F"])),
    ("mufu_ex2", exp(&["MUFU"])),
    ("MufuSinE", exp(&["MUFU", "FMUL"])),
    ("MufuRcpE", exp(&["MUFU", "FFMA", "FADD"])),
    ("MufuCosE", exp(&["MUFU", "FMUL"])),
    ("mufu_lat_kernel", exp(&["MUFU"])),
    ("mufu_tput_kernel", exp(&["MUFU"])),
    ("bar_kernel", exp_m(&["BAR"], &[], 8)),
    (
        "bar_direct_kernel",
        exp_m(&["BAR"], &["CS2R", "S2R", "MOV", "SHF", "LEA", "IMAD"], 1),
    ),
    ("bar_tput_kernel", exp_m(&["BAR"], &[], 8)),
    (
        "vote_lat_kernel",
        exp_m(
            &["VOTE", "VOTEU"],
            &["SHF", "LOP3", "ISETP", "MOV", "PLOP3", "P2R", "R2P", "SEL"],
            16,
        ),
    ),
    (
        "vote_tput_kernel",
        exp_m(
            &["VOTE", "VOTEU"],
            &["SHF", "LOP3", "ISETP", "MOV", "PLOP3", "P2R", "R2P", "SEL"],
            16,
        ),
    ),
    ("ldsm_lat_kernel", exp_m(&["LDSM"], &["LOP3", "MOV", "SHF", "LEA", "IMAD"], 16)),
    ("ldsm_tput_kernel", exp_m(&["LDSM"], &["LOP3", "MOV", "SHF", "LEA", "IMAD"], 16)),
    ("line_chase_kernel", exp(&["LDG"])),
    ("line_ring_init", None),
    ("atom_shared_cas_lat_kernel", exp_m(&["ATOMS"], &[], 16)),
    ("atom_shared_tput_kernel", exp_m(&["ATOMS", "REDS"], &[], 16)),
    ("atom_global_tput_kernel", exp_m(&["RED", "ATOMG"], &[], 16)),
    ("peer_cas_chase", exp_m(&["ATOMG"], &[], 8)),
    ("peer_atom_tput", exp_m(&["RED", "ATOMG"], &[], 8)),
    ("empty_kernel", None),
    ("k_noargs", None),
    ("k_args", None),
    ("shfl_lat_kernel", exp(&["SHFL"])),
    (
        "branch_div_kernel",
        exp_m(
            &["FFMA"],
            &["BSSY", "BSYNC", "SEL", "MOV", "IMAD", "PLOP3", "SHF", "LOP3"],
            4,
        ),
    ),
    (
        "branch_pred_kernel",
        exp_m(&["FFMA"], &["SEL", "MOV", "IMAD", "PLOP3", "SHF", "LOP3"], 4),
    ),
    ("shfl_tput_kernel", exp(&["SHFL"])),
    ("fa_mini_kernel", None),
    ("ffma_anchor", exp(&["FFMA"])),
    (
        "stream_anchor",
        exp_m(&["LDG"], &["FADD", "IMAD", "LEA", "SHF", "LOP3", "MOV"], 12),
    ),
    (
        "smemtile_anchor",
        exp_m(&["LDS", "FFMA"], &["IMAD", "LEA", "SHF", "LOP3", "MOV"], 48),
    ),
    ("capmix_anchor", exp(&["FFMA", "LOP3"])),
    ("latbound_demo", exp(&["FFMA"])),
    (
        "mixp_popc_ldg",
        exp_m(&["POPC", "LDG"], &["FADD", "IMAD", "LEA", "SHF", "MOV", "LOP3"], 64),
    ),
    ("mixp_hmma_hfma2", exp_m(&["HMMA", "HFMA2"], &[], 32)),
    (
        "inject_kernel",
        exp_m(
            &["FFMA", "LOP3", "LDG"],
            &["IDP", "POPC", "SEL", "ISETP", "IMAD", "LEA", "MOV", "SHF", "FADD"],
            40,
        ),
    ),
];

/// pipe-class groups for the census-match mode (proxy-validity gate)
const MATCH_GROUPS: &[(&str, &[&str])] = &[
    ("fma", &["FFMA", "FADD", "FMUL", "IMAD", "IDP", "FSETP", "FMNMX"]),
    ("half", &["HADD2", "HMUL2", "HFMA2", "HMNMX2"]),
    (
        "alu",
        &[
            "IADD3", "LOP3", "SHF", "SEL", "ISETP", "PRMT", "LEA", "MOV", "FLO", "POPC",
            "BFE", "BFI", "IABS", "IMNMX",
        ],
    ),
    ("xu", &["MUFU", "F2F", "F2I", "I2F", "I2I"]),
    ("lsu", &["LDG", "STG", "LDS", "STS", "LDC", "LDSM", "LDL", "STL"]),
    (
        "control",
        &[
            "BRA", "NOP", "EXIT", "BSSY", "BSYNC", "CS2R", "S2R", "BAR", "DEPBAR",
            "YIELD", "WARPSYNC", "RET",
        ],
    ),
];

#[derive(Clone, Copy)]
pub struct GateOpts {
    /// minimum primary-op count in the loop body
    pub min_primary: u32,
    /// loop-body size budget (L0 i-cache fit)
    pub l0_bytes: u64,
    /// max non-primary non-control instrs tolerated in the loop
    pub staging_budget: usize,
}

impl Default for GateOpts {
    fn default() -> GateOpts {
        GateOpts { min_primary: 64, l0_bytes: 8192, staging_budget: 6 }
    }
}

fn sorted_dedup(items: &[&str]) -> Vec<String> {
    let mut v: Vec<String> = items.iter().map(|s| s.to_string()).collect();
    v.sort();
    v.dedup();
    v
}

fn body_size(body: &[Instr]) -> u64 {
    body.last().expect("body is non-empty").addr - body[0].addr + 16
}

struct GateCase<'a> {
    label: String,
    expect: Option<&'a Expect>,
    /// `Some(min)`: the bespoke-kernel form of the failure strings; `None`:
    /// the template form (which also prints the primary list and the L0 budget)
    fn_min: Option<u32>,
    /// mix kernels build their expectation on the fly
    union: Option<(Vec<&'static str>, Vec<&'static str>)>,
}

/// The purity gate over one binary's disassembly: returns (report, exit code),
/// byte-for-byte the Python's stdout. `basename` is the binary's file name
/// (the exemption list keys on it).
pub fn check(basename: &str, sass: &str, opts: GateOpts) -> (String, i32) {
    if let Some((_, reason)) = EXEMPT_BINARIES.iter().find(|(b, _)| *b == basename) {
        return (format!("EXEMPT {}: {}\n", basename, reason), 0);
    }

    let mut out = String::new();
    let mut failures = 0u32;
    let mut checked = 0u32;
    for (name, instrs) in parse_functions(sass) {
        let case = if let Some((key, e)) = EXPECT_FN.iter().find(|(k, _)| name.contains(k)) {
            let Some(e) = e else { continue };
            GateCase {
                label: format!("fn<{}>", key),
                expect: Some(e),
                fn_min: Some(e.min.unwrap_or(opts.min_primary)),
                union: None,
            }
        } else {
            let Some(kind) = ["lat_kernel", "tput_kernel", "pure_kernel", "mix_kernel"]
                .iter()
                .find(|k| name.contains(*k))
            else {
                continue;
            };
            let mut matches: Vec<&(&str, Option<Expect>)> = EXPECT
                .iter()
                .filter(|(o, _)| name.contains(&format!("{}E", o)))
                .collect();
            // drop substring shadows (OpSEL inside OpISETPSEL's symbol, ...)
            let shadowed: Vec<&str> = matches
                .iter()
                .filter(|(o, _)| {
                    matches.iter().any(|(p, _)| {
                        o != p && p.contains(o) && name.contains(&format!("{}E", p))
                    })
                })
                .map(|(o, _)| *o)
                .collect();
            matches.retain(|(o, _)| !shadowed.contains(o));
            if matches.is_empty() {
                continue;
            }
            if *kind == "mix_kernel" {
                if matches.iter().any(|(_, e)| e.is_none()) {
                    continue;
                }
                let mut names: Vec<&str> = matches.iter().map(|(o, _)| *o).collect();
                names.sort();
                let primary: Vec<&'static str> = matches
                    .iter()
                    .flat_map(|(_, e)| e.as_ref().expect("filtered above").primary)
                    .copied()
                    .collect();
                let companions: Vec<&'static str> = matches
                    .iter()
                    .flat_map(|(_, e)| e.as_ref().expect("filtered above").companions)
                    .copied()
                    .collect();
                GateCase {
                    label: format!("{}<{}>", kind, names.join("+")),
                    expect: None,
                    fn_min: None,
                    union: Some((primary, companions)),
                }
            } else {
                // Python max(matches, key=len): the FIRST longest match wins
                let mut best = 0;
                for i in 1..matches.len() {
                    if matches[i].0.len() > matches[best].0.len() {
                        best = i;
                    }
                }
                let (op, e) = (matches[best].0, &matches[best].1);
                GateCase {
                    label: format!("{}<{}>", kind, op),
                    expect: e.as_ref(),
                    fn_min: None,
                    union: None,
                }
            }
        };

        checked += 1;
        let Some((lo, hi)) = hot_loop(&instrs) else {
            out.push_str(&format!("FAIL {}: no loop found\n", case.label));
            failures += 1;
            continue;
        };
        let body = &instrs[lo..=hi];
        let size = body_size(body);

        let (primary, companions, min_primary, is_fn) = match (&case.union, case.expect) {
            (Some((p, c)), _) => (p.clone(), c.clone(), opts.min_primary, false),
            (None, Some(e)) => (
                e.primary.to_vec(),
                e.companions.to_vec(),
                case.fn_min.unwrap_or(opts.min_primary),
                case.fn_min.is_some(),
            ),
            (None, None) => {
                // template sequence op: size-gated only
                if size > opts.l0_bytes {
                    out.push_str(&format!(
                        "FAIL {}: loop body {} B exceeds L0 budget {} B\n",
                        case.label, size, opts.l0_bytes
                    ));
                    failures += 1;
                } else {
                    out.push_str(&format!(
                        "PASS {}: sequence op, {} instrs, {} B (no purity gate)\n",
                        case.label,
                        body.len(),
                        size
                    ));
                }
                continue;
            }
        };

        if is_fn {
            if size > opts.l0_bytes {
                out.push_str(&format!(
                    "FAIL {}: loop body {} B exceeds L0 budget\n",
                    case.label, size
                ));
                failures += 1;
                continue;
            }
        } else if size > opts.l0_bytes {
            out.push_str(&format!(
                "FAIL {}: loop body {} B exceeds L0 budget {} B\n",
                case.label, size, opts.l0_bytes
            ));
            failures += 1;
            continue;
        }

        let n_primary = body.iter().filter(|x| primary.contains(&x.base.as_str())).count();
        let aliens: Vec<&str> = body
            .iter()
            .map(|x| x.base.as_str())
            .filter(|b| {
                !primary.contains(b) && !companions.contains(b) && !CONTROL.contains(b)
            })
            .collect();
        if (n_primary as u32) < min_primary {
            if is_fn {
                out.push_str(&format!(
                    "FAIL {}: only {} primary ops in loop\n",
                    case.label, n_primary
                ));
            } else {
                let mut ps = primary.clone();
                ps.sort();
                out.push_str(&format!(
                    "FAIL {}: only {} primary ops ({}) in loop, need >= {}\n",
                    case.label,
                    n_primary,
                    ps.join("/"),
                    opts.min_primary
                ));
            }
            failures += 1;
        } else if aliens.len() > opts.staging_budget {
            let listed = sorted_dedup(&aliens).join(" ");
            if is_fn {
                out.push_str(&format!(
                    "FAIL {}: {} non-primary ops (budget {}): {}\n",
                    case.label,
                    aliens.len(),
                    opts.staging_budget,
                    listed
                ));
            } else {
                out.push_str(&format!(
                    "FAIL {}: {} non-primary ops in timed loop (budget {}): {}\n",
                    case.label,
                    aliens.len(),
                    opts.staging_budget,
                    listed
                ));
            }
            failures += 1;
        } else {
            let extra = if aliens.is_empty() {
                String::new()
            } else if is_fn {
                format!(" (+{} staging)", aliens.len())
            } else {
                format!(" (+{} staging: {})", aliens.len(), sorted_dedup(&aliens).join(" "))
            };
            out.push_str(&format!(
                "PASS {}: {} primary, {} instrs, {} B{}\n",
                case.label,
                n_primary,
                body.len(),
                size,
                extra
            ));
        }
    }

    if checked == 0 {
        out.push_str("FAIL: no bench kernels found in binary\n");
        return (out, 1);
    }
    out.push_str(&format!("{} kernels checked, {} failures\n", checked, failures));
    (out, if failures > 0 { 1 } else { 0 })
}

/// One `--census-match` side: `path:kernel_regex` (an empty regex means all).
fn shares(spec: &str) -> Result<(Vec<(String, f64)>, usize), String> {
    let (path, regex) = match spec.split_once(':') {
        Some((p, r)) => (p, if r.is_empty() { ".*" } else { r }),
        None => (spec, ".*"),
    };
    let sass = sass_text(std::path::Path::new(path)).map_err(|e| e.to_string())?;
    let (mix, matched) = loop_mix(&sass, &Pattern::new(regex));
    if matched == 0 {
        return Err(format!("FAIL census-match: no kernels matched in {}\n", spec));
    }
    let total: u64 = mix.items.iter().map(|(_, n)| n).sum();
    let mut g: Vec<(String, f64)> = MATCH_GROUPS
        .iter()
        .map(|(k, _)| (k.to_string(), 0.0))
        .chain(std::iter::once(("other".to_string(), 0.0)))
        .collect();
    for (base, n) in &mix.items {
        let grp = MATCH_GROUPS
            .iter()
            .find(|(_, ops)| ops.contains(&base.as_str()))
            .map(|(k, _)| *k)
            .unwrap_or("other");
        let slot = g.iter_mut().find(|(k, _)| k == grp).expect("group exists");
        slot.1 += 100.0 * *n as f64 / total as f64;
    }
    Ok((g, matched))
}

/// `check_sass.py --census-match`: group shares within tolerance -> exit 0.
pub fn census_match(spec_a: &str, spec_b: &str, tolerance_pp: f64) -> (String, i32) {
    let (ga, na) = match shares(spec_a) {
        Ok(v) => v,
        Err(msg) => return (msg, 1),
    };
    let (gb, nb) = match shares(spec_b) {
        Ok(v) => v,
        Err(msg) => return (msg, 1),
    };
    let mut out = format!(
        "census-match: A={} kernel(s), B={} kernel(s); tolerance +-{}pp\n",
        na,
        nb,
        repr_f64(tolerance_pp)
    );
    let mut worst = 0.0f64;
    let mut fail = 0;
    for (i, (grp, a)) in ga.iter().enumerate() {
        let b = gb[i].1;
        let d = (a - b).abs();
        if d > worst {
            worst = d;
        }
        let status = if d <= tolerance_pp { "ok" } else { "EXCEEDS" };
        if d > tolerance_pp {
            fail = 1;
        }
        out.push_str(&format!(
            "  {:<8} A {:>5.1}%  B {:>5.1}%  |d| {:>5.1}pp  {}\n",
            grp, a, b, d, status
        ));
    }
    out.push_str(&format!(
        "census-match: {} (worst {:.1}pp)\n",
        if fail != 0 { "FAIL" } else { "PASS" },
        worst
    ));
    (out, fail)
}
