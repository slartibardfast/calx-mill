//! `verify_projection.py` port: the absolute projection gate. PPM within
//! ±20% of measured cycles/iter on every gate kernel; the latency-bound demo
//! is excluded by design and reported with its error. ADD is reported
//! alongside, never gated.

use crate::nvidia::mktable::median;
use crate::nvidia::pattern::Pattern;
use crate::nvidia::projection::{project, Census, MemClass};
use crate::nvidia::sass::{loop_mix, sass_text};
use crate::nvidia::table::Rates;
use std::path::Path;

const TOL: f64 = 0.20;

/// (measured row_id, binary, kernel regex, mem class, warps, gated)
const KERNELS: &[(&str, &str, &str, MemClass, u32, bool)] = &[
    ("proj.anchor.ffma", "bench/proj/anchors.bin", "ffma_anchor", MemClass::None, 8, true),
    ("proj.anchor.stream", "bench/proj/anchors.bin", "stream_anchor", MemClass::Dram, 8, true),
    ("proj.anchor.smemtile", "bench/proj/anchors.bin", "smemtile_anchor", MemClass::None, 8, true),
    ("proj.anchor.capmix", "bench/proj/anchors.bin", "capmix_anchor", MemClass::None, 8, true),
    ("proj.anchor.latbound", "bench/proj/anchors.bin", "latbound_demo", MemClass::None, 1, false),
    ("proj.fa_mini.base", "bench/proj/fa_mini.bin", "fa_mini_kernelILi0", MemClass::L1, 8, true),
    ("proj.fa_mini.dp4a", "bench/proj/fa_mini.bin", "fa_mini_kernelILi1", MemClass::L1, 8, true),
    ("proj.fa_mini.staged", "bench/proj/fa_mini.bin", "fa_mini_kernelILi2", MemClass::L1, 8, true),
];

const INJECT: &[(&str, u32)] = &[
    ("base", 0),
    ("ffma", 8),
    ("ffma", 16),
    ("ffma", 24),
    ("idp4a", 8),
    ("idp4a", 16),
    ("idp4a", 24),
    ("lop3", 8),
    ("lop3", 16),
    ("lop3", 24),
    ("popc", 4),
    ("popc", 8),
    ("popc", 12),
];

fn opsym(op: &str) -> &'static str {
    match op {
        "base" => "OpNONEELi0",
        "ffma" => "OpFFMAELi",
        "idp4a" => "OpIDP4A_S8ELi",
        "lop3" => "OpLOP3ELi",
        "popc" => "OpPOPCELi",
        _ => unreachable!("unknown inject op"),
    }
}

/// A bench path relative to the root; falls back to a cached `.sass`
/// disassembly next to where the binary would be (the parity fixtures carry
/// disassemblies, not binaries).
fn bench_sass(root: &Path, binary: &str) -> std::io::Result<String> {
    let bin = root.join(binary);
    if bin.exists() {
        return sass_text(&bin);
    }
    let sass = bin.with_extension("sass");
    if sass.exists() {
        return sass_text(&sass);
    }
    sass_text(&bin) // let the cuobjdump attempt produce the error
}

/// The gate over a tu102-shaped tree (`table/tu102_ops.csv`,
/// `data/results/t5820-2xrtx6000/proj.csv`, `bench/proj/*`). Returns
/// (report, exit code), byte-for-byte the Python's stdout.
pub fn verify_projection(root: &Path) -> std::io::Result<(String, i32)> {
    let rates = Rates::parse(&std::fs::read_to_string(root.join("table/tu102_ops.csv"))?)
        .map_err(std::io::Error::other)?;
    let results =
        std::fs::read_to_string(root.join("data/results/t5820-2xrtx6000/proj.csv"))?;
    let t = crate::nvidia::csvio::Table::parse(&results);
    let (id_col, val_col) = (t.col("row_id"), t.col("value"));
    let mut meas: Vec<(String, Vec<f64>)> = Vec::new();
    for row in &t.rows {
        let v: f64 = row[val_col].parse().map_err(|_| {
            std::io::Error::other(format!("bad value {:?}", row[val_col]))
        })?;
        match meas.iter_mut().find(|(k, _)| *k == row[id_col]) {
            Some((_, vals)) => vals.push(v),
            None => meas.push((row[id_col].clone(), vec![v])),
        }
    }
    let measured = |row_id: &str| {
        meas.iter().find(|(k, _)| k == row_id).map(|(_, v)| median(v))
    };

    let mut rows: Vec<(String, String, String, MemClass, u32, bool)> = KERNELS
        .iter()
        .map(|&(r, b, x, m, w, g)| {
            (r.to_string(), b.to_string(), x.to_string(), m, w, g)
        })
        .collect();
    for &(op, k) in INJECT {
        let sym = if op == "base" {
            opsym(op).to_string()
        } else {
            format!("{}{}E", opsym(op), k)
        };
        rows.push((
            format!("proj.inject.{}.k{}", op, k),
            "bench/proj/inject.bin".to_string(),
            format!("inject_kernelINS_\\d*{}", sym),
            MemClass::L1,
            8,
            true,
        ));
    }

    let mut out = format!(
        "{:<28} {:>10} {:>10} {:>7}  {:>10} {:<12}\n",
        "kernel", "measured", "PPM", "err", "ADD", "bound"
    );
    let mut failures = 0u32;
    for (row_id, binary, regex, mc, warps, gated) in rows {
        let Some(m) = measured(&row_id) else {
            out.push_str(&format!("{:<28} NO MEASUREMENT — run the bench\n", row_id));
            failures += gated as u32;
            continue;
        };
        let sass = bench_sass(root, &binary)?;
        let (mix, n) = loop_mix(&sass, &Pattern::new(&regex));
        if n == 0 {
            out.push_str(&format!("{:<28} NO KERNEL MATCH ({})\n", row_id, regex));
            failures += gated as u32;
            continue;
        }
        let census: Census = mix;
        let r = project(&census, warps as f64, mc, &rates, 5.82);
        let err = (r.ppm_cycles - m) / m;
        let ok = err.abs() <= TOL;
        let tag = if gated {
            if ok {
                "PASS"
            } else {
                "FAIL"
            }
        } else {
            "demo"
        };
        if gated && !ok {
            failures += 1;
        }
        out.push_str(&format!(
            "{:<28} {:>10.1} {:>10.1} {:>+6.1}%  {:>10.1} {:<12} {}\n",
            row_id,
            m,
            r.ppm_cycles,
            100.0 * err,
            r.add_cycles,
            r.ppm_bound,
            tag
        ));
    }
    out.push_str(&format!("\nverify_projection: {} gate failure(s)\n", failures));
    Ok((out, if failures > 0 { 1 } else { 0 }))
}
