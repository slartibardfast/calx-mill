//! SASS text parsing: the op-mix census (`sass_census.py`) and the hot-loop
//! locator shared with the purity gate (`check_sass.py`). Input is `cuobjdump
//! -sass` output, either read from a `.sass` file or piped from cuobjdump.

use crate::nvidia::pattern::Pattern;
use crate::nvidia::projection::Census;

/// One decoded SASS instruction: its address, mnemonic base, and the
/// instruction text up to (not including) the closing `;`.
pub struct Instr {
    pub addr: u64,
    pub base: String,
    pub text: String,
}

fn function_name(line: &str) -> Option<&str> {
    // ^\s*Function : (\S+)
    let t = line.trim_start();
    let rest = t.strip_prefix("Function : ")?;
    let name = rest.split_whitespace().next()?;
    Some(name)
}

/// Scan one line for `/*<hex>*/ <ws> [@!?P<d> <ws>] MNEMONIC...`, returning
/// (addr, byte offset of the predicate-or-mnemonic start, byte offset just
/// past the mnemonic). Mirrors the reference tools' instruction regex,
/// including its refusal of uniform predicates (`@UP0` does not match).
fn scan_instr(line: &str) -> Option<(u64, usize, usize)> {
    let b = line.as_bytes();
    let mut from = 0;
    while let Some(open) = line[from..].find("/*").map(|p| p + from) {
        from = open + 2;
        let mut i = open + 2;
        let hex_start = i;
        while i < b.len() && (b[i].is_ascii_digit() || (b'a'..=b'f').contains(&b[i])) {
            i += 1;
        }
        if i == hex_start || !line[i..].starts_with("*/") {
            continue;
        }
        let addr = u64::from_str_radix(&line[hex_start..i], 16).expect("hex digits parse");
        i += 2;
        let ws_start = i;
        while i < b.len() && (b[i] as char).is_whitespace() {
            i += 1;
        }
        if i == ws_start {
            continue;
        }
        let start = i;
        // optional predicate: @!?P\d+ then whitespace
        if i < b.len() && b[i] == b'@' {
            let mut j = i + 1;
            if j < b.len() && b[j] == b'!' {
                j += 1;
            }
            if j < b.len() && b[j] == b'P' {
                j += 1;
                let digits = j;
                while j < b.len() && b[j].is_ascii_digit() {
                    j += 1;
                }
                if j == digits {
                    continue; // no digits: not a predicate
                }
                let ws = j;
                while j < b.len() && (b[j] as char).is_whitespace() {
                    j += 1;
                }
                if j == ws {
                    continue; // no whitespace after the predicate
                }
                i = j;
            } else {
                continue; // '@' not followed by a plain P-predicate
            }
        }
        // mnemonic: [A-Z][A-Z0-9.]*
        if i >= b.len() || !b[i].is_ascii_uppercase() {
            continue;
        }
        let mn_start = i;
        i += 1;
        while i < b.len() && (b[i].is_ascii_uppercase() || b[i].is_ascii_digit() || b[i] == b'.') {
            i += 1;
        }
        let _ = mn_start;
        return Some((addr, start, i));
    }
    None
}

/// `check_sass.py::parse_functions`: (mangled name, decoded instructions) per
/// function, in file order.
pub fn parse_functions(sass: &str) -> Vec<(String, Vec<Instr>)> {
    let mut out: Vec<(String, Vec<Instr>)> = Vec::new();
    for line in sass.lines() {
        if let Some(name) = function_name(line) {
            out.push((name.to_string(), Vec::new()));
            continue;
        }
        let Some(last) = out.last_mut() else { continue };
        if let Some((addr, start, _)) = scan_instr(line) {
            let tail = &line[start..];
            let text = match tail.find(';') {
                Some(p) => &tail[..p],
                None => tail,
            };
            let text = text.trim().to_string();
            let mnemonic = if text.starts_with('@') {
                text.split_whitespace().nth(1).unwrap_or("")
            } else {
                text.split_whitespace().next().unwrap_or("")
            };
            let base = mnemonic.split('.').next().unwrap_or("").to_string();
            last.1.push(Instr { addr, base, text });
        }
    }
    out
}

/// `check_sass.py::hot_loop`: the largest backward-branch body as an inclusive
/// index pair, first-found on span ties.
pub fn hot_loop(instrs: &[Instr]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for (i, instr) in instrs.iter().enumerate() {
        if instr.base != "BRA" {
            continue;
        }
        let Some(tgt) = bra_target(&instr.text) else { continue };
        if tgt >= instr.addr {
            continue;
        }
        let Some(j) = instrs.iter().position(|x| x.addr == tgt) else { continue };
        if best.is_none_or(|(bj, bi)| (i - j) > (bi - bj)) {
            best = Some((j, i));
        }
    }
    best
}

/// First `BRA\s+0x<hex>` target in the instruction text.
fn bra_target(text: &str) -> Option<u64> {
    let b = text.as_bytes();
    let mut from = 0;
    while let Some(p) = text[from..].find("BRA").map(|p| p + from) {
        from = p + 3;
        let mut i = p + 3;
        let ws = i;
        while i < b.len() && (b[i] as char).is_whitespace() {
            i += 1;
        }
        if i == ws || !text[i..].starts_with("0x") {
            continue;
        }
        i += 2;
        let hex = i;
        while i < b.len() && (b[i].is_ascii_digit() || (b'a'..=b'f').contains(&b[i])) {
            i += 1;
        }
        if i == hex {
            continue;
        }
        return Some(u64::from_str_radix(&text[hex..i], 16).expect("hex digits parse"));
    }
    None
}

/// `check_sass.py::loop_mix`: mnemonic-base counts over the hot-loop bodies of
/// matching kernels, plus how many kernels matched.
pub fn loop_mix(sass: &str, kernel: &Pattern) -> (Census, usize) {
    let mut mix = Census::default();
    let mut matched = 0;
    for (name, instrs) in parse_functions(sass) {
        if !kernel.is_match(&name) {
            continue;
        }
        let Some((lo, hi)) = hot_loop(&instrs) else { continue };
        matched += 1;
        for instr in &instrs[lo..=hi] {
            mix.add(&instr.base, 1);
        }
    }
    (mix, matched)
}

/// `sass_census.py`: per-kernel mnemonic histogram over whole functions.
/// `full` keeps full mnemonics (`FFMA.FTZ`); otherwise bases (`FFMA`).
pub fn census_per_kernel(sass: &str, kernel: &Pattern, full: bool) -> Vec<(String, Census)> {
    let mut counts: Vec<(String, Census)> = Vec::new();
    let mut current: Option<usize> = None;
    for line in sass.lines() {
        if let Some(name) = function_name(line) {
            current = if kernel.is_match(name) {
                match counts.iter().position(|(k, _)| k == name) {
                    Some(i) => Some(i),
                    None => {
                        counts.push((name.to_string(), Census::default()));
                        Some(counts.len() - 1)
                    }
                }
            } else {
                None
            };
            continue;
        }
        let Some(idx) = current else { continue };
        if let Some((_, start, mn_end)) = scan_instr(line) {
            let tail = &line[start..mn_end];
            let mnemonic = if tail.starts_with('@') {
                // scan_instr's span starts at the predicate; the mnemonic is
                // the last whitespace-separated token in the span
                tail.split_whitespace().last().unwrap_or("")
            } else {
                tail
            };
            let op = if full { mnemonic } else { mnemonic.split('.').next().unwrap_or("") };
            counts[idx].1.add(op, 1);
        }
    }
    counts
}

/// `sass_census.py`'s CSV output: kernels sorted, ops by descending count
/// (insertion order on ties, as `Counter.most_common`).
pub fn census_csv(counts: &[(String, Census)]) -> String {
    let mut out = String::from("kernel,op,count,share_pct\n");
    let mut kernels: Vec<&(String, Census)> = counts.iter().collect();
    kernels.sort_by(|a, b| a.0.cmp(&b.0));
    for (kernel, census) in kernels {
        let total: u64 = census.items.iter().map(|(_, n)| n).sum();
        let mut items: Vec<&(String, u64)> = census.items.iter().collect();
        items.sort_by(|a, b| b.1.cmp(&a.1));
        for (op, n) in items {
            out.push_str(&format!(
                "{},{},{},{:.2}\n",
                kernel,
                op,
                n,
                (100.0 * *n as f64) / total as f64
            ));
        }
    }
    out
}

/// Disassembly acquisition: a `.sass` file is read as text; anything else is
/// piped through `cuobjdump -sass` (override the binary with `CUOBJDUMP`).
pub fn sass_text(path: &std::path::Path) -> std::io::Result<String> {
    if path.extension().is_some_and(|e| e == "sass") {
        return std::fs::read_to_string(path);
    }
    let cuobjdump = std::env::var("CUOBJDUMP").unwrap_or_else(|_| "cuobjdump".into());
    let out = std::process::Command::new(&cuobjdump)
        .arg("-sass")
        .arg(path)
        .output()?;
    if !out.status.success() {
        return Err(std::io::Error::other(format!(
            "{} -sass {} failed: {}",
            cuobjdump,
            path.display(),
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    String::from_utf8(out.stdout).map_err(std::io::Error::other)
}
