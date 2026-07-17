//! `project.py` port: a SASS op-mix census folded through the measured rate
//! table into per-resource cycle demands, projected by the core's fractional
//! per-pipe-max. The mnemonic -> (pipe, table row) map is the measured
//! contention-probe map; ops without a measured rate fall back to their pipe's
//! class rate with a `defaulted` marker, exactly as the Python does.

use crate::nvidia::csvio::Table;
use crate::nvidia::pattern::Pattern;
use crate::nvidia::table::Rates;
use crate::ResourceKind;

/// instructions per SM per cycle (4 schedulers, single issue)
pub const ISSUE_CAP: f64 = 4.0;

/// mnemonic base -> table row for the rate (BAR has no rate row: it is
/// latency-handled and follows the defaulted path, as in the Python).
const OP_TABLE_ROW: &[(&str, &str)] = &[
    ("FFMA", "alu.ffma.tput"),
    ("FADD", "alu.fadd.tput"),
    ("FMUL", "alu.fmul.tput"),
    ("IMAD", "alu.imad.tput"),
    ("IDP", "alu.idp4a.tput"),
    ("HFMA2", "alu.hfma2.tput"),
    ("LOP3", "alu.lop3.tput"),
    ("SHF", "alu.shf.tput"),
    ("SEL", "alu.sel.tput"),
    ("IADD3", "alu.iadd3.tput"),
    ("ISETP", "alu.isetp.tput"),
    ("PRMT", "alu.prmt.tput"),
    ("POPC", "alu.popc.tput"),
    ("FLO", "alu.flo.tput"),
    ("DADD", "alu.dadd.tput"),
    ("DFMA", "alu.dfma.tput"),
    ("MUFU", "sfu.mufu.ex2.tput"),
    ("F2F", "cvt.f2f.tput"),
    ("HADD2", "cvt.f2f.tput"),
    ("F2I", "cvt.i2f_f2i.tput"),
    ("I2F", "cvt.i2f_f2i.tput"),
    ("HMMA", "tensor.hmma.1688.tput"),
    ("IMMA", "tensor.imma.8816.tput"),
];

/// Turing tensor-pipe rate in warpinst/SM/clk (PAPER.md tab:tensor, m16n8k8) — the
/// unmeasured-table default for the `tensor` pipe, and the one constant the
/// telemetry side's GPU-global HMMA denominator is derived from (the two halves
/// must not disagree about what a tensor op costs).
pub const TENSOR_HMMA_PER_SM_CLK: f64 = 0.5;

const PIPE_OF: &[(&str, &str)] = &[
    ("HMMA", "tensor"),
    ("IMMA", "tensor"),
    ("BMMA", "tensor"),
    ("FFMA", "fma"),
    ("FADD", "fma"),
    ("FMUL", "fma"),
    ("IMAD", "fma"),
    ("IDP", "fma"),
    ("FSETP", "fma"),
    ("FSEL", "alu"),
    ("FMNMX", "fma"),
    ("HADD2", "own"),
    ("HMUL2", "own"),
    ("HFMA2", "own"),
    ("LOP3", "alu"),
    ("SHF", "alu"),
    ("SEL", "alu"),
    ("IADD3", "alu"),
    ("ISETP", "alu"),
    ("PRMT", "alu"),
    ("LEA", "alu"),
    ("MOV", "alu"),
    ("BFE", "alu"),
    ("BFI", "alu"),
    ("POPC", "own_xu"),
    ("FLO", "own_xu"),
    ("MUFU", "own_xu"),
    ("F2F", "own_xu"),
    ("F2I", "own_xu"),
    ("I2F", "own_xu"),
    ("DADD", "own_fp64"),
    ("DFMA", "own_fp64"),
    ("DMUL", "own_fp64"),
];

const CONTROL: &[&str] = &["BRA", "NOP", "EXIT", "CS2R", "S2R", "BSSY", "BSYNC", "YIELD"];
const LSU: &[&str] = &["LDG", "STG", "LDS", "STS", "LDC", "LDL", "STL", "LDSM"];

/// bytes per thread for memory mnemonics by width suffix
const WIDTH_BYTES: &[(&str, u64)] = &[
    ("8", 1),
    ("U8", 1),
    ("S8", 1),
    ("16", 2),
    ("U16", 2),
    ("S16", 2),
    ("32", 4),
    ("64", 8),
    ("128", 16),
];

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MemClass {
    None,
    Dram,
    L1,
}

impl MemClass {
    pub fn parse(s: &str) -> Option<MemClass> {
        match s {
            "none" => Some(MemClass::None),
            "dram" => Some(MemClass::Dram),
            "l1" => Some(MemClass::L1),
            _ => None,
        }
    }
}

/// An insertion-ordered counter: Python's `collections.Counter` keeps first-seen
/// key order, and the projection's float accumulation order (and tie-breaks)
/// depend on it.
#[derive(Default, Clone)]
pub struct Census {
    pub items: Vec<(String, u64)>,
}

impl Census {
    pub fn add(&mut self, op: &str, n: u64) {
        if let Some((_, count)) = self.items.iter_mut().find(|(k, _)| k == op) {
            *count += n;
        } else {
            self.items.push((op.to_string(), n));
        }
    }

    /// `project.py::census_from_csv`: sum the `count` column over rows whose
    /// kernel matches.
    pub fn from_census_csv(text: &str, kernel: &Pattern) -> Census {
        let t = Table::parse(text);
        let (k, op, count) = (t.col("kernel"), t.col("op"), t.col("count"));
        let mut census = Census::default();
        for row in &t.rows {
            if kernel.is_match(&row[k]) {
                census.add(&row[op], row[count].parse().expect("count is an integer"));
            }
        }
        census
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

pub struct ProjectResult {
    pub ppm_cycles: f64,
    pub ppm_bound: String,
    pub add_cycles: f64,
    pub per_resource: Vec<(String, f64)>,
    pub defaulted: Vec<String>,
}

fn lookup<'a>(map: &'a [(&str, &str)], key: &str) -> Option<&'a str> {
    map.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

fn mem_bytes(mnemonic_full: &str) -> u64 {
    let parts: Vec<&str> = mnemonic_full.split('.').collect();
    for p in parts.iter().rev() {
        if let Some((_, b)) = WIDTH_BYTES.iter().find(|(w, _)| w == p) {
            return *b;
        }
    }
    4 // unsuffixed LDG/LDS default to 32-bit
}

fn lds_inst_rate(bytes_per_thread: u64) -> f64 {
    // measured: 64 B/clk/SM smem ceiling; inst rate = 64 / (32 * width)
    64.0 / (32.0 * bytes_per_thread as f64)
}

pub fn l1_budget(rates: &Rates, defaulted: &mut Vec<String>) -> f64 {
    if let Some(r) = rates.get("mem.l1.bw") {
        return r.value.parse().expect("mem.l1.bw value parses");
    }
    defaulted.push("mem.l1.bw(default32)".into());
    32.0
}

/// The per-pipe-max projection of one loop body's census at `warps` warps/SM,
/// mirroring `project.py::project` operation for operation (the demand fold
/// and the max/sum live in the substrate-generic core).
pub fn project(
    census: &Census,
    warps: f64,
    mem_class: MemClass,
    rates: &Rates,
    dram_budget: f64,
) -> ProjectResult {
    let mut pipe_names: Vec<String> = Vec::new();
    let mut pipe_cycles: Vec<f64> = Vec::new();
    let mut mem_bytes_total = 0.0f64;
    let mut smem_cycles = 0.0f64;
    let mut total_insts: u64 = 0;
    let mut defaulted: Vec<String> = Vec::new();

    for (mn, n) in &census.items {
        let base = mn.split('.').next().expect("split yields at least one part");
        if CONTROL.contains(&base) {
            total_insts += n;
            continue;
        }
        total_insts += n;
        if LSU.contains(&base) {
            let b = mem_bytes(mn) * 32; // per warp
            if base == "LDS" || base == "STS" {
                smem_cycles += *n as f64 * (1.0 / lds_inst_rate(mem_bytes(mn)));
            } else if base == "LDSM" {
                // ldmatrix moves SHARED-memory tiles (not DRAM): 128 B/warp per 8x8
                // b16 matrix, x1/x2/x4 by the trailing count suffix, so 4*count
                // B/thread against the same 64 B/clk/SM smem ceiling. x4 -> 8
                // cyc/inst, reproducing the measured `tensor.ldsm.tput` 0.125.
                let count = mn
                    .rsplit('.')
                    .next()
                    .and_then(|s| s.parse::<u64>().ok())
                    .filter(|c| matches!(c, 1 | 2 | 4))
                    .unwrap_or(1);
                smem_cycles += *n as f64 * (1.0 / lds_inst_rate(4 * count));
            } else if base == "LDL" || base == "STL" {
                defaulted.push(mn.clone()); // local traffic: flagged, not modelled
            } else {
                mem_bytes_total += (*n * b) as f64;
            }
            continue;
        }
        // Tensor mnemonics carry their MMA shape (`HMMA.1688.F32`); a measured row
        // for that exact shape wins, an unmeasured shape falls back to the base-shape
        // row with a `defaulted` marker (costed, but flagged as approximate).
        let tensor_shape = if base == "HMMA" || base == "IMMA" {
            mn.split('.')
                .nth(1)
                .filter(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
        } else {
            None
        };
        let shape_measured = tensor_shape.and_then(|shape| {
            rates.get(&format!("tensor.{}.{}.tput", base.to_ascii_lowercase(), shape))
        });
        let base_measured = lookup(OP_TABLE_ROW, base).and_then(|id| rates.get(id));
        if shape_measured.is_none() && tensor_shape.is_some() && base_measured.is_some() {
            defaulted.push(mn.clone()); // tensor shape without its own measured row
        }
        let (pipe, rate) = match shape_measured.or(base_measured) {
            Some(r) => {
                let rate: f64 = r.value.parse().expect("rate value parses");
                let pipe = if r.pipe.is_empty() {
                    lookup(PIPE_OF, base).unwrap_or("alu")
                } else {
                    r.pipe.as_str()
                };
                (pipe.to_string(), rate)
            }
            None => {
                let pipe = lookup(PIPE_OF, base).unwrap_or("alu");
                let rate = if pipe == "fma" || pipe == "alu" {
                    2.0
                } else if pipe == "tensor" {
                    TENSOR_HMMA_PER_SM_CLK
                } else {
                    0.5
                };
                defaulted.push(mn.clone());
                (pipe.to_string(), rate)
            }
        };
        let idx = match pipe_names.iter().position(|p| *p == pipe) {
            Some(i) => i,
            None => {
                pipe_names.push(pipe);
                pipe_cycles.push(0.0);
                pipe_names.len() - 1
            }
        };
        pipe_cycles[idx] += *n as f64 / rate;
    }

    // the byte budget is resolved only when there is memory traffic, so the
    // l1-default marker appears exactly when the Python's would
    let budget = if mem_bytes_total != 0.0 {
        match mem_class {
            MemClass::Dram => dram_budget,
            _ => l1_budget(rates, &mut defaulted),
        }
    } else {
        1.0
    };

    let ss = crate::project_mix(
        warps,
        &pipe_cycles,
        smem_cycles,
        mem_bytes_total,
        budget,
        total_insts as f64,
        ISSUE_CAP,
    );
    let name = |kind: ResourceKind| -> String {
        match kind {
            ResourceKind::Pipe(i) => format!("pipe:{}", pipe_names[i]),
            ResourceKind::LocalStoreBw => "smem".into(),
            ResourceKind::MemoryBw => "mem".into(),
            ResourceKind::Issue => "issue".into(),
        }
    };
    ProjectResult {
        ppm_cycles: ss.ppm_cycles,
        ppm_bound: name(ss.per_resource[ss.ppm_bound].0),
        add_cycles: ss.add_cycles,
        per_resource: ss.per_resource.iter().map(|&(k, v)| (name(k), v)).collect(),
        defaulted,
    }
}

/// `project.py::main`'s stdout for one projection, byte for byte.
pub fn report(r: &ProjectResult) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "PPM: {:.1} cycles/iter (bound: {})\n",
        r.ppm_cycles, r.ppm_bound
    ));
    out.push_str(&format!("ADD: {:.1} cycles/iter\n", r.add_cycles));
    let mut items = r.per_resource.clone();
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("demands are never NaN"));
    for (k, v) in &items {
        out.push_str(&format!("  {:<12} {:>10.1}\n", k, v));
    }
    if !r.defaulted.is_empty() {
        let mut set: Vec<&str> = r.defaulted.iter().map(|s| s.as_str()).collect();
        set.sort_unstable();
        set.dedup();
        out.push_str(&format!("  defaulted rates: {}\n", set.join(" ")));
    }
    out
}
