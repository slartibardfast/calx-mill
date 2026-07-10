//! `ptxas -v` (nvcc `-Xptxas -v`) parsing: per-kernel register/smem/spill
//! usage, the static side of the occupancy fold. A real fixture costs nothing
//! without a GPU - nvcc compiles and reports without one.

use crate::{ceil_div, Substrate, WorkUnit};

/// One kernel's `ptxas info` resource report.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct KernelUsage {
    pub name: String,
    pub arch: String,
    pub registers: u32,
    pub barriers: u32,
    pub smem_bytes: u32,
    /// constant-bank bytes as (bank, bytes), in report order
    pub cmem_bytes: Vec<(u32, u32)>,
    pub lmem_bytes: u32,
    pub stack_frame: u32,
    pub spill_stores: u32,
    pub spill_loads: u32,
}

fn leading_u32(s: &str) -> Option<u32> {
    let t = s.trim_start();
    let end = t.find(|c: char| !c.is_ascii_digit()).unwrap_or(t.len());
    if end == 0 {
        return None;
    }
    t[..end].parse().ok()
}

/// Parse the stderr of `nvcc -Xptxas -v` (or `ptxas -v` directly): every
/// `Compiling entry function` opens a kernel; `Function properties` and
/// `Used ...` lines fill it.
pub fn parse_ptxas_v(text: &str) -> Vec<KernelUsage> {
    let mut out: Vec<KernelUsage> = Vec::new();
    let mut in_properties = false;
    for line in text.lines() {
        let info = line
            .trim_start()
            .strip_prefix("ptxas info")
            .map(|r| r.trim_start_matches([' ', ':']).trim());
        if let Some(rest) = info {
            in_properties = false;
            if let Some(r) = rest.strip_prefix("Compiling entry function '") {
                let Some((name, tail)) = r.split_once('\'') else { continue };
                let arch = tail
                    .split('\'')
                    .nth(1)
                    .unwrap_or("")
                    .to_string();
                out.push(KernelUsage { name: name.to_string(), arch, ..Default::default() });
            } else if rest.starts_with("Function properties for ") {
                in_properties = true;
            } else if let Some(r) = rest.strip_prefix("Used ") {
                let Some(k) = out.last_mut() else { continue };
                for clause in r.split(", ") {
                    let clause = clause.trim().trim_start_matches("used ");
                    let Some(n) = leading_u32(clause) else { continue };
                    if clause.ends_with("registers") {
                        k.registers = n;
                    } else if clause.ends_with("barriers") {
                        k.barriers = n;
                    } else if clause.ends_with("bytes smem") {
                        k.smem_bytes = n;
                    } else if clause.ends_with("bytes lmem") {
                        k.lmem_bytes = n;
                    } else if let Some(bank) = clause
                        .split_once("bytes cmem[")
                        .and_then(|(_, b)| b.strip_suffix(']'))
                        .and_then(|b| b.parse().ok())
                    {
                        k.cmem_bytes.push((bank, n));
                    }
                }
            }
            continue;
        }
        if in_properties {
            // "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads"
            if let Some(k) = out.last_mut() {
                for clause in line.trim().split(", ") {
                    let Some(n) = leading_u32(clause) else { continue };
                    if clause.ends_with("stack frame") {
                        k.stack_frame = n;
                    } else if clause.ends_with("spill stores") {
                        k.spill_stores = n;
                    } else if clause.ends_with("spill loads") {
                        k.spill_loads = n;
                    }
                }
            }
            in_properties = false;
        }
    }
    out
}

/// The adapter's TU102 SM characterization (sm_75): 64K registers at 256/warp
/// allocation, 64 KiB shared memory, 32 warps/SM, 4 single-issue schedulers,
/// 64 B/clk local-store streaming.
pub fn tu102_sm() -> Substrate {
    Substrate {
        register_capacity: 65536,
        register_granularity: 256,
        local_store_bytes: 65536,
        local_store_granularity: 128,
        concurrency_ceiling: 32,
        issue_cap: 4,
        mem_bandwidth: 64,
    }
}

/// Fold one kernel's static usage into the core's per-warp work unit:
/// registers become regs/thread x 32 (the core rounds to the 256-reg warp
/// allocation unit); block shared memory is divided across the block's warps
/// (ceiling), an approximation of CUDA's per-block allocation that the
/// GPU-side measured close refines.
pub fn work_unit(usage: &KernelUsage, block_threads: u32) -> WorkUnit {
    let warps_per_block = ceil_div(block_threads.max(1), 32).max(1);
    WorkUnit {
        registers: usage.registers * 32,
        local_store_bytes: ceil_div(usage.smem_bytes, warps_per_block),
    }
}
