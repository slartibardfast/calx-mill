//! examples/bonsai-split.rs — the layer-split sweep for Ternary-Bonsai-2-27B
//! (PTQ1_0) between a small CUDA GPU and an AVX/AVX2 host CPU.
//!
//! Provenance of every constant is named at its definition:
//! - model bytes: the GGUF tensor table of
//!   prism-ml/Ternary-Bonsai-2-27B-gguf PTQ1_0 (header parsed 2026-09-20;
//!   per-layer means, class split 16 full-attention / 48 linear).
//! - CPU axes: refs/agner/extract.md rows in the agentic-llama.cpp-prism host
//!   (Agner Fog microarchitecture tables, pinned by SHA256).
//! - GPU axes: the TU102 SM substrate verbatim from this crate's tests; device
//!   constants from datasheets, effective-DRAM anchored the way
//!   tests/fixtures/tu102 does (5.82 B/clk/SM measured vs 609 GB/s board).
//!
//! Units: nanoseconds and bytes, f64 throughout (the u32 `project` overflows
//! at one token's 5.9e9 bytes; LACUNAE notes the model is unit-agnostic).
//!
//! Run: cargo run --release --example bonsai-split [-- --full]

use calx_mill::{mix_demands, select_bound, Substrate};

// ---------------------------------------------------------------- model
// PTQ1_0 block: 28 bytes per 128 weights = 0.21875 B/weight (ggml-common.h).
const N_LAYERS: usize = 64;
const FULL_ATTN_LAYERS: usize = 16; // layers 3,7,...,63 (qwen35.full_attention_interval = 4)
const LINEAR_LAYERS: usize = 48;
const BYTES_FULL_ATTN_LAYER: f64 = 81.47e6;
const BYTES_LINEAR_LAYER: f64 = 84.91e6;
const BYTES_OUTPUT_HEAD: f64 = 278.1e6; // streamed every generated token (logits matmul)
const BYTES_EMBD_ROW: f64 = 5120.0 * 0.21875; // one row per token via get_rows
const D_MODEL: f64 = 5120.0;

// VRAM budget arithmetic for the 4 GiB card (bytes):
const VRAM_TOTAL: f64 = 4096.0 * 1024.0 * 1024.0;
const VRAM_CUDA_CONTEXT: f64 = 260e6; // measured class default for llama.cpp CUDA
const VRAM_COMPUTE_BUFFERS: f64 = 120e6; // batch-1 decode buffers
const CTX_TOKENS: f64 = 8192.0;
const KV_BYTES_PER_TOKEN_PER_FULL_LAYER: f64 = 4.0 * 256.0 * 2.0 * 2.0; // 4 kv heads x 256 x K+V x f16
const BYTES_GDN_STATE_PER_LINEAR_LAYER: f64 = 8e6; // 16 x 128 x 128 x f32, resident with its layer
fn vram_weights_budget(n_gpu_linear: usize, n_gpu_full: usize) -> f64 {
    let kv = FULL_ATTN_LAYERS as f64 * KV_BYTES_PER_TOKEN_PER_FULL_LAYER * CTX_TOKENS;
    VRAM_TOTAL - VRAM_CUDA_CONTEXT - VRAM_COMPUTE_BUFFERS - kv
        - n_gpu_linear as f64 * BYTES_GDN_STATE_PER_LINEAR_LAYER
        - n_gpu_full as f64 * 0.0 // full-attn KV counted above; no extra state
}

// ---------------------------------------------------------------- kernels
// AVX2 vec_dot for PTQ1_0 (no VNNI): per 128-weight block, the base-3 decode is
// 5 passes of (vpmullo16, vpand, vpmullo16, vpsrl16) on 16+8 lanes plus the
// pack/permute staging and the maddubs/madd dot. The landed kernel's
// instruction mix measures ~115 uops single-block, ~90 marginal with the
// intended two-block interleave: budget 90 per 28-byte block.
const VEC_DOT_UOPS_PER_BLOCK: f64 = 90.0;
// PQ2_0 AVX2 branch: shift+and extract + maddubs + madd: ~18 uops per 34-byte block.
const PQ2_VEC_DOT_UOPS_PER_BLOCK: f64 = 18.0;
// Repack path: decode paid once at load; steady-state dot is ~8 uops per block
// against Q8_0 and the loop is issue-comfortable, so the leg is memory-bound.
const REPACK_UOPS_PER_BLOCK: f64 = 8.0;

#[derive(Clone, Copy, PartialEq)]
enum Kernel {
    VecDot,  // plan/0002 outcome: decode every token (issue-heavy)
    Repacked, // plan/0003 outcome: decode once at load (bandwidth-bound)
}

// ---------------------------------------------------------------- substrates
// Per-core axes; every row cites refs/agner/extract.md (the pinned Agner Fog
// tables). register_capacity = physical vector file; concurrency_ceiling = ROB;
// issue_cap = rename width; mem_bandwidth = L1-class bytes/cycle (the DRAM leg
// enters as an explicit bytes/ns demand below, never through this field).
struct Cpu {
    name: &'static str,
    substrate: Substrate,
    cores: f64,
    clock_ghz: f64,
    dram_gbps: f64, // theoretical peak
    dram_eff: f64,  // STREAM-class achieved fraction
    // uop multiplier for the SIMD ladder this core runs: 1.0 where 256-bit
    // integer ops are native (AVX2), 1.5 where they crack to 2x128 (Zen 1),
    // 2.0 for the AVX-only ladder (Sandy/Ivy Bridge: 128-bit integer + 256-bit
    // FP, the plan/0006 stretch kernels).
    uops_scale: f64,
}

fn sandy_bridge_core() -> Substrate {
    // extract.md: ROB 168, 144 vector regs, 4-wide; AVX only (no AVX2 integer,
    // no FMA): the 128-bit maddubs + 256-bit FP ladder doubles the uop count.
    Substrate {
        register_capacity: 144,
        register_granularity: 1,
        local_store_bytes: 32 * 1024,
        local_store_granularity: 64,
        concurrency_ceiling: 168,
        issue_cap: 4,
        mem_bandwidth: 32, // 2x 128-bit loads/cycle
    }
}

fn haswell_core() -> Substrate {
    // extract.md: ROB 192, RS 60, 168 vector regs, 4-wide, 2x 256-bit FMA.
    Substrate {
        register_capacity: 168,
        register_granularity: 1,
        local_store_bytes: 32 * 1024,
        local_store_granularity: 64,
        concurrency_ceiling: 192,
        issue_cap: 4,
        mem_bandwidth: 32, // 2x 256-bit loads/cycle
    }
}

fn comet_lake_core() -> Substrate {
    // extract.md: the Skylake client core (ROB 224, RS 97, 168 vector regs,
    // 4-wide). The i9-10885H, primary target: AVX2+FMA, no VNNI.
    Substrate {
        register_capacity: 168,
        register_granularity: 1,
        local_store_bytes: 32 * 1024,
        local_store_granularity: 64,
        concurrency_ceiling: 224,
        issue_cap: 4,
        mem_bandwidth: 64, // 2x 256-bit loads/cycle
    }
}

fn skylake_x_core() -> Substrate {
    // The local Xeon W-2140B (same axes as the Skylake client row; wider
    // memory subsystem modeled at the platform level, quad-channel).
    comet_lake_core()
}

fn zen1_core() -> Substrate {
    // extract.md: 192 uops in flight, 4-wide dispatch, 256-bit ops crack to 2.
    Substrate {
        register_capacity: 160,
        register_granularity: 1,
        local_store_bytes: 32 * 1024,
        local_store_granularity: 64,
        concurrency_ceiling: 192,
        issue_cap: 4,
        mem_bandwidth: 32,
    }
}

fn zen2_core() -> Substrate {
    Substrate {
        register_capacity: 168,
        register_granularity: 1,
        local_store_bytes: 32 * 1024,
        local_store_granularity: 64,
        concurrency_ceiling: 224,
        issue_cap: 4,
        mem_bandwidth: 64,
    }
}

fn zen3_core() -> Substrate {
    // extract.md: ROB 256, 6-wide dispatch.
    Substrate {
        register_capacity: 192,
        register_granularity: 1,
        local_store_bytes: 32 * 1024,
        local_store_granularity: 64,
        concurrency_ceiling: 256,
        issue_cap: 6,
        mem_bandwidth: 64,
    }
}

fn cpus() -> Vec<Cpu> {
    vec![
        Cpu { name: "i9-10885H (CML, AVX2+FMA)", substrate: comet_lake_core(), cores: 8.0, clock_ghz: 2.4, dram_gbps: 46.9, dram_eff: 0.75, uops_scale: 1.0 }, // 8C/16T, sustained all-core turbo
        Cpu { name: "i7-4670K (HSW, AVX2+FMA)", substrate: haswell_core(), cores: 4.0, clock_ghz: 3.4, dram_gbps: 25.6, dram_eff: 0.75, uops_scale: 1.0 },
        Cpu { name: "i7-2600K (SNB, AVX)", substrate: sandy_bridge_core(), cores: 4.0, clock_ghz: 3.4, dram_gbps: 21.3, dram_eff: 0.72, uops_scale: 2.0 },
        Cpu { name: "Zen 1 1700-class", substrate: zen1_core(), cores: 8.0, clock_ghz: 3.0, dram_gbps: 42.6, dram_eff: 0.75, uops_scale: 1.5 },
        Cpu { name: "Zen 2 3700X-class", substrate: zen2_core(), cores: 8.0, clock_ghz: 3.8, dram_gbps: 47.7, dram_eff: 0.78, uops_scale: 1.0 },
        Cpu { name: "Zen 3 5800X-class", substrate: zen3_core(), cores: 8.0, clock_ghz: 4.2, dram_gbps: 47.7, dram_eff: 0.80, uops_scale: 1.0 },
        Cpu { name: "Xeon W-2140B (local, SKX)", substrate: skylake_x_core(), cores: 8.0, clock_ghz: 3.2, dram_gbps: 85.3, dram_eff: 0.75, uops_scale: 1.0 }, // anchor machine
    ]
}

struct Gpu {
    name: &'static str,
    sms: f64,
    dram_gbps: f64,
    dram_eff: f64,
    // Batch-1 TG decode-effective GB/s for PTQ1_0: Turing's MMVQ path is
    // instruction/launch-bound, not bandwidth-bound. Anchor: the local RTX
    // 6000 measures 31.84 tok/s at full offload (llama-bench, 2026-09-20) =
    // 5.66 GB / 31.5 ms = 180 GB/s against 609 GB/s measured DRAM (0.30).
    // The 16-SM TU117 scales the instruction side by SM count.
    // Per-SM axes remain tu102_sm() verbatim for every Turing part here.
    decode_gbps: f64,
}

fn gpus() -> Vec<Gpu> {
    vec![
        Gpu { name: "TU117-4GiB (T1000/GTX1650 class)", sms: 16.0, dram_gbps: 160.0, dram_eff: 0.85, decode_gbps: (16.0 / 72.0) * 180.0 },
        Gpu { name: "RTX 6000 (TU102, local anchor)", sms: 72.0, dram_gbps: 672.0, dram_eff: 0.907, decode_gbps: 180.0 }, // measured (see above); 609 measured / 672 peak DRAM, calx-mill fixture
    ]
}

const PCIE_GEN3_X16_GBPS: f64 = 11.0; // effective, both directions summed

// Fixed per-layer scheduling overhead, calibrated from the anchor run
// (full-offload TG128 on the local RTX 6000; see plan/0001 anchor-local).
// Updated when the anchor lands; starts at the llama.cpp class default.
fn per_layer_fixed_ns(calibrated: f64) -> f64 {
    calibrated
}

// ---------------------------------------------------------------- legs
struct Leg {
    ns: f64,
    bound: String,
}

/// CPU leg: memory demand (weights streamed from DRAM) against the pipe demand
/// (decode + dot uops across all cores), named by the PPM bound. Every demand
/// is pre-scaled to nanoseconds: the Issue axis enters as machine uops/ns
/// (cores x issue_cap x clock), so total_ops / issue_cap is already ns.
fn cpu_leg(cpu: &Cpu, bytes: f64, kernel: Kernel) -> Leg {
    let blocks = bytes / 28.0; // PTQ1_0: one 128-weight block per 28 B
    let uops_per_block = match kernel {
        Kernel::VecDot => VEC_DOT_UOPS_PER_BLOCK,
        Kernel::Repacked => REPACK_UOPS_PER_BLOCK,
    } * cpu.uops_scale;
    let total_ops = blocks * uops_per_block;
    let uops_per_ns = cpu.cores * cpu.substrate.issue_cap as f64 * cpu.clock_ghz;
    let issue_ns = total_ops / uops_per_ns;
    let demands = mix_demands(
        1.0,
        &[issue_ns],
        0.0,
        bytes,
        cpu.dram_gbps * cpu.dram_eff, // bytes per ns
        total_ops,
        uops_per_ns,
    );
    let ss = select_bound(&demands);
    Leg { ns: ss.ppm_cycles, bound: format!("{:?}", ss.per_resource[ss.ppm_bound].0) }
}

fn gpu_leg(gpu: &Gpu, bytes: f64) -> Leg {
    let ns = bytes / gpu.decode_gbps;
    Leg { ns, bound: "MemoryBw".into() }
}

fn pcie_leg(bytes: f64) -> f64 {
    bytes / PCIE_GEN3_X16_GBPS
}

// ---------------------------------------------------------------- sweep
fn layer_bytes(n_gpu: usize) -> (f64, f64, usize, usize) {
    // llama.cpp offloads layers 0..n_gpu-1; classes interleave as 3,7,.. full.
    let mut g = 0.0;
    let mut c = 0.0;
    let mut gl = 0usize;
    let mut gf = 0usize;
    for i in 0..N_LAYERS {
        let is_full = i % 4 == 3;
        let b = if is_full { BYTES_FULL_ATTN_LAYER } else { BYTES_LINEAR_LAYER };
        if i < n_gpu {
            g += b;
            if is_full { gf += 1 } else { gl += 1 }
        } else {
            c += b;
        }
    }
    (g, c, gl, gf)
}

fn sweep(cpu: &Cpu, gpu: &Gpu, kernel: Kernel, fixed_ns: f64, output_on_gpu: bool) -> Vec<(usize, f64, f64, String)> {
    let mut rows = Vec::new();
    for n_gpu in 0..=N_LAYERS {
        let (mut gbytes, mut cbytes, gl, gf) = layer_bytes(n_gpu);
        // the output head streams wherever it lives; llama.cpp keeps it with
        // the CPU unless the whole model is offloaded (both variants printed)
        if output_on_gpu || n_gpu == N_LAYERS {
            gbytes += BYTES_OUTPUT_HEAD;
        } else {
            cbytes += BYTES_OUTPUT_HEAD;
        }
        cbytes += BYTES_EMBD_ROW;
        // VRAM feasibility at the 4 GiB budget (only binding for the small card)
        let budget = vram_weights_budget(gl, gf);
        if gpu.sms < 30.0 && gbytes > budget {
            continue; // infeasible split for a 4 GiB card
        }
        let g = gpu_leg(gpu, gbytes);
        let c = cpu_leg(cpu, cbytes, kernel);
        let pcie = pcie_leg(D_MODEL * 4.0 * 2.0); // activation crossing, both ways
        let t = g.ns + c.ns + pcie + (n_gpu as f64 + (N_LAYERS - n_gpu) as f64) * fixed_ns;
        rows.push((n_gpu, 1e9 / t, t / 1e6, format!("{}|{}", g.bound, c.bound)));
    }
    rows
}

fn main() {
    let full = std::env::args().any(|a| a == "--full");
    let fixed = per_layer_fixed_ns(40_000.0); // placeholder until the anchor lands; see plan/0001
    println!("Ternary-Bonsai-2-27B PTQ1_0 layer-split sweep (context {:.0}, PCIe Gen3 x16)", CTX_TOKENS);
    println!("model: {} layers ({} full-attn x {:.2} MB, {} linear x {:.2} MB), output head {:.0} MB\n",
        N_LAYERS, FULL_ATTN_LAYERS, BYTES_FULL_ATTN_LAYER / 1e6, LINEAR_LAYERS, BYTES_LINEAR_LAYER / 1e6, BYTES_OUTPUT_HEAD / 1e6);

    println!("== primary target: 4 GiB Turing + i9-10885H ==");
    let cpu = &cpus()[0];
    let gpu = &gpus()[0];
    for kernel in [Kernel::VecDot, Kernel::Repacked] {
        let rows = sweep(cpu, gpu, kernel, fixed, false);
        if let Some(best) = rows.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
            println!("{:>9}: best split ngl={:>2}  {:>6.2} tok/s  ({:.2} ms/token, bound {})",
                if kernel == Kernel::VecDot { "vec_dot" } else { "repacked" }, best.0, best.1, best.2, best.3);
        } else {
            println!("{:>9}: no feasible split at the 4 GiB budget", if kernel == Kernel::VecDot { "vec_dot" } else { "repacked" });
        }
        if full {
            for (n, tps, ms, b) in rows.iter().step_by(4) {
                println!("           ngl={:>2}  {:>6.2} tok/s  {:>6.2} ms  {}", n, tps, ms, b);
            }
        }
    }

    println!("\n== divergence matrix: best tok/s by CPU class and kernel (4 GiB Turing) ==");
    println!("{:<32} {:>12} {:>12} {:>14}", "CPU", "vec_dot", "repacked", "repack gain");
    for cpu in cpus() {
        let mut line = format!("{:<32}", cpu.name);
        let mut vals = Vec::new();
        for kernel in [Kernel::VecDot, Kernel::Repacked] {
            let rows = sweep(&cpu, &gpus()[0], kernel, fixed, false);
            let best = rows.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|r| r.1).unwrap_or(0.0);
            vals.push(best);
            line += &format!(" {:>12.2}", best);
        }
        let gain = if vals[0] > 0.0 { vals[1] / vals[0] } else { f64::INFINITY };
        println!("{} {:>13.2}x", line, gain);
    }

    println!("\n== local anchor rig: RTX 6000 + Xeon W-2140B ==");
    for kernel in [Kernel::VecDot, Kernel::Repacked] {
        let rows = sweep(&cpus()[6], &gpus()[1], kernel, fixed, false);
        if let Some(best) = rows.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
            println!("{:>9}: best split ngl={:>2}  {:>6.2} tok/s (bound {})",
                if kernel == Kernel::VecDot { "vec_dot" } else { "repacked" }, best.0, best.1, best.3);
        }
    }

    // the pure-GPU end of the local rig, against the measured anchor
    // (llama-bench tg128, ngl 99: 31.84 tok/s, 2026-09-20)
    let all_gpu = gpu_leg(&gpus()[1], layer_bytes(64).0 + BYTES_OUTPUT_HEAD);
    println!("pure GPU (ngl 64): {:.2} tok/s at the measured decode-effective rate (anchor: 31.84 tok/s)", 1e9 / all_gpu.ns);
}
