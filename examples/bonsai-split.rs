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
// Repack path: decode paid once at load into 2-bit slots; the steady-state
// gemv expands each 8-byte code row to a ymm (dword spread) and dots with
// maddubs/madd: about 60 uops per block per row at the landed kernel's mix.
const REPACK_UOPS_PER_BLOCK: f64 = 60.0;

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
    dram_eff: f64, // STREAM-class achieved fraction
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

#[derive(Clone, Copy, PartialEq)]
enum Packing {
    // 28 B per 128 weights; TG128 anchor on the local RTX 6000 (2026-09-20):
    // 31.84 tok/s = 5.66 GB / 31.5 ms = 180 GB/s decode-effective (0.30 of
    // the 609 GB/s measured DRAM): Turing decode is instruction-bound.
    Ptq10,
    // 34 B per 128 weights; same rig: 41.90 tok/s = 268 GB/s (0.44). The
    // cheaper unpacking wins both phases on Turing; the trade is 21 percent
    // more bytes per layer.
    Pq20,
}

impl Packing {
    fn linear_layer(self) -> f64 {
        match self { Packing::Ptq10 => 84.91e6, Packing::Pq20 => 84.91e6 * 34.0 / 28.0 }
    }
    fn full_layer(self) -> f64 {
        match self { Packing::Ptq10 => 81.47e6, Packing::Pq20 => 81.47e6 * 34.0 / 28.0 }
    }
    fn output(self) -> f64 {
        match self { Packing::Ptq10 => 278.1e6, Packing::Pq20 => 278.1e6 * 34.0 / 28.0 }
    }
    fn blocks_per_byte(self) -> f64 {
        match self { Packing::Ptq10 => 1.0 / 28.0, Packing::Pq20 => 1.0 / 34.0 }
    }
}

struct Gpu {
    name: &'static str,
    sms: f64,
    dram_gbps: f64,
    dram_eff: f64,
    ptq10_decode_gbps: f64,
    pq20_decode_gbps: f64,
    // Batch-1 TG decode-effective GB/s for PTQ1_0: Turing's MMVQ path is
    // instruction/launch-bound, not bandwidth-bound. Anchor: the local RTX
    // 6000 measures 31.84 tok/s at full offload (llama-bench, 2026-09-20) =
    // 5.66 GB / 31.5 ms = 180 GB/s against 609 GB/s measured DRAM (0.30).
    // The 16-SM TU117 scales the instruction side by SM count.
    // Per-SM axes remain tu102_sm() verbatim for every Turing part here.
}

fn gpus() -> Vec<Gpu> {
    let sm_scale = 16.0 / 72.0;
    vec![
        Gpu { name: "TU117-4GiB (T1000/GTX1650 class)", sms: 16.0, dram_gbps: 160.0, dram_eff: 0.85,
              ptq10_decode_gbps: sm_scale * 180.0, pq20_decode_gbps: sm_scale * 268.0 },
        Gpu { name: "RTX 6000 (TU102, local anchor)", sms: 72.0, dram_gbps: 672.0, dram_eff: 0.907,
              ptq10_decode_gbps: 180.0, pq20_decode_gbps: 268.0 }, // both measured 2026-09-20
    ]
}

const PCIE_GEN3_X16_GBPS: f64 = 11.0; // effective, both directions summed

// Fixed per-layer scheduling overhead, calibrated from the anchor run
// (full-offload TG128 on the local RTX 6000; see plan/0001 anchor-local).
// Updated when the anchor lands; starts at the llama.cpp class default.
fn per_layer_fixed_ns(calibrated: f64) -> f64 {
    calibrated
}

// The LACUNAE-named blind spot, now measured (plan/0008, 2026-09-24): the
// mixed-split CPU leg is per-op-overhead-bound, not compute-bound. Anchor
// calibration: ngl 37, repack on, op-offload off, t16, elision-era binary on
// the local rig measures 4.07 tok/s (245.7 ms/token) against this sweep's
// compute-bound prediction of ~73 ms at the same split; the ~172 ms gap over
// 27 CPU-side layers is scheduler splits (~175 per compute, measured with
// GGML_SCHED_DEBUG), per-op barriers, and the host node walk. Charged as a
// fixed ns per CPU-side layer. Transfer assumption, stated not hidden: the
// i9-10885H target is also an 8-core AVX2 host at comparable clock, so the
// anchor's per-layer constant carries over; re-calibrate per host class when
// a target measurement exists.
const CPU_LEG_OVERHEAD_NS_PER_CPU_LAYER: f64 = 6_390_000.0;

// ---------------------------------------------------------------- legs
struct Leg {
    ns: f64,
    bound: String,
}

/// CPU leg: memory demand (weights streamed from DRAM) against the pipe demand
/// (decode + dot uops across all cores), named by the PPM bound. Every demand
/// is pre-scaled to nanoseconds: the Issue axis enters as machine uops/ns
/// (cores x issue_cap x clock), so total_ops / issue_cap is already ns.
fn cpu_leg(cpu: &Cpu, bytes: f64, kernel: Kernel, packing: Packing) -> Leg {
    let blocks = bytes * packing.blocks_per_byte();
    let uops_per_block = match (kernel, packing) {
        (Kernel::VecDot, Packing::Ptq10) => VEC_DOT_UOPS_PER_BLOCK,
        (Kernel::VecDot, Packing::Pq20) => PQ2_VEC_DOT_UOPS_PER_BLOCK,
        (Kernel::Repacked, _) => REPACK_UOPS_PER_BLOCK,
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

fn gpu_leg(gpu: &Gpu, bytes: f64, packing: Packing) -> Leg {
    let rate = match packing {
        Packing::Ptq10 => gpu.ptq10_decode_gbps,
        Packing::Pq20 => gpu.pq20_decode_gbps,
    };
    Leg { ns: bytes / rate, bound: "MemoryBw".into() }
}

fn pcie_leg(bytes: f64) -> f64 {
    bytes / PCIE_GEN3_X16_GBPS
}

// ---------------------------------------------------------------- sweep
fn layer_bytes(n_gpu: usize, packing: Packing) -> (f64, f64, usize, usize) {
    // llama.cpp offloads layers 0..n_gpu-1; classes interleave as 3,7,.. full.
    let mut g = 0.0;
    let mut c = 0.0;
    let mut gl = 0usize;
    let mut gf = 0usize;
    for i in 0..N_LAYERS {
        let is_full = i % 4 == 3;
        let b = if is_full { packing.full_layer() } else { packing.linear_layer() };
        if i < n_gpu {
            g += b;
            if is_full { gf += 1 } else { gl += 1 }
        } else {
            c += b;
        }
    }
    (g, c, gl, gf)
}

fn sweep(cpu: &Cpu, gpu: &Gpu, kernel: Kernel, packing: Packing, fixed_ns: f64) -> Vec<(usize, f64, f64, String)> {
    let mut rows = Vec::new();
    for n_gpu in 0..=N_LAYERS {
        let (mut gbytes, mut cbytes, gl, gf) = layer_bytes(n_gpu, packing);
        // the output head streams wherever it lives; llama.cpp keeps it with
        // the CPU unless the whole model is offloaded
        if n_gpu == N_LAYERS {
            gbytes += packing.output();
        } else {
            cbytes += packing.output();
        }
        cbytes += BYTES_EMBD_ROW;
        // VRAM feasibility at the 4 GiB budget (only binding for the small card)
        let budget = vram_weights_budget(gl, gf);
        if gpu.sms < 30.0 && gbytes > budget {
            continue; // infeasible split for a 4 GiB card
        }
        let g = gpu_leg(gpu, gbytes, packing);
        let c = cpu_leg(cpu, cbytes, kernel, packing);
        let pcie = pcie_leg(D_MODEL * 4.0 * 2.0); // activation crossing, both ways
        // the measured per-op overhead of the CPU leg (see the constant's
        // provenance): charged per CPU-side layer, zero at full offload
        let cpu_overhead = (N_LAYERS - n_gpu) as f64 * CPU_LEG_OVERHEAD_NS_PER_CPU_LAYER;
        let t = g.ns + c.ns + pcie + cpu_overhead + (n_gpu as f64 + (N_LAYERS - n_gpu) as f64) * fixed_ns;
        rows.push((n_gpu, 1e9 / t, t / 1e6, format!("{}|{}", g.bound, c.bound)));
    }
    rows
}

fn main() {
    let full = std::env::args().any(|a| a == "--full");
    let fixed = per_layer_fixed_ns(40_000.0);
    println!("Ternary-Bonsai-2-27B layer-split sweep (context {:.0}, PCIe Gen3 x16)", CTX_TOKENS);
    println!("model: {} layers (16 full-attn / 48 linear; PTQ1_0 {:.2}/{:.2} MB, PQ2_0 {:.2}/{:.2} MB)\n",
        N_LAYERS,
        BYTES_FULL_ATTN_LAYER / 1e6, BYTES_LINEAR_LAYER / 1e6,
        BYTES_FULL_ATTN_LAYER * 34.0 / 28.0 / 1e6, BYTES_LINEAR_LAYER * 34.0 / 28.0 / 1e6);

    println!("== primary target: 4 GiB Turing + i9-10885H, both packings ==");
    let cpu = &cpus()[0];
    let gpu = &gpus()[0];
    for packing in [Packing::Ptq10, Packing::Pq20] {
        for kernel in [Kernel::VecDot, Kernel::Repacked] {
            let rows = sweep(cpu, gpu, kernel, packing, fixed);
            let kname = if kernel == Kernel::VecDot { "vec_dot" } else { "repacked" };
            match rows.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
                Some(best) => println!("{:>6}/{:>9}: best ngl={:>2}  {:>6.2} tok/s  ({:.1} ms/token, bound {})",
                    if packing == Packing::Ptq10 { "PTQ1_0" } else { "PQ2_0" }, kname, best.0, best.1, best.2, best.3),
                None => println!("{:>6}/{:>9}: no feasible split at the 4 GiB budget",
                    if packing == Packing::Ptq10 { "PTQ1_0" } else { "PQ2_0" }, kname),
            }
            if full {
                for (n, tps, ms, b) in rows.iter().step_by(4) {
                    println!("           ngl={:>2}  {:>6.2} tok/s  {:>6.2} ms  {}", n, tps, ms, b);
                }
            }
        }
    }

    println!("\n== divergence matrix: best tok/s by CPU class and kernel (4 GiB Turing, PTQ1_0) ==");
    println!("{:<32} {:>12} {:>12} {:>14}", "CPU", "vec_dot", "repacked", "repack gain");
    for cpu in cpus() {
        let mut line = format!("{:<32}", cpu.name);
        let mut vals = Vec::new();
        for kernel in [Kernel::VecDot, Kernel::Repacked] {
            let rows = sweep(&cpu, &gpus()[0], kernel, Packing::Ptq10, fixed);
            let best = rows.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|r| r.1).unwrap_or(0.0);
            vals.push(best);
            line += &format!(" {:>12.2}", best);
        }
        let gain = if vals[0] > 0.0 { vals[1] / vals[0] } else { f64::INFINITY };
        println!("{} {:>13.2}x", line, gain);
    }

    println!("\n== local anchor rig: RTX 6000 + Xeon W-2140B ==");
    // calibration check: the corrected sweep must reproduce the measured
    // mixed point that set CPU_LEG_OVERHEAD_NS_PER_CPU_LAYER (plan/0008,
    // 2026-09-24: 4.07 tok/s at ngl 37, repack on, op-offload off, t16)
    {
        let rows = sweep(&cpus()[6], &gpus()[1], Kernel::Repacked, Packing::Ptq10, fixed);
        if let Some(row) = rows.iter().find(|r| r.0 == 37) {
            println!("calibration check, PTQ1_0 repacked ngl 37: {:.2} tok/s (measured 4.07)", row.1);
        }
    }
    for packing in [Packing::Ptq10, Packing::Pq20] {
        let rows = sweep(&cpus()[6], &gpus()[1], Kernel::Repacked, packing, fixed);
        if let Some(best) = rows.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
            println!("{:>6}: best split ngl={:>2}  {:>6.2} tok/s",
                if packing == Packing::Ptq10 { "PTQ1_0" } else { "PQ2_0" }, best.0, best.1);
        }
    }

    // the pure-GPU ends, against the measured anchors (llama-bench, 2026-09-20)
    let a = gpu_leg(&gpus()[1], layer_bytes(64, Packing::Ptq10).0 + Packing::Ptq10.output(), Packing::Ptq10);
    let b = gpu_leg(&gpus()[1], layer_bytes(64, Packing::Pq20).0 + Packing::Pq20.output(), Packing::Pq20);
    println!("pure GPU (ngl 64): PTQ1_0 {:.2} tok/s (anchor 31.84), PQ2_0 {:.2} tok/s (anchor 41.90)",
        1e9 / a.ns, 1e9 / b.ns);
}
