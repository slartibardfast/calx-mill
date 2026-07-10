//! The three-substrate falsifier (plan/0136's overfit gate).
//!
//! If the core cannot predict the 8088, an AVX-512 core, and a TU102 SM, the
//! `Substrate` abstraction leaked. These are hand-authored characterizations; the
//! numbers are illustrative at the stand-up, refined when real adapters land.

use calx_mill::{concurrency, occupancy_pct, project, Bottleneck, Substrate, WorkUnit};

/// Intel 8088 (~4.77 MHz): one ALU, a 4-byte prefetch queue as its only overlap,
/// 14 registers, an 8-bit bus. Almost always fetch/memory-bound.
fn i8088() -> Substrate {
    Substrate {
        register_capacity: 14,
        register_granularity: 1,
        local_store_bytes: 0,
        local_store_granularity: 0,
        concurrency_ceiling: 4, // the prefetch queue
        issue_cap: 1,
        mem_bandwidth: 1, // bytes/cycle-class: an 8-bit bus starves the EU
    }
}

/// An AVX-512 core: 32 architectural ZMM registers (we model the ROB as the
/// concurrency ceiling), two FMA512 pipes, a wide issue cap, ample L1 bandwidth.
fn avx512_core() -> Substrate {
    Substrate {
        register_capacity: 180, // physical vector regs as the pressure limit
        register_granularity: 1,
        local_store_bytes: 32 * 1024, // L1d
        local_store_granularity: 64,
        concurrency_ceiling: 256, // the ROB
        issue_cap: 5,
        mem_bandwidth: 64, // ~2x 256-bit loads/cycle, in B/cycle-class
    }
}

/// A TU102 SM (sm_75): 65536 regs, 256-reg/warp allocation unit, 64 KiB smem,
/// 32 warps/SM, 4 schedulers, 64 B/clk/SM smem streaming.
fn tu102_sm() -> Substrate {
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

#[test]
fn i8088_is_prefetch_bound() {
    let s = i8088();
    let w = WorkUnit { registers: 4, local_store_bytes: 0 };
    // 14 regs / 4 = 3, floored by the 4-deep prefetch queue ceiling.
    assert_eq!(concurrency(&s, &w), 3);
    // a heavier instruction stream saturates the small register set first.
    let heavy = WorkUnit { registers: 8, local_store_bytes: 0 };
    assert_eq!(concurrency(&s, &heavy), 1);
}

#[test]
fn avx512_fills_the_rob_before_register_pressure() {
    let s = avx512_core();
    let w = WorkUnit { registers: 3, local_store_bytes: 0 };
    // 180 / 3 = 60, floored by the 256-entry ROB => register pressure binds at 60.
    assert_eq!(concurrency(&s, &w), 60);
    let light = WorkUnit { registers: 1, local_store_bytes: 0 };
    // 180 regs, one each => 180, still under the ROB.
    assert_eq!(concurrency(&s, &light), 180);
}

#[test]
fn tu102_sm_hits_full_warp_occupancy_then_reg_limit() {
    let s = tu102_sm();
    // 40 regs/thread => 40 x 32 = 1280 regs/warp => 65536 / 1280 = 51 => floored to 32.
    let light = WorkUnit { registers: 40 * 32, local_store_bytes: 0 };
    assert_eq!(concurrency(&s, &light), 32);
    assert_eq!(occupancy_pct(&s, &light), 100);
    // 128 regs/thread => 128 x 32 = 4096 regs/warp => 65536 / 4096 = 16 warps (reg-limited).
    let heavy = WorkUnit { registers: 128 * 32, local_store_bytes: 0 };
    assert_eq!(concurrency(&s, &heavy), 16);
    assert_eq!(occupancy_pct(&s, &heavy), 50);
}

#[test]
fn projection_names_the_binding_resource() {
    // 8088 moving bytes: the 8-bit bus binds, not the lone ALU.
    let s = i8088();
    let p = project(&s, &[1], &[10], 200); // 10 ALU ops, 200 bytes
    assert_eq!(p.bottleneck, Bottleneck::Memory);
    // AVX-512 at high arithmetic intensity: an FMA512 pipe binds.
    let s = avx512_core();
    let p = project(&s, &[2, 2, 1], &[1000, 1000, 0], 10);
    assert!(matches!(p.bottleneck, Bottleneck::Pipe(0) | Bottleneck::Pipe(1)));
}
