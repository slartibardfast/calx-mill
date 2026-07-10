//! calx-mill - a generic substrate-throughput modeler.
//!
//! Two halves:
//!   - [`concurrency`]: the substrate-generic "occupancy" - how many concurrent
//!     work units fit, a `min` of resource limits. The 8088 prefetch queue, the
//!     AVX-512 ROB, and the GPU SM's warps are instances of the same quantity.
//!   - [`project`]: the per-pipe-max (PPM) throughput projection, floored by the
//!     issue cap and the memory byte budget. Generalizes the Roofline model to
//!     multiple pipes.
//!
//! The core knows nothing of CUDA, x86, or the 8088; a [`Substrate`] is just its
//! resource axes. Kani proves the arithmetic universal over substrate specs (see
//! the `proofs` module, run with `cargo kani`).

#![forbid(unsafe_code)]

/// A compute substrate: the resource axes shared by every machine we model, from
/// an Intel 8088 to an AVX-512 core to a GPU SM.
#[derive(Clone, Copy)]
pub struct Substrate {
    /// register-file capacity, in allocation units (registers).
    pub register_capacity: u32,
    /// register allocation granularity (e.g. 256 regs/warp on sm_75).
    pub register_granularity: u32,
    /// fast local store capacity, in bytes (shared memory / L1 scratch).
    pub local_store_bytes: u32,
    /// local-store allocation granularity (e.g. 128 B on sm_75).
    pub local_store_granularity: u32,
    /// the in-flight-work ceiling (warps/SM, ROB entries, prefetch-queue depth).
    pub concurrency_ceiling: u32,
    /// peak issue width, in uops/cycle (the scheduler cap, all pipes summed).
    pub issue_cap: u32,
    /// memory bandwidth, in bytes/cycle.
    pub mem_bandwidth: u32,
}

/// Per-concurrency-unit resource demand. One "concurrency unit" is whatever the
/// substrate runs in parallel to hide latency: a GPU warp, a CPU loop iteration,
/// an 8088 instruction stream. The core never sees threads: a GPU adapter folds
/// `regs/thread x threads/warp` into `registers` before constructing this.
#[derive(Clone, Copy)]
pub struct WorkUnit {
    /// register-file demand for one concurrency unit (GPU: regs/thread x 32).
    pub registers: u32,
    /// local-store (smem / scratch) demand for one concurrency unit, in bytes.
    pub local_store_bytes: u32,
}

/// Round `x` up to the next multiple of `unit`. A `unit` of 0 means "no
/// granularity" and leaves `x` unchanged. Saturates rather than overflow.
pub fn round_up(x: u32, unit: u32) -> u32 {
    if unit == 0 {
        return x;
    }
    let rem = x % unit;
    if rem == 0 {
        x
    } else {
        x.saturating_add(unit - rem)
    }
}

/// Ceiling division: ceil(a / b). `b` of 0 yields 0 (treated as "no cap").
pub fn ceil_div(a: u32, b: u32) -> u32 {
    if b == 0 {
        return 0;
    }
    // (a + b - 1) / b, without the overflow near u32::MAX.
    let q = a / b;
    if a % b == 0 {
        q
    } else {
        q.saturating_add(1)
    }
}

/// How many concurrent work units fit on `s` given demand `w`. This is the
/// substrate-generic occupancy: the saturation of the concurrency dimension. It is
/// the `min` of the register limit, the local-store limit, and the hard ceiling.
pub fn concurrency(s: &Substrate, w: &WorkUnit) -> u32 {
    let reg_per = round_up(w.registers.max(1), s.register_granularity);
    let by_reg = s.register_capacity / reg_per;
    let by_local = if w.local_store_bytes == 0 || s.local_store_bytes == 0 {
        s.concurrency_ceiling
    } else {
        let ls_per = round_up(w.local_store_bytes, s.local_store_granularity).max(1);
        s.local_store_bytes / ls_per
    };
    by_reg.min(by_local).min(s.concurrency_ceiling)
}

/// Occupancy fraction in `[0, 100]`: concurrency realised over the ceiling.
pub fn occupancy_pct(s: &Substrate, w: &WorkUnit) -> u32 {
    let c = concurrency(s, w);
    if s.concurrency_ceiling == 0 {
        return 0;
    }
    // c <= ceiling by construction, so this is exact without overflow at 100%.
    (c * 100) / s.concurrency_ceiling
}

/// Which resource binds the projection first.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bottleneck {
    /// A specific compute pipe (its index in the rate slice).
    Pipe(usize),
    /// The scheduler issue cap (total uops/cycle across all pipes).
    IssueCap,
    /// The memory byte budget.
    Memory,
}

/// A steady-state throughput projection: predicted cycles and the binding resource.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Projection {
    pub cycles: u32,
    pub bottleneck: Bottleneck,
}

/// Per-pipe-max (PPM) projection. `pipe_rates[i]` is pipe i's issue rate
/// (ops/cycle); `ops_per_pipe[i]` is the workload's op count on that pipe. The
/// result is the per-pipe demand floored by the substrate issue cap and the memory
/// byte budget (`total_bytes` / `mem_bandwidth`).
pub fn project(
    s: &Substrate,
    pipe_rates: &[u32],
    ops_per_pipe: &[u32],
    total_bytes: u32,
) -> Projection {
    debug_assert_eq!(pipe_rates.len(), ops_per_pipe.len());
    let mut cyc_pipe = 0u32;
    let mut bind = 0usize;
    for i in 0..pipe_rates.len() {
        let rate = pipe_rates[i].max(1);
        let ops = *ops_per_pipe.get(i).unwrap_or(&0);
        let c = ceil_div(ops, rate);
        if c > cyc_pipe {
            cyc_pipe = c;
            bind = i;
        }
    }
    let total_ops = ops_per_pipe.iter().copied().fold(0u32, |a, b| a.saturating_add(b));
    let cyc_issue = ceil_div(total_ops, s.issue_cap.max(1));
    let cyc_mem = ceil_div(total_bytes, s.mem_bandwidth);

    if cyc_mem >= cyc_pipe && cyc_mem >= cyc_issue {
        Projection { cycles: cyc_mem, bottleneck: Bottleneck::Memory }
    } else if cyc_issue >= cyc_pipe {
        Projection { cycles: cyc_issue, bottleneck: Bottleneck::IssueCap }
    } else {
        Projection { cycles: cyc_pipe, bottleneck: Bottleneck::Pipe(bind) }
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    // Keep the state space tractable: the theorems are about arithmetic
    // structure, not specific magnitudes.
    const BOUND: u32 = 1 << 20;

    fn any_substrate() -> Substrate {
        let s = Substrate {
            register_capacity: kani::any(),
            register_granularity: kani::any(),
            local_store_bytes: kani::any(),
            local_store_granularity: kani::any(),
            concurrency_ceiling: kani::any(),
            issue_cap: kani::any(),
            mem_bandwidth: kani::any(),
        };
        kani::assume(s.register_granularity <= BOUND);
        kani::assume(s.local_store_granularity <= BOUND);
        s
    }

    fn any_work_unit() -> WorkUnit {
        WorkUnit {
            registers: kani::any(),
            local_store_bytes: kani::any(),
        }
    }

    #[kani::proof]
    fn round_up_is_a_multiple_and_ge_x() {
        let x: u32 = kani::any();
        let unit: u32 = kani::any();
        kani::assume(x <= BOUND);
        kani::assume(unit <= BOUND);
        let r = round_up(x, unit);
        if unit == 0 {
            kani::assert(r == x, "unit 0 leaves x unchanged");
        } else {
            kani::assert(r % unit == 0, "round_up yields a multiple of unit");
            kani::assert(r >= x, "round_up >= x");
            kani::assert(r == x || r - x < unit, "round_up is the tightest multiple >= x");
        }
    }

    #[kani::proof]
    fn concurrency_never_exceeds_the_ceiling() {
        let s = any_substrate();
        let w = any_work_unit();
        let c = concurrency(&s, &w);
        kani::assert(c <= s.concurrency_ceiling, "concurrency bounded by the ceiling");
    }

    #[kani::proof]
    fn occupancy_is_in_unit_range() {
        let s = any_substrate();
        let w = any_work_unit();
        let p = occupancy_pct(&s, &w);
        kani::assume(s.concurrency_ceiling > 0);
        kani::assert(p <= 100, "occupancy in [0,100]");
    }

    #[kani::proof]
    fn fewer_registers_never_lowers_concurrency() {
        // monotonicity: shrinking per-unit register demand cannot reduce the
        // concurrent count (catches a sign-flip in the resource limit).
        let s = any_substrate();
        let mut a = any_work_unit();
        let mut b = any_work_unit();
        kani::assume(a.registers <= b.registers);
        kani::assume(a.local_store_bytes == b.local_store_bytes);
        let ca = concurrency(&s, &a);
        let cb = concurrency(&s, &b);
        kani::assert(ca >= cb, "fewer regs/work-unit => concurrency non-decreasing");
    }
}
