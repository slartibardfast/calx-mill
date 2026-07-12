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

/// A block of concurrency units that co-allocate one local-store slab (GPU: a
/// CTA of `units` warps sharing its shared-memory allocation). Blocks are the
/// co-residency quantum of a cooperative launch: registers allocate per unit,
/// local store once per block. Where [`WorkUnit`] folds a block's local store
/// per unit as an approximation, this form is exact.
#[derive(Clone, Copy)]
pub struct Block {
    /// concurrency units per block (GPU: warps per CTA). Zero fits nowhere.
    pub units: u32,
    /// per-unit register demand (GPU: regs/thread x 32).
    pub unit_registers: u32,
    /// local store per block, in bytes, allocated once for the whole block.
    pub local_store_bytes: u32,
}

/// How many whole blocks fit on one substrate instance: the `min` of the
/// register limit (units x rounded per-unit demand), the block-granular
/// local-store limit, and the concurrency ceiling in whole blocks.
pub fn blocks_per_instance(s: &Substrate, b: &Block) -> u32 {
    if b.units == 0 {
        return 0;
    }
    let reg_per_unit = round_up(b.unit_registers.max(1), s.register_granularity);
    let reg_per_block = reg_per_unit.saturating_mul(b.units);
    let by_reg = s.register_capacity / reg_per_block;
    let by_local = if b.local_store_bytes == 0 || s.local_store_bytes == 0 {
        u32::MAX
    } else {
        let ls_per = round_up(b.local_store_bytes, s.local_store_granularity).max(1);
        s.local_store_bytes / ls_per
    };
    let by_ceiling = s.concurrency_ceiling / b.units;
    by_reg.min(by_local).min(by_ceiling)
}

/// Cooperative-launch residency: a grid of `grid_blocks` blocks across
/// `instances` substrate instances is feasible only if every block is
/// co-resident at once — a cooperative grid barrier deadlocks otherwise.
/// This is the check an occupancy number alone does not give you.
pub fn cooperative_fits(s: &Substrate, b: &Block, grid_blocks: u32, instances: u32) -> bool {
    blocks_per_instance(s, b).saturating_mul(instances) >= grid_blocks
}

/// The precision configuration of an op: the knob that DERIVES the three
/// resource axes the core already models, rather than a fourth axis. It is
/// strictly the PERFORMANCE half of the precision-performance frontier; the
/// accuracy half is measured, not modelled here (see
/// `reference/megakernel/precision_ledger.md`), and no accuracy term crosses
/// this seam. Widths are in bytes so the core stays dtype-agnostic (an adapter
/// maps q8/f16/f32/bf16 to these); `pipe` indexes the caller's own pipe naming
/// (e.g. a tensor pipe for f16 HMMA vs a CUDA-core FMA pipe for f32).
#[derive(Clone, Copy)]
pub struct Precision {
    /// accumulator element width in bytes (fp16=2, fp32=4, fp64=8). Sets the
    /// accumulator's register footprint via [`acc_registers`].
    pub acc_bytes: u32,
    /// streamed element width in bytes (q8=1, f16=2, f32=4). Sets the streamed
    /// byte total via [`stream_bytes`].
    pub data_bytes: u32,
    /// the compute pipe this precision dispatches to, an index into the
    /// caller's `pipe_rates` (tensor vs CUDA-core, etc.).
    pub pipe: usize,
}

/// Streamed bytes for `elems` elements at this precision's data width. This is
/// the `total_bytes` / `mem_bytes` the projection floors by the memory budget,
/// so halving the data width (f16 -> q8 KV) halves the memory-bound cycles.
pub fn stream_bytes(elems: u32, p: &Precision) -> u32 {
    elems.saturating_mul(p.data_bytes)
}

/// Per-concurrency-unit register demand at this precision: the
/// precision-independent `base_registers` plus the accumulator's cost, which is
/// `acc_elems` elements of `acc_bytes` each, packed into 32-bit registers
/// (ceil). An fp32 accumulator (4 B) costs ~2x the registers of fp16 (2 B), so
/// this is where accumulator precision buys back accuracy at an occupancy cost.
pub fn acc_registers(base_registers: u32, acc_elems: u32, p: &Precision) -> u32 {
    let acc_regs = ceil_div(acc_elems.saturating_mul(p.acc_bytes), 4);
    base_registers.saturating_add(acc_regs)
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

/// Which resource class a fractional steady-state demand belongs to. Substrate-
/// generic: a pipe is an index into the caller's own pipe naming; the other three
/// axes (local-store bandwidth, memory bandwidth, the issue cap) exist on every
/// substrate that has them at all.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ResourceKind {
    Pipe(usize),
    LocalStoreBw,
    MemoryBw,
    Issue,
}

/// A fractional steady-state projection: the per-resource cycle demands in
/// evaluation order, the per-pipe-max (PPM) result, and the naive-additive (ADD)
/// result reported alongside (never gated).
#[derive(Clone, Debug, PartialEq)]
pub struct SteadyState {
    /// every resource's cycle demand, in the order they were evaluated.
    pub per_resource: Vec<(ResourceKind, f64)>,
    /// the PPM projection: the largest per-resource demand.
    pub ppm_cycles: f64,
    /// index into `per_resource` of the binding resource (first max wins).
    pub ppm_bound: usize,
    /// the additive model: every non-issue demand summed; if that sum is not
    /// positive, the issue demand (an all-control workload still issues).
    pub add_cycles: f64,
}

/// Build the fractional per-resource demand vector for one work unit scaled by
/// `concurrency` concurrent units. `pipe_cycles[i]` is pipe i's accumulated
/// issue-cycle demand (the caller's op-mix fold, `sum(ops / rate)`); a local-store
/// or memory demand of exactly zero means "absent" and emits no entry. Order:
/// pipes, local store, memory, issue - `select_bound`'s first-max-wins tie-break
/// binds on this order.
pub fn mix_demands(
    concurrency: f64,
    pipe_cycles: &[f64],
    local_store_cycles: f64,
    mem_bytes: f64,
    mem_bytes_per_cycle: f64,
    total_ops: f64,
    issue_cap: f64,
) -> Vec<(ResourceKind, f64)> {
    let mut v = Vec::with_capacity(pipe_cycles.len() + 3);
    for (i, &c) in pipe_cycles.iter().enumerate() {
        v.push((ResourceKind::Pipe(i), concurrency * c));
    }
    if local_store_cycles != 0.0 {
        v.push((ResourceKind::LocalStoreBw, concurrency * local_store_cycles));
    }
    if mem_bytes != 0.0 {
        v.push((ResourceKind::MemoryBw, concurrency * mem_bytes / mem_bytes_per_cycle));
    }
    v.push((ResourceKind::Issue, concurrency * total_ops / issue_cap));
    v
}

/// PPM + ADD selection over a demand vector. The binding resource is the first
/// entry with the maximal demand (a stable max scan: a later entry must be
/// strictly greater to displace an earlier one). An empty vector yields a zero
/// projection rather than panicking.
pub fn select_bound(per_resource: &[(ResourceKind, f64)]) -> SteadyState {
    if per_resource.is_empty() {
        return SteadyState {
            per_resource: Vec::new(),
            ppm_cycles: 0.0,
            ppm_bound: 0,
            add_cycles: 0.0,
        };
    }
    let mut bound = 0usize;
    for i in 1..per_resource.len() {
        if per_resource[i].1 > per_resource[bound].1 {
            bound = i;
        }
    }
    let mut add = 0.0f64;
    let mut issue = None;
    for &(kind, cycles) in per_resource {
        if kind == ResourceKind::Issue {
            issue = Some(cycles);
        } else {
            add += cycles;
        }
    }
    let add_cycles = if add > 0.0 { add } else { issue.unwrap_or(add) };
    SteadyState {
        per_resource: per_resource.to_vec(),
        ppm_cycles: per_resource[bound].1,
        ppm_bound: bound,
        add_cycles,
    }
}

/// The fractional-rate per-pipe-max projection: `mix_demands` folded through
/// `select_bound`. This is `project` generalized to fractional rates and an
/// explicit local-store bandwidth axis.
pub fn project_mix(
    concurrency: f64,
    pipe_cycles: &[f64],
    local_store_cycles: f64,
    mem_bytes: f64,
    mem_bytes_per_cycle: f64,
    total_ops: f64,
    issue_cap: f64,
) -> SteadyState {
    select_bound(&mix_demands(
        concurrency,
        pipe_cycles,
        local_store_cycles,
        mem_bytes,
        mem_bytes_per_cycle,
        total_ops,
        issue_cap,
    ))
}

pub mod nvidia;

#[cfg(kani)]
mod proofs {
    use super::*;

    // Keep the state space tractable: the theorems are about arithmetic structure,
    // not specific magnitudes, so a representative bounded range is sufficient (and
    // still exercises every code path and edge case: 0, 1, granularity boundaries).
    const BOUND: u32 = 1 << 10;

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
        kani::assume(s.register_capacity <= BOUND);
        kani::assume(s.register_granularity <= BOUND);
        kani::assume(s.local_store_bytes <= BOUND);
        kani::assume(s.local_store_granularity <= BOUND);
        kani::assume(s.concurrency_ceiling <= BOUND);
        kani::assume(s.issue_cap <= BOUND);
        kani::assume(s.mem_bandwidth <= BOUND);
        s
    }

    fn any_work_unit() -> WorkUnit {
        let w = WorkUnit {
            registers: kani::any(),
            local_store_bytes: kani::any(),
        };
        kani::assume(w.registers <= BOUND);
        kani::assume(w.local_store_bytes <= BOUND);
        w
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

    // ---- precision dimension (call/0022) ----
    // The two-axis theorem: data precision buys bandwidth, accumulator
    // precision costs concurrency. Both DERIVE the existing resource axes, so
    // the proofs compose with the core's own monotonicity, not a new model.

    // Precision widths are byte widths of real dtypes (q8=1..fp64=8), and
    // element counts fit a fragment/tile; bounding the MULTIPLICANDS to these
    // realistic ranges keeps the u32 products small enough for CBMC while still
    // exercising 0/1/boundary edges. The monotonicity theorems are structural,
    // so this loses no generality.
    const DTYPE_BYTES: u32 = 8; // fp64 is the widest we model
    const ELEMS: u32 = 256; // accumulator elements / streamed elements per unit

    // Underlying lemma: `project` is monotone non-decreasing in total_bytes,
    // because cyc_mem = ceil_div(bytes, bw) is, and project == max(pipe, issue,
    // mem). Bytes carried directly (no multiply): the full BOUND range is cheap.
    #[kani::proof]
    fn project_is_monotone_in_bytes() {
        let s = any_substrate();
        let rate: u32 = kani::any();
        let ops: u32 = kani::any();
        let b_lo: u32 = kani::any();
        let b_hi: u32 = kani::any();
        kani::assume(rate <= BOUND && ops <= BOUND && b_lo <= b_hi && b_hi <= BOUND);
        let clo = project(&s, &[rate], &[ops], b_lo).cycles;
        let chi = project(&s, &[rate], &[ops], b_hi).cycles;
        kani::assert(clo <= chi, "fewer streamed bytes => projection never slower");
    }

    // Less data precision never makes a memory-bound projection slower: fewer
    // bytes per element floor the projection at fewer memory cycles (composes
    // stream_bytes monotonicity with the lemma above). q8 vs f16 KV is a
    // projection, not a measurement.
    #[kani::proof]
    fn data_precision_buys_bandwidth() {
        let s = any_substrate();
        let elems: u32 = kani::any();
        let d_lo: u32 = kani::any(); // lower precision (fewer bytes/elem)
        let d_hi: u32 = kani::any();
        kani::assume(elems <= ELEMS && d_lo <= d_hi && d_hi <= DTYPE_BYTES);
        let lo = Precision { acc_bytes: 4, data_bytes: d_lo, pipe: 0 };
        let hi = Precision { acc_bytes: 4, data_bytes: d_hi, pipe: 0 };
        let rate: u32 = kani::any();
        let ops: u32 = kani::any();
        kani::assume(rate <= BOUND && ops <= BOUND);
        let clo = project(&s, &[rate], &[ops], stream_bytes(elems, &lo)).cycles;
        let chi = project(&s, &[rate], &[ops], stream_bytes(elems, &hi)).cycles;
        kani::assert(clo <= chi, "lower data precision => projection never slower");
    }

    // More accumulator precision never raises concurrency: a wider accumulator
    // costs more registers, and concurrency is monotone non-increasing in
    // registers (composes acc_registers monotonicity with the core). fp32
    // accumulate is an occupancy cost.
    #[kani::proof]
    fn acc_precision_costs_concurrency() {
        let s = any_substrate();
        let base: u32 = kani::any();
        let acc_elems: u32 = kani::any();
        let a_lo: u32 = kani::any(); // narrower accumulator
        let a_hi: u32 = kani::any(); // wider accumulator
        kani::assume(base <= BOUND && acc_elems <= ELEMS && a_lo <= a_hi && a_hi <= DTYPE_BYTES);
        let ls: u32 = kani::any();
        kani::assume(ls <= BOUND);
        let lo = Precision { acc_bytes: a_lo, data_bytes: 2, pipe: 0 };
        let hi = Precision { acc_bytes: a_hi, data_bytes: 2, pipe: 0 };
        let w_lo = WorkUnit { registers: acc_registers(base, acc_elems, &lo), local_store_bytes: ls };
        let w_hi = WorkUnit { registers: acc_registers(base, acc_elems, &hi), local_store_bytes: ls };
        kani::assert(w_lo.registers <= w_hi.registers, "wider acc => more registers");
        kani::assert(
            concurrency(&s, &w_lo) >= concurrency(&s, &w_hi),
            "wider accumulator precision => concurrency non-increasing",
        );
    }

    #[kani::proof]
    fn occupancy_is_in_unit_range() {
        let s = any_substrate();
        let w = any_work_unit();
        let p = occupancy_pct(&s, &w);
        kani::assume(s.concurrency_ceiling > 0);
        kani::assert(p <= 100, "occupancy in [0,100]");
    }

    fn any_demand_value() -> f64 {
        // finite: a demand is cycles of work; the ADD sum of two opposite
        // infinities would manufacture a NaN no real workload can produce.
        let v: f64 = kani::any();
        kani::assume(v.is_finite());
        v
    }

    #[kani::proof]
    fn select_bound_picks_the_first_maximal_demand() {
        let demands = [
            (ResourceKind::Pipe(0), any_demand_value()),
            (ResourceKind::Pipe(1), any_demand_value()),
            (ResourceKind::LocalStoreBw, any_demand_value()),
            (ResourceKind::MemoryBw, any_demand_value()),
            (ResourceKind::Issue, any_demand_value()),
        ];
        let r = select_bound(&demands);
        kani::assert(r.ppm_bound < demands.len(), "bound index is in range");
        kani::assert(
            r.ppm_cycles == demands[r.ppm_bound].1,
            "PPM equals the demand it names",
        );
        let mut i = 0;
        while i < demands.len() {
            kani::assert(!(demands[i].1 > r.ppm_cycles), "no demand exceeds the PPM");
            i += 1;
        }
        let mut j = 0;
        while j < r.ppm_bound {
            kani::assert(!(demands[j].1 >= r.ppm_cycles), "first max wins");
            j += 1;
        }
    }

    #[kani::proof]
    fn select_bound_add_falls_back_to_issue_when_no_work() {
        let issue = any_demand_value();
        let demands = [
            (ResourceKind::Pipe(0), 0.0),
            (ResourceKind::LocalStoreBw, 0.0),
            (ResourceKind::Issue, issue),
        ];
        let r = select_bound(&demands);
        kani::assert(
            r.add_cycles == issue,
            "an all-control workload's ADD is its issue demand",
        );
    }

    #[kani::proof]
    fn select_bound_handles_the_empty_vector() {
        let r = select_bound(&[]);
        kani::assert(r.ppm_cycles == 0.0, "empty demand vector projects zero");
        kani::assert(r.add_cycles == 0.0, "empty demand vector adds zero");
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

    fn any_block() -> Block {
        let b = Block {
            units: kani::any(),
            unit_registers: kani::any(),
            local_store_bytes: kani::any(),
        };
        kani::assume(b.units <= BOUND);
        kani::assume(b.unit_registers <= BOUND);
        kani::assume(b.local_store_bytes <= BOUND);
        b
    }

    #[kani::proof]
    fn blocks_respect_every_capacity() {
        let s = any_substrate();
        let b = any_block();
        kani::assume(b.units > 0);
        let n = blocks_per_instance(&s, &b);
        kani::assert(
            n.saturating_mul(b.units) <= s.concurrency_ceiling,
            "resident blocks respect the concurrency ceiling",
        );
        let reg_per_unit = round_up(b.unit_registers.max(1), s.register_granularity);
        let total = (n as u64) * (reg_per_unit as u64) * (b.units as u64);
        kani::assert(
            total <= s.register_capacity as u64,
            "resident blocks respect the register file",
        );
    }

    #[kani::proof]
    fn fewer_unit_registers_never_lower_blocks() {
        let s = any_substrate();
        let a = any_block();
        let b = any_block();
        kani::assume(a.units == b.units);
        kani::assume(a.local_store_bytes == b.local_store_bytes);
        kani::assume(a.unit_registers <= b.unit_registers);
        kani::assert(
            blocks_per_instance(&s, &a) >= blocks_per_instance(&s, &b),
            "fewer regs/unit => blocks/instance non-decreasing",
        );
    }

    #[kani::proof]
    fn cooperative_fits_is_monotone_in_instances() {
        let s = any_substrate();
        let b = any_block();
        let g: u32 = kani::any();
        let i1: u32 = kani::any();
        let i2: u32 = kani::any();
        kani::assume(g <= BOUND);
        kani::assume(i1 <= BOUND);
        kani::assume(i2 <= BOUND);
        kani::assume(i1 <= i2);
        if cooperative_fits(&s, &b, g, i1) {
            kani::assert(
                cooperative_fits(&s, &b, g, i2),
                "more instances never break residency",
            );
        }
    }

    #[kani::proof]
    fn one_block_per_instance_is_the_persistent_kernel_shape() {
        // grid == instances (one block per instance, the persistent-kernel
        // launch) is feasible exactly when at least one block fits.
        let s = any_substrate();
        let b = any_block();
        let n: u32 = kani::any();
        kani::assume(n >= 1);
        kani::assume(n <= BOUND);
        kani::assert(
            cooperative_fits(&s, &b, n, n) == (blocks_per_instance(&s, &b) >= 1),
            "grid == instances iff one block fits per instance",
        );
    }
}

#[cfg(test)]
mod precision_tests {
    use super::*;

    // The two-axis frontier with concrete numbers (call/0022).
    #[test]
    fn data_width_halves_the_bytes() {
        let q8 = Precision { acc_bytes: 4, data_bytes: 1, pipe: 0 };
        let f16 = Precision { acc_bytes: 4, data_bytes: 2, pipe: 0 };
        let f32 = Precision { acc_bytes: 4, data_bytes: 4, pipe: 0 };
        assert_eq!(stream_bytes(1024, &q8), 1024);
        assert_eq!(stream_bytes(1024, &f16), 2048);
        assert_eq!(stream_bytes(1024, &f32), 4096);
        // q8 KV is exactly half the streamed bytes of f16 KV (the +45% lever).
        assert_eq!(2 * stream_bytes(1024, &q8), stream_bytes(1024, &f16));
    }

    #[test]
    fn fp32_accumulate_costs_registers() {
        // a 16x16 WMMA accumulator holds 8 elements per lane. base = the op's
        // precision-independent registers.
        let fp16 = Precision { acc_bytes: 2, data_bytes: 2, pipe: 0 };
        let fp32 = Precision { acc_bytes: 4, data_bytes: 2, pipe: 0 };
        assert_eq!(acc_registers(40, 8, &fp16), 40 + 4); // 8*2/4 = 4 regs
        assert_eq!(acc_registers(40, 8, &fp32), 40 + 8); // 8*4/4 = 8 regs
        // fp32 accumulate costs strictly more registers -> never more concurrency.
        let s = Substrate {
            register_capacity: 65536, register_granularity: 256,
            local_store_bytes: 0, local_store_granularity: 128,
            concurrency_ceiling: 32, issue_cap: 4, mem_bandwidth: 624,
        };
        let w16 = WorkUnit { registers: acc_registers(40, 8, &fp16) * 32, local_store_bytes: 0 };
        let w32 = WorkUnit { registers: acc_registers(40, 8, &fp32) * 32, local_store_bytes: 0 };
        assert!(concurrency(&s, &w16) >= concurrency(&s, &w32));
    }
}
