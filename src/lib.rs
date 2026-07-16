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
    /// The dependency-latency (recurrence) critical path: a chain is
    /// under-parallelized, so `depth * op_latency` exceeds the throughput term.
    Latency,
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

// ---------------------------------------------------------------- latency / ILP
//
// The dimension the throughput core lacked: dependency-limited (latency-bound)
// execution. The initiation-interval bound of modulo scheduling (Rau 1994; Lam
// 1988; survey Allan et al. 1995) is `MII = max(ResMII, RecMII)` — the resource/
// throughput term the core already models (`project`) vs the recurrence/latency
// term this adds. Little's Law (Little 1961) fixes when the recurrence is hidden:
// the independent in-flight work must reach `latency / (cycles-per-op)`.
//
// UNITS: `cyc_per_op` is R, the reciprocal throughput (cycles/op) — sub-1 op/cycle
// rates (a Turing HMMA issues one per 2 cycles) are integer here where an ops/cycle
// rate would round to zero. `op_latency` is L, cycles from issue to result. A
// chain of `depth` D dependent ops has critical path `D*L` regardless of how many
// such chains run — they overlap under it; the chain COUNT enters only through the
// throughput term (total ops). So `cycles = max(C*D*R, D*L)`, and utilization
// `= C*R/L` capped at 1: full only at `C >= L/R` (7 independent HMMA chains on
// TU102: L/R = 14/2). This is calx-mill's model of the plan/0143 FATTN result.

/// A dependency chain: `depth` dependent ops on the critical path, each
/// `op_latency` cycles issue->result. The recurrence (RecMII) term.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DepChain {
    pub depth: u32,
    pub op_latency: u32,
}

impl DepChain {
    /// The recurrence floor `depth * op_latency`: one chain's critical path,
    /// independent of how many chains run. Saturating (no overflow/panic).
    pub fn latency_floor(&self) -> u32 {
        self.depth.saturating_mul(self.op_latency)
    }
}

/// Compose a throughput projection with a dependency-latency floor: the
/// initiation-interval bound `max(ResMII, RecMII)`. `throughput` is the resource
/// term (from `project`/`project_mix`); `chain` gives the recurrence term. Returns
/// the slower. REDUCTION: when the chain is hidden (`latency_floor <= throughput`)
/// the throughput projection is returned UNCHANGED — no regression on
/// throughput-bound work (the Kani `latency_reduces_to_throughput_when_hidden`).
pub fn project_latency(throughput: Projection, chain: &DepChain) -> Projection {
    let floor = chain.latency_floor();
    if floor > throughput.cycles {
        Projection { cycles: floor, bottleneck: Bottleneck::Latency }
    } else {
        throughput
    }
}

/// The MII bound for `chains` independent dependency chains, each of depth `depth`,
/// on a pipe of reciprocal-throughput `cyc_per_op` (R, cycles/op) and per-op
/// `op_latency` (L). ResMII = `chains*depth*R` (all ops through the one pipe);
/// RecMII = `depth*L` (one chain's path). `cycles = max(ResMII, RecMII)`.
/// Saturating throughout.
pub fn chain_cycles(chains: u32, depth: u32, cyc_per_op: u32, op_latency: u32) -> Projection {
    let total_ops = chains.saturating_mul(depth);
    let throughput = Projection {
        cycles: total_ops.saturating_mul(cyc_per_op),
        bottleneck: Bottleneck::Pipe(0),
    };
    project_latency(throughput, &DepChain { depth, op_latency })
}

/// Utilization percent: `ResMII / MII` = the fraction of peak throughput reached.
/// Little's Law as a fraction: `min(1, C*R/L)` — 100% only when the chain is
/// hidden (`C >= L/R`). Reproduces the plan/0143 FATTN QK figure: (4,64,2,14) -> 57.
pub fn chain_utilization_pct(chains: u32, depth: u32, cyc_per_op: u32, op_latency: u32) -> u32 {
    let throughput = chains
        .saturating_mul(depth)
        .saturating_mul(cyc_per_op) as u64;
    let mii = chain_cycles(chains, depth, cyc_per_op, op_latency).cycles.max(1) as u64;
    ((throughput * 100) / mii) as u32
}

/// ADVISORY (compiler-dependent, deliberately NOT a Kani lower-bound proof): the
/// live register demand a hot loop inflates to when the compiler software-pipelines
/// `unroll` iterations, each carrying `per_iter_live` registers across the overlap
/// on top of the loop-invariant `base`. ONE-WAY CONSERVATIVE by construction: it may
/// only OVER-estimate — `base + unroll*per_iter_live`, the worst case where no
/// iteration's registers free before the next issues — so it can predict "won't
/// co-reside" wrongly but never "will co-reside" wrongly (a false "fits" launches a
/// kernel that then won't). Feed the result to `concurrency`/`blocks_per_instance`
/// so the co-residency verdict is available BEFORE ptxas runs. The proof obligation
/// (`unroll_registers_never_lowers_the_estimate`) is only that it is monotone
/// non-decreasing in `unroll` — more unroll never PREDICTS more co-residency.
pub fn unrolled_registers(base: u32, unroll: u32, per_iter_live: u32) -> u32 {
    base.saturating_add(unroll.saturating_mul(per_iter_live))
}

/// Independent dependency chains a register file can sustain.
///
/// [`OpTemplate::chains`] is a *request*; the scheduler keeps only as many live
/// accumulator chains as the budget holds. This couples the register dimension to the
/// latency dimension: raising `base_registers` — e.g. an overlap lever spending
/// registers on a prefetch buffer — shrinks the chains left, which can re-expose a
/// recurrence the isolated latency model treated as hidden (the register-file instance
/// of "shared resources do not compose by `max`"; see call/0031).
///
/// # Arguments
/// * `reg_budget` — per-thread register ceiling for the target occupancy (e.g.
///   `65536 / block_threads` at one block per SM).
/// * `base_registers` — registers held regardless of chain count (indices, pointers,
///   shared operands).
/// * `regs_per_chain` — live registers one independent accumulator chain costs
///   (clamped to a minimum of 1 to avoid division by zero).
///
/// # Returns
/// `(reg_budget - base_registers) / regs_per_chain`, floored at 0. Saturating; never
/// panics.
///
/// # Guarantees
/// One-way conservative: monotone non-decreasing in `reg_budget` and non-increasing in
/// `base_registers`, so spending registers can only lower the result, never raise it
/// (proof `spending_registers_never_raises_achievable_chains`).
pub fn achievable_chains(reg_budget: u32, base_registers: u32, regs_per_chain: u32) -> u32 {
    reg_budget.saturating_sub(base_registers) / regs_per_chain.max(1)
}

/// A composed op template: a dependency-structured op reduced to its latency-model
/// inputs — `chains` independent chains, each `depth` dependent ops, on a pipe of
/// reciprocal-throughput `cyc_per_op` and per-op `op_latency`. The convenience
/// layer (call/0028): build the inputs once per op family, project in one call.
/// Falsified by BREADTH, not by one family — a single flash-attention template
/// overfits (FA chains are latency-hideable, so it learns "always hideable"); the
/// breadth gate drives it across distinct structures, incl. a recurrence at
/// `chains == 1` that MUST stay latency-bound.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OpTemplate {
    pub chains: u32,
    pub depth: u32,
    pub cyc_per_op: u32,
    pub op_latency: u32,
}

impl OpTemplate {
    pub fn cycles(&self) -> Projection {
        chain_cycles(self.chains, self.depth, self.cyc_per_op, self.op_latency)
    }
    pub fn utilization_pct(&self) -> u32 {
        chain_utilization_pct(self.chains, self.depth, self.cyc_per_op, self.op_latency)
    }
    pub fn latency_bound(&self) -> bool {
        self.cycles().bottleneck == Bottleneck::Latency
    }
    /// The hide threshold in independent chains: `ceil(op_latency / cyc_per_op)` =
    /// L/R (Little's Law). At or above it the recurrence is hidden.
    pub fn chains_to_hide(&self) -> u32 {
        ceil_div(self.op_latency, self.cyc_per_op.max(1))
    }
    /// Utilization once the register budget clamps the requested chains.
    ///
    /// Bounds `self.chains` by [`achievable_chains`], then returns
    /// [`chain_utilization_pct`] at the clamped count. When a lever spends registers so
    /// that the sustainable chains fall below `self.chains`, this drops below
    /// [`utilization_pct`](Self::utilization_pct); since wall-time for fixed work scales
    /// as `1 / utilization`, that drop is the leg's slowdown.
    ///
    /// # Arguments
    /// * `reg_budget`, `base_registers`, `regs_per_chain` — see [`achievable_chains`].
    ///
    /// # Returns
    /// Utilization percent (`0..=100`) at `min(self.chains, achievable_chains(..))`
    /// chains, floored at one chain.
    pub fn utilization_under_registers(
        &self,
        reg_budget: u32,
        base_registers: u32,
        regs_per_chain: u32,
    ) -> u32 {
        let ach = achievable_chains(reg_budget, base_registers, regs_per_chain)
            .min(self.chains)
            .max(1);
        chain_utilization_pct(ach, self.depth, self.cyc_per_op, self.op_latency)
    }
    /// Whether an overlap lever that spends `spend` registers keeps the recurrence hidden.
    ///
    /// Composes the two dimensions the leg-level model keeps apart: overlapping the load
    /// (the register `spend`) is only free if the compute leg still sustains enough
    /// chains to reach the hide threshold.
    ///
    /// # Arguments
    /// * `reg_budget`, `base_registers`, `regs_per_chain` — see [`achievable_chains`].
    /// * `spend` — registers the overlap mechanism adds to `base_registers`.
    ///
    /// # Returns
    /// `true` iff [`achievable_chains`] after the spend is at least
    /// [`chains_to_hide`](Self::chains_to_hide); `false` flags a spend that re-exposes
    /// the recurrence.
    pub fn overlap_keeps_hidden(
        &self,
        reg_budget: u32,
        base_registers: u32,
        regs_per_chain: u32,
        spend: u32,
    ) -> bool {
        let ach = achievable_chains(
            reg_budget,
            base_registers.saturating_add(spend),
            regs_per_chain,
        );
        ach >= self.chains_to_hide()
    }
}

/// Which lane a [`Phase`] runs on. A multi-phase op interleaves memory-bound tile
/// loads and compute-bound math; a double-buffer overlaps the two lanes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lane {
    Memory,
    Compute,
}

/// One phase of a multi-phase op, reduced to a cycle cost and its lane. Ops like
/// the megakernel FATTN decode are NOT one dependency chain: they interleave
/// memory phases (K/V tile loads) with compute phases (QK, PV HMMA), serialized
/// by barriers. The single-chain [`OpTemplate`] mispredicts such an op's wall
/// time (call/0032, the three overnight FATTN regressions that were each
/// single-chain-green yet regressed). Build a phase per barrier-separated stage;
/// an [`OpComposition`] then projects the serial sum and the double-buffer floor.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Phase {
    pub cycles: u32,
    pub lane: Lane,
}

impl Phase {
    /// A compute phase whose cost is an [`OpTemplate`]'s MII.
    pub fn compute(t: OpTemplate) -> Phase {
        Phase { cycles: t.cycles().cycles, lane: Lane::Compute }
    }
    /// A memory phase whose cost the caller derives from `bytes / bandwidth`
    /// (e.g. via [`project`] or `stream_bytes` over the substrate's `mem_bandwidth`).
    pub fn memory(cycles: u32) -> Phase {
        Phase { cycles, lane: Lane::Memory }
    }
}

/// A multi-phase op as an ordered list of barrier-separated [`Phase`]s. The
/// composition the single-chain model lacks (call/0032): it sums the phases the
/// hardware runs serially today, and projects the floor a two-lane double-buffer
/// reaches by overlapping the memory lane under the compute lane. It does the
/// lane arithmetic only; the register feasibility of the overlap is the existing
/// [`OpTemplate::overlap_keeps_hidden`] gate, wired in by [`Self::overlapped_if`]
/// — a smem double-buffer spends no registers (creditable), a register-prefetch
/// spends registers and may not be (the call/0031 trap that regressed the rig).
#[derive(Clone, Copy, Debug)]
pub struct OpComposition<'a> {
    pub phases: &'a [Phase],
}

impl<'a> OpComposition<'a> {
    /// Total cycles with no overlap: every phase runs in series (today's kernel).
    ///
    /// # Returns
    /// The saturating sum of every phase's cycles.
    pub fn serial_cycles(&self) -> u32 {
        self.phases
            .iter()
            .fold(0u32, |acc, p| acc.saturating_add(p.cycles))
    }

    /// Cycles spent in one lane.
    ///
    /// # Arguments
    /// * `lane` — [`Lane::Memory`] or [`Lane::Compute`].
    ///
    /// # Returns
    /// The saturating sum of the cycles of phases on `lane`.
    pub fn lane_cycles(&self, lane: Lane) -> u32 {
        self.phases
            .iter()
            .filter(|p| p.lane == lane)
            .fold(0u32, |acc, p| acc.saturating_add(p.cycles))
    }

    /// The pipelined floor a perfect two-lane double-buffer reaches: the memory
    /// lane hides under the compute lane (or vice versa), so the op costs the
    /// larger of the two lane sums rather than their sum.
    ///
    /// # Returns
    /// `max(memory_lane_cycles, compute_lane_cycles)`.
    pub fn overlapped_cycles(&self) -> u32 {
        self.lane_cycles(Lane::Memory)
            .max(self.lane_cycles(Lane::Compute))
    }

    /// The projection to trust, given whether the overlap is register-creditable.
    ///
    /// The load↔compute overlap only earns [`Self::overlapped_cycles`] if the
    /// mechanism keeps the compute lane's ILP — a smem double-buffer does (spends
    /// no registers), a register-prefetch may not. Compute `creditable` as the
    /// AND over the compute phases' [`OpTemplate::overlap_keeps_hidden`] for the
    /// overlap's register spend.
    ///
    /// # Arguments
    /// * `creditable` — whether the overlap keeps every compute phase's recurrence
    ///   hidden after its register spend.
    ///
    /// # Returns
    /// [`Self::overlapped_cycles`] when `creditable`, else [`Self::serial_cycles`].
    ///
    /// # Guarantees
    /// `overlapped_if(false) == serial_cycles() >= overlapped_cycles() ==
    /// overlapped_if(true)`: a non-creditable overlap never yields an optimistic
    /// number (proof `overlap_gate_never_optimistic`), so a register-spending
    /// lever cannot be green-lit by the composition.
    pub fn overlapped_if(&self, creditable: bool) -> u32 {
        if creditable {
            self.overlapped_cycles()
        } else {
            self.serial_cycles()
        }
    }
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

    // ---- latency / ILP dimension (call/0028) ----
    // MII = max(ResMII, RecMII): the recurrence floor `depth*op_latency` composed
    // with the throughput term. depth*op_latency and chains*depth*cyc_per_op are
    // u32 products, so bound the multiplicands to a fragment/chain scale (CBMC
    // stays tractable, 0/1/boundary edges still hit).
    const CHAIN_N: u32 = 24; // chains / depth scale (proof-only; theorems are structural,
    const LAT: u32 = 16; // op-latency / cyc-per-op scale — a small bound keeps CBMC's
                         // multiplication reasoning (chains*depth, depth*latency) tractable
                         // while still hitting 0/1/boundary and the C>=L/R threshold cases.

    // (a) Reduction: when the chain is hidden (recurrence floor <= throughput), the
    // latency projection is EXACTLY the throughput projection — no regression on
    // throughput-bound work (MII collapses to ResMII == the core's PPM).
    #[kani::proof]
    fn latency_reduces_to_throughput_when_hidden() {
        let t_cycles: u32 = kani::any();
        let t_bind: usize = kani::any();
        let depth: u32 = kani::any();
        let op_latency: u32 = kani::any();
        kani::assume(t_cycles <= BOUND && t_bind < 8 && depth <= CHAIN_N && op_latency <= LAT);
        let chain = DepChain { depth, op_latency };
        kani::assume(chain.latency_floor() <= t_cycles); // hidden
        let thr = Projection { cycles: t_cycles, bottleneck: Bottleneck::Pipe(t_bind) };
        kani::assert(
            project_latency(thr, &chain) == thr,
            "hidden chain => projection unchanged (== PPM)",
        );
    }

    // (b) The projection is the max of both terms: never below either bound (a
    // correct MII lower bound); saturating (Kani also checks no panic/overflow).
    #[kani::proof]
    fn latency_is_the_max_of_both_bounds() {
        let t_cycles: u32 = kani::any();
        let depth: u32 = kani::any();
        let op_latency: u32 = kani::any();
        kani::assume(t_cycles <= BOUND && depth <= CHAIN_N && op_latency <= LAT);
        let chain = DepChain { depth, op_latency };
        let thr = Projection { cycles: t_cycles, bottleneck: Bottleneck::Pipe(0) };
        let out = project_latency(thr, &chain).cycles;
        kani::assert(out >= t_cycles, "never below the throughput (ResMII) term");
        kani::assert(out >= chain.latency_floor(), "never below the recurrence (RecMII) term");
    }

    // (c) Monotone in depth (at fixed chains/rate/latency): a deeper dependency
    // chain is never faster — deepening a recurrence cannot lower cycles.
    #[kani::proof]
    fn latency_monotone_in_depth() {
        let chains: u32 = kani::any();
        let d_lo: u32 = kani::any();
        let d_hi: u32 = kani::any();
        let r: u32 = kani::any();
        let l: u32 = kani::any();
        kani::assume(chains <= CHAIN_N && d_lo <= d_hi && d_hi <= CHAIN_N && r <= LAT && l <= LAT);
        let lo = chain_cycles(chains, d_lo, r, l).cycles;
        let hi = chain_cycles(chains, d_hi, r, l).cycles;
        kani::assert(lo <= hi, "deeper chain (fixed chains) => never faster");
    }

    // (d) Monotone in chains AT FIXED TOTAL WORK — the load-bearing statement (the
    // interleave / depth-reduction lever): with C1*D1 == C2*D2 and D1 >= D2, the
    // throughput term is identical (same total ops) and the recurrence floor D*L is
    // non-increasing, so more/shorter chains never raise cycles. Stated at fixed
    // work because with C,D free "more chains" adds work (the naive form is false).
    #[kani::proof]
    fn latency_monotone_in_chains_at_fixed_work() {
        let c1: u32 = kani::any();
        let d1: u32 = kani::any();
        let c2: u32 = kani::any();
        let d2: u32 = kani::any();
        let r: u32 = kani::any();
        let l: u32 = kani::any();
        kani::assume(
            c1 <= CHAIN_N && d1 <= CHAIN_N && c2 <= CHAIN_N && d2 <= CHAIN_N && r <= LAT && l <= LAT,
        );
        kani::assume(c1.saturating_mul(d1) == c2.saturating_mul(d2)); // same total work W
        kani::assume(d1 >= d2); // fewer/deeper (c1) vs more/shallower (c2)
        let fewer_deeper = chain_cycles(c1, d1, r, l).cycles;
        let more_shallow = chain_cycles(c2, d2, r, l).cycles;
        kani::assert(
            more_shallow <= fewer_deeper,
            "more, shorter chains at fixed work => never slower",
        );
    }

    // (e) The unroll->register advisory is one-way conservative: monotone
    // NON-DECREASING in unroll, so more unroll never predicts FEWER registers, and
    // (composing with `fewer_registers_never_lowers_concurrency`) never predicts
    // MORE co-residency — a false "fits" is the dangerous direction. This is the
    // ONLY property proven of the advisory; it is not a lower bound on the real
    // (compiler-chosen) register count.
    #[kani::proof]
    fn unroll_registers_never_lowers_the_estimate() {
        let base: u32 = kani::any();
        let u_lo: u32 = kani::any();
        let u_hi: u32 = kani::any();
        let per: u32 = kani::any();
        kani::assume(base <= BOUND && u_lo <= u_hi && u_hi <= CHAIN_N && per <= LAT);
        kani::assert(
            unrolled_registers(base, u_lo, per) <= unrolled_registers(base, u_hi, per),
            "more unroll => never fewer registers (never falsely promises co-residency)",
        );
        kani::assert(unrolled_registers(base, u_lo, per) >= base, "estimate >= base");
    }

    // `achievable_chains` is one-way conservative in `base_registers`: spending
    // registers can only lower the chains the file sustains (call/0031).
    #[kani::proof]
    fn spending_registers_never_raises_achievable_chains() {
        let budget: u32 = kani::any();
        let base: u32 = kani::any();
        let spend: u32 = kani::any();
        let per: u32 = kani::any();
        kani::assume(budget <= BOUND && base <= BOUND && spend <= BOUND && per <= LAT);
        kani::assert(
            achievable_chains(budget, base.saturating_add(spend), per)
                <= achievable_chains(budget, base, per),
            "spending registers never raises the chains the file can sustain",
        );
    }

    // `achievable_chains` is monotone non-decreasing in `reg_budget`.
    #[kani::proof]
    fn more_budget_never_lowers_achievable_chains() {
        let b_lo: u32 = kani::any();
        let b_hi: u32 = kani::any();
        let base: u32 = kani::any();
        let per: u32 = kani::any();
        kani::assume(b_lo <= b_hi && b_hi <= BOUND && base <= BOUND && per <= LAT);
        kani::assert(
            achievable_chains(b_lo, base, per) <= achievable_chains(b_hi, base, per),
            "more register budget never sustains fewer chains",
        );
    }

    // `overlap_keeps_hidden` is monotone in `spend`: if a larger spend keeps the
    // recurrence hidden, a smaller one does too (no false-safe cliff).
    #[kani::proof]
    fn overlap_hiding_is_monotone_in_spend() {
        let op = OpTemplate {
            chains: kani::any(),
            depth: kani::any(),
            cyc_per_op: kani::any(),
            op_latency: kani::any(),
        };
        let budget: u32 = kani::any();
        let base: u32 = kani::any();
        let per: u32 = kani::any();
        let s_lo: u32 = kani::any();
        let s_hi: u32 = kani::any();
        kani::assume(op.chains <= CHAIN_N && op.depth <= CHAIN_N);
        kani::assume(op.cyc_per_op >= 1 && op.cyc_per_op <= LAT && op.op_latency <= LAT);
        kani::assume(budget <= BOUND && base <= BOUND && per <= LAT);
        kani::assume(s_lo <= s_hi && s_hi <= BOUND);
        if op.overlap_keeps_hidden(budget, base, per, s_hi) {
            kani::assert(
                op.overlap_keeps_hidden(budget, base, per, s_lo),
                "a smaller register spend can only keep the leg more hidden",
            );
        }
    }

    // OpComposition (call/0032): the two-lane overlap floor never falls below the
    // larger lane and never rises above the serial sum. Bounds a 4-phase op so the
    // saturating sums stay well inside u32.
    #[kani::proof]
    fn overlap_between_max_lane_and_serial() {
        let c: [u32; 4] = [kani::any(), kani::any(), kani::any(), kani::any()];
        let l: [bool; 4] = [kani::any(), kani::any(), kani::any(), kani::any()];
        for i in 0..4 {
            kani::assume(c[i] <= BOUND);
        }
        let phases = [
            Phase { cycles: c[0], lane: if l[0] { Lane::Memory } else { Lane::Compute } },
            Phase { cycles: c[1], lane: if l[1] { Lane::Memory } else { Lane::Compute } },
            Phase { cycles: c[2], lane: if l[2] { Lane::Memory } else { Lane::Compute } },
            Phase { cycles: c[3], lane: if l[3] { Lane::Memory } else { Lane::Compute } },
        ];
        let comp = OpComposition { phases: &phases };
        let ov = comp.overlapped_cycles();
        let ser = comp.serial_cycles();
        let mem = comp.lane_cycles(Lane::Memory);
        let cmp = comp.lane_cycles(Lane::Compute);
        kani::assert(ov == mem.max(cmp), "overlap is the max of the two lanes");
        kani::assert(ov <= ser, "overlap never exceeds the serial sum");
        kani::assert(mem <= ser && cmp <= ser, "each lane is within the serial sum");
    }

    // The register gate never produces an optimistic number: a non-creditable
    // overlap falls back to the serial sum, which is >= the overlapped floor.
    #[kani::proof]
    fn overlap_gate_never_optimistic() {
        let c: [u32; 3] = [kani::any(), kani::any(), kani::any()];
        for i in 0..3 {
            kani::assume(c[i] <= BOUND);
        }
        let phases = [
            Phase { cycles: c[0], lane: Lane::Memory },
            Phase { cycles: c[1], lane: Lane::Compute },
            Phase { cycles: c[2], lane: Lane::Memory },
        ];
        let comp = OpComposition { phases: &phases };
        kani::assert(
            comp.overlapped_if(false) == comp.serial_cycles(),
            "a non-creditable overlap yields the serial sum",
        );
        kani::assert(
            comp.overlapped_if(false) >= comp.overlapped_if(true),
            "gating off never yields a smaller (optimistic) projection",
        );
    }
}

#[cfg(test)]
mod register_coupling_tests {
    use super::*;

    #[test]
    fn achievable_chains_arithmetic() {
        assert_eq!(achievable_chains(170, 87, 8), 10); // (170-87)/8
        assert_eq!(achievable_chains(87, 87, 8), 0); // no headroom
        assert_eq!(achievable_chains(50, 87, 8), 0); // base > budget (saturating)
        assert_eq!(achievable_chains(100, 0, 0), 100); // regs_per_chain clamped to 1
    }

    // The register-prefetch regression as a worked case: a spend that fits the
    // occupancy budget can still starve the compute leg's chains and re-expose the
    // recurrence — the cost the isolated max(load, compute) projection missed.
    #[test]
    fn register_spend_reexposes_latency() {
        // FATTN QK compute leg after depth-reduction: 7 chains hide the 14-cyc HMMA
        // latency at 2 cyc/op, so it runs at 100% utilization.
        let qk = OpTemplate { chains: 7, depth: 64, cyc_per_op: 2, op_latency: 14 };
        assert_eq!(qk.chains_to_hide(), 7);
        assert_eq!(qk.utilization_pct(), 100);
        // A budget that just fits the 7 chains (4 regs/chain over a 100-reg base):
        // all sustained, fully hidden.
        let (budget, base, per) = (128, 100, 4);
        assert_eq!(achievable_chains(budget, base, per), 7);
        assert!(qk.overlap_keeps_hidden(budget, base, per, 0));
        assert_eq!(qk.utilization_under_registers(budget, base, per), 100);
        // Spend 12 registers on an overlap buffer -> only 4 chains sustained -> the
        // recurrence re-exposes: keeps_hidden is false and utilization falls to 57%
        // (a ~1.75x leg slowdown). The spend still co-resides — occupancy-OK, ILP-not.
        assert!(!qk.overlap_keeps_hidden(budget, base, per, 12));
        assert_eq!(qk.utilization_under_registers(budget, base + 12, per), 57);
    }

    #[test]
    fn spend_within_headroom_stays_hidden() {
        let qk = OpTemplate { chains: 7, depth: 64, cyc_per_op: 2, op_latency: 14 };
        // 160-reg budget over a 100-reg base sustains 15 chains: a 12-reg spend leaves
        // 12 >= 7, still hidden, and the clamp keeps all 7 requested chains at 100%.
        assert!(qk.overlap_keeps_hidden(160, 100, 4, 12));
        assert_eq!(qk.utilization_under_registers(160, 100, 4), 100);
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

#[cfg(test)]
mod latency_tests {
    use super::*;

    // Reproduce the plan/0143 FATTN result from the TU102 paper's constants (HMMA:
    // R = 2 cyc/op, L = 14 cyc). QK is NT = 4 independent position-tile chains, each
    // depth 64 (both-split: 2 HMMAs x 32 ks-steps). C = 4 < L/R = 7 -> latency-bound.
    #[test]
    fn fattn_qk_is_latency_bound_at_four_chains() {
        let p = chain_cycles(4, 64, 2, 14);
        assert_eq!(p.cycles, 896); // max(ResMII 4*64*2=512, RecMII 64*14=896) = 896
        assert_eq!(p.bottleneck, Bottleneck::Latency);
        assert_eq!(chain_utilization_pct(4, 64, 2, 14), 57); // 512/896: the measured ~57%
    }

    // The hide threshold C >= L/R = 7: at 7 chains the throughput term catches the
    // recurrence floor (100% util, throughput-bound); below it, latency-bound.
    #[test]
    fn hmma_hides_at_seven_chains() {
        assert_eq!(chain_utilization_pct(6, 64, 2, 14), 85); // 768/896, still latency-bound
        assert_eq!(chain_cycles(6, 64, 2, 14).bottleneck, Bottleneck::Latency);
        assert_eq!(chain_utilization_pct(7, 64, 2, 14), 100); // 896/896: hidden
        assert_eq!(chain_cycles(7, 64, 2, 14).bottleneck, Bottleneck::Pipe(0));
        assert_eq!(chain_cycles(8, 64, 2, 14).bottleneck, Bottleneck::Pipe(0)); // throughput-bound
    }

    // The depth-reduction / interleave lever at FIXED WORK W = 256 ops (the plan/0143
    // "P.V G>=7 / QK partial-sum" prediction): splitting one depth-64 chain into more,
    // shorter chains crosses the threshold and reaches the throughput floor.
    #[test]
    fn splitting_fixed_work_reaches_the_throughput_floor() {
        let deep = chain_cycles(4, 64, 2, 14); // fewer/deeper: latency-bound
        let split = chain_cycles(8, 32, 2, 14); // more/shorter, SAME 256 ops
        assert_eq!(deep.cycles, 896);
        assert_eq!(split.cycles, 512); // max(512, 32*14=448) = 512, throughput-bound
        assert_eq!(split.bottleneck, Bottleneck::Pipe(0));
        assert!(split.cycles < deep.cycles); // the win the latency model predicts
    }
}

#[cfg(test)]
mod overfit_gate_latency {
    use super::*;
    // The latency model must predict a dependency case on EACH substrate, or it
    // overfit TU102. (R = cyc/op, L = latency; same time-unit within a case.)

    // TU102 HMMA: R=2, L=14 (PAPER.md tab:tensor). One accumulator (C=1) deeply
    // latency-bound; hidden only at C >= L/R = 7.
    #[test]
    fn tu102_hmma_dependency_chain() {
        let t = OpTemplate { chains: 1, depth: 32, cyc_per_op: 2, op_latency: 14 };
        assert!(t.latency_bound());
        assert_eq!(t.chains_to_hide(), 7);
    }
    // AVX-512 FMA: 4-cyc latency, 2 FMA ports (~2/cyc) -> R=0.5; integer half-cycle
    // units R=1,L=8 keep the ratio. Textbook "8 accumulators saturate AVX FMA".
    #[test]
    fn avx512_fma_dependency_chain() {
        let t = OpTemplate { chains: 1, depth: 64, cyc_per_op: 1, op_latency: 8 };
        assert!(t.latency_bound());
        assert_eq!(t.chains_to_hide(), 8);
    }
    // ARM SME ZA outer-product-accumulate: same tile-accumulate chain as HMMA, a
    // different vendor's matrix engine -> the same model. Constants illustrative,
    // to be pinned from the SME microbench characterizations (Hello SME!/Demystify).
    #[test]
    fn arm_sme_za_dependency_chain() {
        let t = OpTemplate { chains: 1, depth: 32, cyc_per_op: 1, op_latency: 6 };
        assert!(t.latency_bound());
        assert_eq!(t.chains_to_hide(), 6);
    }
    // 8088 EU: unpipelined -> issue-to-issue == latency (R == L) and NO ILP (C is
    // structurally 1). The model reports fully serial (util 100%, no hiding
    // headroom): it does NOT falsely promise hiding where there is no ILP.
    #[test]
    fn i8088_unpipelined_chain_has_no_ilp() {
        let t = OpTemplate { chains: 1, depth: 8, cyc_per_op: 118, op_latency: 118 };
        assert_eq!(t.cycles().cycles, 8 * 118); // serial
        assert_eq!(t.utilization_pct(), 100);
        assert_eq!(t.chains_to_hide(), 1);
    }
}

#[cfg(test)]
mod op_templates_breadth {
    use super::*;
    // Falsify the template layer by BREADTH across distinct latency structures; a
    // single flash-attention template would overfit ("always hideable").

    // (1) FA-decode QK: NT=4 hideable chains, depth 64. Latency-bound at C=4 (<7);
    // splitting fixed work to C>=7 reaches the throughput floor.
    #[test]
    fn fa_decode_hideable() {
        let t = OpTemplate { chains: 4, depth: 64, cyc_per_op: 2, op_latency: 14 };
        assert!(t.latency_bound());
        assert_eq!(t.utilization_pct(), 57);
        let hidden = OpTemplate { chains: 8, depth: 32, cyc_per_op: 2, op_latency: 14 };
        assert!(!hidden.latency_bound());
    }
    // (2) DeltaNet/GDN recurrence -- THE anti-overfit case: each step depends on the
    // prior state, so independent chains = 1 by data dependency. MUST stay
    // latency-bound and MUST NOT be reported hideable. Chunk-parallel scan raises C
    // to the CHUNK COUNT (a ceiling), never to one-chain-per-step.
    #[test]
    fn deltanet_recurrence_stays_latency_bound() {
        let seq = OpTemplate { chains: 1, depth: 128, cyc_per_op: 1, op_latency: 8 };
        assert!(seq.latency_bound()); // C=1: maximally latency-bound
        assert_eq!(seq.utilization_pct(), 12); // 128/(128*8)
        let chunked = OpTemplate { chains: 8, depth: 16, cyc_per_op: 1, op_latency: 8 };
        assert!(!chunked.latency_bound()); // C=8=L/R hidden -- but the chunk ceiling
    }
    // (3) MMVQ dp4a GEMV tail: CUDA-core pipe (not tensor), ILP from output columns.
    #[test]
    fn mmvq_dp4a_tail() {
        let one = OpTemplate { chains: 1, depth: 32, cyc_per_op: 1, op_latency: 6 };
        assert!(one.latency_bound());
        let many = OpTemplate { chains: 6, depth: 32, cyc_per_op: 1, op_latency: 6 };
        assert!(!many.latency_bound()); // 6 columns hide the 6-cyc dp4a latency
    }
    // (4) Cross-GPU XCHG/NVLink: interconnect latency, few in-flight messages -- a
    // different RESOURCE than compute, same model. Constants illustrative.
    #[test]
    fn xchg_nvlink_latency_bound() {
        let t = OpTemplate { chains: 2, depth: 4, cyc_per_op: 10, op_latency: 400 };
        assert!(t.latency_bound()); // 4*400 >> 2*4*10
    }
    // (5) cross-substrate is covered by overfit_gate_latency (AVX-512, 8088). Five
    // distinct structures through ONE primitive (OpTemplate/chain_cycles): if any
    // could not be expressed, the layer leaked.
}

#[cfg(test)]
mod op_composition_tests {
    use super::*;

    // Validation against the measured FATTN_DECODE sub-phase profile (plan/0143,
    // capture/fattn-phase-profile.txt, deep 256K dual-GPU). Phase cycles are the
    // measured per-op times in centi-milliseconds (0.01 ms units); the model is
    // unit-agnostic, so the ratios are what matter. Memory lane = K+V loads;
    // compute lane = QK + softmax + PV + setup.
    fn fattn_phases() -> [Phase; 6] {
        [
            Phase::memory(758),  // KLOAD   7.58 ms
            Phase::compute(OpTemplate { chains: 1, depth: 1055, cyc_per_op: 1, op_latency: 1 }), // QK 10.55
            Phase::compute(OpTemplate { chains: 1, depth: 162, cyc_per_op: 1, op_latency: 1 }),  // SOFTMAX 1.62
            Phase::memory(655),  // VLOAD   6.55 ms
            Phase::compute(OpTemplate { chains: 1, depth: 724, cyc_per_op: 1, op_latency: 1 }),  // PV 7.24
            Phase::compute(OpTemplate { chains: 1, depth: 4, cyc_per_op: 1, op_latency: 1 }),    // SETUP 0.04
        ]
    }

    #[test]
    fn serial_reproduces_the_measured_whole_op() {
        let phases = fattn_phases();
        let comp = OpComposition { phases: &phases };
        // Measured FATTN_DECODE whole-op wall: 33.68 ms = 3368 centi-ms. The phase
        // laps sum to 3358; within the one-thread-sample tolerance (< 0.5%).
        assert_eq!(comp.serial_cycles(), 3358);
        assert!((comp.serial_cycles() as i32 - 3368).abs() <= 20);
    }

    #[test]
    fn overlap_projects_the_double_buffer_floor() {
        let phases = fattn_phases();
        let comp = OpComposition { phases: &phases };
        // memory lane = 758 + 655 = 1413; compute lane = 1055 + 162 + 724 + 4 = 1945.
        assert_eq!(comp.lane_cycles(Lane::Memory), 1413);
        assert_eq!(comp.lane_cycles(Lane::Compute), 1945);
        // A perfect double-buffer hides the 14.1 ms of load under the 19.45 ms of
        // compute: op -> 19.45 ms (from 33.58), the ~30% decode prize.
        assert_eq!(comp.overlapped_cycles(), 1945);
    }

    // The discriminator the single-chain model lacked: the smem double-buffer
    // spends no registers (creditable -> the overlap floor); the register-prefetch
    // spends registers, fails the compute leg's hide gate (not creditable -> the
    // serial sum), which is exactly what regressed three times on the rig.
    #[test]
    fn register_gate_separates_double_buffer_from_prefetch() {
        let phases = fattn_phases();
        let comp = OpComposition { phases: &phases };
        // FATTN QK compute leg: 7 chains hide the 14-cyc HMMA at 2 cyc/op (the
        // worked case of register_spend_reexposes_latency). Budget 128 over a
        // 100-reg base just fits 7 chains.
        let qk = OpTemplate { chains: 7, depth: 64, cyc_per_op: 2, op_latency: 14 };
        let (budget, base, per) = (128u32, 100u32, 4u32);
        // smem double-buffer: spends no registers -> 7 chains sustained -> creditable.
        let db_creditable = qk.overlap_keeps_hidden(budget, base, per, 0);
        assert!(db_creditable);
        assert_eq!(comp.overlapped_if(db_creditable), 1945); // the win

        // register-prefetch: spends 12 registers -> only 4 chains sustained -> the
        // recurrence re-exposes -> NOT creditable -> the composition falls back to
        // the serial sum, matching the three rig regressions.
        let pf_creditable = qk.overlap_keeps_hidden(budget, base, per, 12);
        assert!(!pf_creditable);
        assert_eq!(comp.overlapped_if(pf_creditable), 3358); // no credit -> serial
    }
}
