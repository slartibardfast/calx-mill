//! Primitive-telemetry ingest — measured `SteadyState` anchors from the megakernel's
//! on-device primitives (`clock64` / `%globaltimer` / `%smid` + schedule-known
//! `bytes`/`ops`). Realizes `plan/0144/spec/primitive-telemetry-ingest.md`.
//!
//! The megakernel is a persistent cooperative kernel no external HW profiler can attach
//! to (ncu can't replay a cooperative launch; nsys deadlocks the host-doorbell loop;
//! CUPTI is proprietary, out of scope). So the anchor is produced by the kernel measuring
//! itself, and this module turns those records into the *measured image of calx-mill's own
//! projection vector* — a measured [`SteadyState`] in the same [`ResourceKind`]/
//! [`Bottleneck`] columns the model projects — so the gate is column-by-column, not a
//! scalar compare. It gates the CLASSIFICATION (T5) and the overlap RE-BIND (T6), the two
//! axes every plan/0143 mispredict got wrong.

use crate::validity::Verdict;
use crate::{select_bound, Bottleneck, Lane, ResourceKind, SteadyState};

/// The stamped SM clock in GHz — the single source for every cycle<->ns conversion
/// and clock-derived constant here (it was previously three inline 1.455 literals).
/// Per-record conversions prefer the DERIVED clock when plausible; this stamp is
/// the fallback and the plausibility reference. See [`TeleRecord::effective_clock_ghz`].
pub const STAMPED_CLOCK_GHZ: f64 = 1.455;

/// Fractional tolerance for a record's derived clock against the stamp: outside it
/// the record is clock-implausible (counted, stamped-clock fallback, no anchor).
pub const CLOCK_TOL_FRAC: f64 = 0.05;

/// DRAM roofline as bytes moved per SM-clock cycle: 609 GB/s at the stamped clock,
/// GPU-global (a record's `bytes` is the op's whole DRAM traffic). Provenance: TU102
/// spec bandwidth / SM clock; consistent with the out-of-band `nvidia-smi dmon`
/// mem-controller-77%-busy observation (plan/0143 capture). The numerator that turns
/// measured bytes into a `MemoryBw` cycle demand and bounds T1.
pub const PEAK_DRAM_BYTES_PER_CYCLE: f64 = 609.0e9 / (STAMPED_CLOCK_GHZ * 1.0e9); // ~418.6

/// GPU-global peak HMMA (tensor-pipe) rate in ops/cycle: the projection side's
/// [`crate::nvidia::projection::TENSOR_HMMA_PER_SM_CLK`] (PAPER.md tab:tensor,
/// m16n8k8) × 72 SMs (`nvidia::ptxas::TU102_SMS`) — ONE shared per-SM constant, so
/// the census projection and this measured-column denominator cannot disagree. The
/// Pipe-column denominator for a tensor op's SCHEDULE-KNOWN whole-op HMMA count,
/// matching the GPU-global convention `PEAK_DRAM_BYTES_PER_CYCLE` uses.
pub const TU102_HMMA_PER_CYCLE: f64 =
    crate::nvidia::projection::TENSOR_HMMA_PER_SM_CLK * 72.0; // = 36.0
/// GPU-global peak fma-pipe rate in ops/cycle: `2.0 op/SM/clk` (PAPER.md tab:alu — FFMA
/// and the fma-pipe-bound `IDP.4A`/dp4a both saturate at 2/SM/clk) × 72 SMs. The
/// Pipe-column denominator for the CUDA-core GEMV/dp4a ops.
pub const TU102_FMA_PER_CYCLE: f64 = 2.0 * 72.0; // = 144.0

/// One per-op telemetry record — the frozen kernel↔calx-mill seam (spec §seam). `bytes`
/// or `ops` == 0 means "column absent" (attached in a later pass; the ingest then emits
/// no `MemoryBw`/`Pipe` demand, matching `mix_demands`' zero convention).
///
/// `op_wall_ns` is a HOST-ATTACHED aggregate (like `bytes`/`ops`, NOT a new device-emitted
/// field — the four-value device ring is unchanged): the op's TRUE whole-op wall (the max
/// block span) for a block-uneven op, where block-0's own `%globaltimer` span under-counts
/// it. `0` means absent → the roofline falls back to block-0's `span_ns`, the prior
/// behaviour. See [`TeleRecord::wall_ns`].
#[derive(Clone, Debug, PartialEq)]
pub struct TeleRecord {
    pub op_index: u32,
    pub kind: String,
    pub lane: Lane,
    pub gt_start_ns: u64,
    pub gt_end_ns: u64,
    pub cycles: u64,
    pub bytes: u64,
    pub ops: u64,
    pub op_wall_ns: u64,
}

impl TeleRecord {
    /// The `%globaltimer` span in ns. Saturating, so it is `>= 0` for any inputs (T2).
    pub fn span_ns(&self) -> u64 {
        self.gt_end_ns.saturating_sub(self.gt_start_ns)
    }
    /// The op's TRUE whole-op wall in ns: the host-attached `op_wall_ns` (max block span)
    /// when present, else block-0's `%globaltimer` `span_ns`. For a BLOCK-UNEVEN op — a
    /// GEMV whose blocks own unequal row counts — block-0 finishes early, so `span_ns`
    /// under-counts the op wall and `bytes / span_ns` OVER-counts the achieved rate, a
    /// false T1 roofline reject (`whole-op-bytes` are GPU-global, but the span is one
    /// block's). Using the true wall makes the numerator and denominator commensurate.
    /// `max` keeps the accessor monotone (`>= span_ns`) even against a mis-attached value,
    /// so the correction can only LOWER the achieved rate — it removes false rejects, it
    /// can never manufacture a false accept.
    pub fn wall_ns(&self) -> u64 {
        if self.op_wall_ns > 0 {
            self.op_wall_ns.max(self.span_ns())
        } else {
            self.span_ns()
        }
    }
    /// Achieved DRAM GB/s = bytes / wall-ns (1 byte/ns == 1 GB/s), over the TRUE op wall
    /// ([`Self::wall_ns`]), not block-0's span. Zero when the wall or bytes are absent.
    pub fn achieved_gbps(&self) -> f64 {
        let ns = self.wall_ns();
        if ns == 0 || self.bytes == 0 {
            return 0.0;
        }
        self.bytes as f64 / ns as f64
    }
    /// T1 roofline sanity: a record implying a DRAM rate above the wall is an instrument
    /// error, not an anchor. `false` ⇒ reject the record (do not build an anchor from it).
    pub fn roofline_ok(&self) -> bool {
        self.achieved_gbps() <= PEAK_DRAM_BYTES_PER_CYCLE * STAMPED_CLOCK_GHZ // == 609 GB/s
    }
    /// The record's own realized clock in GHz: `cycles / span_ns`. Both counters are
    /// block-0's (`clock64` and `%globaltimer`), so they are commensurate by
    /// construction — never derive against the host-attached `op_wall_ns`, which is
    /// another block's span. `None` when either counter is absent (zero).
    pub fn derived_clock_ghz(&self) -> Option<f64> {
        let ns = self.span_ns();
        if ns == 0 || self.cycles == 0 {
            return None;
        }
        Some(self.cycles as f64 / ns as f64)
    }
    /// Clock plausibility against the stamp: a derivable clock must sit within
    /// `tol_frac` of [`STAMPED_CLOCK_GHZ`] (DVFS boost or throttle otherwise corrupts
    /// every conversion silently); an underivable clock is not checkable (`true`).
    pub fn clock_plausible(&self, tol_frac: f64) -> bool {
        match self.derived_clock_ghz() {
            None => true,
            Some(c) => (c - STAMPED_CLOCK_GHZ).abs() <= tol_frac * STAMPED_CLOCK_GHZ,
        }
    }
    /// The clock this record's conversions should use: derived when plausible, else
    /// the stamp. The fallback keeps an implausible record on the stamped
    /// denominators — a broken instrument must not re-scale the model around itself.
    pub fn effective_clock_ghz(&self) -> f64 {
        match self.derived_clock_ghz() {
            Some(c) if self.clock_plausible(CLOCK_TOL_FRAC) => c,
            _ => STAMPED_CLOCK_GHZ,
        }
    }
    /// The per-record DRAM bytes-per-cycle denominator at the effective clock.
    pub fn dram_bytes_per_cycle(&self) -> f64 {
        609.0e9 / (self.effective_clock_ghz() * 1.0e9)
    }
}

/// Reconstruct the measured [`SteadyState`] (spec §reconstruction): the per-`ResourceKind`
/// cycle demands from measured `bytes` + schedule-known `ops`, with `ppm_cycles` set to the
/// OBSERVED wall `cycles` — the wall is the independent realized bound, not the argmax of
/// the computed columns (those classify; the wall adjudicates). Absent (zero) columns are
/// omitted. `peak_pipe_rate` is ops/cycle for the op's compute pipe (0 ⇒ no `Pipe` column).
pub fn measured_steady_state(r: &TeleRecord, peak_pipe_rate: f64) -> SteadyState {
    let mut v: Vec<(ResourceKind, f64)> = Vec::new();
    if r.bytes > 0 {
        v.push((ResourceKind::MemoryBw, r.bytes as f64 / r.dram_bytes_per_cycle()));
    }
    if r.ops > 0 && peak_pipe_rate > 0.0 {
        v.push((ResourceKind::Pipe(0), r.ops as f64 / peak_pipe_rate));
    }
    let mut ss = select_bound(&v);
    ss.ppm_cycles = r.cycles as f64; // the measured wall IS the realized bound (spec)
    ss
}

/// The measured [`Bottleneck`] (spec §reconstruction): the column whose computed demand
/// explains the wall; `Latency` when the wall exceeds every throughput column by more than
/// `tol` (recurrence exposed — `D*L` dominates). The wall adjudicates; the computed
/// columns classify. `Memory` wins ties (the roofline is the floor we design against).
pub fn measured_bottleneck(r: &TeleRecord, peak_pipe_rate: f64, tol: f64) -> Bottleneck {
    let wall = r.cycles as f64;
    let mem = if r.bytes > 0 {
        r.bytes as f64 / r.dram_bytes_per_cycle()
    } else {
        0.0
    };
    let pipe = if r.ops > 0 && peak_pipe_rate > 0.0 {
        r.ops as f64 / peak_pipe_rate
    } else {
        0.0
    };
    let top = mem.max(pipe);
    if wall > top + tol {
        return Bottleneck::Latency; // wall exceeds all throughput terms
    }
    if mem >= pipe {
        Bottleneck::Memory
    } else {
        Bottleneck::Pipe(0)
    }
}

/// The GPU-global peak pipe rate (ops/cycle) for an op KIND — the per-op pipe-rate wiring
/// that lets the measured `Pipe` column populate from the schedule's ops (spec
/// §reconstruction: `Pipe(i) = ops / peak_rate[i]`). The schedule knows which pipe an op
/// dispatches to; calx-mill supplies the substrate rate. Tensor ops (FATTN / HMMA GEMM)
/// bind the HMMA pipe; the CUDA-core weight-stream GEMVs (MMVQ / dp4a) bind the fma pipe.
/// An unrecognized kind (or one with no `ops` numerator) returns `0.0` → no `Pipe` column,
/// matching the zero convention (`mix_demands` / [`measured_steady_state`]).
pub fn op_pipe_rate(kind: &str) -> f64 {
    let k = kind.to_ascii_uppercase();
    if k.contains("FATTN") || k.contains("HMMA") || k.contains("GEMM") || k.contains("ATTN") {
        TU102_HMMA_PER_CYCLE
    } else if k.contains("MMVQ") || k.contains("GEMV") || k.contains("DP4A") || k.contains("IDP") {
        TU102_FMA_PER_CYCLE
    } else {
        0.0
    }
}

/// Measured realized overlap of two op spans (spec §reconstruction): intersection /
/// min(len), clamped to [0, 1] (T2), and 0 when the spans are disjoint. This is the
/// *measured* concurrency `overlapped_contended` projects — the megakernel-unique
/// measurement no external profiler yields.
pub fn overlap_fraction(a: &TeleRecord, b: &TeleRecord) -> f64 {
    let lo = a.gt_start_ns.max(b.gt_start_ns);
    let hi = a.gt_end_ns.min(b.gt_end_ns);
    if hi <= lo {
        return 0.0; // disjoint
    }
    let inter = (hi - lo) as f64;
    let m = (a.span_ns().min(b.span_ns())) as f64;
    if m <= 0.0 {
        return 0.0;
    }
    (inter / m).min(1.0)
}

/// Gate a projected [`SteadyState`] against an op's measured record (spec §gate, T5). The
/// column-aligned verdict:
/// - `Certified` — the projected bound reproduces the measured wall within `tol_cyc` AND
///   the projected bottleneck matches the measured binding column.
/// - `Provisional` — the bound reproduces but the projected column is WRONG (right number,
///   wrong reason). Decide with caution; the bench confirms. Never `Certified`.
/// - `Refused` — the bound misses. Only the bench/oracle gates here.
///
/// T4 (no-launder) is the caller's: an op with no record has no anchor and must be
/// `Refused` — never reconstruct a default record to manufacture a pass.
pub fn gate_op(
    projected: &SteadyState,
    proj_bottleneck: Bottleneck,
    r: &TeleRecord,
    peak_pipe_rate: f64,
    tol_cyc: f64,
    tol_cls: f64,
) -> Verdict {
    let bound_ok = (projected.ppm_cycles - r.cycles as f64).abs() <= tol_cyc;
    let class_ok = proj_bottleneck == measured_bottleneck(r, peak_pipe_rate, tol_cls);
    if bound_ok && class_ok {
        Verdict::Certified
    } else if bound_ok {
        Verdict::Provisional // right number, wrong binding column
    } else {
        Verdict::Refused
    }
}

/// Parse a `.tele` TSV (the header line names the columns; see the spec seam). Unknown
/// lanes default to `Compute`. Malformed rows are skipped — see
/// [`parse_tele_counted`] for the loss surfaced. Returns the records in file order.
pub fn parse_tele(text: &str) -> Vec<TeleRecord> {
    parse_tele_counted(text).0
}

/// [`parse_tele`] with the loss surfaced: `(records, skipped)`, where `skipped`
/// counts data-shaped rows dropped as malformed (short or unparseable). Header,
/// comment, and blank lines are not losses. An instrument that drops records must
/// say so: a truncated or corrupted `.tele` no longer passes silently.
pub fn parse_tele_counted(text: &str) -> (Vec<TeleRecord>, usize) {
    let mut out = Vec::new();
    let mut skipped = 0usize;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("op_index") || line.starts_with('#') {
            continue;
        }
        // schema: op_index op_kind lane gt_start_ns gt_end_ns cycles bytes ops [op_wall_ns]
        // op_wall_ns is an OPTIONAL trailing column (host-attached true-op-wall); older
        // 8-column files parse unchanged with op_wall_ns == 0.
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 8 {
            skipped += 1;
            continue;
        }
        let (Some(op_index), Some(gs), Some(ge), Some(cyc), Some(bytes), Some(ops)) = (
            f[0].parse::<u32>().ok(),
            f[3].parse::<u64>().ok(),
            f[4].parse::<u64>().ok(),
            f[5].parse::<u64>().ok(),
            f[6].parse::<u64>().ok(),
            f[7].parse::<u64>().ok(),
        ) else {
            skipped += 1;
            continue;
        };
        out.push(TeleRecord {
            op_index,
            kind: f[1].to_string(),
            lane: if f[2] == "mem" { Lane::Memory } else { Lane::Compute },
            gt_start_ns: gs,
            gt_end_ns: ge,
            cycles: cyc,
            bytes,
            ops,
            op_wall_ns: f.get(8).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0),
        });
    }
    (out, skipped)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(gs: u64, ge: u64, cyc: u64, bytes: u64, ops: u64) -> TeleRecord {
        TeleRecord {
            op_index: 0,
            kind: "X".into(),
            lane: Lane::Memory,
            gt_start_ns: gs,
            gt_end_ns: ge,
            cycles: cyc,
            bytes,
            ops,
            op_wall_ns: 0,
        }
    }

    // T2 span well-formedness: span >= 0 (saturating), overlap in [0,1], 0 disjoint.
    #[test]
    fn t2_span_and_overlap_wellformed() {
        assert_eq!(rec(100, 50, 0, 0, 0).span_ns(), 0); // reversed -> saturates to 0
        let a = rec(0, 100, 0, 0, 0);
        let b = rec(200, 300, 0, 0, 0);
        assert_eq!(overlap_fraction(&a, &b), 0.0); // disjoint
        let c = rec(50, 150, 0, 0, 0);
        let o = overlap_fraction(&a, &c); // intersect [50,100]=50 / min(100,100)=100
        assert!((o - 0.5).abs() < 1e-9);
        let d = rec(10, 40, 0, 0, 0); // fully inside a -> inter 30 / min(100,30)=30 -> 1.0
        assert_eq!(overlap_fraction(&a, &d), 1.0);
    }

    // T1 roofline sanity: a record implying > 609 GB/s is rejected.
    #[test]
    fn t1_roofline_rejects_superluminal() {
        assert!(rec(0, 100, 0, 1_000, 0).roofline_ok()); // 10 GB/s, fine
        assert!(!rec(0, 1, 0, 100_000, 0).roofline_ok()); // 100000 GB/s, impossible
    }

    // T3 bound faithfulness: measured ppm == the observed wall; columns == bytes/rate.
    #[test]
    fn t3_reconstruct_is_exact() {
        let r = rec(0, 1000, 4242, 41_860, 84); // arbitrary bytes/ops
        let ss = measured_steady_state(&r, 2.0);
        assert_eq!(ss.ppm_cycles, 4242.0); // wall, not the argmax column (no fudge)
        let mem = ss.per_resource.iter().find(|(k, _)| *k == ResourceKind::MemoryBw).unwrap().1;
        let pipe = ss.per_resource.iter().find(|(k, _)| *k == ResourceKind::Pipe(0)).unwrap().1;
        // faithfulness: each column is EXACTLY numerator/rate, no fudge factor.
        assert_eq!(mem, 41_860.0 / PEAK_DRAM_BYTES_PER_CYCLE);
        assert_eq!(pipe, 84.0 / 2.0);
    }

    // T5 classification gate: right-number-wrong-column is Provisional, never Certified.
    #[test]
    fn t5_classification_gate() {
        // memory-bound op: bytes big, wall == the memory demand.
        let r = rec(0, 1000, 100, 41_860, 0); // mem demand 100 cyc == wall
        let ss = measured_steady_state(&r, 0.0);
        // projection that agrees on BOUND and COLUMN -> Certified.
        assert_eq!(
            gate_op(&ss, Bottleneck::Memory, &r, 0.0, 5.0, 5.0),
            Verdict::Certified
        );
        // right bound, WRONG column (says Pipe) -> Provisional.
        assert_eq!(
            gate_op(&ss, Bottleneck::Pipe(0), &r, 0.0, 5.0, 5.0),
            Verdict::Provisional
        );
        // wrong bound -> Refused.
        let mut ss2 = ss.clone();
        ss2.ppm_cycles = 500.0;
        assert_eq!(
            gate_op(&ss2, Bottleneck::Memory, &r, 0.0, 5.0, 5.0),
            Verdict::Refused
        );
    }

    // measured_bottleneck: latency when the wall exceeds every throughput column.
    #[test]
    fn measured_bottleneck_latency_when_wall_exceeds_throughput() {
        // small bytes/ops (tiny throughput demand) but a big wall -> recurrence exposed.
        let r = rec(0, 10, 1000, 418, 0); // mem demand ~1 cyc, wall 1000
        assert_eq!(measured_bottleneck(&r, 0.0, 5.0), Bottleneck::Latency);
    }

    #[test]
    fn counted_parse_surfaces_losses() {
        // one good row, one truncated (data-shaped, short), one unparseable field;
        // header/comment/blank lines are not losses.
        let t = "op_index\top_kind\tlane\tgt_start_ns\tgt_end_ns\tcycles\tbytes\tops\n\
                 # comment\n\
                 \n\
                 6\tMMVQ_Q4_0\tmem\t100\t200\t75282\t0\t0\n\
                 7\tGEMV_F16\tmem\t100\t200\n\
                 8\tGEMV_F16\tmem\t100\t200\tnot-a-number\t0\t0\n";
        let (recs, skipped) = parse_tele_counted(t);
        assert_eq!(recs.len(), 1);
        assert_eq!(skipped, 2);
        // the delegating wrapper is loss-blind by design (call sites unchanged)
        assert_eq!(parse_tele(t).len(), 1);
    }

    #[test]
    fn parse_roundtrip() {
        let t = "op_index\top_kind\tlane\tgt_start_ns\tgt_end_ns\tcycles\tbytes\tops\n\
                 6\tMMVQ_Q4_0\tmem\t100\t200\t75282\t0\t0\n";
        let v = parse_tele(t);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].op_index, 6);
        assert_eq!(v[0].cycles, 75282);
        assert_eq!(v[0].lane, Lane::Memory);
        assert_eq!(v[0].op_wall_ns, 0); // 8-column file -> absent -> block-0 span
    }

    // (b) true-op-wall: a block-uneven GEMV whose WHOLE-op bytes divided by BLOCK-0's short
    // span implies a super-peak rate — a false T1 reject. The host-attached true op wall
    // (max block span) makes numerator and denominator commensurate and clears the reject.
    #[test]
    fn true_op_wall_clears_block_uneven_false_reject() {
        // 100 kB whole-op over block-0's 100 ns span => 1000 GB/s > 609 => FALSE reject.
        let block0 = rec(0, 100, 0, 100_000, 0);
        assert!(!block0.roofline_ok());
        assert_eq!(block0.wall_ns(), 100);
        // The op's true wall (slowest block) is 200 ns => 500 GB/s <= 609 => accepted.
        let mut whole = block0.clone();
        whole.op_wall_ns = 200;
        assert_eq!(whole.wall_ns(), 200);
        assert!(whole.roofline_ok());
        // Monotone/robust: a mis-attached wall SHORTER than block-0's span never raises the
        // rate (max keeps wall_ns >= span_ns), so it cannot manufacture a false accept.
        let mut bad = block0.clone();
        bad.op_wall_ns = 50; // < span 100
        assert_eq!(bad.wall_ns(), 100); // clamped up to the span
        assert!(!bad.roofline_ok());
    }

    // clock derivation: derived when both counters present, plausibility-gated
    // against the stamp, stamped-clock fallback for implausible/underivable records.
    #[test]
    fn clock_derives_and_gates_plausibility() {
        // 1500 cycles over 1000 ns = 1.500 GHz: ~3% off the stamp -> plausible, used.
        let boost = rec(0, 1000, 1500, 0, 0);
        assert!((boost.derived_clock_ghz().unwrap() - 1.5).abs() < 1e-12);
        assert!(boost.clock_plausible(CLOCK_TOL_FRAC));
        assert!((boost.effective_clock_ghz() - 1.5).abs() < 1e-12);
        // 4242 cycles over 1000 ns = 4.242 GHz: a shifted-clock record is FLAGGED
        // (implausible) and falls back to the stamped denominators.
        let shifted = rec(0, 1000, 4242, 0, 0);
        assert!(!shifted.clock_plausible(CLOCK_TOL_FRAC));
        assert_eq!(shifted.effective_clock_ghz(), STAMPED_CLOCK_GHZ);
        // underivable (no cycles or no span): not checkable, stamped fallback.
        let absent = rec(0, 0, 0, 0, 0);
        assert_eq!(absent.derived_clock_ghz(), None);
        assert!(absent.clock_plausible(CLOCK_TOL_FRAC));
        assert_eq!(absent.effective_clock_ghz(), STAMPED_CLOCK_GHZ);
    }

    #[test]
    fn clock_rescales_memory_demand_when_plausible() {
        // At a plausible 1.5 GHz boost the same bytes cost MORE cycles of demand
        // (fewer bytes move per cycle), and the roofline reference stays 609 GB/s.
        let r = rec(0, 1000, 1500, 418_600, 0);
        let expected = 418_600.0 / (609.0e9 / 1.5e9);
        let ss = measured_steady_state(&r, 0.0);
        let mem = ss.per_resource.iter().find(|(k, _)| *k == ResourceKind::MemoryBw).unwrap().1;
        assert!((mem - expected).abs() < 1e-9);
    }

    #[test]
    fn parse_reads_optional_op_wall_column() {
        // 9-column row: the trailing op_wall_ns is picked up.
        let t = "op_index\top_kind\tlane\tgt_start_ns\tgt_end_ns\tcycles\tbytes\tops\top_wall_ns\n\
                 6\tMMVQ_Q4_0\tmem\t100\t200\t75282\t100000\t0\t340\n";
        let v = parse_tele(t);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].op_wall_ns, 340);
        assert_eq!(v[0].wall_ns(), 340); // 340 > span 100
    }

    // (a) per-op pipe-rate wiring: the schedule's op kind selects the pipe rate that
    // populates the measured Pipe column, so the bottleneck classifies correctly.
    #[test]
    fn op_pipe_rate_routes_by_kind() {
        assert_eq!(op_pipe_rate("FATTN_DECODE"), TU102_HMMA_PER_CYCLE);
        assert_eq!(op_pipe_rate("HMMA_QK"), TU102_HMMA_PER_CYCLE);
        assert_eq!(op_pipe_rate("MMVQ_Q4_0"), TU102_FMA_PER_CYCLE);
        assert_eq!(op_pipe_rate("GEMV_F16"), TU102_FMA_PER_CYCLE);
        assert_eq!(op_pipe_rate("RMSNORM"), 0.0); // unrecognized -> no Pipe column
    }

    #[test]
    fn pipe_column_populates_and_classifies() {
        // A compute-bound FATTN-like op: big HMMA count, small bytes, wall == the pipe demand.
        // ops 3600 / 36 HMMA-per-cyc = 100 cyc pipe demand; wall 100 => Pipe-bound.
        let r = TeleRecord {
            op_index: 0,
            kind: "FATTN_DECODE".into(),
            lane: Lane::Compute,
            gt_start_ns: 0,
            gt_end_ns: 0,
            cycles: 100,
            bytes: 418, // ~1 cyc of DRAM — negligible
            ops: 3600,
            op_wall_ns: 0,
        };
        let rate = op_pipe_rate(&r.kind);
        let ss = measured_steady_state(&r, rate);
        // the Pipe column is present (populated from ops / rate) and equals 3600/36 = 100.
        let pipe = ss
            .per_resource
            .iter()
            .find(|(k, _)| *k == ResourceKind::Pipe(0))
            .unwrap()
            .1;
        assert_eq!(pipe, 100.0);
        assert_eq!(measured_bottleneck(&r, rate, 5.0), Bottleneck::Pipe(0));
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    fn any_rec() -> TeleRecord {
        TeleRecord {
            op_index: 0,
            kind: String::new(),
            lane: Lane::Memory,
            gt_start_ns: kani::any(),
            gt_end_ns: kani::any(),
            cycles: kani::any(),
            bytes: 0,
            ops: 0,
            op_wall_ns: kani::any(),
        }
    }

    // T2 (integer core): span is saturating-nonnegative, and overlap of disjoint spans
    // is exactly 0. Bounded to keep the u64 arithmetic tractable.
    #[kani::proof]
    fn t2_disjoint_overlap_is_zero() {
        let mut a = any_rec();
        let mut b = any_rec();
        kani::assume(a.gt_start_ns <= (1 << 20) && a.gt_end_ns <= (1 << 20));
        kani::assume(b.gt_start_ns <= (1 << 20) && b.gt_end_ns <= (1 << 20));
        // force disjoint: a entirely before b.
        kani::assume(a.gt_end_ns <= b.gt_start_ns);
        assert!(overlap_fraction(&a, &b) == 0.0);
        // span is never "negative": saturating.
        let _ = a.span_ns();
        assert!(a.span_ns() == a.gt_end_ns.saturating_sub(a.gt_start_ns));
    }

    // (b) true-op-wall never falls below block-0's span, for ANY host-attached op_wall_ns
    // (incl. a mis-attached one shorter than the span). This is the anti-optimism guarantee
    // of the roofline correction: wall_ns >= span_ns, so bytes/wall <= bytes/span, so the
    // true-wall achieved rate is never ABOVE the block-0-span rate — the correction can
    // only remove a false T1 reject, never manufacture a false accept. Integer/structural
    // (no FP division; the rate-monotonicity is the accompanying unit test, matching the
    // codebase convention of keeping FP-division facts in tests).
    #[kani::proof]
    fn true_wall_never_below_block0_span() {
        let mut r = any_rec();
        kani::assume(r.gt_start_ns <= (1 << 20) && r.gt_end_ns <= (1 << 20));
        kani::assume(r.op_wall_ns <= (1 << 20));
        assert!(r.wall_ns() >= r.span_ns());
        // absence (op_wall_ns == 0) reduces to the prior block-0-span behaviour.
        r.op_wall_ns = 0;
        assert!(r.wall_ns() == r.span_ns());
    }
}
