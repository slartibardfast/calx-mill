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

/// DRAM roofline as bytes moved per SM-clock cycle: 609 GB/s at 1.455 GHz, GPU-global
/// (a record's `bytes` is the op's whole DRAM traffic). Provenance: TU102 spec bandwidth
/// / SM clock; consistent with the out-of-band `nvidia-smi dmon` mem-controller-77%-busy
/// observation (plan/0143 capture). The numerator that turns measured bytes into a
/// `MemoryBw` cycle demand and bounds T1.
pub const PEAK_DRAM_BYTES_PER_CYCLE: f64 = 609.0e9 / 1.455e9; // ~418.6

/// One per-op telemetry record — the frozen kernel↔calx-mill seam (spec §seam). `bytes`
/// or `ops` == 0 means "column absent" (attached in a later pass; the ingest then emits
/// no `MemoryBw`/`Pipe` demand, matching `mix_demands`' zero convention).
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
}

impl TeleRecord {
    /// The `%globaltimer` span in ns. Saturating, so it is `>= 0` for any inputs (T2).
    pub fn span_ns(&self) -> u64 {
        self.gt_end_ns.saturating_sub(self.gt_start_ns)
    }
    /// Achieved DRAM GB/s = bytes / span-ns (1 byte/ns == 1 GB/s). Zero when the span or
    /// bytes are absent.
    pub fn achieved_gbps(&self) -> f64 {
        let ns = self.span_ns();
        if ns == 0 || self.bytes == 0 {
            return 0.0;
        }
        self.bytes as f64 / ns as f64
    }
    /// T1 roofline sanity: a record implying a DRAM rate above the wall is an instrument
    /// error, not an anchor. `false` ⇒ reject the record (do not build an anchor from it).
    pub fn roofline_ok(&self) -> bool {
        self.achieved_gbps() <= PEAK_DRAM_BYTES_PER_CYCLE * 1.455 // == 609 GB/s
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
        v.push((ResourceKind::MemoryBw, r.bytes as f64 / PEAK_DRAM_BYTES_PER_CYCLE));
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
        r.bytes as f64 / PEAK_DRAM_BYTES_PER_CYCLE
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
/// lanes default to `Compute`. Malformed rows are skipped. Returns the records in file
/// order.
pub fn parse_tele(text: &str) -> Vec<TeleRecord> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("op_index") || line.starts_with('#') {
            continue;
        }
        // schema: op_index op_kind lane gt_start_ns gt_end_ns cycles bytes ops
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 8 {
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
        });
    }
    out
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
    fn parse_roundtrip() {
        let t = "op_index\top_kind\tlane\tgt_start_ns\tgt_end_ns\tcycles\tbytes\tops\n\
                 6\tMMVQ_Q4_0\tmem\t100\t200\t75282\t0\t0\n";
        let v = parse_tele(t);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].op_index, 6);
        assert_eq!(v[0].cycles, 75282);
        assert_eq!(v[0].lane, Lane::Memory);
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
}
