# Lacunae register and improvement set

Reviewed 2026-07-17 against `main` @ `1f648db`. Each item states the gap, the
evidence, the remediation, and the check that closes it. The register follows
the recorded failure pattern (call/0028, 0031, 0032, 0033, 0035 in the host):
every real miss so far came from a projection composed or extrapolated outside
its anchored regime. The gate mode was built to refuse that class; the first
group below closes the gaps in the gate itself.

Items marked **tranche-first** are the highest leverage per line changed.

## Gate integrity

The arbiter's promise is that an unvalidated projection cannot be laundered
into a pass. Four mechanical gaps undercut that promise today.

### PROVISIONAL shares exit code 0 with CERTIFIED (tranche-first)

`gate` exits 0 for both CERTIFIED and PROVISIONAL and 2 for REFUSED
(src/main.rs:509-526). An exit-code-driven manifest consumer treats
PROVISIONAL as a terminal pass, which is exactly what the spec says it is not.

Remediation: three distinct exit codes (0 CERTIFIED, 3 PROVISIONAL, 2
REFUSED), or a `--require certified` flag under which anything below Gate
exits non-zero. Verify by: a parity test asserting the three codes are
pairwise distinct.

### Degenerate defaults gate to PROVISIONAL (tranche-first)

`calx-mill gate` with no flags emits PROVISIONAL exit 0: value 0 reproduces
anchor 0 within tol 0, and fit defaults to in-domain (src/main.rs:492-501).

Remediation: `--value`, `--anchor`, `--tol` become required; a missing flag is
a usage error. Verify by: zero-argument invocation exits non-zero with usage
text.

### Domain fit is self-declared; the anchor registry has no code realization (tranche-first)

`--fit` is an operator-typed enum and nothing computes membership; anchors are
per-invocation floats (src/validity.rs:24-29, src/main.rs:496-501). The spec's
anchor registry (projection-validity A5) exists only as prose. The
no-launder property therefore holds against a dishonest model but not a
mistaken operator; the prefill-addendum laundering caught in the 2026-07-17
adversarial review went through this exact hole.

Remediation: a committed `anchors.csv` registry (id, measured, tol, units,
domain bounds as regime keys: ctx, batch, gpu-count, dtype). `gate --anchor-id
<id> --at ctx=262144,...` computes the fit mechanically; an unknown key or an
out-of-bounds value is OutOfDomain. Keep manual `--fit` only as
`--fit-override`, echoed loudly in the output. Verify by: unit tests for the
membership computation; the recorded 10^5 op-precision case (anchor KL 2.00e-2
vs projection 1e-7) refuses without any operator honesty required.

### Composite claims bypass the composition cap

`validity::compose` (A4: two Gates meet at CrossChecked unless the composite
is re-anchored) is not reachable from the CLI (src/validity.rs:80-87); an
operator gates a composite with one `gate` call and a self-declared fit.

Remediation: `gate --compose <verdict>,<verdict>` (or anchor-id pair) emitting
the composed verdict. Verify by: two CERTIFIED inputs compose to PROVISIONAL
unless a composite anchor is supplied.

### Tolerance is absolute, unitless, and sign-blind

`Anchor::reproduces` takes `|tol|` of a bare float (src/validity.rs:49-51). A
tolerance intended as relative silently becomes absolute; a tok/s anchor and a
KL anchor need different regimes and nothing checks units; a negative tol is
silently accepted.

Remediation: `--tol-rel <pct>` as the alternative form; registry rows carry
units and `gate` echoes them; negative tolerance is an error. Verify by: unit
mismatch between claim and anchor refuses.

## Model dimensions

Gaps where the model would confidently mispredict, in the style of the
recorded lacunae. Ordered by expected impact on the current lever queue.

### Tensor pipe missing from the census projection (tranche-first)

HMMA and IMMA appear in neither OP_TABLE_ROW nor PIPE_OF
(src/nvidia/projection.rs:17-72); they fall to the default pipe "alu" at rate
2.0 ops/clk against a real Turing HMMA rate near 0.5, roughly 4x optimistic on
the flagship pipe of any tensor kernel. LDSM sits in LSU and is costed as
global-memory bytes (projection.rs:75). The miss is flagged in the `defaulted`
list but still costed. Meanwhile the telemetry side carries its own HMMA
constant (src/telemetry.rs:28), so the two halves disagree about what a tensor
op costs, and that constant is the f32-accumulate rate applied to f16-acc ops
too.

Remediation: measured `hmma` table rows (f16-acc and f32-acc separately),
PIPE_OF entries for HMMA/IMMA, LDSM routed to the shared-memory lane, and one
shared rate table cited by both projection and telemetry. Verify by: a census
of the FATTN kernel projects HMMA pipe cycles within the verify tolerance of
the sub-phase profiler anchors already committed in the host's .tele captures.

### No L2 memory class

MemClass is none, dram, or l1 (src/nvidia/projection.rs:91-95); an L2-resident
working set is forced to either the DRAM budget (5.82 B/clk/SM) or the L1 rate
(32 B/clk/SM). The l2bw bench binary exists in the suite (it is even
purity-exempt, src/nvidia/check.rs:19-21) and its measurement feeds nothing.

Remediation: `--mem-class l2` with a measured `mem.l2.bw` row, plumbed like
`l1_budget`. Verify by: the existing l2bw microbench projects within
tolerance; DRAM and L1 parity tests unchanged.

### Memory latency and outstanding loads are unmodeled

The latency dimension (chain_cycles, Little's Law) applies only to register
accumulator chains (src/lib.rs:356-558). DRAM or L2 latency and the in-flight
load limit appear nowhere, so a low-occupancy, latency-bound loader projects
as cheap MemoryBw demand.

Remediation: apply the same chain machinery to loads (L = measured memory
latency, C = in-flight loads) as an additional lane in bound selection. Verify
by: a small pointer-chase microbench, mispredicted today, projects within
tolerance after; Kani monotonicity for the new lane.

### Warp count scales demand purely linearly

`mix_demands` multiplies every demand by the warp count and tests/parity.rs
asserts the linearity; `concurrency()` output never feeds `project()`. There
is no latency-hiding knee, which is the same shape as the occupancy-OK vs
ILP-OK failure (call/0031) at the whole-kernel level.

Remediation: effective warps capped by a hide threshold on the bound lane,
one-way conservative like the register coupling. Verify by: reproduces the
measured warp sweep already in the tu102 ops table; the linearity assertion
becomes a knee assertion.

### The composition Lane enum cannot express shared-memory contention

Lane is Memory or Compute only (src/lib.rs:563-566), yet the contention lesson
that forced `overlapped_contended` was a shared-memory writer against a
shared-memory reader; that case is representable only by dropping to raw
demand vectors (the pv_smem test does exactly this).

Remediation: a LocalStore lane wired through OpComposition and
`overlapped_contended`. Verify by: the three measured dbuf verdicts reproduce
at Phase level, not just at demand-vector level.

### Overlap crediting is all-or-nothing

`overlapped_if` returns fully-overlapped or fully-serial (src/lib.rs:662-668).
The telemetry side measures a real `overlap_fraction` per adjacent pair
(src/telemetry.rs:160-172) and no projection consumes it.

Remediation: fractional crediting, overlapped = serial - credit * (serial -
max(lanes)), credit taken from a measured overlap anchor. Verify by: credit 0
equals serial and credit 1 equals the current overlap (a bracket proof in the
style of contention_brackets_overlap).

### Spill traffic costs nothing

LDL/STL are flagged and excluded from demand (src/nvidia/projection.rs:208-209)
while ptxas spill counts are parsed and then unused (src/nvidia/ptxas.rs).
Register-pressure cases are the tool's core clientele; a spilling kernel
projects as if the spills were free.

Remediation: cost LDL/STL as L1-class bytes and fold the ptxas spill counts
into `project` when both inputs are supplied. Verify by: a forced-spill
microbench moves from under-projection to within tolerance.

### Q4 effective bytes overstate streams by ~1.78x

`dtype_bytes` ceils 4.5-bpw block quants to 1 byte per element
(src/nvidia/mod.rs:30-39). The comment declares the approximation, but the
error exceeds every gate tolerance in use (verify 20 percent, census-match 10
points), so a q4 memory-bound anchor either refuses a correct model or passes
a wrong one.

Remediation: exact rational bytes per block (Q4_0 = 18 bytes per 32 elements),
stream bytes computed as elems * num / den with saturation. Verify by: a q4
GEMV census matches the measured bytes column from the .tele seam.

## Instrument integrity

### Clock is assumed, never measured (tranche-first)

1.455 GHz is hardcoded in three places (src/telemetry.rs:22, :89,
src/main.rs:548) and stamped into every mk-table row. DVFS boost or thermal
throttle silently corrupts the roofline check, the ms conversions, and the
DRAM bytes-per-cycle constant at once. The fix is nearly free: every .tele
record already carries both globaltimer ns and SM cycles, so the realized
clock is derivable per record.

Remediation: derive clock per record from cycles / (gt_end - gt_start), use it
for the conversions, and reject (T-law style) any record whose implied clock
deviates from the stamped one beyond a small tolerance. Verify by: a fixture
with a shifted clock is flagged; existing fixtures unchanged.

### Telemetry classification biases

Ties in `measured_bottleneck` resolve to Memory, including the all-zero
degenerate record (src/telemetry.rs:115-136); the CLI hardcodes tol 0.0
(src/main.rs:567) which biases the displayed binding toward latency; the
summary sums per-op cycles as if serial (src/main.rs:542-549); overlap
detection is adjacent-pairs only (src/main.rs:590-599).

Remediation: degenerate records classify as rejected rather than Memory; the
tolerance becomes a flag with a sane default; the summary reports wall span
alongside the serial sum; overlap detection sweeps all interval pairs. Verify
by: unit tests per case; the GEMV_F16 block-0 over-count fixture stays
rejected.

### Parsers degrade silently (tranche-first)

Malformed .tele rows are skipped without a count, so a truncated file passes
(src/telemetry.rs:216-229). CSV bytes are pushed `as char`, mangling non-ASCII
input to mojibake (src/nvidia/csvio.rs:59, :79). The pattern module treats
unsupported regex constructs as literals, so a `[0-9]`-style kernel filter
matches nothing with no warning (src/nvidia/pattern.rs). In the SASS scanner,
uniform-datapath `@UP0`-predicated instructions are dropped to zero cost while
`@P0`-predicated ones count at full cost, two opposite-signed silent errors
(src/nvidia/sass.rs:53-77).

Remediation: parse-error counters surfaced in every output and a strict mode
that exits non-zero on any malformed input; the pattern parser rejects
metacharacters it does not implement; the census reports the uniform-datapath
drop count. Verify by: one fixture per failure mode.

### ncu ingestion has never seen real output

Self-declared: the fixture is hand-built from documented format, no raw export
was ever captured (src/nvidia/ncu.rs:5-12). The megakernel itself cannot be
profiled, but any plain kernel can supply the fixture.

Remediation: capture one real `ncu --csv` export and commit it as the parity
fixture. Verify by: the parity test runs over the real fixture.

### verify-projection exempts the model's weakest axis

The latency-bound demo is excluded from the absolute gate by design
(src/nvidia/verify.rs:21) and the gate checks the number but not the bound
column, so a mis-attributed bottleneck can hide inside the 20 percent band.
The results path also bakes in a host name (verify.rs:76).

Remediation: fold the telemetry column-gate (right number and right column)
into verify; parameterize the tree root. Verify by: a fixture with the right
cycles and the wrong column fails verify.

## Host-side alignments

These live in the host repo (yarn-agentic), not this tree; recorded here so
the set is complete.

- The recorded pin trails the pushed main tip (3eb55ea vs 1f648db); advancing
  it needs a rebuild and artifact re-hash under the recorded recipe.
- The plan/0136 precision-dimension box is still open while call/0022 records
  it landed. Related fact from this review: the `Precision` type is reachable
  from no CLI path or adapter (tests only), so either wire it (a `--dtype`
  path into `project`) and close the box on that evidence, or record the box
  honestly as scaffolding.
- Open plan/0144 obligations this set would discharge: populate the anchor
  registry; wire `calx-mill gate` into the gate manifest as a listed arbiter;
  retire the pvround/qkround numpy scripts into anchored tests.
- Non-TU102 latency constants in the overfit gate remain representative
  values, owed pins from their cited microbenchmarks when a consumer needs
  them.
