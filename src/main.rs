//! calx-mill CLI: the NVIDIA adapter's projection and parse commands. Output
//! parity with the reference Python tools (project.py, sass_census.py,
//! check_sass.py, mk_table.py, verify_projection.py) is held by the
//! differential tests; this binary is the orchestration surface.

use calx_mill::nvidia::check::{census_match, check, GateOpts};
use calx_mill::nvidia::mktable::mk_table;
use calx_mill::nvidia::ncu::{achieved_occupancy, parse_ncu_csv};
use calx_mill::nvidia::pattern::Pattern;
use calx_mill::nvidia::projection::{project, report, Census, MemClass};
use calx_mill::nvidia::ptxas::{block, parse_ptxas_v, tu102_sm, work_unit, TU102_SMS};
use calx_mill::nvidia::sass::{census_csv, census_per_kernel, sass_text};
use calx_mill::nvidia::table::Rates;
use calx_mill::nvidia::verify::verify_projection;
use calx_mill::{blocks_per_instance, concurrency, cooperative_fits, occupancy_pct};
use calx_mill::{achievable_chains, Bottleneck, Lane, OpComposition, OpTemplate, Phase};
use calx_mill::validity::{
    authority, compose as compose_authority, computed_fit, exit_code, parse_registry, verdict,
    Anchor, AnchorRow, Authority, DomainFit, Verdict, USAGE_EXIT,
};
use calx_mill::telemetry::{
    measured_bottleneck, op_pipe_rate, overlap_fraction, parse_tele_counted, TeleRecord,
    CLOCK_TOL_FRAC,
};
use std::path::Path;
use std::process::ExitCode;

const USAGE: &str = "\
usage: calx-mill <command> [args]

commands:
  project <census.csv> --table <ops.csv> [--kernel PAT] [--warps N]
          [--mem-class none|dram|l1] [--dram-budget B]
          [--regs-per-thread N --smem-block B --block-threads N]
      PPM/ADD projection of an op-mix census against a measured op table
      (project.py). The optional resource triple adds the core's
      concurrency/occupancy line.
  census <bin|.sass> [--kernel PAT] [--full] [-o FILE]
      per-kernel SASS op histogram (sass_census.py)
  check-sass <bin|.sass> [--min-primary N] [--l0-bytes N] [--staging-budget N]
  check-sass --census-match BIN:PAT BIN:PAT [--tolerance-pp F]
      the timed-loop purity gate (check_sass.py)
  mk-table <results-dir> --priors FILE [--na FILE] -o FILE
      aggregate measurement CSVs into the ops table (mk_table.py)
  verify-projection <root>
      the absolute projection gate over a tu102-shaped tree
      (verify_projection.py)
  ptxas <ptxas-v-output> [--block-threads N] [--grid-blocks N]
        [--instances N] [--block-smem B]
      parse `nvcc -Xptxas -v` resource usage; with a block size, fold each
      kernel through the core's occupancy; with a grid, check cooperative
      co-residency (blocks/instance x instances vs the grid, default 72
      SMs; --block-smem overrides static smem with the dynamic launch value)
  ncu <export.csv>
      parse an `ncu --csv` metric export; achieved occupancy per launch
  telemetry <file.tele> [--strict]
      summarize the megakernel's on-device records: per-op wall, heavy hitters,
      realized overlap; counts malformed rows, roofline (T1) and clock-implausible
      rejects. --strict exits non-zero when any row was skipped.
  latency --chains C --depth D --cyc-per-op R --op-latency L
          [--reg-budget B --base-regs BR --regs-per-chain RC --spend S]
      the dependency-latency projection MII = max(C*D*R, D*L) (call/0028):
      cycles, bottleneck, utilization (Little's Law C*R/L), and the L/R hide
      threshold. e.g. the FATTN QK chain: --chains 4 --depth 64 --cyc-per-op 2
      --op-latency 14 -> latency-bound, 57% (7 chains to hide).
      With --reg-budget, also derives the chains the register file sustains
      ((B-BR)/RC) and, with --spend, whether an overlap lever spending S
      registers keeps the recurrence hidden or re-exposes it (call/0031).

  compose --mem-cycles M --compute-cycles C
          [--chains .. --depth .. --cyc-per-op .. --op-latency ..
           --reg-budget B --base-regs BR --regs-per-chain RC --spend S]
      whole-op phase composition (call/0033): serial = M+C (today's serial
      kernel) vs overlapped = max(M,C) (the double-buffer floor). With
      --reg-budget, gates the overlap on the compute lane keeping its ILP after
      the --spend registers: a smem double-buffer (spend 0) earns the floor, a
      register-prefetch (spend > headroom) falls back to serial. e.g. FATTN deep:
      --mem-cycles 1413 --compute-cycles 1945 -> serial 3358, overlapped 1945.

  gate --value V --anchor M (--tol T | --tol-rel PCT) --fit-override FIT [--units U]
  gate --value V --registry FILE --anchor-id ID --at k=v,... [--fit-override FIT]
  gate --compose VERDICT,VERDICT [composite anchor flags as above]
      projection-validity ARBITER (plan/0144/spec/projection-validity.md; contract
      call/0036): judge a claim against its measured anchor + domain and emit
      CERTIFIED (exit 0) | PROVISIONAL (exit 3 -- build, bench confirms, NOT a
      terminal pass) | REFUSED (exit 2, the bench gates); operator error exits 4.
      `anchored` = the model's value AT the anchor (--at-anchor, default V)
      reproduces M±T. In registry mode the domain FIT is COMPUTED from the row
      (an unknown or missing axis is out-of-domain); --fit-override substitutes
      it and tags the verdict `[fit-override]`. --compose runs the A4 cap: the
      composite keeps Gate only if it reproduces its OWN anchor. Makes an
      unvalidated projection un-launderable into a pass.

kernel PAT is the subset the tools use: literals, '.', '\\d', with '*'/'+'.
Binaries are disassembled via cuobjdump (override with CUOBJDUMP); .sass
files are read as cached disassembly.";

struct Args {
    positional: Vec<String>,
    flags: Vec<(String, Option<String>)>,
}

impl Args {
    fn parse(raw: &[String], value_flags: &[&str]) -> Result<Args, String> {
        let mut positional = Vec::new();
        let mut flags = Vec::new();
        let mut i = 0;
        while i < raw.len() {
            let a = &raw[i];
            if let Some(name) = a.strip_prefix("--") {
                if value_flags.contains(&name) {
                    i += 1;
                    let v = raw.get(i).ok_or_else(|| format!("--{} needs a value", name))?;
                    flags.push((name.to_string(), Some(v.clone())));
                } else {
                    flags.push((name.to_string(), None));
                }
            } else if a == "-o" {
                i += 1;
                let v = raw.get(i).ok_or("-o needs a value")?;
                flags.push(("o".to_string(), Some(v.clone())));
            } else {
                positional.push(a.clone());
            }
            i += 1;
        }
        Ok(Args { positional, flags })
    }

    fn value(&self, name: &str) -> Option<&str> {
        self.flags
            .iter()
            .find(|(k, _)| k == name)
            .and_then(|(_, v)| v.as_deref())
    }

    fn has(&self, name: &str) -> bool {
        self.flags.iter().any(|(k, _)| k == name)
    }

    fn number<T: std::str::FromStr>(&self, name: &str, default: T) -> Result<T, String> {
        match self.value(name) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| format!("--{}: bad value {:?}", name, v)),
        }
    }
}

const GATE_USAGE: &str = "\
usage: calx-mill gate --value V --anchor M (--tol T | --tol-rel PCT) --fit-override FIT
       calx-mill gate --value V --registry FILE --anchor-id ID --at k=v,...
       calx-mill gate --compose VERDICT,VERDICT [composite anchor flags as above]
FIT is at-anchor|in-domain|out-of-domain. Registry mode COMPUTES the fit from the
row's domain (--fit-override substitutes it and tags the verdict [fit-override]).
exits: 0 CERTIFIED, 3 PROVISIONAL, 2 REFUSED, 4 usage (never adjudicated).";

/// A resolved anchor source for `gate`: from the registry (fit computed) or the
/// manual flags (fit operator-declared via --fit-override). `Err` is a usage message.
struct GateAnchor {
    anchor: Anchor,
    fit: DomainFit,
    computed: Option<DomainFit>, // Some(..) in registry mode
    overridden: bool,
    units: String,
    id: String, // registry row id; empty in manual mode
}

fn parse_fit(s: &str) -> Option<DomainFit> {
    match s {
        "at-anchor" => Some(DomainFit::AtAnchor),
        "in-domain" => Some(DomainFit::InDomain),
        "out-of-domain" | "out" => Some(DomainFit::OutOfDomain),
        _ => None,
    }
}

fn fit_name(f: DomainFit) -> &'static str {
    match f {
        DomainFit::AtAnchor => "at-anchor",
        DomainFit::InDomain => "in-domain",
        DomainFit::OutOfDomain => "out-of-domain",
    }
}

fn parse_query(at: &str) -> Result<Vec<(String, String)>, String> {
    let mut query = Vec::new();
    for part in at.split(',').filter(|p| !p.is_empty()) {
        let Some((k, v)) = part.split_once('=') else {
            return Err(format!("--at: {:?} is not key=value", part));
        };
        query.push((k.trim().to_string(), v.trim().to_string()));
    }
    if query.is_empty() {
        return Err("--at needs at least one key=value".into());
    }
    Ok(query)
}

/// Resolve `gate`'s anchor source. `Ok(None)` = no anchor flags at all (legal only
/// under --compose); `Err` = an operator/usage error (exit 4 at the caller).
fn resolve_gate_anchor(args: &Args) -> Result<Option<GateAnchor>, String> {
    let registry_mode =
        args.has("registry") || args.has("anchor-id") || args.has("at");
    if registry_mode {
        for conflicting in ["anchor", "tol", "tol-rel"] {
            if args.has(conflicting) {
                return Err(format!(
                    "--{} conflicts with --registry (one source of truth)",
                    conflicting
                ));
            }
        }
        let (Some(file), Some(id), Some(at)) =
            (args.value("registry"), args.value("anchor-id"), args.value("at"))
        else {
            return Err("registry mode needs --registry, --anchor-id, and --at".into());
        };
        let text = std::fs::read_to_string(file)
            .map_err(|e| format!("--registry {}: {}", file, e))?;
        let rows = parse_registry(&text)?;
        let row: AnchorRow = rows
            .into_iter()
            .find(|r| r.id == id)
            .ok_or_else(|| format!("--anchor-id {:?} not in {}", id, file))?;
        let query = parse_query(at)?;
        let computed = computed_fit(&row, &query);
        let (fit, overridden) = match args.value("fit-override") {
            Some(s) => {
                let f = parse_fit(s).ok_or_else(|| {
                    format!("--fit-override: {:?} (at-anchor|in-domain|out-of-domain)", s)
                })?;
                (f, true)
            }
            None => (computed, false),
        };
        return Ok(Some(GateAnchor {
            anchor: row.anchor,
            fit,
            computed: Some(computed),
            overridden,
            units: row.units,
            id: row.id,
        }));
    }
    let any_manual = args.has("anchor") || args.has("tol") || args.has("tol-rel")
        || args.has("fit-override");
    if !any_manual {
        return Ok(None);
    }
    let Some(measured) = args.value("anchor") else {
        return Err("manual mode needs --anchor".into());
    };
    let measured: f64 = measured
        .parse()
        .map_err(|_| format!("--anchor: bad value {:?}", measured))?;
    let tol = match (args.value("tol"), args.value("tol-rel")) {
        (Some(_), Some(_)) => return Err("--tol and --tol-rel are exclusive".into()),
        (None, None) => return Err("manual mode needs --tol or --tol-rel".into()),
        (Some(t), None) => {
            let t: f64 = t.parse().map_err(|_| format!("--tol: bad value {:?}", t))?;
            if t < 0.0 {
                return Err("--tol must be non-negative".into());
            }
            t
        }
        (None, Some(p)) => {
            let p: f64 = p.parse().map_err(|_| format!("--tol-rel: bad value {:?}", p))?;
            if p < 0.0 {
                return Err("--tol-rel must be non-negative".into());
            }
            p / 100.0 * measured.abs()
        }
    };
    let Some(fit) = args.value("fit-override") else {
        return Err("manual mode needs --fit-override (there is no computed fit)".into());
    };
    let fit = parse_fit(fit).ok_or_else(|| {
        format!("--fit-override: {:?} (at-anchor|in-domain|out-of-domain)", fit)
    })?;
    Ok(Some(GateAnchor {
        anchor: Anchor { measured, tol },
        fit,
        computed: None,
        overridden: true, // manual fit is by definition operator-declared
        units: args.value("units").unwrap_or("").to_string(),
        id: String::new(),
    }))
}

fn run() -> Result<i32, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = raw.first().map(|s| s.as_str()) else {
        eprintln!("{}", USAGE);
        return Ok(2);
    };
    let rest = &raw[1..];
    match command {
        "project" => {
            let args = Args::parse(
                rest,
                &[
                    "table",
                    "kernel",
                    "warps",
                    "mem-class",
                    "dram-budget",
                    "regs-per-thread",
                    "smem-block",
                    "block-threads",
                ],
            )?;
            let [census_path] = args.positional.as_slice() else {
                return Err("project: exactly one census CSV expected".into());
            };
            let table_path = args.value("table").ok_or("project: --table is required")?;
            let kernel = Pattern::try_new(args.value("kernel").unwrap_or(".*"))?;
            let warps: f64 = args.number("warps", 8.0)?;
            let mem_class = MemClass::parse(args.value("mem-class").unwrap_or("none"))
                .ok_or("--mem-class: none, dram, or l1")?;
            let dram_budget: f64 = args.number("dram-budget", 5.82)?;
            let rates = Rates::parse(&read(table_path)?)?;
            let census = Census::from_census_csv(&read(census_path)?, &kernel);
            if census.is_empty() {
                eprintln!("no ops matched");
                return Ok(1);
            }
            let r = project(&census, warps, mem_class, &rates, dram_budget);
            print!("{}", report(&r));
            if args.has("regs-per-thread") {
                let usage = calx_mill::nvidia::ptxas::KernelUsage {
                    registers: args.number("regs-per-thread", 0u32)?,
                    smem_bytes: args.number("smem-block", 0u32)?,
                    ..Default::default()
                };
                let block_threads: u32 = args.number("block-threads", 32)?;
                let s = tu102_sm();
                let w = work_unit(&usage, block_threads);
                let c = concurrency(&s, &w);
                println!(
                    "occupancy: {}/{} warps ({}%)",
                    c,
                    s.concurrency_ceiling,
                    occupancy_pct(&s, &w)
                );
            }
            Ok(0)
        }
        "census" => {
            let args = Args::parse(rest, &["kernel"])?;
            let [path] = args.positional.as_slice() else {
                return Err("census: exactly one binary or .sass expected".into());
            };
            let kernel = Pattern::try_new(args.value("kernel").unwrap_or(".*"))?;
            let sass = sass_text(Path::new(path)).map_err(|e| e.to_string())?;
            // stderr, never stdout: the census CSV is golden-pinned byte for byte.
            let drops = calx_mill::nvidia::sass::uniform_predicated_drops(&sass);
            if drops > 0 {
                eprintln!(
                    "census: {} uniform-datapath predicated instructions dropped \
                     (@UP*: refused by the scanner, costed at zero)",
                    drops
                );
            }
            let counts = census_per_kernel(&sass, &kernel, args.has("full"));
            if counts.is_empty() {
                eprintln!("no kernels matched '{}'", args.value("kernel").unwrap_or(".*"));
                return Ok(1);
            }
            let csv = census_csv(&counts);
            match args.value("o") {
                Some(out) if out != "-" => {
                    std::fs::write(out, csv).map_err(|e| e.to_string())?
                }
                _ => print!("{}", csv),
            }
            Ok(0)
        }
        "check-sass" => {
            let args = Args::parse(
                rest,
                &["min-primary", "l0-bytes", "staging-budget", "tolerance-pp"],
            )?;
            if args.has("census-match") {
                let [a, b] = args.positional.as_slice() else {
                    return Err("check-sass --census-match: two BIN:PAT specs expected".into());
                };
                let tol: f64 = args.number("tolerance-pp", 10.0)?;
                let (out, exit) = census_match(a, b, tol);
                print!("{}", out);
                return Ok(exit);
            }
            let [path] = args.positional.as_slice() else {
                return Err("check-sass: exactly one binary or .sass expected".into());
            };
            let opts = GateOpts {
                min_primary: args.number("min-primary", 64)?,
                l0_bytes: args.number("l0-bytes", 8192)?,
                staging_budget: args.number("staging-budget", 6)?,
            };
            let p = Path::new(path);
            let basename = p
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or("check-sass: bad path")?;
            // the exemption list keys on the .bin name; a cached .sass keeps it
            let basename = basename.strip_suffix(".sass").map_or_else(
                || basename.to_string(),
                |stem| format!("{}.bin", stem),
            );
            let sass = sass_text(p).map_err(|e| e.to_string())?;
            let (out, exit) = check(&basename, &sass, opts);
            print!("{}", out);
            Ok(exit)
        }
        "mk-table" => {
            let args = Args::parse(rest, &["priors", "na"])?;
            let [results_dir] = args.positional.as_slice() else {
                return Err("mk-table: exactly one results directory expected".into());
            };
            let priors = read(args.value("priors").ok_or("mk-table: --priors is required")?)?;
            let na = args.value("na").map(read).transpose()?;
            let out_path = args.value("o").ok_or("mk-table: -o is required")?;
            let mut results = Vec::new();
            for entry in std::fs::read_dir(results_dir).map_err(|e| e.to_string())? {
                let entry = entry.map_err(|e| e.to_string())?;
                let name = entry
                    .file_name()
                    .into_string()
                    .map_err(|n| format!("non-utf8 file name {:?}", n))?;
                if !name.ends_with(".csv") {
                    continue;
                }
                results.push((name, read(entry.path().to_str().unwrap_or_default())?));
            }
            let (csv, n_rows) = mk_table(&results, &priors, na.as_deref());
            std::fs::write(out_path, csv).map_err(|e| e.to_string())?;
            println!("{}: {} rows", out_path, n_rows);
            Ok(0)
        }
        "verify-projection" => {
            let args = Args::parse(rest, &[])?;
            let [root] = args.positional.as_slice() else {
                return Err("verify-projection: exactly one root directory expected".into());
            };
            let (out, exit) = verify_projection(Path::new(root)).map_err(|e| e.to_string())?;
            print!("{}", out);
            Ok(exit)
        }
        "ptxas" => {
            let args = Args::parse(
                rest,
                &["block-threads", "grid-blocks", "instances", "block-smem"],
            )?;
            let [path] = args.positional.as_slice() else {
                return Err("ptxas: exactly one ptxas -v output file expected".into());
            };
            let usage = parse_ptxas_v(&read(path)?);
            if usage.is_empty() {
                eprintln!("no kernels found");
                return Ok(1);
            }
            let block_threads: Option<u32> = match args.value("block-threads") {
                Some(v) => Some(v.parse().map_err(|_| "--block-threads: bad value")?),
                None => None,
            };
            let grid_blocks: Option<u32> = match args.value("grid-blocks") {
                Some(v) => Some(v.parse().map_err(|_| "--grid-blocks: bad value")?),
                None => None,
            };
            let instances: u32 = args.number("instances", TU102_SMS)?;
            let block_smem: Option<u32> = match args.value("block-smem") {
                Some(v) => Some(v.parse().map_err(|_| "--block-smem: bad value")?),
                None => None,
            };
            if grid_blocks.is_some() && block_threads.is_none() {
                return Err("ptxas: --grid-blocks needs --block-threads".into());
            }
            let s = tu102_sm();
            let mut all_fit = true;
            for k in &usage {
                print!(
                    "{}: {} regs, {} B smem, {} barriers, spills {}/{}",
                    k.name, k.registers, k.smem_bytes, k.barriers, k.spill_stores, k.spill_loads
                );
                if let Some(bt) = block_threads {
                    let w = work_unit(k, bt);
                    print!(
                        ", occupancy {}/{} warps ({}%) at {} threads/block",
                        concurrency(&s, &w),
                        s.concurrency_ceiling,
                        occupancy_pct(&s, &w),
                        bt
                    );
                    if let Some(g) = grid_blocks {
                        let b = block(k, bt, block_smem);
                        let per = blocks_per_instance(&s, &b);
                        let fits = cooperative_fits(&s, &b, g, instances);
                        all_fit &= fits;
                        print!(
                            ", {} blocks/instance, cooperative grid {} on {} instances: {}",
                            per,
                            g,
                            instances,
                            if fits { "fits" } else { "DEADLOCK" }
                        );
                    }
                }
                println!();
            }
            Ok(if all_fit { 0 } else { 1 })
        }
        "ncu" => {
            let args = Args::parse(rest, &[])?;
            let [path] = args.positional.as_slice() else {
                return Err("ncu: exactly one ncu --csv export expected".into());
            };
            let rows = parse_ncu_csv(&read(path)?)?;
            for r in &rows {
                println!("[{}] {}: {} = {} {}", r.launch, r.kernel, r.metric, r.value, r.unit);
            }
            for (launch, kernel, pct) in achieved_occupancy(&rows) {
                println!("achieved occupancy [{}] {}: {}%", launch, kernel, pct);
            }
            Ok(0)
        }
        "latency" => {
            // MII = max(ResMII, RecMII): the dependency-latency projection (call/0028).
            let args = Args::parse(
                rest,
                &[
                    "chains",
                    "depth",
                    "cyc-per-op",
                    "op-latency",
                    "reg-budget",
                    "base-regs",
                    "regs-per-chain",
                    "spend",
                ],
            )?;
            let t = OpTemplate {
                chains: args.number("chains", 1u32)?,
                depth: args.number("depth", 1u32)?,
                cyc_per_op: args.number("cyc-per-op", 1u32)?,
                op_latency: args.number("op-latency", 1u32)?,
            };
            let res = t.chains.saturating_mul(t.depth).saturating_mul(t.cyc_per_op);
            let rec = t.depth.saturating_mul(t.op_latency);
            let p = t.cycles();
            let bound = match p.bottleneck {
                Bottleneck::Latency => "latency-bound (RecMII, under-parallelized)",
                Bottleneck::Pipe(_) => "throughput-bound (ResMII)",
                Bottleneck::IssueCap => "issue-bound",
                Bottleneck::Memory => "memory-bound",
            };
            println!("MII = max(ResMII {}, RecMII {}) = {} cycles [{}]", res, rec, p.cycles, bound);
            println!(
                "utilization {}%; hidden at chains >= L/R = {} (have {})",
                t.utilization_pct(),
                t.chains_to_hide(),
                t.chains,
            );
            // Register->chains coupling (call/0031): with a budget, report the chains
            // the file sustains and whether an overlap `spend` keeps the recurrence hidden.
            let budget = args.number("reg-budget", 0u32)?;
            if budget > 0 {
                let base = args.number("base-regs", 0u32)?;
                let per = args.number("regs-per-chain", 1u32)?;
                let spend = args.number("spend", 0u32)?;
                let ach = achievable_chains(budget, base, per);
                println!(
                    "registers: budget {} - base {} @ {}/chain -> sustains {} chains (requested {})",
                    budget, base, per, ach, t.chains,
                );
                println!(
                    "under registers: utilization {}%",
                    t.utilization_under_registers(budget, base, per),
                );
                if spend > 0 {
                    let keeps = t.overlap_keeps_hidden(budget, base, per, spend);
                    let after = achievable_chains(budget, base.saturating_add(spend), per);
                    println!(
                        "overlap spend {}: sustains {} chains -> {} (hide threshold {})",
                        spend,
                        after,
                        if keeps { "STAYS HIDDEN" } else { "RE-EXPOSES LATENCY" },
                        t.chains_to_hide(),
                    );
                }
            }
            Ok(0)
        }
        "compose" => {
            // Whole-op phase composition (call/0033): serial sum vs the two-lane
            // double-buffer overlap floor, gated on the compute lane keeping its ILP.
            let args = Args::parse(
                rest,
                &[
                    "mem-cycles",
                    "compute-cycles",
                    "chains",
                    "depth",
                    "cyc-per-op",
                    "op-latency",
                    "reg-budget",
                    "base-regs",
                    "regs-per-chain",
                    "spend",
                ],
            )?;
            let mem = args.number("mem-cycles", 0u32)?;
            let cmp = args.number("compute-cycles", 0u32)?;
            let phases = [
                Phase { cycles: mem, lane: Lane::Memory },
                Phase { cycles: cmp, lane: Lane::Compute },
            ];
            let comp = OpComposition { phases: &phases };
            println!(
                "serial = mem {} + compute {} = {} cycles (no overlap; today's kernel)",
                mem, cmp, comp.serial_cycles(),
            );
            println!(
                "overlapped = max(mem {}, compute {}) = {} cycles (perfect double-buffer floor)",
                mem, cmp, comp.overlapped_cycles(),
            );
            // The compute lane's register gate decides whether the overlap is real.
            let budget = args.number("reg-budget", 0u32)?;
            if budget > 0 {
                let t = OpTemplate {
                    chains: args.number("chains", 1u32)?,
                    depth: args.number("depth", 1u32)?,
                    cyc_per_op: args.number("cyc-per-op", 1u32)?,
                    op_latency: args.number("op-latency", 1u32)?,
                };
                let base = args.number("base-regs", 0u32)?;
                let per = args.number("regs-per-chain", 1u32)?;
                let spend = args.number("spend", 0u32)?;
                let creditable = t.overlap_keeps_hidden(budget, base, per, spend);
                println!(
                    "register gate: overlap spend {} -> compute lane {} -> projection {} cycles",
                    spend,
                    if creditable { "STAYS HIDDEN (creditable)" } else { "RE-EXPOSES (not creditable)" },
                    comp.overlapped_if(creditable),
                );
            }
            Ok(0)
        }
        "gate" => {
            // projection-validity ARBITER (plan/0144/spec/projection-validity.md; process
            // contract host call/0036): judge one claim against its anchor + domain and
            // emit CERTIFIED (0) | PROVISIONAL (3) | REFUSED (2); an operator error is
            // USAGE (4) — the arbiter never adjudicated. The tool adjudicates; the agent
            // attaches the verdict, never overrides it.
            let args = Args::parse(
                rest,
                &["value", "anchor", "tol", "tol-rel", "at-anchor", "units",
                  "registry", "anchor-id", "at", "fit-override", "compose"],
            )?;
            let usage = |msg: String| -> Result<i32, String> {
                eprintln!("calx-mill gate: {}", msg);
                eprintln!("{}", GATE_USAGE);
                Ok(USAGE_EXIT as i32)
            };
            // Every gate flag takes a value, so a valueless (unknown) flag or a stray
            // positional is an operator error — closing the silent-boolean fallthrough
            // that let `--anchor-id` be ignored and defaults adjudicate.
            if let Some((k, _)) = args.flags.iter().find(|(_, v)| v.is_none()) {
                return usage(format!("unknown flag --{}", k));
            }
            if let Some(p) = args.positional.first() {
                return usage(format!("unexpected argument {:?}", p));
            }
            let ga = match resolve_gate_anchor(&args) {
                Err(msg) => return usage(msg),
                Ok(ga) => ga,
            };
            // --compose VERDICT,VERDICT: A4 composition. The composite keeps the meet
            // (may stay Gate) only when the composite claim itself earns Gate from its
            // own anchor in this same invocation; else capped at CrossChecked.
            if let Some(spec) = args.value("compose") {
                let parse_verdict = |s: &str| match s.to_ascii_lowercase().as_str() {
                    "certified" => Some(Authority::Gate),
                    "provisional" => Some(Authority::CrossChecked),
                    "refused" => Some(Authority::Advisory),
                    _ => None,
                };
                let Some((sa, sb)) = spec.split_once(',') else {
                    return usage("--compose needs VERDICT,VERDICT".into());
                };
                let (Some(ta), Some(tb)) = (parse_verdict(sa.trim()), parse_verdict(sb.trim()))
                else {
                    return usage(format!(
                        "--compose: verdicts are certified|provisional|refused, got {:?}",
                        spec
                    ));
                };
                let composite_anchored = match &ga {
                    None => false,
                    Some(ga) => {
                        let Some(value) = args.value("value") else {
                            return usage("a composite anchor needs --value".into());
                        };
                        let Ok(value) = value.parse::<f64>() else {
                            return usage(format!("--value: bad value {:?}", value));
                        };
                        let at_anchor = match args.value("at-anchor") {
                            None => value,
                            Some(s) => match s.parse::<f64>() {
                                Ok(x) => x,
                                Err(_) => {
                                    return usage(format!("--at-anchor: bad value {:?}", s))
                                }
                            },
                        };
                        authority(ga.anchor.reproduces(at_anchor), ga.fit) == Authority::Gate
                    }
                };
                let v = verdict(compose_authority(ta, tb, composite_anchored));
                let name = match v {
                    Verdict::Certified => "CERTIFIED (re-anchored composite)",
                    Verdict::Provisional => {
                        "PROVISIONAL (A4: composition is capped below Gate -- the \
                         interaction is unvalidated; the bench confirms)"
                    }
                    Verdict::Refused => "REFUSED (the meet includes an Advisory input)",
                };
                println!("COMPOSED: {}", name);
                return Ok(exit_code(v) as i32);
            }
            // Single-claim adjudication.
            let Some(ga) = ga else {
                return usage(
                    "need an anchor source: --anchor (--tol|--tol-rel) --fit-override FIT, \
                     or --registry FILE --anchor-id ID --at k=v,..."
                        .into(),
                );
            };
            let Some(value) = args.value("value") else {
                return usage("need --value (the claim being judged)".into());
            };
            let Ok(value) = value.parse::<f64>() else {
                return usage(format!("--value: bad value {:?}", value));
            };
            let at_anchor = match args.value("at-anchor") {
                None => value, // model's value AT the anchor defaults to the claim
                Some(s) => match s.parse::<f64>() {
                    Ok(x) => x,
                    Err(_) => return usage(format!("--at-anchor: bad value {:?}", s)),
                },
            };
            // A claim speaking different units than its anchor is a failed adjudication
            // (REFUSED), not a usage slip: the number cannot mean what the anchor means.
            if let (Some(u), false) = (args.value("units"), ga.units.is_empty()) {
                if u != ga.units {
                    println!(
                        "REFUSED: claim units {:?} do not match anchor units {:?} \
                         -> the bench/oracle gates, not the model.",
                        u, ga.units
                    );
                    return Ok(exit_code(Verdict::Refused) as i32);
                }
            }
            if let Some(computed) = ga.computed {
                if ga.overridden {
                    println!(
                        "fit: {} (computed) -> {} (OPERATOR OVERRIDE)",
                        fit_name(computed),
                        fit_name(ga.fit)
                    );
                } else {
                    println!(
                        "fit: {} (computed from registry row {:?})",
                        fit_name(computed),
                        ga.id
                    );
                }
            }
            let tag = if ga.overridden { " [fit-override]" } else { "" };
            let units = if ga.units.is_empty() {
                String::new()
            } else {
                format!(" {}", ga.units)
            };
            let anchored = ga.anchor.reproduces(at_anchor);
            let v = verdict(authority(anchored, ga.fit));
            match v {
                Verdict::Certified => println!(
                    "CERTIFIED value={} (Gate: reproduces anchor {}±{}{} at this query){}",
                    value, ga.anchor.measured, ga.anchor.tol, units, tag
                ),
                Verdict::Provisional => println!(
                    "PROVISIONAL value={} (CrossChecked: anchored model, extrapolated \
                     -- decide/build on it, the bench confirms; NOT a terminal gate){}",
                    value, tag
                ),
                Verdict::Refused => {
                    let why = if !anchored {
                        format!(
                            "does not reproduce anchor {}±{}{} (got {})",
                            ga.anchor.measured, ga.anchor.tol, units, at_anchor
                        )
                    } else {
                        "out of the anchored domain".to_string()
                    };
                    println!(
                        "REFUSED: {} -> the bench/oracle gates, not the model.{}",
                        why, tag
                    );
                }
            }
            Ok(exit_code(v) as i32)
        }
        "telemetry" => {
            // plan/0144 primitive-telemetry ingest: read the megakernel's on-device
            // .tele record and summarize the MEASURED anchors — per-op wall, the
            // heavy hitters, and the realized overlap of adjacent lane-crossing pairs
            // (the measurement no external profiler yields). bytes/ops==0 records show
            // no MemoryBw/Pipe column yet (attached in a later pass), so the memory
            // classification is reported only where bytes are present.
            let args = Args::parse(rest, &[])?;
            let path = args
                .positional
                .first()
                .ok_or("telemetry: need a .tele file path")?;
            let strict = args.has("strict");
            let text = read(path)?;
            let (recs, skipped) = parse_tele_counted(&text);
            if skipped > 0 {
                eprintln!(
                    "telemetry: {} malformed rows skipped (truncated/corrupt input)",
                    skipped
                );
            }
            if recs.is_empty() {
                return Err(format!("{}: no telemetry records parsed", path));
            }
            let n_bad: usize = recs.iter().filter(|r| !r.roofline_ok()).count();
            let n_clock: usize = recs
                .iter()
                .filter(|r| !r.clock_plausible(CLOCK_TOL_FRAC))
                .count();
            let n_bytes: usize = recs.iter().filter(|r| r.bytes > 0).count();
            let sum_cyc: u64 = recs.iter().map(|r| r.cycles).sum();
            // ms from each record's own effective clock (derived-else-stamped), so a
            // plausible DVFS excursion converts honestly instead of at the stamp.
            let sum_ms: f64 = recs
                .iter()
                .map(|r| r.cycles as f64 / (r.effective_clock_ghz() * 1.0e9) * 1e3)
                .sum();
            let sum_span: u64 = recs.iter().map(|r| r.span_ns()).sum();
            println!(
                "telemetry: {} op records; Σcycles {} ({:.3} ms, per-record derived-else-stamped \
                 clock); Σgt-span {:.3} ms; {} with bytes; {} roofline-violating (T1 reject); \
                 {} clock-implausible (stamped fallback, no anchor)",
                recs.len(),
                sum_cyc,
                sum_ms,
                sum_span as f64 / 1e6,
                n_bytes,
                n_bad,
                n_clock
            );
            // Heavy hitters by measured wall.
            let mut idx: Vec<usize> = (0..recs.len()).collect();
            idx.sort_by(|&a, &b| recs[b].cycles.cmp(&recs[a].cycles));
            println!(
                "  top ops by measured wall (op_index kind lane cycles wall-ns bind):"
            );
            for &i in idx.iter().take(8) {
                let r: &TeleRecord = &recs[i];
                // per-op pipe-rate wiring: the op kind selects the pipe rate that populates
                // the measured Pipe column, so the binding column classifies (mem/pipe/lat).
                let rate = op_pipe_rate(&r.kind);
                let bind = if r.bytes == 0 && (r.ops == 0 || rate == 0.0) {
                    "-".to_string() // no bytes/ops columns yet -> unclassified
                } else {
                    match measured_bottleneck(r, rate, 0.0) {
                        Bottleneck::Memory => "mem".to_string(),
                        Bottleneck::Pipe(_) => "pipe".to_string(),
                        Bottleneck::Latency => "lat".to_string(),
                        Bottleneck::IssueCap => "issue".to_string(),
                    }
                };
                println!(
                    "    {:>5} {:<16} {:<7} {:>10} {:>10} {:>5}",
                    r.op_index,
                    r.kind,
                    if r.lane == Lane::Memory { "mem" } else { "compute" },
                    r.cycles,
                    r.wall_ns(),
                    bind,
                );
            }
            // Realized overlap of adjacent lane-crossing pairs (the T6 input): how much
            // the schedule's consecutive mem/compute ops actually ran concurrently.
            let mut pairs = 0usize;
            let mut overlapped = 0usize;
            let mut sum_ov = 0.0f64;
            for w in recs.windows(2) {
                if w[0].lane != w[1].lane {
                    let o = overlap_fraction(&w[0], &w[1]);
                    pairs += 1;
                    sum_ov += o;
                    if o > 0.05 {
                        overlapped += 1;
                    }
                }
            }
            if pairs > 0 {
                println!(
                    "  adjacent lane-crossing pairs: {}; realized-overlap mean {:.3}, {} with >5% overlap \
                     (measured concurrency for overlapped_contended / T6)",
                    pairs,
                    sum_ov / pairs as f64,
                    overlapped
                );
            }
            // --strict: a lossy parse is a failed run, not a footnote.
            Ok(if strict && skipped > 0 { 1 } else { 0 })
        }
        "--help" | "-h" | "help" => {
            println!("{}", USAGE);
            Ok(0)
        }
        other => Err(format!("unknown command {:?}\n{}", other, USAGE)),
    }
}

fn read(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code as u8),
        Err(msg) => {
            eprintln!("calx-mill: {}", msg);
            ExitCode::from(2)
        }
    }
}
