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
            let kernel = Pattern::new(args.value("kernel").unwrap_or(".*"));
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
            let kernel = Pattern::new(args.value("kernel").unwrap_or(".*"));
            let sass = sass_text(Path::new(path)).map_err(|e| e.to_string())?;
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
