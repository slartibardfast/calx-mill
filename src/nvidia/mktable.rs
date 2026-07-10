//! `mk_table.py` port: aggregate `data/results/<host>/` measurements plus the
//! priors into the published ops table. Deterministic: same inputs, byte-
//! identical CSV (the differential test holds it to the committed
//! `tu102_ops.csv` byte for byte).

use crate::nvidia::csvio::{write_record, Table};
use crate::nvidia::pyfmt::g6;

/// row_id prefix -> SASS instruction (as proven by the purity gate); longest
/// prefix wins.
const INSTRUCTION: &[(&str, &str)] = &[
    ("alu.ffma", "FFMA"),
    ("alu.fadd", "FADD"),
    ("alu.fmul", "FMUL"),
    ("alu.iadd3_lop3", "IADD3+LOP3"),
    ("alu.iadd3", "IADD3"),
    ("alu.imad", "IMAD"),
    ("alu.lop3", "LOP3"),
    ("alu.shf", "SHF"),
    ("alu.sel", "SEL"),
    ("alu.fsel", "SEL"),
    ("alu.isetp_sel", "ISETP+SEL"),
    ("alu.isetp", "ISETP"),
    ("alu.fsetp_sel", "FSETP+SEL"),
    ("alu.fsetp", "FSETP"),
    ("alu.popc", "POPC"),
    ("alu.flo", "FLO"),
    ("alu.prmt", "PRMT"),
    ("alu.idp4a", "IDP.4A"),
    ("alu.hfma2", "HFMA2"),
    ("alu.dadd", "DADD"),
    ("alu.dfma", "DFMA"),
    ("alu.idiv", "(IDIV sequence)"),
];

fn instruction_for(row_id: &str) -> &'static str {
    let mut keys: Vec<&(&str, &str)> = INSTRUCTION.iter().collect();
    keys.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    for (prefix, inst) in keys {
        if row_id.starts_with(prefix) {
            return inst;
        }
    }
    ""
}

/// `w8_s8 -> Some("_s8")` (sweep family); `w8 -> Some("")`. Non-sweep
/// variants are their own group.
fn sweep_base(variant: &str) -> Option<&str> {
    let b = variant.as_bytes();
    if b.len() >= 2 && b[0] == b'w' && b[1].is_ascii_digit() {
        Some(variant.trim_start_matches(|c: char| c == 'w' || c.is_ascii_digit()))
    } else {
        None
    }
}

/// `statistics.median`: sorted midpoint, mean of the middle two on even length.
pub fn median(vals: &[f64]) -> f64 {
    let mut v = vals.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("values are never NaN"));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

struct Meas {
    run_id: String,
    gpu_index: String,
    row_id: String,
    kind: String,
    variant: String,
    value: String,
    unit: String,
    cv_pct: String,
    bench_src: String,
    git_sha: String,
    notes: String,
    family: String,
}

fn fval(s: &str) -> f64 {
    s.parse().unwrap_or_else(|_| panic!("not a number: {:?}", s))
}

struct Prior {
    prior_value: String,
    prior_src: String,
}

enum GroupKey {
    Sweep(String),
    Plain(String),
}

impl GroupKey {
    /// Python sorts groups by `str(key)`: a sweep key is a tuple whose str is
    /// `('sweep', '<suffix>')`; a plain key is the variant itself.
    fn sort_key(&self) -> String {
        match self {
            GroupKey::Sweep(s) => format!("('sweep', '{}')", s),
            GroupKey::Plain(v) => v.clone(),
        }
    }
}

/// Generate the ops-table CSV. `results` is (file name, content) for every
/// file in the results directory (the caller lists the directory); `priors`
/// and `na` are the priors table and the explicit-absence table.
pub fn mk_table(
    results: &[(String, String)],
    priors_csv: &str,
    na_csv: Option<&str>,
) -> (String, usize) {
    let pt = Table::parse(priors_csv);
    let (p_id, p_val, p_src) =
        (pt.col("row_id"), pt.col("prior_value"), pt.col("prior_src"));
    let mut priors: Vec<(String, Prior)> = Vec::new();
    for row in &pt.rows {
        let prior = Prior { prior_value: row[p_val].clone(), prior_src: row[p_src].clone() };
        if let Some((_, p)) = priors.iter_mut().find(|(k, _)| *k == row[p_id]) {
            *p = prior;
        } else {
            priors.push((row[p_id].clone(), prior));
        }
    }
    let prior_of = |key: &str| priors.iter().find(|(k, _)| k == key).map(|(_, p)| p);

    // measurements[row_id] in first-seen order across the name-sorted files
    let mut files: Vec<&(String, String)> = results.iter().collect();
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let mut measurements: Vec<(String, Vec<Meas>)> = Vec::new();
    for (fname, content) in files {
        if fname == "runs.csv" || !fname.ends_with(".csv") {
            continue;
        }
        if fname == "proj.csv" {
            continue; // diagnostic family (differential experiment); not table rows
        }
        let family = &fname[..fname.len() - 4];
        let t = Table::parse(content);
        let col = |n: &str| t.col(n);
        let (run_id, gpu, row_id, kind, variant, value, unit, cv, src, sha, notes) = (
            col("run_id"),
            col("gpu_index"),
            col("row_id"),
            col("kind"),
            col("variant"),
            col("value"),
            col("unit"),
            col("cv_pct"),
            col("bench_src"),
            col("git_sha"),
            col("notes"),
        );
        for row in &t.rows {
            let m = Meas {
                run_id: row[run_id].clone(),
                gpu_index: row[gpu].clone(),
                row_id: row[row_id].clone(),
                kind: row[kind].clone(),
                variant: row[variant].clone(),
                value: row[value].clone(),
                unit: row[unit].clone(),
                cv_pct: row[cv].clone(),
                bench_src: row[src].clone(),
                git_sha: row[sha].clone(),
                notes: row[notes].clone(),
                family: family.to_string(),
            };
            match measurements.iter_mut().find(|(k, _)| *k == m.row_id) {
                Some((_, v)) => v.push(m),
                None => measurements.push((m.row_id.clone(), vec![m])),
            }
        }
    }

    // pipe labels from the contention probes: rows named <op>.pipe carry
    // "pipe=<label>; ..." in notes and stay out of the table body
    let mut pipe_label: Vec<(String, String)> = Vec::new();
    measurements.retain(|(row_id, rows)| {
        if !row_id.ends_with(".pipe") {
            return true;
        }
        let note = &rows[0].notes;
        if let Some(rest) = note.strip_prefix("pipe=") {
            let label = rest.split(';').next().expect("split is non-empty");
            let label = label.split(" (").next().expect("split is non-empty");
            pipe_label.push((row_id[..row_id.len() - ".pipe".len()].to_string(), label.to_string()));
        }
        false
    });

    let mut row_ids: Vec<&str> = measurements.iter().map(|(k, _)| k.as_str()).collect();
    row_ids.sort_unstable();

    let fields = [
        "row_id", "class", "instruction", "variant", "kind", "value", "unit", "cv_pct",
        "pipe", "prior_value", "prior_src", "deviation_pct", "flag", "measured_by",
        "clock_mhz", "notes",
    ];
    let mut out = String::new();
    write_record(&mut out, &fields);
    let mut n_rows = 0usize;

    for row_id in row_ids {
        let rows = &measurements.iter().find(|(k, _)| k == row_id).expect("key exists").1;
        let kind = rows[0].kind.clone();
        let unit = rows[0].unit.clone();
        let family = rows[0].family.clone();
        let bench_src = rows[0].bench_src.clone();

        let mut groups: Vec<(GroupKey, Vec<&Meas>)> = Vec::new();
        if kind == "recip_tput" || kind == "bandwidth" {
            // w-prefixed variants are occupancy sweeps: peak per suffix; the
            // rest (stride18, conflict4, broadcast) group standalone
            let mut sweeps: Vec<(String, Vec<&Meas>)> = Vec::new();
            let mut standalone: Vec<(String, Vec<&Meas>)> = Vec::new();
            for m in rows.iter() {
                match sweep_base(&m.variant) {
                    Some(sb) => match sweeps.iter_mut().find(|(k, _)| k == sb) {
                        Some((_, v)) => v.push(m),
                        None => sweeps.push((sb.to_string(), vec![m])),
                    },
                    None => match standalone.iter_mut().find(|(k, _)| *k == m.variant) {
                        Some((_, v)) => v.push(m),
                        None => standalone.push((m.variant.clone(), vec![m])),
                    },
                }
            }
            for (k, v) in sweeps {
                groups.push((GroupKey::Sweep(k), v));
            }
            for (k, v) in standalone {
                groups.push((GroupKey::Plain(k), v));
            }
        } else {
            // latency rows: every variant is its own row
            for m in rows.iter() {
                match groups.iter_mut().find(|(k, _)| match k {
                    GroupKey::Plain(v) => *v == m.variant,
                    GroupKey::Sweep(_) => false,
                }) {
                    Some((_, v)) => v.push(m),
                    None => groups.push((GroupKey::Plain(m.variant.clone()), vec![m])),
                }
            }
        }
        groups.sort_by(|a, b| a.0.sort_key().cmp(&b.0.sort_key()));

        for (variant_key, grp) in &groups {
            let (run_vals, gpu_vals, cvs, variant): (Vec<f64>, Vec<(String, Vec<f64>)>, Vec<f64>, String);
            match variant_key {
                GroupKey::Sweep(_) => {
                    // per invocation: take the sweep peak
                    let mut per_run: Vec<((String, String), Vec<&Meas>)> = Vec::new();
                    for m in grp {
                        let key = (m.run_id.clone(), m.gpu_index.clone());
                        match per_run.iter_mut().find(|(k, _)| *k == key) {
                            Some((_, v)) => v.push(m),
                            None => per_run.push((key, vec![m])),
                        }
                    }
                    let mut rv = Vec::new();
                    let mut gv: Vec<(String, Vec<f64>)> = Vec::new();
                    let mut peak_variants: Vec<String> = Vec::new();
                    let mut cv = Vec::new();
                    for ((_, gpu), rr) in &per_run {
                        let mut peak = 0usize;
                        for i in 1..rr.len() {
                            if fval(&rr[i].value) > fval(&rr[peak].value) {
                                peak = i;
                            }
                        }
                        let p = rr[peak];
                        rv.push(fval(&p.value));
                        match gv.iter_mut().find(|(k, _)| k == gpu) {
                            Some((_, v)) => v.push(fval(&p.value)),
                            None => gv.push((gpu.clone(), vec![fval(&p.value)])),
                        }
                        peak_variants.push(p.variant.clone());
                        cv.push(fval(&p.cv_pct));
                    }
                    // Counter.most_common(1): highest count, first-seen on ties
                    let mut counted: Vec<(&str, usize)> = Vec::new();
                    for v in &peak_variants {
                        match counted.iter_mut().find(|(k, _)| k == v) {
                            Some((_, n)) => *n += 1,
                            None => counted.push((v, 1)),
                        }
                    }
                    let mut best = 0usize;
                    for i in 1..counted.len() {
                        if counted[i].1 > counted[best].1 {
                            best = i;
                        }
                    }
                    (run_vals, gpu_vals, cvs, variant) =
                        (rv, gv, cv, counted[best].0.to_string());
                }
                GroupKey::Plain(vk) => {
                    let rv: Vec<f64> = grp.iter().map(|m| fval(&m.value)).collect();
                    let mut gv: Vec<(String, Vec<f64>)> = Vec::new();
                    for m in grp {
                        match gv.iter_mut().find(|(k, _)| *k == m.gpu_index) {
                            Some((_, v)) => v.push(fval(&m.value)),
                            None => gv.push((m.gpu_index.clone(), vec![fval(&m.value)])),
                        }
                    }
                    let cv: Vec<f64> = grp.iter().map(|m| fval(&m.cv_pct)).collect();
                    (run_vals, gpu_vals, cvs, variant) = (rv, gv, cv, vk.clone());
                }
            }

            // provenance binds per published row (this variant group), not per
            // row_id
            let mut grp_shas: Vec<&str> = grp.iter().map(|m| m.git_sha.as_str()).collect();
            grp_shas.sort_unstable();
            grp_shas.dedup();
            let grp_runs = {
                let mut runs: Vec<&str> = grp.iter().map(|m| m.run_id.as_str()).collect();
                runs.sort_unstable();
                runs.dedup();
                runs.len()
            };

            let value = median(&run_vals);
            let within_cv = if cvs.is_empty() { 0.0 } else { median(&cvs) };
            let mut flag = "ok".to_string();
            let mut extra: Vec<String> = Vec::new();
            let note = grp
                .iter()
                .filter(|m| !m.notes.is_empty())
                .next_back() // latest annotation wins
                .map(|m| m.notes.clone())
                .unwrap_or_default();

            // cycle-domain rows are deterministic (0.1% floor); wall-clock
            // bandwidth rows carry real DRAM refresh/thermal variation (0.5%);
            // host-domain time rows carry scheduler jitter (5%)
            let floor = if kind == "time_us" {
                5.0
            } else if kind == "bandwidth" && unit == "GB/s" {
                0.5
            } else {
                0.1
            };
            if run_vals.len() < 2 {
                flag = "UNVERIFIED".into();
                extra.push("between-run rule unmet (single invocation)".into());
            } else {
                let hi = run_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let lo = run_vals.iter().cloned().fold(f64::INFINITY, f64::min);
                let spread = if value != 0.0 { 100.0 * (hi - lo) / value } else { 0.0 };
                if spread > within_cv.max(floor) {
                    extra.push(format!(
                        "between-run spread {:.2}% exceeds within-run cv {:.2}%",
                        spread, within_cv
                    ));
                    flag = "UNVERIFIED".into();
                }
            }

            if gpu_vals.len() >= 2 {
                let meds: Vec<f64> = gpu_vals.iter().map(|(_, v)| median(v)).collect();
                let gpu_diff =
                    if value != 0.0 { 100.0 * (meds[0] - meds[1]).abs() / value } else { 0.0 };
                if gpu_diff > (2.0 * within_cv).max(0.3) {
                    extra.push(format!("GPU0-vs-GPU1 medians differ {:.2}%", gpu_diff));
                    flag = "UNVERIFIED".into();
                }
            }

            // variant-specific priors: <row_id>.<variant> beats <row_id>
            let with_variant = format!("{}.{}", row_id, variant);
            let mut prior = prior_of(&with_variant).or_else(|| prior_of(row_id));
            if prior.is_some()
                && !variant.is_empty()
                && prior_of(&with_variant).is_none()
                && (kind == "latency_cycles" || kind == "latency_ns")
                && !["", "l1hit", "broadcast", "conflict1", "derived"]
                    .contains(&variant.as_str())
            {
                prior = None; // a bare-row prior binds only the base variant
            }
            let prior_value = prior.map(|p| p.prior_value.clone()).unwrap_or_default();
            let prior_src = prior.map(|p| p.prior_src.clone()).unwrap_or_default();
            let mut deviation = String::new();
            if let Some(p) = prior {
                let pv = fval(&p.prior_value);
                let dev = 100.0 * (value - pv) / pv;
                deviation = format!("{:.1}", dev);
                if dev.abs() > 25.0 && flag == "ok" {
                    flag = "DEV>25%".into();
                }
            }

            let mut parts: Vec<String> = Vec::new();
            if !note.is_empty() {
                parts.push(note);
            }
            parts.extend(extra);
            let all_notes = parts.join("; ");

            let pipe = pipe_label
                .iter()
                .find(|(k, _)| row_id.starts_with(&format!("{}.", k)))
                .map(|(_, p)| p.as_str())
                .unwrap_or("");
            let measured_by =
                format!("{}@{} n_runs={}", bench_src, grp_shas.join("+"), grp_runs);
            write_record(
                &mut out,
                &[
                    row_id,
                    &family,
                    instruction_for(row_id),
                    &variant,
                    &kind,
                    &g6(value),
                    &unit,
                    &format!("{:.3}", within_cv),
                    pipe,
                    &prior_value,
                    &prior_src,
                    &deviation,
                    &flag,
                    &measured_by,
                    "1455",
                    &all_notes,
                ],
            );
            n_rows += 1;
        }
    }

    // explicit-absence rows: features that do not exist on sm_75 are real rows
    // (kind=na, flag=NA_SM75), never missing keys
    if let Some(na) = na_csv {
        let t = Table::parse(na);
        let (id, inst, src, notes) =
            (t.col("row_id"), t.col("instruction"), t.col("prior_src"), t.col("notes"));
        for row in &t.rows {
            let class = row[id].split('.').next().expect("split is non-empty");
            write_record(
                &mut out,
                &[
                    &row[id],
                    class,
                    &row[inst],
                    "",
                    "na",
                    "",
                    "",
                    "",
                    "",
                    "",
                    &row[src],
                    "",
                    "NA_SM75",
                    "table/na_sm75.csv",
                    "",
                    &row[notes],
                ],
            );
            n_rows += 1;
        }
    }
    (out, n_rows)
}
