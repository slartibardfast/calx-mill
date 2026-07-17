//! Projection validity — calx-mill as a calibrated instrument.
//!
//! Realizes `plan/0143.../` sibling spec `plan/0144/spec/projection-validity.md`: a
//! projection may GATE a decision only if it reproduces a measured **anchor** within its
//! declared **domain**; else it is advisory (a guess) or provisional (anchored but
//! extrapolated). Authority is not an ignorable field — it is the tier the `gate` mode
//! (main.rs) reasons over, emitting `CERTIFIED`/`PROVISIONAL`/`REFUSED` as a gate-manifest
//! arbiter, so an unvalidated projection cannot be laundered into a decision.
//!
//! Laws (spec A1–A5), discharged by the `#[cfg(kani)]` proofs below and the anchor tests:
//! - A1 anchored-gate, A2 in-domain, A3 no-launder, A4 monotone-down (capped at
//!   CrossChecked under composition), A5 falsifiability (the anchor registry).

/// The authority tier — the lattice `Advisory < CrossChecked < Gate` the `gate` mode
/// reasons over. `derive(Ord)` gives the order; `min` is the meet.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Authority {
    Advisory,     // unanchored / out-of-domain / fails its anchor -> a guess
    CrossChecked, // anchored model, extrapolated to this in-domain query, or a composition
    Gate,         // directly anchored AND at the anchor point -> reproduces a measured value
}

/// How the current query relates to where the model was anchored (A2).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DomainFit {
    AtAnchor,    // this query is (a) the anchor point
    InDomain,    // same validated domain, a different query (extrapolation)
    OutOfDomain, // outside the regime the anchor covers
}

/// The `gate`-mode verdict for a tier. `Refused` is a refusal to gate, not "the lever is
/// bad"; the bench/oracle gates whatever the model refuses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    Certified,   // Gate
    Provisional, // CrossChecked -- decide/build on it, the bench confirms
    Refused,     // Advisory -- only the bench/oracle can gate here
}

/// A measured ground-truth calibration point (spec anchor registry).
#[derive(Clone, Copy, Debug)]
pub struct Anchor {
    pub measured: f64,
    pub tol: f64,
}

impl Anchor {
    /// A1: does a projected value reproduce this anchor within tolerance?
    pub fn reproduces(&self, projected: f64) -> bool {
        (projected - self.measured).abs() <= self.tol.abs()
    }
}

/// The authority a projection earns from (A1) anchor reproduction and (A2) domain fit.
/// `anchored` = "the model reproduces its anchor within tol" (from [`Anchor::reproduces`]).
///
/// # Guarantees
/// A3: `!anchored` ⇒ `Advisory` (proof `unvalidated_never_gates`). A2: `OutOfDomain` ⇒
/// never `Gate` (proof `out_of_domain_is_advisory`).
pub fn authority(anchored: bool, fit: DomainFit) -> Authority {
    if !anchored {
        return Authority::Advisory; // A1/A3: no reproduced anchor -> a guess
    }
    match fit {
        // A2:
        DomainFit::AtAnchor => Authority::Gate,
        DomainFit::InDomain => Authority::CrossChecked,
        DomainFit::OutOfDomain => Authority::Advisory,
    }
}

/// A4: compose two projections. The meet of their tiers, **capped at `CrossChecked`**
/// (never `Gate`) unless the *composite itself* reproduces its own anchor
/// (`composite_anchored`). You cannot gain a `Gate` by stacking two `Gate`s — the
/// interaction is unvalidated (register→ILP, whole-op, contention lessons).
///
/// # Guarantees
/// `compose(a, b, _) <= a` and `<= b`; `compose(Gate, Gate, false) == CrossChecked`
/// (proof `composition_never_raises_authority`, `compose_caps_at_crosschecked`).
pub fn compose(a: Authority, b: Authority, composite_anchored: bool) -> Authority {
    let m = a.min(b); // meet
    if composite_anchored {
        m // re-anchored composite keeps the meet (may be Gate if both were)
    } else {
        m.min(Authority::CrossChecked) // else cap strictly below Gate
    }
}

/// The `gate`-mode verdict for a tier.
pub fn verdict(a: Authority) -> Verdict {
    match a {
        Authority::Gate => Verdict::Certified,
        Authority::CrossChecked => Verdict::Provisional,
        Authority::Advisory => Verdict::Refused,
    }
}

/// The `gate` process exit code per verdict — the contract a gate manifest keys on
/// (host `call/0036`). Pairwise distinct: PROVISIONAL is not a terminal pass and must
/// not share CERTIFIED's 0; REFUSED keeps 2 (the established no-launder exit).
pub fn exit_code(v: Verdict) -> u8 {
    match v {
        Verdict::Certified => 0,
        Verdict::Refused => 2,
        Verdict::Provisional => 3,
    }
}

/// `gate` operator/usage error: the arbiter never adjudicated (missing or unknown
/// flag, malformed registry or `--at`, conflicting tolerance forms, negative tol).
pub const USAGE_EXIT: u8 = 4;

/// One axis bound of an anchor's validated domain: a closed numeric range
/// (`ctx=1..8192`) or an exact value (`dtype=q4_0`, string-compared).
#[derive(Clone, Debug, PartialEq)]
pub enum DomainBound {
    Range(f64, f64),
    Exact(String),
}

/// One registry row (spec A5 realized): a measured anchor plus its own query point
/// (`at`) and the domain it validates. `anchor.tol` is stored resolved-absolute
/// (a `rel` tolerance is resolved against `measured` at load).
#[derive(Clone, Debug)]
pub struct AnchorRow {
    pub id: String,
    pub anchor: Anchor,
    pub units: String,
    pub at: Vec<(String, String)>,
    pub domain: Vec<(String, DomainBound)>,
}

fn parse_kv(field: &str, what: &str) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    for part in field.split(';').filter(|p| !p.is_empty()) {
        let (k, v) = part
            .split_once('=')
            .ok_or_else(|| format!("{}: {:?} is not key=value", what, part))?;
        out.push((k.trim().to_string(), v.trim().to_string()));
    }
    Ok(out)
}

/// Parse the anchor registry CSV: columns
/// `id,measured,tol,tol_kind,units,at,domain` with `at` = `key=value;...` and
/// `domain` = `key=lo..hi;...` (inclusive) or `key=value` (exact). A duplicate id,
/// a negative tolerance, or an unknown `tol_kind` is a load error — never a default.
/// (The CSV reader is the adapter layer's `csvio`; it is substrate-neutral, imported
/// here so the registry format has exactly one parser.)
pub fn parse_registry(text: &str) -> Result<Vec<AnchorRow>, String> {
    let t = crate::nvidia::csvio::Table::parse(text);
    for need in ["id", "measured", "tol", "tol_kind", "units", "at", "domain"] {
        if !t.header.iter().any(|h| h == need) {
            return Err(format!("registry: missing column {:?}", need));
        }
    }
    let (ci, cm, ct, ck, cu, ca, cd) = (
        t.col("id"), t.col("measured"), t.col("tol"), t.col("tol_kind"),
        t.col("units"), t.col("at"), t.col("domain"),
    );
    let mut rows: Vec<AnchorRow> = Vec::new();
    for r in &t.rows {
        let id = r[ci].clone();
        if rows.iter().any(|x| x.id == id) {
            return Err(format!("registry: duplicate id {:?}", id));
        }
        let measured: f64 = r[cm]
            .parse()
            .map_err(|_| format!("registry {}: bad measured {:?}", id, r[cm]))?;
        let tol_raw: f64 = r[ct]
            .parse()
            .map_err(|_| format!("registry {}: bad tol {:?}", id, r[ct]))?;
        if tol_raw < 0.0 {
            return Err(format!("registry {}: negative tol", id));
        }
        let tol = match r[ck].as_str() {
            "abs" => tol_raw,
            "rel" => tol_raw / 100.0 * measured.abs(),
            other => return Err(format!("registry {}: tol_kind {:?} (abs|rel)", id, other)),
        };
        let mut domain = Vec::new();
        for (k, v) in parse_kv(&r[cd], &format!("registry {} domain", id))? {
            let bound = match v.split_once("..") {
                Some((lo, hi)) => {
                    let lo: f64 = lo.trim().parse()
                        .map_err(|_| format!("registry {}: bad range {:?}", id, v))?;
                    let hi: f64 = hi.trim().parse()
                        .map_err(|_| format!("registry {}: bad range {:?}", id, v))?;
                    DomainBound::Range(lo, hi)
                }
                None => DomainBound::Exact(v),
            };
            domain.push((k, bound));
        }
        rows.push(AnchorRow {
            id,
            anchor: Anchor { measured, tol },
            units: r[cu].clone(),
            at: parse_kv(&r[ca], "registry at")?,
            domain,
        });
    }
    Ok(rows)
}

fn value_eq(a: &str, b: &str) -> bool {
    match (a.parse::<f64>(), b.parse::<f64>()) {
        (Ok(x), Ok(y)) => x == y,
        _ => a == b,
    }
}

/// A2/A5 made mechanical: the fit is COMPUTED from the registry row, never typed.
/// A query key the domain does not cover, a domain key the query does not supply,
/// or a value outside its bound is `OutOfDomain` (membership unverifiable is
/// non-membership); the anchor's own query point is `AtAnchor`; else `InDomain`.
pub fn computed_fit(row: &AnchorRow, query: &[(String, String)]) -> DomainFit {
    for (k, _) in query {
        if !row.domain.iter().any(|(dk, _)| dk == k) {
            return DomainFit::OutOfDomain; // an axis the anchor never covered
        }
    }
    for (dk, bound) in &row.domain {
        let Some((_, qv)) = query.iter().find(|(k, _)| k == dk) else {
            return DomainFit::OutOfDomain; // unverifiable axis
        };
        match bound {
            DomainBound::Range(lo, hi) => {
                let Ok(x) = qv.parse::<f64>() else { return DomainFit::OutOfDomain };
                if x < *lo || x > *hi {
                    return DomainFit::OutOfDomain;
                }
            }
            DomainBound::Exact(s) => {
                if !value_eq(qv, s) {
                    return DomainFit::OutOfDomain;
                }
            }
        }
    }
    let at_anchor = query.len() == row.at.len()
        && query
            .iter()
            .all(|(k, v)| row.at.iter().any(|(ak, av)| ak == k && value_eq(av, v)));
    if at_anchor {
        DomainFit::AtAnchor
    } else {
        DomainFit::InDomain
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    fn any_auth() -> Authority {
        match kani::any::<u8>() % 3 {
            0 => Authority::Advisory,
            1 => Authority::CrossChecked,
            _ => Authority::Gate,
        }
    }
    fn any_fit() -> DomainFit {
        match kani::any::<u8>() % 3 {
            0 => DomainFit::AtAnchor,
            1 => DomainFit::InDomain,
            _ => DomainFit::OutOfDomain,
        }
    }

    // A3 no-launder: without a reproduced anchor, authority is Advisory (never gates).
    #[kani::proof]
    fn unvalidated_never_gates() {
        kani::assert(authority(false, any_fit()) == Authority::Advisory,
                     "unanchored -> Advisory");
    }

    // A2 in-domain: an out-of-domain query is never Gate (advisory even if anchored).
    #[kani::proof]
    fn out_of_domain_is_advisory() {
        let anchored: bool = kani::any();
        kani::assert(authority(anchored, DomainFit::OutOfDomain) == Authority::Advisory,
                     "out-of-domain -> Advisory");
        kani::assert(authority(anchored, any_fit()) != Authority::Gate || anchored,
                     "Gate requires anchored");
    }

    // A4 monotone-down: composition never rises above either input.
    #[kani::proof]
    fn composition_never_raises_authority() {
        let a = any_auth(); let b = any_auth(); let anc: bool = kani::any();
        let c = compose(a, b, anc);
        kani::assert(c <= a && c <= b, "compose <= meet of inputs");
    }

    // A4 cap: an un-re-anchored composition is never Gate; two Gates compose to CrossChecked.
    #[kani::proof]
    fn compose_caps_at_crosschecked() {
        let a = any_auth(); let b = any_auth();
        kani::assert(compose(a, b, false) <= Authority::CrossChecked,
                     "un-re-anchored composition never Gate");
        kani::assert(compose(Authority::Gate, Authority::Gate, false) == Authority::CrossChecked,
                     "cannot gain a Gate by stacking Gates");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Authority::*;
    use DomainFit::*;

    #[test]
    fn tier_from_anchor_and_domain() {
        assert_eq!(authority(true, AtAnchor), Gate);
        assert_eq!(authority(true, InDomain), CrossChecked);
        assert_eq!(authority(true, OutOfDomain), Advisory);
        assert_eq!(authority(false, AtAnchor), Advisory); // A3: no anchor beats domain
    }

    #[test]
    fn verdicts() {
        assert_eq!(verdict(Gate), Verdict::Certified);
        assert_eq!(verdict(CrossChecked), Verdict::Provisional);
        assert_eq!(verdict(Advisory), Verdict::Refused);
    }

    #[test]
    fn compose_cannot_launder_a_gate() {
        assert_eq!(compose(Gate, Gate, false), CrossChecked);   // stacking two gates
        assert_eq!(compose(Gate, Advisory, false), Advisory);   // meet with a guess
        assert_eq!(compose(Gate, Gate, true), Gate);            // re-anchored composite
    }

    // The session's conformance instances (spec) as anchored cases.
    #[test]
    fn conformance_op_precision_is_refused() {
        // The op-precision model predicted KL 1.8e-7; the full-stack anchor is 0.0200.
        let full_stack_kl = Anchor { measured: 0.0200, tol: 0.002 };
        assert!(!full_stack_kl.reproduces(1.8e-7));             // A1: does NOT reproduce
        // ...and a KL claim is out of the op-local domain anyway:
        assert_eq!(verdict(authority(false, OutOfDomain)), Verdict::Refused);
    }

    #[test]
    fn conformance_pipe_overlap_is_provisional() {
        // overlapped_contended is anchored on FATTN (reproduces the dbuf profile);
        // extrapolated to pipe-overlap -> CrossChecked -> Provisional (build, bench confirms).
        assert_eq!(verdict(authority(true, InDomain)), Verdict::Provisional);
    }

    #[test]
    fn conformance_2limb_reproduces_its_anchor() {
        // The shipping 2-limb KL anchor; a model claiming ~0.01836 reproduces it.
        let two_limb = Anchor { measured: 0.01836, tol: 0.0005 };
        assert!(two_limb.reproduces(0.0184));
        assert!(!two_limb.reproduces(0.0200));                  // naive-f16 does not
    }

    #[test]
    fn exit_codes_pairwise_distinct() {
        let codes = [
            exit_code(Verdict::Certified),
            exit_code(Verdict::Provisional),
            exit_code(Verdict::Refused),
        ];
        assert_eq!(codes, [0, 3, 2]);
        assert!(codes[0] != codes[1] && codes[1] != codes[2] && codes[0] != codes[2]);
        assert!(!codes.contains(&USAGE_EXIT)); // operator error is its own lane
    }

    const REGISTRY: &str = "\
id,measured,tol,tol_kind,units,at,domain\n\
mmvq-deep-us,68.19,20,rel,us,ctx=4096;batch=1,ctx=1..8192;batch=1..1\n\
fullstack-kl-naive-f16,0.0200,10,rel,kl,limbs=1;stack=full,limbs=1..1;stack=full\n";

    fn q(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn registry_loads_and_resolves_rel_tol() {
        let rows = parse_registry(REGISTRY).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].id, "mmvq-deep-us");
        assert!((rows[0].anchor.tol - 13.638).abs() < 1e-9); // 20% of 68.19
        assert_eq!(rows[0].units, "us");
        assert_eq!(rows[0].domain[0], ("ctx".to_string(), DomainBound::Range(1.0, 8192.0)));
    }

    #[test]
    fn registry_rejects_duplicate_and_negative() {
        let dup = format!("{}mmvq-deep-us,1,1,abs,us,ctx=1,ctx=1..2\n", REGISTRY);
        assert!(parse_registry(&dup).is_err());
        let neg = "id,measured,tol,tol_kind,units,at,domain\nx,1,-1,abs,,,\n";
        assert!(parse_registry(neg).is_err());
        let kind = "id,measured,tol,tol_kind,units,at,domain\nx,1,1,pct,,,\n";
        assert!(parse_registry(kind).is_err());
    }

    #[test]
    fn fit_is_mechanical() {
        let rows = parse_registry(REGISTRY).unwrap();
        let r = &rows[0];
        assert_eq!(computed_fit(r, &q(&[("ctx", "4096"), ("batch", "1")])), AtAnchor);
        assert_eq!(computed_fit(r, &q(&[("ctx", "2048"), ("batch", "1")])), InDomain);
        assert_eq!(computed_fit(r, &q(&[("ctx", "100000"), ("batch", "1")])), OutOfDomain);
        // an axis the anchor never covered:
        assert_eq!(
            computed_fit(r, &q(&[("ctx", "4096"), ("batch", "1"), ("gpus", "2")])),
            OutOfDomain
        );
        // an unverifiable axis (domain key missing from the query):
        assert_eq!(computed_fit(r, &q(&[("ctx", "4096")])), OutOfDomain);
        // a non-numeric value against a range:
        assert_eq!(computed_fit(r, &q(&[("ctx", "deep"), ("batch", "1")])), OutOfDomain);
    }

    // The 10^5 case end-to-end with NO operator honesty: the claim is at the anchor's
    // own query point, but fails A1 against the recorded measured value -> Refused.
    #[test]
    fn registry_refuses_the_op_precision_launder() {
        let rows = parse_registry(REGISTRY).unwrap();
        let r = &rows[1];
        let query = q(&[("limbs", "1"), ("stack", "full")]);
        let fit = computed_fit(r, &query);
        assert_eq!(fit, AtAnchor);
        let anchored = r.anchor.reproduces(1e-7);
        assert!(!anchored);
        assert_eq!(verdict(authority(anchored, fit)), Verdict::Refused);
    }
}
