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
}
