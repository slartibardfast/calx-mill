//! A tiny pattern matcher for kernel-name selection, covering exactly the
//! constructs the reference tools' patterns use: literal characters, `.`,
//! `\d`, and the `*`/`+` quantifiers, matched anywhere in the text (Python
//! `re.search` semantics). A full regex engine would be a dependency spent on
//! constructs no caller uses; any other escape is taken as the literal
//! escaped character.

#[derive(Clone, Copy, PartialEq, Eq)]
enum Atom {
    Lit(u8),
    Any,
    Digit,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Quant {
    One,
    Star,
    Plus,
}

pub struct Pattern {
    atoms: Vec<(Atom, Quant)>,
}

fn atom_ok(atom: Atom, b: u8) -> bool {
    match atom {
        Atom::Lit(l) => b == l,
        Atom::Any => b != b'\n',
        Atom::Digit => b.is_ascii_digit(),
    }
}

fn match_here(atoms: &[(Atom, Quant)], text: &[u8]) -> bool {
    let Some((&(atom, quant), rest)) = atoms.split_first() else {
        return true;
    };
    match quant {
        Quant::One => !text.is_empty() && atom_ok(atom, text[0]) && match_here(rest, &text[1..]),
        Quant::Plus => {
            !text.is_empty()
                && atom_ok(atom, text[0])
                && match_star(atom, rest, &text[1..])
        }
        Quant::Star => match_star(atom, rest, text),
    }
}

fn match_star(atom: Atom, rest: &[(Atom, Quant)], text: &[u8]) -> bool {
    let mut n = 0;
    while n < text.len() && atom_ok(atom, text[n]) {
        n += 1;
    }
    // greedy, backing off one character at a time
    loop {
        if match_here(rest, &text[n..]) {
            return true;
        }
        if n == 0 {
            return false;
        }
        n -= 1;
    }
}

impl Pattern {
    /// [`Pattern::new`] for OPERATOR-supplied patterns: reject the regex constructs
    /// this engine does not implement instead of taking them as literals (a
    /// `[0-9]`-style filter previously matched nothing, silently).
    pub fn try_new(pat: &str) -> Result<Pattern, String> {
        let bytes = pat.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            match bytes[i] {
                b'\\' => i += 1, // an escaped character is always a literal
                c if br"[](){}|?^$".contains(&c) => {
                    return Err(format!(
                        "pattern {:?}: unsupported regex construct {:?} \
                         (supported: literals, '.', '\\d', '*', '+')",
                        pat, c as char
                    ));
                }
                _ => {}
            }
            i += 1;
        }
        Ok(Pattern::new(pat))
    }

    pub fn new(pat: &str) -> Pattern {
        let bytes = pat.as_bytes();
        let mut atoms = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            let atom = match bytes[i] {
                b'\\' if i + 1 < bytes.len() => {
                    i += 1;
                    match bytes[i] {
                        b'd' => Atom::Digit,
                        other => Atom::Lit(other),
                    }
                }
                b'.' => Atom::Any,
                other => Atom::Lit(other),
            };
            i += 1;
            let quant = match bytes.get(i) {
                Some(b'*') => {
                    i += 1;
                    Quant::Star
                }
                Some(b'+') => {
                    i += 1;
                    Quant::Plus
                }
                _ => Quant::One,
            };
            atoms.push((atom, quant));
        }
        Pattern { atoms }
    }

    /// Python `re.search`: does the pattern match at any position?
    pub fn is_match(&self, text: &str) -> bool {
        let bytes = text.as_bytes();
        (0..=bytes.len()).any(|start| match_here(&self.atoms, &bytes[start..]))
    }
}

#[cfg(test)]
mod tests {
    use super::Pattern;

    #[test]
    fn searches_like_python_re() {
        assert!(Pattern::new("ffma_anchor").is_match("_ZN5tu10218ffma_anchor_kernelE"));
        assert!(!Pattern::new("ffma_anchor").is_match("_ZN5tu102stream_anchorE"));
        assert!(Pattern::new(".*").is_match("anything"));
        assert!(Pattern::new(".*").is_match(""));
        let inject = Pattern::new("inject_kernelINS_\\d*OpFFMAELi8E");
        assert!(inject.is_match("_ZN5tu10213inject_kernelINS_6OpFFMAELi8EEEvPKfPxPf"));
        assert!(inject.is_match("prefix_inject_kernelINS_OpFFMAELi8E"));
        assert!(!inject.is_match("_ZN5tu10213inject_kernelINS_6OpFFMAELi16EEEv"));
        assert!(Pattern::new("fa_mini_kernelILi0").is_match("_ZN5tu10214fa_mini_kernelILi0EEEv"));
        assert!(Pattern::new("a\\d+b").is_match("xxa12byy"));
        assert!(!Pattern::new("a\\d+b").is_match("xxabyy"));
    }

    #[test]
    fn try_new_rejects_unimplemented_constructs() {
        assert!(Pattern::try_new("[0-9]").is_err());
        assert!(Pattern::try_new("a|b").is_err());
        assert!(Pattern::try_new("^anchor$").is_err());
        assert!(Pattern::try_new("x{3}").is_err());
        assert!(Pattern::try_new("a?").is_err());
        // the supported subset still parses, and escapes stay literals
        assert!(Pattern::try_new("fa_mini_kernelILi0").is_ok());
        assert!(Pattern::try_new("inject_kernelINS_\\d*OpFFMAELi8E").is_ok());
        assert!(Pattern::try_new("\\[literal\\]").is_ok());
    }
}
