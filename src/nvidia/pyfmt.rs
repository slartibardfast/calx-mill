//! Formatting helpers that byte-match the reference Python tools' output, so
//! the differential tests can compare whole reports byte for byte.

/// Python `format(v, '.6g')` (printf `%.6g`): six significant digits, trailing
/// zeros stripped, exponent form when the decimal exponent is below -4 or at
/// least 6 (with a sign and at least two exponent digits, e.g. `1e+06`).
pub fn g6(v: f64) -> String {
    if v == 0.0 {
        return if v.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf".into() } else { "inf".into() };
    }
    let neg = v < 0.0;
    // {:.5e} rounds to six significant digits (nearest, ties to even), exactly
    // the rounding %.6g performs on the binary value.
    let sci = format!("{:.5e}", v.abs());
    let (mant, exp) = sci.split_once('e').expect("LowerExp always has an exponent");
    let exp: i32 = exp.parse().expect("exponent is an integer");
    let digits: Vec<u8> = mant.bytes().filter(|b| *b != b'.').collect();
    debug_assert_eq!(digits.len(), 6);
    let digits = String::from_utf8(digits).expect("digits are ASCII");
    let body = if !(-4..6).contains(&exp) {
        let trimmed = digits.trim_end_matches('0');
        let (head, tail) = trimmed.split_at(1);
        let mantissa = if tail.is_empty() {
            head.to_string()
        } else {
            format!("{}.{}", head, tail)
        };
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", mantissa, sign, exp.abs())
    } else if exp >= 0 {
        let point = exp as usize + 1;
        if point >= digits.len() {
            digits.clone()
        } else {
            let frac = digits[point..].trim_end_matches('0');
            if frac.is_empty() {
                digits[..point].to_string()
            } else {
                format!("{}.{}", &digits[..point], frac)
            }
        }
    } else {
        let zeros = "0".repeat((-exp - 1) as usize);
        format!("0.{}{}", zeros, digits.trim_end_matches('0'))
    };
    if neg {
        format!("-{}", body)
    } else {
        body
    }
}

/// Python `str(float)` for the magnitudes the tools print: Rust's shortest
/// round-trip digits with a `.0` suffix restored on integral values (Python
/// prints `10.0` where Rust prints `10`). Python switches to exponent notation
/// outside [1e-4, 1e16); the tools never print such values, so this does not.
pub fn repr_f64(v: f64) -> String {
    let s = format!("{}", v);
    if s.bytes().all(|b| b.is_ascii_digit() || b == b'-') {
        format!("{}.0", s)
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g6_matches_python_format_6g() {
        assert_eq!(g6(4.07032), "4.07032");
        assert_eq!(g6(0.0526396), "0.0526396");
        assert_eq!(g6(624.0), "624");
        assert_eq!(g6(43.9999), "43.9999");
        assert_eq!(g6(0.0), "0");
        assert_eq!(g6(123456.0), "123456");
        assert_eq!(g6(1234567.0), "1.23457e+06");
        assert_eq!(g6(999999.9), "1e+06");
        assert_eq!(g6(0.000012345), "1.2345e-05");
        assert_eq!(g6(-2.5), "-2.5");
        assert_eq!(g6(1455.0), "1455");
    }

    #[test]
    fn repr_matches_python_str() {
        assert_eq!(repr_f64(10.0), "10.0");
        assert_eq!(repr_f64(0.2), "0.2");
        assert_eq!(repr_f64(-3.0), "-3.0");
        assert_eq!(repr_f64(12.5), "12.5");
    }
}
