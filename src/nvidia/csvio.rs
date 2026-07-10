//! Minimal CSV: a reader with Python-csv semantics (quoted fields, CRLF or LF,
//! blank records skipped) and a writer that byte-matches Python's
//! `csv.DictWriter` defaults (minimal quoting, `\r\n` row terminator).

/// A parsed CSV file with a header row: `rows` holds the records in file
/// order, each padded to the header's width.
pub struct Table {
    pub header: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl Table {
    pub fn parse(text: &str) -> Table {
        let mut records = parse_records(text);
        if records.is_empty() {
            return Table { header: Vec::new(), rows: Vec::new() };
        }
        let header = records.remove(0);
        for row in &mut records {
            while row.len() < header.len() {
                row.push(String::new());
            }
        }
        Table { header, rows: records }
    }

    /// Column index by name; the reference CSVs are fixed-schema, so a missing
    /// column is a caller bug.
    pub fn col(&self, name: &str) -> usize {
        self.header
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("no column {:?} in header {:?}", name, self.header))
    }
}

/// Split CSV text into records (including the header). Handles quoted fields,
/// doubled quotes, embedded newlines, CRLF/LF terminators; skips blank records
/// the way `csv.DictReader` does.
pub fn parse_records(text: &str) -> Vec<Vec<String>> {
    let mut records = Vec::new();
    let mut record: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut field_started = false;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if in_quotes {
            if c == b'"' {
                if bytes.get(i + 1) == Some(&b'"') {
                    field.push('"');
                    i += 1;
                } else {
                    in_quotes = false;
                }
            } else {
                field.push(c as char);
            }
        } else {
            match c {
                b'"' => {
                    in_quotes = true;
                    field_started = true;
                }
                b',' => {
                    record.push(std::mem::take(&mut field));
                    field_started = true;
                }
                b'\r' => {}
                b'\n' => {
                    if field_started || !field.is_empty() || !record.is_empty() {
                        record.push(std::mem::take(&mut field));
                        records.push(std::mem::take(&mut record));
                    }
                    field_started = false;
                }
                _ => field.push(c as char),
            }
        }
        i += 1;
    }
    if field_started || !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }
    records
}

/// Append one CSV record with Python `csv.writer` defaults: minimal quoting
/// (a field is quoted only if it contains the delimiter, a quote, or a line
/// break) and a `\r\n` terminator.
pub fn write_record(out: &mut String, fields: &[&str]) {
    for (i, f) in fields.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        if f.contains([',', '"', '\r', '\n']) {
            out.push('"');
            out.push_str(&f.replace('"', "\"\""));
            out.push('"');
        } else {
            out.push_str(f);
        }
    }
    out.push_str("\r\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_and_quoted_fields() {
        let t = Table::parse("a,b,c\r\n1,\"x,y\",3\n\n4,,\"q\"\"q\"\n");
        assert_eq!(t.header, ["a", "b", "c"]);
        assert_eq!(t.rows.len(), 2);
        assert_eq!(t.rows[0], ["1", "x,y", "3"]);
        assert_eq!(t.rows[1], ["4", "", "q\"q"]);
    }

    #[test]
    fn writes_with_minimal_quoting_and_crlf() {
        let mut s = String::new();
        write_record(&mut s, &["a", "b,c", "d\"e", ""]);
        assert_eq!(s, "a,\"b,c\",\"d\"\"e\",\r\n");
    }
}
