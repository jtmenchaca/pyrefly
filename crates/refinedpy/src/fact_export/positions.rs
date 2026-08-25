//! Byte offset to 1-based line number, for the provenance `line` field.

/// Every line's own start offset in `source`, so a byte offset reads
/// back as a 1-based line number.
pub(super) fn line_starts_of(source: &[u8]) -> Vec<usize> {
    let mut starts = vec![0usize];
    for (index, byte) in source.iter().enumerate() {
        if *byte == b'\n' {
            starts.push(index + 1);
        }
    }
    starts
}

/// The 1-based line `offset` sits on.
pub(super) fn line_of(line_starts: &[usize], offset: usize) -> i64 {
    match line_starts.binary_search(&offset) {
        Ok(index) => (index + 1) as i64,
        Err(index) => index as i64,
    }
}
