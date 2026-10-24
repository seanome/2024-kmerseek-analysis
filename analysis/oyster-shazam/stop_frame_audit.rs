//! Count stop-containing translation frames in FASTQ records from standard input.
use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};
fn frames_with_stop(dna: &[u8]) -> usize {
    let reverse: Vec<u8> = dna
        .iter()
        .rev()
        .map(|b| match b.to_ascii_uppercase() {
            b'A' => b'T',
            b'T' => b'A',
            b'C' => b'G',
            b'G' => b'C',
            _ => b'N',
        })
        .collect();
    [dna, reverse.as_slice()]
        .iter()
        .map(|strand| {
            (0..3)
                .filter(|offset| {
                    strand
                        .get(*offset..)
                        .unwrap_or_default()
                        .chunks_exact(3)
                        .any(|c| {
                            c[0].eq_ignore_ascii_case(&b'T')
                                && ((c[1].eq_ignore_ascii_case(&b'A')
                                    && (c[2].eq_ignore_ascii_case(&b'A')
                                        || c[2].eq_ignore_ascii_case(&b'G')))
                                    || (c[1].eq_ignore_ascii_case(&b'G')
                                        && c[2].eq_ignore_ascii_case(&b'A')))
                        })
                })
                .count()
        })
        .sum()
}
fn main() -> io::Result<()> {
    let mut input = io::stdin().lock();
    let mut header = String::new();
    let mut seq = String::new();
    let mut plus = String::new();
    let mut quality = String::new();
    let mut counts: BTreeMap<(usize, usize), (u64, u64)> = BTreeMap::new();
    loop {
        header.clear();
        seq.clear();
        plus.clear();
        quality.clear();
        if input.read_line(&mut header)? == 0 {
            break;
        }
        for line in [&mut seq, &mut plus, &mut quality] {
            if input.read_line(line)? == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Truncated FASTQ",
                ));
            }
        }
        let dna = seq.trim_end().as_bytes();
        if !header.starts_with('@')
            || !plus.starts_with('+')
            || dna.len() != quality.trim_end().len()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid four-line FASTQ",
            ));
        }
        let ambiguous = dna
            .iter()
            .any(|b| !matches!(b.to_ascii_uppercase(), b'A' | b'C' | b'G' | b'T'));
        let count = counts
            .entry((dna.len(), frames_with_stop(dna)))
            .or_default();
        count.0 += 1;
        count.1 += u64::from(ambiguous);
    }
    let mut out = io::BufWriter::new(io::stdout().lock());
    writeln!(out, "read_length\tframes_with_stop\treads\tambiguous_reads")?;
    for ((length, frames), (reads, ambiguous)) in counts {
        writeln!(out, "{length}\t{frames}\t{reads}\t{ambiguous}")?;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_stops() {
        assert_eq!(frames_with_stop(b"TAA"), 1);
        assert_eq!(frames_with_stop(b"TTA"), 1);
        assert_eq!(frames_with_stop(b"CCC"), 0);
        assert_eq!(frames_with_stop(b"taa"), 1);
        assert_eq!(frames_with_stop(b"NNN"), 0);
    }
    #[test]
    fn reverse_symmetry() {
        assert_eq!(
            frames_with_stop(b"ATGTAACCTTAG"),
            frames_with_stop(b"CTAAGGTTACAT")
        );
    }
}
