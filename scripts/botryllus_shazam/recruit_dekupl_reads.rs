//! Validate synchronized, four-line FASTQ records. gzip handles compressed integrity.
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::process::{Command, Stdio};

fn record<R: BufRead>(r: &mut R) -> io::Result<Option<[String; 4]>> {
    let mut lines = std::array::from_fn(|_| String::new());
    if r.read_line(&mut lines[0])? == 0 {
        return Ok(None);
    }
    for line in &mut lines[1..] {
        if r.read_line(line)? == 0 {
            return Err(io::Error::other("truncated FASTQ"));
        }
    }
    for line in &mut lines {
        line.truncate(line.trim_end_matches(['\r', '\n']).len());
    }
    if !lines[0].starts_with('@')
        || !lines[2].starts_with('+')
        || lines[1].is_empty()
        || lines[1].len() != lines[3].len()
        || !lines[1].bytes().all(|b| b"ACGTNacgtn".contains(&b))
        || !lines[3].bytes().all(|b| (33..=126).contains(&b))
    {
        return Err(io::Error::other("invalid FASTQ record"));
    }
    Ok(Some(lines))
}

fn read_id(header: &str) -> &str {
    let id = header.split_whitespace().next().unwrap_or("");
    id.strip_suffix("/1")
        .or_else(|| id.strip_suffix("/2"))
        .unwrap_or(id)
}

fn canonical(seq: &[u8]) -> Option<u64> {
    let (mut f, mut r) = (0u64, 0u64);
    if seq.len() != 31 {
        return None;
    }
    for (i, b) in seq.iter().enumerate() {
        let n = match b.to_ascii_uppercase() {
            b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => return None,
        };
        f = (f << 2) | n;
        r |= (3 - n) << (2 * i);
    }
    Some(f.min(r))
}
fn hits(seq: &[u8], keys: &HashMap<u64, String>) -> Vec<(usize, String, char)> {
    let (mut f, mut r, mut valid) = (0u64, 0u64, 0usize);
    let mask = (1u64 << 62) - 1;
    let mut out = Vec::new();
    for (i, b) in seq.iter().enumerate() {
        let n = match b.to_ascii_uppercase() {
            b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => {
                f = 0;
                r = 0;
                valid = 0;
                continue;
            }
        };
        f = ((f << 2) | n) & mask;
        r = (r >> 2) | ((3 - n) << 60);
        valid += 1;
        if valid >= 31 {
            if let Some(id) = keys.get(&f.min(r)) {
                out.push((i + 1 - 31, id.clone(), if f <= r { '+' } else { '-' }));
            }
        }
    }
    out
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        return Err("usage: recruit seeds.tsv R1 R2 out1.gz out2.gz hits.tsv.gz".into());
    }
    let mut keys = HashMap::new();
    for line in BufReader::new(File::open(&args[1])?).lines() {
        let line = line?;
        let p: Vec<_> = line.split('\t').collect();
        if p.len() != 2 {
            return Err("Invalid seed row".into());
        }
        let key = canonical(p[0].as_bytes()).ok_or("Invalid 31-mer")?;
        if keys.insert(key, p[1].to_string()).is_some() {
            return Err("Duplicate canonical seed".into());
        }
    }
    let mut a = Command::new("gzip")
        .args(["-dc", "--", &args[2]])
        .stdout(Stdio::piped())
        .spawn()?;
    let mut b = Command::new("gzip")
        .args(["-dc", "--", &args[3]])
        .stdout(Stdio::piped())
        .spawn()?;
    let mut ra = BufReader::with_capacity(1 << 20, a.stdout.take().unwrap());
    let mut rb = BufReader::with_capacity(1 << 20, b.stdout.take().unwrap());
    let mut children = Vec::new();
    let mut outputs = Vec::new();
    for path in &args[4..7] {
        let mut c = Command::new("gzip")
            .arg("-c")
            .stdin(Stdio::piped())
            .stdout(File::create(path)?)
            .spawn()?;
        outputs.push(BufWriter::new(c.stdin.take().unwrap()));
        children.push(c);
    }
    writeln!(
        outputs[2],
        "query_id\tread_id\tmate\tposition_0based\tstrand"
    )?;
    let (mut pairs, mut recruited, mut occurrences) = (0u64, 0u64, 0u64);
    loop {
        match (record(&mut ra)?, record(&mut rb)?) {
            (None, None) => break,
            (Some(x), Some(y)) if read_id(&x[0]) == read_id(&y[0]) => {
                pairs += 1;
                let hx = hits(x[1].as_bytes(), &keys);
                let hy = hits(y[1].as_bytes(), &keys);
                if !hx.is_empty() || !hy.is_empty() {
                    recruited += 1;
                    for line in &x {
                        writeln!(outputs[0], "{}", line)?;
                    }
                    for line in &y {
                        writeln!(outputs[1], "{}", line)?;
                    }
                    for (mate, hh) in [(1, hx), (2, hy)] {
                        for (pos, id, strand) in hh {
                            occurrences += 1;
                            writeln!(
                                outputs[2],
                                "{}\t{}\t{}\t{}\t{}",
                                id,
                                read_id(&x[0]),
                                mate,
                                pos,
                                strand
                            )?;
                        }
                    }
                }
            }
            _ => return Err("Mate IDs or counts differ".into()),
        }
    }
    drop(outputs);
    for mut c in children {
        if !c.wait()?.success() {
            return Err("gzip output failed".into());
        }
    }
    if !a.wait()?.success() || !b.wait()?.success() {
        return Err("gzip input failed".into());
    }
    println!(
        "{{\"input_pairs\":{},\"recruited_pairs\":{},\"occurrences\":{}}}",
        pairs, recruited, occurrences
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn forward_reverse_and_n_reset() {
        let seq = b"ACGTTGCATGTCAGTACGATCGTAGCTAGCA";
        let mut keys = HashMap::new();
        keys.insert(canonical(seq).unwrap(), "DK1".into());
        assert_eq!(hits(seq, &keys).len(), 1);
        let rc: Vec<u8> = seq
            .iter()
            .rev()
            .map(|b| match b {
                b'A' => b'T',
                b'C' => b'G',
                b'G' => b'C',
                _ => b'A',
            })
            .collect();
        assert_eq!(hits(&rc, &keys).len(), 1);
        let mut joined = seq.to_vec();
        joined.push(b'N');
        joined.extend(seq);
        let h = hits(&joined, &keys);
        assert_eq!(h.len(), 2);
        assert_eq!(h[1].0, 32);
        assert!(hits(b"ACGT", &keys).is_empty());
    }
}
