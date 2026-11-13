//! Validate synchronized, four-line FASTQ records. gzip handles compressed integrity.
use std::io::{self, BufRead, BufReader};
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

fn validate<A: BufRead, B: BufRead>(a: &mut A, b: &mut B) -> io::Result<(u64, u64)> {
    let (mut pairs, mut bases) = (0, 0);
    loop {
        match (record(a)?, record(b)?) {
            (None, None) => return Ok((pairs, bases)),
            (Some(x), Some(y)) if read_id(&x[0]) == read_id(&y[0]) => {
                pairs += 1;
                bases += (x[1].len() + y[1].len()) as u64;
            }
            _ => return Err(io::Error::other("mate counts or identifiers differ")),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: validate_pairs R1.fastq.gz R2.fastq.gz".into());
    }
    let mut a = Command::new("gzip")
        .args(["-dc", "--", &args[1]])
        .stdout(Stdio::piped())
        .spawn()?;
    let mut b = Command::new("gzip")
        .args(["-dc", "--", &args[2]])
        .stdout(Stdio::piped())
        .spawn()?;
    let result = validate(
        &mut BufReader::with_capacity(1 << 20, a.stdout.take().unwrap()),
        &mut BufReader::with_capacity(1 << 20, b.stdout.take().unwrap()),
    );
    if result.is_err() {
        let _ = a.kill();
        let _ = b.kill();
    }
    let status_a = a.wait()?;
    let status_b = b.wait()?;
    let (pairs, bases) = result?;
    if !status_a.success() || !status_b.success() {
        return Err("gzip integrity check failed".into());
    }
    println!("{{\"pairs\":{pairs},\"bases\":{bases}}}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn synchronized_and_empty() {
        let a = b"@one/1\nAC\n+\nII\n";
        let b = b"@one/2\nTG\n+\nII\n";
        assert_eq!(validate(&mut &a[..], &mut &b[..]).unwrap(), (1, 4));
        assert_eq!(validate(&mut &b""[..], &mut &b""[..]).unwrap(), (0, 0));
    }
    #[test]
    fn rejects_mismatch_truncation_and_bad_quality() {
        for b in [
            b"@two\nAC\n+\nII\n".as_slice(),
            b"@one\nAC\n+\nI\n",
            b"@one\nAC\n+",
        ] {
            assert!(validate(&mut &b"@one\nAC\n+\nII\n"[..], &mut &b[..]).is_err());
        }
        assert!(validate(&mut &b"@one\nAC\n+\nII\n"[..], &mut &b""[..]).is_err());
    }
}
