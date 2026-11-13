//! kmer-codes and count-kmers: the two halves of the Botryllus-to-human DE k-mer test.
//!
//! `kmer-codes` lists the distinct k-mer codes of a set of peptides, so the k-mers of the Botryllus
//! reads that went up and of those that went down can be listed apart. `count-kmers` counts how often each of a given set of codes occurs
//! in a run's framed reads, so human groups can be compared. Both use `for_each_kmer`, so a code
//! means the same k-mer in both. Several alphabet-k settings go in one pass over the reads.
use std::collections::HashMap;
use std::fs::File;
use std::hash::BuildHasherDefault;
use std::io::{BufRead, BufReader, BufWriter, Write};

use crate::best_frame::{check_k_fits, for_each_kmer, KmerSet, MulHasher};
use crate::Res;

type Counts = HashMap<u64, u64, BuildHasherDefault<MulHasher>>;

/// Parses `protein:7,dayhoff:13,hp:25`.
fn settings(s: &str) -> Res<Vec<(String, usize)>> {
    s.split(',')
        .map(|x| {
            let (a, k) = x.split_once(':').ok_or("setting must be alphabet:k")?;
            let k: usize = k.parse()?;
            check_k_fits(a, k)?;
            Ok((a.to_string(), k))
        })
        .collect()
}

/// Calls `f(id, seq)` for every FASTA record read from `r`.
fn for_each_record(r: impl BufRead, mut f: impl FnMut(&str, &[u8]) -> Res<()>) -> Res<()> {
    let (mut id, mut seq) = (String::new(), Vec::new());
    for line in r.lines() {
        let line = line?;
        if let Some(h) = line.strip_prefix('>') {
            if !id.is_empty() {
                f(&id, &seq)?;
            }
            id = h.split_whitespace().next().unwrap_or("").to_string();
            seq.clear();
        } else {
            seq.extend(line.trim().bytes());
        }
    }
    if !id.is_empty() {
        f(&id, &seq)?;
    }
    Ok(())
}

pub fn cmd_kmer_codes(args: &[String]) -> Res<()> {
    if args.len() != 1 {
        return Err("usage: pec-tools kmer-codes <alphabet:k[,alphabet:k..]> < peptides.faa > alphabet_k_code.tsv".into());
    }
    // Distinct codes over the whole input, per setting, sorted.
    let sets = settings(&args[0])?;
    let mut seen: Vec<KmerSet> = sets.iter().map(|_| KmerSet::default()).collect();
    let stdin = std::io::stdin();
    for_each_record(BufReader::with_capacity(1 << 22, stdin.lock()), |_, seq| {
        for (j, (a, k)) in sets.iter().enumerate() {
            let s = &mut seen[j];
            for_each_kmer(seq, a, *k, |c| {
                s.insert(c);
            });
        }
        Ok(())
    })?;
    let mut out = BufWriter::with_capacity(1 << 20, std::io::stdout().lock());
    writeln!(out, "alphabet\tk\tcode")?;
    for (j, (a, k)) in sets.iter().enumerate() {
        let mut v: Vec<u64> = seen[j].iter().copied().collect();
        v.sort_unstable();
        for c in v {
            writeln!(out, "{a}\t{k}\t{c}")?;
        }
    }
    out.flush()?;
    Ok(())
}

pub fn cmd_count_kmers(args: &[String]) -> Res<()> {
    if args.len() != 2 {
        return Err("usage: pec-tools count-kmers <alphabet_k_code.tsv> <out.tsv> < framed.faa".into());
    }
    // One table of counts per setting, holding only the codes asked for.
    let mut keys: Vec<(String, usize)> = Vec::new();
    let mut tables: Vec<Counts> = Vec::new();
    for (i, line) in BufReader::new(File::open(&args[0])?).lines().enumerate() {
        let line = line?;
        let f: Vec<&str> = line.split('\t').collect();
        if i == 0 || f.len() < 3 {
            continue; // header
        }
        let (a, k, c) = (f[0], f[1].parse::<usize>()?, f[2].parse::<u64>()?);
        let j = match keys.iter().position(|(ka, kk)| ka == a && *kk == k) {
            Some(j) => j,
            None => {
                check_k_fits(a, k)?;
                keys.push((a.to_string(), k));
                tables.push(Counts::default());
                keys.len() - 1
            }
        };
        tables[j].insert(c, 0);
    }
    // Valid windows per setting: the run's size in k-mers, to normalise counts between runs.
    let mut windows = vec![0u64; keys.len()];
    let stdin = std::io::stdin();
    for_each_record(BufReader::with_capacity(1 << 22, stdin.lock()), |_, seq| {
        for (j, (a, k)) in keys.iter().enumerate() {
            let t = &mut tables[j];
            let w = &mut windows[j];
            for_each_kmer(seq, a, *k, |c| {
                *w += 1;
                if let Some(n) = t.get_mut(&c) {
                    *n += 1;
                }
            });
        }
        Ok(())
    })?;
    let mut out = BufWriter::new(File::create(&args[1])?);
    writeln!(out, "alphabet\tk\tcode\tcount")?;
    for (j, (a, k)) in keys.iter().enumerate() {
        // code -1 holds the run's total valid windows for this setting
        writeln!(out, "{a}\t{k}\t-1\t{}", windows[j])?;
        let mut rows: Vec<(&u64, &u64)> = tables[j].iter().filter(|(_, n)| **n > 0).collect();
        rows.sort_unstable();
        for (c, n) in rows {
            writeln!(out, "{a}\t{k}\t{c}\t{n}")?;
        }
    }
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_parse_and_reject_overflow() {
        assert_eq!(settings("protein:7,hp:25").unwrap(), vec![("protein".into(), 7), ("hp".into(), 25)]);
        assert!(settings("protein:20").is_err());
        assert!(settings("protein").is_err());
    }

    #[test]
    fn records_are_split_at_headers() {
        let mut got = Vec::new();
        for_each_record(&b">a x\nMK\nV\n>b\nW\n"[..], |id, s| {
            got.push((id.to_string(), s.to_vec()));
            Ok(())
        })
        .unwrap();
        assert_eq!(got, vec![("a".into(), b"MKV".to_vec()), ("b".into(), b"W".to_vec())]);
    }
}
