//! Build one k-mer sketch per RNA-seq run, and measure how much of each protein it contains.
//!
//! `sketch`: read a run's FASTQ files, translate every read in six frames
//! (`translate.rs`), keep pieces of at least 30 residues, and store every k-mer hash of every
//! alphabet in `k_for_protein7_bits.tsv` (scaled = 1, so no hash is left out). Writes
//! `<out_dir>/<accession>.sig.gz`, a gzipped JSON list in sourmash's signature layout with
//! one signature per alphabet, and one row of counts to stdout for `index_log.tsv`.
//!
//! `query`: for each protein in a FASTA and each alphabet, the containment of the protein
//! in each sketch: shared distinct hashes / the protein's distinct hashes.

mod kmers;
mod translate;

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use flate2::{read::MultiGzDecoder, write::GzEncoder, Compression};
use needletail::parse_fastx_file;
use rayon::prelude::*;
use serde::Deserialize;

use kmers::Alphabet;
use translate::{kept_translations, CodonTable, RULE_NAMES};

const READS_PER_BATCH: usize = 200_000;

/// The alphabets and their k, from `k_for_protein7_bits.tsv`.
fn read_alphabets(path: &Path) -> Result<Vec<Alphabet>> {
    let mut out = Vec::new();
    let mut header: Option<Vec<String>> = None;
    for line in BufReader::new(File::open(path)?).lines() {
        let line = line?;
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let cols: Vec<String> = line.split('\t').map(str::to_string).collect();
        match &header {
            None => header = Some(cols),
            Some(h) => {
                let get = |name: &str| -> Result<&str> {
                    let i = h.iter().position(|c| c == name).context(name.to_string())?;
                    Ok(cols[i].as_str())
                };
                out.push(Alphabet::new(get("alphabet")?, get("k")?.parse()?)?);
            }
        }
    }
    Ok(out)
}

/// sourmash's name for the three alphabets it encodes itself; kmerseek's name otherwise.
fn molecule(name: &str) -> &str {
    match name {
        "protein20" => "protein",
        "dayhoff6" => "dayhoff",
        "hp_lehninger2" => "hp",
        other => other,
    }
}

/// sourmash's md5sum of a sketch: md5 over the ksize, then each hash, as decimal text.
fn sourmash_md5(ksize: usize, mins: &[u64]) -> String {
    let mut ctx = md5::Context::new();
    ctx.consume(ksize.to_string());
    for m in mins {
        ctx.consume(m.to_string());
    }
    format!("{:x}", ctx.compute())
}

fn write_sig(path: &Path, name: &str, alphabets: &[Alphabet], sets: Vec<HashSet<u64>>) -> Result<Vec<usize>> {
    let tmp = path.with_extension("tmp");
    let mut w = BufWriter::new(GzEncoder::new(File::create(&tmp)?, Compression::fast()));
    write!(w, "[{{\"class\":\"sourmash_signature\",\"email\":\"\",\"hash_function\":\"0.murmur64\",\
               \"filename\":\"{name}\",\"name\":\"{name}\",\"license\":\"CC0\",\"signatures\":[")?;
    let mut sizes = Vec::new();
    for (i, (a, set)) in alphabets.iter().zip(sets).enumerate() {
        let mut mins: Vec<u64> = set.into_iter().collect();
        mins.sort_unstable();
        // sourmash stores a protein sketch's ksize in nucleotides, 3 per residue.
        let ksize = 3 * a.k;
        if i > 0 {
            write!(w, ",")?;
        }
        write!(w, "{{\"num\":0,\"ksize\":{ksize},\"seed\":{},\"max_hash\":{},\"md5sum\":\"{}\",\
                   \"molecule\":\"{}\",\"mins\":[",
               kmerseek::SEED, u64::MAX, sourmash_md5(ksize, &mins), molecule(&a.name))?;
        for (j, m) in mins.iter().enumerate() {
            if j > 0 {
                write!(w, ",")?;
            }
            write!(w, "{m}")?;
        }
        write!(w, "]}}")?;
        sizes.push(mins.len());
    }
    write!(w, "],\"version\":0.4}}]")?;
    w.into_inner()?.finish()?;
    std::fs::rename(&tmp, path)?;
    Ok(sizes)
}

fn sketch(accession: &str, alphabets_tsv: &Path, out_dir: &Path, fastqs: &[PathBuf]) -> Result<()> {
    let t0 = Instant::now();
    let alphabets = read_alphabets(alphabets_tsv)?;
    let table = CodonTable::new();
    let mut sets: Vec<HashSet<u64>> = alphabets.iter().map(|_| HashSet::new()).collect();
    let mut counts = [0u64; 5];
    let (mut reads, mut reads_kept, mut pieces) = (0u64, 0u64, 0u64);
    for fq in fastqs {
        let mut reader = parse_fastx_file(fq).with_context(|| format!("{}", fq.display()))?;
        let mut batch: Vec<Vec<u8>> = Vec::with_capacity(READS_PER_BATCH);
        loop {
            let rec = reader.next();
            let done = rec.is_none();
            if let Some(rec) = rec {
                batch.push(rec?.seq().into_owned());
            }
            if batch.len() == READS_PER_BATCH || (done && !batch.is_empty()) {
                // translate in parallel, then let each alphabet fill its own set in parallel
                let translated: Vec<(Vec<Vec<u8>>, [u64; 5])> = batch
                    .par_iter()
                    .map(|r| {
                        let mut c = [0u64; 5];
                        let k = kept_translations(&table, r, &mut c);
                        (k.into_iter().map(|p| p.peptide).collect(), c)
                    })
                    .collect();
                for (peps, c) in &translated {
                    reads += 1;
                    reads_kept += (!peps.is_empty()) as u64;
                    pieces += peps.len() as u64;
                    for i in 0..5 {
                        counts[i] += c[i];
                    }
                }
                alphabets.par_iter().zip(sets.par_iter_mut()).for_each(|(a, set)| {
                    let mut buf = Vec::new();
                    for (peps, _) in &translated {
                        for p in peps {
                            a.for_each_hash(p, &mut buf, |h| {
                                set.insert(h);
                            });
                        }
                    }
                });
                batch.clear();
            }
            if done {
                break;
            }
        }
    }
    std::fs::create_dir_all(out_dir)?;
    let sig = out_dir.join(format!("{accession}.sig.gz"));
    let sizes = write_sig(&sig, accession, &alphabets, sets)?;
    let mut row = format!(
        "{accession}\t{}\t{reads}\t{reads_kept}\t{pieces}\t{}\t{:.1}",
        fastqs.len(),
        counts.iter().map(u64::to_string).collect::<Vec<_>>().join("\t"),
        t0.elapsed().as_secs_f64()
    );
    for (a, n) in alphabets.iter().zip(sizes) {
        row.push_str(&format!("\t{}:{}={n}", a.name, a.k));
    }
    println!("accession\tfiles\treads\treads_with_kept_piece\tkept_pieces\t{}\tseconds\thashes_per_alphabet",
             RULE_NAMES.join("\t"));
    println!("{row}");
    Ok(())
}

#[derive(Deserialize)]
struct SigFile {
    name: String,
    signatures: Vec<SigEntry>,
}

#[derive(Deserialize)]
struct SigEntry {
    ksize: usize,
    molecule: String,
    mins: Vec<u64>,
}

fn read_proteins(path: &Path) -> Result<Vec<(String, Vec<u8>)>> {
    let mut out = Vec::new();
    let mut reader = parse_fastx_file(path)?;
    while let Some(rec) = reader.next() {
        let rec = rec?;
        let id = String::from_utf8_lossy(rec.id()).split_whitespace().next().unwrap_or("").to_string();
        out.push((id, rec.seq().to_ascii_uppercase()));
    }
    Ok(out)
}

fn query(proteins_fa: &Path, alphabets_tsv: &Path, sigs: &[PathBuf]) -> Result<()> {
    let alphabets = read_alphabets(alphabets_tsv)?;
    let proteins = read_proteins(proteins_fa)?;
    // each protein's distinct hashes, per alphabet
    let protein_sets: Vec<Vec<HashSet<u64>>> = alphabets
        .par_iter()
        .map(|a| {
            let mut buf = Vec::new();
            proteins
                .iter()
                .map(|(_, s)| {
                    let mut set = HashSet::new();
                    a.for_each_hash(s, &mut buf, |h| {
                        set.insert(h);
                    });
                    set
                })
                .collect()
        })
        .collect();
    let mut out = BufWriter::new(std::io::stdout());
    writeln!(out, "protein\taccession\talphabet\tk\tprotein_kmers\tshared_kmers\tcontainment")?;
    for sig in sigs {
        let f: Vec<SigFile> = serde_json::from_reader(BufReader::new(MultiGzDecoder::new(File::open(sig)?)))?;
        let f = &f[0];
        for (ai, a) in alphabets.iter().enumerate() {
            let e = f
                .signatures
                .iter()
                .find(|e| e.ksize == 3 * a.k && e.molecule == molecule(&a.name))
                .with_context(|| format!("{}: no {} k={}", sig.display(), a.name, a.k))?;
            if !e.mins.windows(2).all(|w| w[0] < w[1]) {
                bail!("{}: hashes not sorted", sig.display());
            }
            for (pi, (pid, _)) in proteins.iter().enumerate() {
                let q = &protein_sets[ai][pi];
                let shared = q.iter().filter(|h| e.mins.binary_search(h).is_ok()).count();
                let c = if q.is_empty() { f64::NAN } else { shared as f64 / q.len() as f64 };
                writeln!(out, "{pid}\t{}\t{}\t{}\t{}\t{shared}\t{c:.6}", f.name, a.name, a.k, q.len())?;
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "expression-indexer sketch ACCESSION ALPHABETS_TSV OUT_DIR FASTQ [FASTQ]\n\
                 expression-indexer query PROTEINS_FASTA ALPHABETS_TSV SIG [SIG ...]";
    match args.get(1).map(String::as_str) {
        Some("sketch") if args.len() >= 6 => {
            let fq: Vec<PathBuf> = args[5..].iter().map(PathBuf::from).collect();
            sketch(&args[2], Path::new(&args[3]), Path::new(&args[4]), &fq)
        }
        Some("query") if args.len() >= 5 => {
            let sigs: Vec<PathBuf> = args[4..].iter().map(PathBuf::from).collect();
            query(Path::new(&args[2]), Path::new(&args[3]), &sigs)
        }
        _ => bail!("usage:\n{usage}"),
    }
}
