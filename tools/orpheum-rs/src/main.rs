//! Standalone, reference-aware six-frame read screening; no assembly or alignment.
mod index;
mod sequence;
use clap::{Parser, Subcommand, ValueEnum};
use index::{words, Alphabet, Index};
use rayon::prelude::*;
use sequence::{six_frames, Record, Records};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(Parser)]
#[command(
    version,
    about = "Streaming six-frame protein-reference screening. Scores are approximate word containment, not Jaccard or gene assignments."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Build a reusable fixed-memory Bloom index from a protein FASTA/FASTQ.
    Index {
        reference: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, value_enum, default_value = "protein")]
        alphabet: Alphabet,
        #[arg(short, long)]
        k: Option<u8>,
        #[arg(long, default_value_t = 64)]
        memory_mib: usize,
    },
    /// Score all frames against a protein reference; emit every qualifying stop-free segment.
    Translate {
        #[arg(short, long)]
        index: PathBuf,
        #[arg(required = true)]
        reads: Vec<PathBuf>,
        #[arg(long)]
        threshold: Option<f64>,
        #[arg(long, value_enum, default_value = "words")]
        stop_policy: StopPolicy,
        #[arg(long)]
        scores: Option<PathBuf>,
        #[arg(long)]
        peptides: Option<PathBuf>,
        #[arg(long)]
        unclassified: Option<PathBuf>,
        #[arg(long, default_value_t = 4)]
        threads: usize,
        #[arg(long, default_value_t = 0.01)]
        max_index_fpr: f64,
    },
    /// Emit stop-free six-frame segments without a reference-dependent coding filter.
    Sixframe {
        #[arg(required = true)]
        reads: Vec<PathBuf>,
        #[arg(long, default_value_t = 7)]
        min_aa: usize,
        #[arg(long, value_enum, default_value = "words")]
        stop_policy: StopPolicy,
    },
}
#[derive(Clone, Copy, Debug, ValueEnum)]
enum StopPolicy {
    Words,
    Frame,
}
struct FrameScore {
    frame: i8,
    unique: usize,
    hits: usize,
    fraction: Option<f64>,
    status: &'static str,
    peptide: Vec<u8>,
}
fn writer(path: Option<&Path>) -> io::Result<Box<dyn Write>> {
    Ok(if let Some(p) = path {
        Box::new(BufWriter::new(
            File::options().write(true).create_new(true).open(p)?,
        ))
    } else {
        Box::new(BufWriter::new(io::stdout()))
    })
}
fn invalid(s: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, s)
}
fn fasta(
    out: &mut dyn Write,
    id: &str,
    frame: i8,
    peptide: &[u8],
    min: usize,
) -> io::Result<usize> {
    let mut n = 0;
    for (segment, p) in peptide.split(|b| *b == b'*' || *b == b'X').enumerate() {
        if p.len() >= min {
            writeln!(
                out,
                ">{id}|frame={frame}|segment={segment}\n{}",
                String::from_utf8_lossy(p)
            )?;
            n += 1
        }
    }
    Ok(n)
}
fn score(record: &Record, index: &Index, threshold: f64, policy: StopPolicy) -> Vec<FrameScore> {
    let dna = &record.sequence;
    let complexity = dna.windows(2).filter(|w| w[0] != w[1]).count() as f64 / dna.len() as f64;
    six_frames(dna)
        .into_iter()
        .enumerate()
        .map(|(i, peptide)| {
            let frame = if i < 3 {
                (i + 1) as i8
            } else {
                -((i - 2) as i8)
            };
            let mut result = FrameScore {
                frame,
                unique: 0,
                hits: 0,
                fraction: None,
                status: "unclassified",
                peptide,
            };
            if complexity < 0.3 {
                result.status = "low_complexity_nucleotide";
                return result;
            }
            if matches!(policy, StopPolicy::Frame) && result.peptide.contains(&b'*') {
                result.status = "frame_has_stop";
                return result;
            }
            let mut keys = words(&result.peptide, index.meta.alphabet, index.meta.k);
            let possible = keys.len();
            keys.sort_unstable();
            keys.dedup();
            result.unique = keys.len();
            if keys.is_empty() {
                result.status = "no_valid_words";
                return result;
            }
            if keys.len() * 2 <= possible {
                result.status = "low_complexity_peptide";
                return result;
            }
            result.hits = keys.iter().filter(|&&w| index.contains(w)).count();
            let fraction = result.hits as f64 / keys.len() as f64;
            result.fraction = Some(fraction);
            if result.hits > 0 && fraction >= threshold {
                result.status = "reference_supported"
            }
            result
        })
        .collect()
}
fn run() -> io::Result<()> {
    let cli = Cli::parse();
    let start = Instant::now();
    match cli.command {
        Command::Index {
            reference,
            output,
            alphabet,
            k,
            memory_mib,
        } => {
            let k = k.unwrap_or(match alphabet {
                Alphabet::Protein => 7,
                Alphabet::Dayhoff => 12,
                Alphabet::Hp => 30,
            });
            let mut index = Index::new(alphabet, k, memory_mib)?;
            let mut source = Records::open(&reference)?;
            let mut digest = Sha256::new();
            while let Some(r) = source.next_record()? {
                digest.update(r.id.as_bytes());
                digest.update([0]);
                digest.update(&r.sequence);
                digest.update([0]);
                for word in words(&r.sequence, alphabet, k) {
                    index.insert(word)
                }
                index.meta.reference_records += 1;
            }
            if index.meta.reference_records == 0 {
                return Err(invalid("Reference has no records"));
            }
            index.meta.reference_sha256 = format!("{:x}", digest.finalize());
            index.save(&output)?;
            eprintln!(
                "{}",
                serde_json::json!({"index":index.meta,"seconds":start.elapsed().as_secs_f64()})
            );
        }
        Command::Sixframe {
            reads,
            min_aa,
            stop_policy,
        } => {
            if min_aa == 0 {
                return Err(invalid("min-aa must be positive"));
            }
            let mut out = writer(None)?;
            let mut records = 0;
            let mut segments = 0;
            for path in reads {
                let mut source = Records::open(&path)?;
                while let Some(r) = source.next_record()? {
                    records += 1;
                    for (i, p) in six_frames(&r.sequence).iter().enumerate() {
                        if matches!(stop_policy, StopPolicy::Frame) && p.contains(&b'*') {
                            continue;
                        }
                        let f = if i < 3 {
                            (i + 1) as i8
                        } else {
                            -((i - 2) as i8)
                        };
                        segments += fasta(&mut out, &r.id, f, p, min_aa)?
                    }
                }
            }
            out.flush()?;
            eprintln!(
                "{}",
                serde_json::json!({"records":records,"segments":segments,"seconds":start.elapsed().as_secs_f64(),"stop_policy":format!("{stop_policy:?}")})
            );
        }
        Command::Translate {
            index,
            reads,
            threshold,
            stop_policy,
            scores,
            peptides,
            unclassified,
            threads,
            max_index_fpr,
        } => {
            if threads == 0 {
                return Err(invalid("threads must be positive"));
            }
            let index = Index::load(&index)?;
            if !max_index_fpr.is_finite()
                || !(0.0..=1.0).contains(&max_index_fpr)
                || index.meta.estimated_fpr > max_index_fpr
            {
                return Err(invalid(
                    "Bloom filter exceeds permitted estimated false-positive rate",
                ));
            }
            let threshold = threshold.unwrap_or(if index.meta.alphabet == Alphabet::Hp {
                0.8
            } else {
                0.5
            });
            if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
                return Err(invalid("threshold must be finite and between zero and one"));
            }
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| io::Error::other(e.to_string()))?;
            let mut peptides = writer(peptides.as_deref())?;
            let mut scores = match scores {
                Some(p) => Some(writer(Some(&p))?),
                None => None,
            };
            let mut unclassified = match unclassified {
                Some(p) => Some(writer(Some(&p))?),
                None => None,
            };
            if let Some(out) = &mut scores {
                writeln!(out,"source\tread_id\tframe\tunique_words\treference_hits\tmembership_fraction\tstatus")?
            }
            let mut n = 0usize;
            let mut supported = 0usize;
            let mut segments = 0;
            for path in reads {
                let mut source = Records::open(&path)?;
                loop {
                    let mut batch = Vec::new();
                    let mut bases = 0;
                    while batch.len() < 4096 && bases < 8_000_000 {
                        match source.next_record()? {
                            Some(r) => {
                                bases += r.sequence.len();
                                batch.push(r)
                            }
                            None => break,
                        }
                    }
                    if batch.is_empty() {
                        break;
                    }
                    let results: Vec<_> = pool.install(|| {
                        batch
                            .par_iter()
                            .map(|r| score(r, &index, threshold, stop_policy))
                            .collect()
                    });
                    for (record, frames) in batch.iter().zip(results) {
                        n += 1;
                        let mut accepted = false;
                        for frame in frames {
                            if let Some(out) = &mut scores {
                                let fraction = frame
                                    .fraction
                                    .map(|x| format!("{x:.8}"))
                                    .unwrap_or_default();
                                writeln!(
                                    out,
                                    "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                                    path.display(),
                                    record.id,
                                    frame.frame,
                                    frame.unique,
                                    frame.hits,
                                    fraction,
                                    frame.status
                                )?
                            }
                            if frame.status == "reference_supported" {
                                accepted = true;
                                segments += fasta(
                                    &mut peptides,
                                    &record.id,
                                    frame.frame,
                                    &frame.peptide,
                                    usize::from(index.meta.k),
                                )?
                            }
                        }
                        if accepted {
                            supported += 1
                        } else if let Some(out) = &mut unclassified {
                            writeln!(
                                out,
                                ">{}\n{}",
                                record.id,
                                String::from_utf8_lossy(&record.sequence)
                            )?
                        }
                    }
                }
            }
            peptides.flush()?;
            if let Some(out) = &mut scores {
                out.flush()?
            }
            if let Some(out) = &mut unclassified {
                out.flush()?
            }
            eprintln!(
                "{}",
                serde_json::json!({"records":n,"reference_supported_reads":supported,"peptide_segments":segments,"seconds":start.elapsed().as_secs_f64(),"threads":threads,"threshold":threshold,"stop_policy":format!("{stop_policy:?}"),"index":index.meta})
            );
        }
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn word_policy_differs_from_frame_policy() {
        let r = Record {
            id: "read".into(),
            sequence: b"ATGTAAGCTGATGAA".to_vec(),
        };
        let mut index = Index::new(Alphabet::Protein, 2, 1).unwrap();
        for w in words(b"ADE", Alphabet::Protein, 2) {
            index.insert(w)
        }
        let a = score(&r, &index, 0.5, StopPolicy::Words);
        let b = score(&r, &index, 0.5, StopPolicy::Frame);
        assert_eq!(a[0].status, "reference_supported");
        assert_eq!(b[0].status, "frame_has_stop");
    }
}
