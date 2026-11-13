"""Rename Salmon columns for shared aggregation; retain quantifier provenance."""
import argparse
import csv
import json
import math
from pathlib import Path


def convert(source, out):
    meta = json.loads((source / 'aux_info/meta_info.json').read_text())
    processed, mapped = meta['num_processed'], meta['num_mapped']
    if processed <= 0 or not 0 <= mapped <= processed:
        raise ValueError('Invalid fragment counts')
    with (source / 'quant.sf').open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    names = [r['Name'] for r in rows]
    if not rows or len(names) != len(set(names)):
        raise ValueError('Empty or duplicate transcript table')
    for r in rows:
        for key in ('TPM', 'NumReads', 'EffectiveLength', 'Length'):
            v = float(r[key])
            if not math.isfinite(v) or v < 0:
                raise ValueError('Invalid abundance or length')
    out.mkdir(parents=True, exist_ok=False)
    keys = ['Name', 'Length', 'EffectiveLength', 'TPM', 'NumReads']
    with (out / 'abundance.tsv').open('w') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['target_id', 'length', 'eff_length', 'tpm', 'est_counts'])
        w.writerows([[r[k] for k in keys] for r in rows])
    (out / 'run_info.json').write_text(json.dumps({
        'n_processed': processed, 'n_pseudoaligned': mapped,
        'p_pseudoaligned': 100 * mapped / processed,
        'source_quantifier': 'salmon', 'source': str(source),
        'note': 'Compatibility names; mapped means Salmon assigned fragments'
    }, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    convert(a.source, a.out)
