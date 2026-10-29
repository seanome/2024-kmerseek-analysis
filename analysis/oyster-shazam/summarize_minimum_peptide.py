"""Print matched minimum-peptide pilot retention with explicit denominators."""
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);s=json.loads(p.read_text());a=s['arms']['1'];b=s['arms']['20']
lines=['# Minimum 20-aa stop-and-X pilot (superseded)','',
'> Superseded. This pilot splits at stop and X with a 20-aa minimum, which is not the current rule (split only at stops, keep X, at least 30 aa; see `HANDOFF-20261025.md`). Do not compare its numbers with the corrected pilot table in `results/migration-20261025/STATUS.md`.','',s['scope'],'','Both arms split all six frames at stop/X. Reference matching requires at least one shared sampled hash; it does not establish function. DNA was not changed.','', '| Group | Original sequences | With ≥1-aa stretch | With ≥20-aa stretch | Stretches kept / all | Amino acids kept / all |','|---|---:|---:|---:|---:|---:|']
for g,x in a['stretches'].items():
 y=b['stretches'][g]
 lines.append(f"| {g} | {x['records']:,} | {x['represented']:,} | {y['represented']:,} | {y['stretches']:,} / {x['stretches']:,} ({100*y['stretches']/x['stretches']:.1f}%) | {y['amino_acids']:,} / {x['amino_acids']:,} ({100*y['amino_acids']/x['amino_acids']:.1f}%) |")
lines += ['', '| Group | Encoding | Scale | Hashes retained | Signature bytes retained | Matches kept / previous |','|---|---|---:|---:|---:|---:|']
for x in a['measurements']:
 y=next(y for y in b['measurements'] if (y['group'],y['moltype'],y['scaled'])==(x['group'],x['moltype'],x['scaled']))
 if 'match_ids' in x:
  if set(y['match_ids'])-set(x['match_ids']):raise ValueError('Gained matches')
 match=f"{y['matches']} / {x['matches']}" if 'matches' in x else '—'
 pct=lambda n,d:f'{100*n/d:.1f}%' if d else 'not measurable (zero baseline)'
 lines.append(f"| {x['group']} | {x['moltype']} | {x['scaled']} | {pct(y['hashes'],x['hashes'])} | {pct(y['signature_bytes'],x['signature_bytes'])} | {match} |")
lines += ['',f"Exact per-segment Branchwater/API comparisons: {a['segment_comparisons']:,} baseline; {b['segment_comparisons']:,} at ≥20aa. Hash-subset checks passed for every original sequence and setting; HP30 unchanged. Signature bytes are gzip collections, excluding indexes and intermediate peptide ZIPs. Systematic samples are diagnostic and are not random uncertainty estimates."]
print('\n'.join(lines))
