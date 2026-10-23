import sys,tempfile,csv,json,hashlib,subprocess
from pathlib import Path
script=Path(__file__).with_name('summarize_extensions.py')
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);src=root/'splash-trimmed';src.mkdir();out=root/'splash-trimmed-all';out.mkdir()
 def anchor(i):
  return ''.join('ACGT'[(i>>(2*j))&3] for j in range(31))
 anchors=[anchor(i) for i in range(213658)]
 scores=src/'splash.after_correction.scores.tsv'
 scores.write_text('anchor\tpval_opt_corrected\n'+''.join(a+'\t0.01\n' for a in anchors))
 sha=hashlib.sha256(scores.read_bytes()).hexdigest()
 for i in range(22):
  p=out/f'batch-{i:03d}';p.mkdir();selection=anchors[i*10000:(i+1)*10000]
  (p/'selected_anchors.tsv').write_text('anchor\n'+'\n'.join(selection)+'\n')
  text='anchor\tid\tcompactor\n'
  if i==0:text+=f'{selection[0]}\t0\t{selection[0]}AAAA\n'
  (p/'compactors.tsv').write_text(text)
  (p/'validated.json').write_text(json.dumps(dict(batch=i,source_sha256=sha,output_sha256=hashlib.sha256(text.encode()).hexdigest(),selected_anchors=len(selection),output_rows=int(i==0),anchors_extended=int(i==0))))
 subprocess.run([sys.executable,str(script),str(root)],check=True,stdout=subprocess.DEVNULL)
 summary=json.loads((out/'summary.json').read_text());assert summary['extended_records']==1 and summary['anchors_without_extension']==213657
 (out/'batch-001/compactors.tsv').write_text('tampered')
 failure=subprocess.run([sys.executable,str(script),str(root)],capture_output=True)
 assert failure.returncode!=0
 print('Complete 213,658-anchor synthetic partition and output-tampering rejection passed')
