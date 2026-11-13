"""Capture each new external command without conflating CPU allocation with CPU use."""
import json
import os
from pathlib import Path
import subprocess
import time


def run_timed(args, log):
    log = Path(log)
    command = list(map(str, args))
    usage = log.with_suffix(log.suffix + '.resource.tsv')
    record = log.with_suffix(log.suffix + '.resource.json')
    started = time.time()
    with log.open('w') as stream:
        completed = subprocess.run(['/usr/bin/time', '-f', '%e\t%U\t%S\t%M\t%x',
            '-o', str(usage), *command], stdout=stream, stderr=subprocess.STDOUT)
    raw = usage.read_text().splitlines() if usage.exists() else []
    fields = raw[-1].split('\t') if raw else []
    measured = len(fields) == 5
    result = dict(command=command, job_id=os.environ.get('SLURM_JOB_ID'),
        array_task=os.environ.get('SLURM_ARRAY_TASK_ID'),
        allocated_cpus=os.environ.get('SLURM_CPUS_PER_TASK'),
        started_unix=started, finished_unix=time.time(), exit_code=completed.returncode,
        measurement='GNU time command-level; do not sum with parent Slurm CPU time',
        wall_seconds=float(fields[0]) if measured else None,
        user_cpu_seconds=float(fields[1]) if measured else None,
        system_cpu_seconds=float(fields[2]) if measured else None,
        peak_rss_kib=int(fields[3]) if measured else None)
    record.write_text(json.dumps(result, indent=2)+'\n')
    completed.check_returncode()
