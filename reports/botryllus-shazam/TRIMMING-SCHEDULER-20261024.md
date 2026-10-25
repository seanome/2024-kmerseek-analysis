# Full-depth trimming scheduler check

24 October 2026, 18:39 PDT: job 49368820 was pending and eligible, with no dependency, no hold, no node constraint and a positive priority. Its download predecessor 49368817 completed successfully. No nf-core output directory existed yet, so trimming had not started.

The job requested 16 CPUs, 48 GB and 24 hours on normal only. The partition inventory reported 59,258 queued CPU cores on normal, compared with 24,796 on hns. These totals describe demand, not the job's position or an estimated start time. Slurm reported Reason=None and StartTime=Unknown; a specific reservation or resource conflict was not identified.

An in-place request to allow hns and normal returned exit status 0, but the follow-up job record still reported normal only. The requested partition change is therefore not verified and must not be reported as applied. Its ID, CPU count, memory, time limit, inputs and downstream dependencies remain unchanged. No duplicate job was submitted. The source sbatch header remains normal until broader eligibility can be confirmed. No resources were reduced: there is no completed full-depth trimming trace supporting a smaller request.

SPLASH 49368823 and DE-kupl / gene DE 49369243 depend on validated trimming. manysearch 49368824 follows SPLASH. Evidence 49369247 and paper-gene recovery 49373978 follow their required analyses; the latter also waits for manysearch. These dependencies remain intentional and should not be bypassed.

18:40 PDT follow-up: trimming remained PENDING / Reason=None / StartTime=Unknown on normal. All 64 raw libraries have validated completion records. The existing dependency chain remains pending; no new full-depth result is available.

Suggested message for SRCC, if the user chooses to request a scheduler check:

> On sh03-ln08 at 2026-10-24 18:40 PDT, my job 49368820 was eligible and pending, with no dependency/hold, Reason=None and StartTime=Unknown. It requests 16 CPUs, 48 GB and 24 h. An scontrol partition update to hns,normal returned success, but the subsequent job record still showed normal only. Could you confirm its partition eligibility and whether a scheduling restriction is preventing allocation?

No support request was sent. No scheduler configuration was changed or probed.
