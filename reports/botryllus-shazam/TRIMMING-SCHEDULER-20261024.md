# Full-depth trimming scheduler check

24 October 2026, 18:39 PDT: trimming job 49368820 was eligible and pending on normal only. Its raw-download dependency had completed. There was no hold or requested node constraint. All 64 raw libraries were validated: 654,672,879 read pairs. No nf-core output directory existed yet.

The request is 16 CPUs, 48 GB and 24 hours. The partition inventory reported 59,258 queued CPU cores on normal and 24,796 on hns. These totals describe demand, not the job's queue position. Slurm initially reported Reason=None and StartTime=Unknown.

An in-place partition update to hns,normal returned success. The first follow-up still showed normal only, so the change was initially unverified. A later scheduler evaluation confirmed hns,normal, Reason=Resources, priority 1017392, and an estimated start at 18:42:59 PDT. That estimate is not a guarantee. The delayed update must not be described as a permanent failure.

The original job ID, CPU count, memory, time limit, inputs and downstream dependencies were preserved. No duplicate job was submitted and no healthy work was restarted. The source sbatch header now retains both verified eligible partitions. Resources were not reduced: no completed full-depth trimming trace supports a smaller request.

SPLASH 49368823 and DE-kupl / gene DE 49369243 depend on validated trimming. manysearch 49368824 follows SPLASH. Evidence 49369247 and paper-gene recovery 49373978 follow their required analyses; recovery also waits for manysearch. These dependencies remain intentional and must not be bypassed.

18:44 PDT follow-up: the broader partition eligibility is retained. Trimming remains pending for Resources; the scheduler start estimate has passed and no trimming log exists yet. No actual runtime or first-result time is measurable before allocation.
