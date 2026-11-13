---
name: nextflow-launch-resources
description: Check Nextflow 25.04 resource overrides when a launch config supplies parameters but resolved process CPU, memory, time or queue values are null.
---

# Verify launch overrides in the generated Slurm script

Observed in oyster migration preflight on 2026-10-25. The launch config populated
`params.task_cpus`, `params.task_memory`, `params.task_time` and `params.partition`,
but process directives assigned directly from those parameters resolved to null.

1. Resolve the actual profile with the pinned Nextflow binary and launch config.
2. Use deferred process closures, for example `process.memory = { params.task_memory }`,
   when the process must read parameters after the launch configuration overrides.
3. Preserve the resolved config and inspect the submitted task's `.command.run`.
   Confirm its `#SBATCH` CPU, memory, time and partition values against the request.
4. A closure printed by `nextflow config` is not proof of the final resource values.
   Require the generated batch script and scheduler evidence.
5. Keep immutable source snapshots for active workflows while fixing the local checkout.

The oyster test job 49525039 generated `-c 4`, `--mem 16384M`, `-t 04:00:00`
and `-p normal,hns` after this fix. This is a first-use staged skill.

Before testing cached resume, wait for the existing Nextflow controller to exit,
not only for the Slurm task to finish. On 2026-10-25, launching during controller
finalization caused a session-lock refusal. Preserve that log, wait for the
controller exit, then retry; never remove the lock or launch a competing session.
