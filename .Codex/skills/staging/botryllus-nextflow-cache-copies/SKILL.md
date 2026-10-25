---
name: botryllus-nextflow-cache-copies
description: Check writable directory copies when changing the Botryllus Nextflow normalization or model processes, especially when resuming cached tasks.
---

Nextflow stages directory inputs as symlinks. In the Botryllus fixture on Sherlock,
`cp -R genes normalized` made `normalized` another symlink to the upstream pooling
output. Writing normalization/model files through it changed the cached input.

When a process needs a writable copy, use `cp -RL -- input output` with a new output
name. Verify the output is not a symlink, edit a file in the copy, and assert the
source file and source directory contents remain unchanged. Keep this check beside
any change to NORMALIZE or BLOCKED_MODEL in `pipelines/botryllus-shazam/main.nf`.

Before resuming a fixture affected by this bug, preserve the failed work directory.
Use a new fixture output or verify the upstream files against their recorded checksums.
Do not treat a successful task exit as proof that cache inputs stayed unchanged.

Added 2026-10-25 during PR 203 migration. Evidence: fixture job 49513387,
NORMALIZE task 72/0b12d69a4a81fe5b768cd5b46773bb; both `genes` and `normalized`
resolved to the POOL_COUNTS task's `models/fixture0/genes` directory.
