"""Entry point, so the benchmark runs as `python -m venom_retrieval <command>`."""

import sys

from .cli import main

sys.exit(main())
