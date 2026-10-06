"""Run notebook 241's own collect script over the searches in sources.D241.

It writes arms.csv, ranks.csv and regions.parquet into D241, ranking each human protein
by its best region exactly as notebook 241 does. The script is imported from the
notebook-241 checkout at sources.NB241_COMMIT and pointed at D241 by setting its OUT.
"""
import importlib.util
import subprocess
import sys

from sources import D241, NB241, NB241_COMMIT

head = subprocess.run(["git", "-C", str(NB241), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
if head != NB241_COMMIT:
    sys.exit(f"{NB241} is at {head}, expected {NB241_COMMIT}")
spec = importlib.util.spec_from_file_location("collect241", NB241 / "notebooks" / "241_alphabet_ranking_collect.py")
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)
collect.OUT = D241
collect.main()
