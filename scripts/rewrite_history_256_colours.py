#!/usr/bin/env python3
"""Rewrite this repo's git history so every past PNG, and every PNG inside a past
notebook, is stored with 256 colours.

PR 123 converts the PNGs on the branch tips and makes new figures 256-colour. Old versions
stay in history, and a clone downloads all of them. Tested on a copy on 2026-10-15: a full
clone goes from 777 MB to 520 MB. PNG versions go from 512 MB to 262 MB (compressed), and
notebook versions from 207 MB to 151 MB. Every commit and branch is kept. Only PNG and
notebook files change, and in a notebook only its images change. Each image is converted
by scripts/shrink_png.py, so the same images are left in full colour as there (colour
bars, photographs).

Every commit gets a new SHA. Commit messages that quote a short SHA are updated by
git-filter-repo; SHAs written anywhere else (PR bodies, comments, notes) still point at the
old commits. commit-map.txt, written by `prepare`, lists old and new SHA for every commit.

Three steps, each run by hand:

    python scripts/rewrite_history_256_colours.py prepare DIR
        Clone GitHub's branches into DIR/before.git, rewrite a copy in DIR/after.git,
        check it, and write DIR/commit-map.txt. About 20 minutes and 12 GB of memory
        (measured 2026-10-15). Changes nothing on GitHub.

    python scripts/rewrite_history_256_colours.py push DIR [--dry-run]
        Force-push every rewritten branch. Refuses if any branch on GitHub moved since
        `prepare`; then run `prepare` again.

    python3 DIR/rewrite_history_256_colours.py migrate DIR/commit-map.txt
        Run inside each existing clone (laptop, Sherlock); `prepare` copies this script
        into DIR. Moves every local branch, and the branch checked out in each worktree,
        to its rewritten commit. Uncommitted edits stay. A worktree with staged changes,
        or a branch with commits that were never pushed, is listed and left alone.

`prepare` needs git-filter-repo, Pillow and numpy (pip install git-filter-repo pillow
numpy) and the scripts/shrink_png.py from PR 123. `migrate` needs only git and Python 3,
and avoids `git -C` and `git worktree` features newer than Sherlock's git 1.8.3.
"""

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

URL = "https://github.com/seanome/2024-kmerseek-analysis.git"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def git(*args, cwd=None, binary=False, check=True):
    out = subprocess.run(["git", *args], cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, check=False)
    if check and out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed in {cwd}:\n{out.stderr.decode()}")
    return out.stdout if binary else out.stdout.decode()


def branch_tips(repo):
    """{refs/heads/name: sha} for a local repo."""
    out = git("for-each-ref", "--format=%(objectname) %(refname)", "refs/heads", cwd=repo)
    return {ref: sha for sha, ref in (line.split() for line in out.splitlines())}


def remote_tips(url):
    out = git("ls-remote", "--heads", url)
    return {ref: sha for sha, ref in (line.split() for line in out.splitlines())}


# --- prepare ---------------------------------------------------------------------------


def load_shrink_png():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import shrink_png

    if not hasattr(shrink_png, "colour_bar_shades"):
        raise SystemExit("scripts/shrink_png.py has no colour-bar test; run this from a "
                         "checkout that has PR 123 merged.")
    return shrink_png


class Converter:
    """Converts one blob at a time. A notebook image seen before is not converted twice."""

    def __init__(self, shrink_png):
        self.sp = shrink_png
        self.cache = {}
        self.stats = dict(png=0, png_changed=0, notebook=0, notebook_changed=0, errors=0,
                          bytes_before=0, bytes_after=0)
        self._shrink = shrink_png.shrink_png_bytes
        # shrink_notebook_text looks this name up when called, so notebook images use
        # the cache too.
        shrink_png.shrink_png_bytes = self.shrink_image

    def shrink_image(self, data):
        key = hashlib.sha1(data).digest()
        if key not in self.cache:
            try:
                self.cache[key] = self._shrink(data)
            except Exception as e:  # an image Pillow cannot read stays as it is
                self.stats["errors"] += 1
                print(f"left as it is, image {key.hex()[:10]}: {e}", file=sys.stderr)
                self.cache[key] = None
        return self.cache[key]

    def convert(self, data):
        """Return the new blob, or None to keep it."""
        if data.startswith(PNG_MAGIC):
            kind = "png"
            self.stats[kind] += 1
            new = self.shrink_image(data)
        elif (data.lstrip()[:1] == b"{" and b'"nbformat"' in data
              and b'"image/png"' in data):
            kind = "notebook"
            self.stats[kind] += 1
            try:
                text, n = self.sp.shrink_notebook_text(data.decode("utf-8"))
            except Exception as e:
                self.stats["errors"] += 1
                print(f"left as it is, notebook blob: {e}", file=sys.stderr)
                return None
            new = text.encode("utf-8") if n else None
        else:
            return None
        if new is None:
            return None
        self.stats[f"{kind}_changed"] += 1
        self.stats["bytes_before"] += len(data)
        self.stats["bytes_after"] += len(new)
        return new


def rewrite(repo, converter):
    import git_filter_repo as fr

    def blob_callback(blob, _metadata):
        new = converter.convert(blob.data)
        if new is not None:
            blob.data = new

    here = os.getcwd()
    os.chdir(repo)  # filter-repo works on the current directory
    try:
        args = fr.FilteringOptions.parse_args(["--force", "--quiet"])
        fr.RepoFilter(args, blob_callback=blob_callback).run()
    finally:
        os.chdir(here)


def tree(repo, sha):
    """{path: (mode, blob sha)} for a commit."""
    out = git("ls-tree", "-r", "-z", sha, cwd=repo)
    files = {}
    for entry in out.split("\0"):
        if entry:
            meta, path = entry.split("\t", 1)
            mode, _type, blob = meta.split()
            files[path] = (mode, blob)
    return files


def without_images(nb):
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            output.get("data", {}).pop("image/png", None)
    return nb


def check(before, after):
    """Raise if the rewrite changed anything but PNG and notebook images."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    old, new = branch_tips(before), branch_tips(after)
    problems = []
    if old.keys() != new.keys():
        problems.append(f"branches differ: {sorted(old.keys() ^ new.keys())}")
    n_old = git("rev-list", "--all", "--count", cwd=before).strip()
    n_new = git("rev-list", "--all", "--count", cwd=after).strip()
    if n_old != n_new:
        problems.append(f"commit count {n_old} before, {n_new} after")
    checked = set()
    for ref in sorted(old.keys() & new.keys()):
        a, b = tree(before, old[ref]), tree(after, new[ref])
        if a.keys() != b.keys():
            problems.append(f"{ref}: files differ: {sorted(a.keys() ^ b.keys())[:5]}")
            continue
        for path in a:
            if a[path] == b[path] or (a[path], b[path]) in checked:
                continue
            checked.add((a[path], b[path]))
            if a[path][0] != b[path][0]:
                problems.append(f"{ref}: {path}: mode changed")
            elif path.endswith(".png"):
                x = Image.open(io.BytesIO(git("cat-file", "blob", a[path][1], cwd=before,
                                              binary=True)))
                y = Image.open(io.BytesIO(git("cat-file", "blob", b[path][1], cwd=after,
                                              binary=True)))
                if x.size != y.size or x.info.get("dpi") != y.info.get("dpi"):
                    problems.append(f"{ref}: {path}: pixel size or dpi changed")
            elif path.endswith(".ipynb"):
                x = json.loads(git("cat-file", "blob", a[path][1], cwd=before, binary=True))
                y = json.loads(git("cat-file", "blob", b[path][1], cwd=after, binary=True))
                if without_images(x) != without_images(y):
                    problems.append(f"{ref}: {path}: changed outside its images")
            else:
                problems.append(f"{ref}: {path}: changed, and is not a PNG or notebook")
    if problems:
        raise SystemExit("Check failed:\n  " + "\n  ".join(problems[:40]))
    print(f"Checked {len(old)} branches and {n_new} commits: only PNGs and notebook "
          f"images changed ({len(checked)} changed files compared).")


def pack_mb(repo):
    sizes = dict(line.split(": ") for line in git("count-objects", "-v", cwd=repo).splitlines())
    return int(sizes["size-pack"]) / 1024  # KiB -> MiB


def prepare(workdir, url):
    shrink_png = load_shrink_png()
    workdir = Path(workdir).resolve()
    if workdir.exists() and any(workdir.iterdir()):
        raise SystemExit(f"{workdir} is not empty")
    workdir.mkdir(parents=True, exist_ok=True)
    before, after = workdir / "before.git", workdir / "after.git"
    git("clone", "--bare", "--quiet", url, str(before))
    git("clone", "--bare", "--quiet", "--no-local", str(before), str(after))
    (workdir / "url.txt").write_text(url + "\n")
    converter = Converter(shrink_png)
    rewrite(after, converter)
    print("  ".join(f"{k}={v}" for k, v in converter.stats.items()))
    check(before, after)
    # filter-repo lists every commit, with old and new SHA, in filter-repo/commit-map.
    (workdir / "commit-map.txt").write_text((after / "filter-repo" / "commit-map").read_text())
    # `migrate` runs from this copy on every machine with a clone (laptop, Sherlock).
    (workdir / Path(__file__).name).write_text(Path(__file__).read_text())
    git("gc", "--quiet", "--prune=now", cwd=after)
    print(f"Pack size: {pack_mb(before):.0f} MiB before, {pack_mb(after):.0f} MiB after.")
    print(f"Next: python {sys.argv[0]} push {workdir} --dry-run")


# --- push ------------------------------------------------------------------------------


def push(workdir, dry_run):
    workdir = Path(workdir).resolve()
    url = (workdir / "url.txt").read_text().strip()
    old, new = branch_tips(workdir / "before.git"), branch_tips(workdir / "after.git")
    now = remote_tips(url)
    moved = sorted(r for r in now if now[r] != old.get(r))
    if moved:
        raise SystemExit("These branches changed on GitHub after `prepare` (new, deleted "
                         "or moved), so the rewrite would lose their commits. Run "
                         "`prepare` again in a new directory.\n  " + "\n  ".join(moved))
    # --force-with-lease refuses each branch that moved between this check and the push.
    leases = [f"--force-with-lease={ref}:{old[ref]}" for ref in sorted(new)]
    refspecs = [f"{new[ref]}:{ref}" for ref in sorted(new)]
    args = ["push", "--porcelain", *(["--dry-run"] if dry_run else []), *leases, url,
            *refspecs]
    out = subprocess.run(["git", *args], cwd=workdir / "after.git",
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(out.stdout[-3000:])
    if out.returncode != 0:
        raise SystemExit(f"push failed:\n{out.stderr}")
    if dry_run:
        print(f"Dry run: {len(refspecs)} branches would be force-pushed. Nothing sent.")
        return
    after_push = remote_tips(url)
    wrong = sorted(r for r in new if after_push.get(r) != new[r])
    if wrong:
        raise SystemExit("These branches on GitHub are not at their rewritten commit:\n  "
                         + "\n  ".join(wrong))
    print(f"All {len(new)} branches on GitHub are at their rewritten commits.")
    print(f"Next, in each clone: python3 {workdir / Path(__file__).name} migrate "
          f"{workdir / 'commit-map.txt'}")


# --- migrate ---------------------------------------------------------------------------


def read_commit_map(path):
    pairs = {}
    for line in Path(path).read_text().splitlines()[1:]:  # first line is a header
        old, new = line.split()
        pairs[old] = new
    return pairs


def worktrees(repo):
    """[(path, branch ref or None, HEAD sha)] for every checkout of this repo."""
    out = git("worktree", "list", "--porcelain", cwd=repo, check=False)
    if not out:  # git 1.8 has no worktree command: just this checkout
        top = git("rev-parse", "--show-toplevel", cwd=repo).strip()
        head = git("rev-parse", "HEAD", cwd=top).strip()
        ref = git("symbolic-ref", "-q", "HEAD", cwd=top, check=False).strip() or None
        return [(top, ref, head)]
    result, path, ref, head, bare = [], None, None, None, False
    for line in out.splitlines() + [""]:
        if line.startswith("worktree "):
            path, ref, head, bare = line[len("worktree "):], None, None, False
        elif line == "bare":
            bare = True
        elif line.startswith("HEAD "):
            head = line[len("HEAD "):]
        elif line.startswith("branch "):
            ref = line[len("branch "):]
        elif line == "" and path:
            if head and not bare and os.path.isdir(path):
                result.append((path, ref, head))
            path = None
    return result


def first_rewritten_ancestor(repo, sha, pairs):
    for commit in git("rev-list", sha, cwd=repo).split():
        if commit in pairs:
            return commit
    return None


def move_checkout(path, ref, old, new):
    """Point a checkout at `new`; keep uncommitted edits; update unedited images."""
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=path).returncode != 0:
        return "SKIPPED, has staged changes: commit or unstage them, then run again"
    changed = [n for n in git("diff", "--name-only", "-z", old, new, cwd=path).split("\0")
               if n]
    replace = []
    for name in changed:
        file = Path(path) / name
        if not file.exists():
            continue
        on_disk = git("hash-object", str(file), cwd=path).strip()
        if on_disk == git("rev-parse", f"{old}:{name}", cwd=path, check=False).strip():
            replace.append(name)  # not edited since the commit: take the new version
    if ref:
        git("update-ref", ref, new, old, cwd=path)
    else:
        git("update-ref", "--no-deref", "HEAD", new, old, cwd=path)
    git("reset", "-q", cwd=path)  # index to the new commit; files on disk untouched
    for i in range(0, len(replace), 200):
        git("checkout", new, "--", *replace[i:i + 200], cwd=path)
    kept = len(changed) - len(replace)
    return f"moved; {len(replace)} images updated" + (
        f", {kept} edited files kept as they are" if kept else "")


def migrate(commit_map, repo):
    pairs = read_commit_map(commit_map)
    rewritten = set(pairs.values())
    git("fetch", "--prune", "origin", cwd=repo)
    checked_out = {}
    report = []
    for path, ref, head in worktrees(repo):
        if ref:
            checked_out[ref] = path
        if head in rewritten:
            report.append(f"already rewritten: {path}")
        elif head in pairs:
            report.append(f"{path}: {move_checkout(path, ref, head, pairs[head])}")
        else:
            base = first_rewritten_ancestor(path, head, pairs)
            report.append(f"NOT MOVED: commits not on GitHub at {path}; rebase with\n"
                          f"    git rebase --onto {pairs.get(base, '?')} {base}")
    for ref, sha in sorted(branch_tips(repo).items()):
        if ref in checked_out or sha in rewritten:
            continue
        name = ref[len("refs/heads/"):]
        if sha in pairs:
            git("update-ref", ref, pairs[sha], sha, cwd=repo)
        else:
            base = first_rewritten_ancestor(repo, sha, pairs)
            report.append(f"NOT MOVED: branch {name} has commits not on GitHub; rebase with\n"
                          f"    git rebase --onto {pairs.get(base, '?')} {base} {name}")
    print("\n".join(report))
    print("Done. The old commits stay on disk until you run:\n"
          "    git reflog expire --expire=now --all && git gc --prune=now")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="step")
    sub.required = True  # the keyword form needs Python 3.7; Sherlock's python3 is 3.6
    p = sub.add_parser("prepare")
    p.add_argument("dir")
    p.add_argument("--url", default=URL)
    p = sub.add_parser("push")
    p.add_argument("dir")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("migrate")
    p.add_argument("commit_map")
    p.add_argument("--repo", default=".")
    args = parser.parse_args(argv)
    if args.step == "prepare":
        prepare(args.dir, args.url)
    elif args.step == "push":
        push(args.dir, args.dry_run)
    else:
        migrate(args.commit_map, args.repo)


if __name__ == "__main__":
    main()
