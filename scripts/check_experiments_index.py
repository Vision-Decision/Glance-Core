"""Check that every repo path and HF file referenced in EXPERIMENTS.md exists.

Usage:
  python scripts/check_experiments_index.py [--md EXPERIMENTS.md] [--branches main,glance-video,...]
         [--tree-list paths.txt] [--no-hf]

Repo paths: backticked tokens whose first segment is a top-level entry of the repo; looked up in the git trees of
the given branches (default: HEAD plus the unmerged branches listed in EXPERIMENTS.md, via origin/ if no local ref).
HF refs: backticked `org/repo:path` tokens; `path/` means "some file under this prefix". Needs huggingface_hub and a
token with read access to the private repos.
"""
import argparse, re, subprocess, sys

BRANCHES = ["HEAD", "glance-video", "v12b-multibox", "worktree-agent-ab225d57487eeae40"]
HF_RE = re.compile(r"^([\w.-]+/[\w.-]+):(.+)$")


def git_tree(ref):
    for r in (ref, f"origin/{ref}"):
        p = subprocess.run(["git", "ls-tree", "-r", "--name-only", r], capture_output=True, text=True)
        if p.returncode == 0:
            return set(p.stdout.split())
    return None


def exists(path, files):
    if path.endswith("/"):
        return any(f.startswith(path) for f in files)
    return path in files or any(f.startswith(path + "/") for f in files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="EXPERIMENTS.md")
    ap.add_argument("--branches", default=",".join(BRANCHES))
    ap.add_argument("--tree-list", help="file with 'branch<TAB>path' lines instead of git")
    ap.add_argument("--no-hf", action="store_true")
    a = ap.parse_args()

    text = open(a.md).read().split("<!-- CHECK-OUTPUT -->")[0]
    fenced = re.findall(r"```.*?\n(.*?)```", text, flags=re.S)
    spans = re.sub(r"```.*?```", " ", text, flags=re.S).split("`")[1::2] + fenced
    toks = set()
    for sp in spans:
        if not re.search(r"\s", sp.strip()):
            toks.add(sp.strip())
        else:  # path-like words inside a command, e.g. `python scripts/x.py --out y`
            toks.update(w.strip("\"'(),;") for w in sp.split() if "/" in w and not w.startswith("-"))
    toks = sorted(toks)

    trees = {}
    if a.tree_list:
        for line in open(a.tree_list):
            b, p = line.rstrip("\n").split("\t", 1)
            trees.setdefault(b, set()).add(p)
    else:
        for b in a.branches.split(","):
            t = git_tree(b)
            if t is None:
                print(f"WARN: branch {b} not found", file=sys.stderr)
            else:
                trees[b] = t
    top = {p.split("/")[0] for t in trees.values() for p in t}

    hf_refs, paths = [], []
    for t in toks:
        m = HF_RE.match(t)
        if m and "/" in m.group(1) and not t.startswith(("http", "~", "/")):
            hf_refs.append((m.group(1), m.group(2)))
        elif "/" in t and t.split("/")[0] in top and not any(c in t for c in "<>*$"):
            paths.append(t)

    missing, on_branch = [], {}
    for p in paths:
        hit = [b for b, f in trees.items() if exists(p, f)]
        if not hit:
            missing.append(("repo", p))
        elif "HEAD" not in hit:
            on_branch[p] = hit

    n_hf = 0
    if not a.no_hf and hf_refs:
        from huggingface_hub import HfApi
        api, cache = HfApi(), {}
        for repo, path in hf_refs:
            if repo not in cache:
                files = None
                for rt in ("model", "dataset"):
                    try:
                        files = set(api.list_repo_files(repo, repo_type=rt))
                        break
                    except Exception:
                        pass
                cache[repo] = files
            n_hf += 1
            if cache[repo] is None:
                missing.append(("hf", f"{repo} (repo not found)"))
            elif not exists(path, cache[repo]):
                missing.append(("hf", f"{repo}:{path}"))

    print(f"checked {len(paths)} repo paths ({', '.join(trees)}) and {n_hf} HF refs")
    if on_branch:
        print(f"{len(on_branch)} paths exist only on a branch (expected for unmerged work):")
        for p, bs in sorted(on_branch.items()):
            print(f"  {p}  [{', '.join(bs)}]")
    print(f"{len(missing)} missing:")
    for kind, p in missing:
        print(f"  {kind}: {p}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
