"""Push a VM directory to the org HF dataset (never the personal account):
   python arch/vm/push.py LOCAL_DIR REPO_PATH [--repo Vision-Decision/glance-train-data] [--type dataset]"""
import argparse
from huggingface_hub import HfApi
ap = argparse.ArgumentParser()
ap.add_argument("local"); ap.add_argument("path")
ap.add_argument("--repo", default="Vision-Decision/glance-train-data"); ap.add_argument("--type", default="dataset")
a = ap.parse_args()
assert a.repo.startswith("Vision-Decision/"), "org repos only"
HfApi().upload_folder(folder_path=a.local, path_in_repo=a.path, repo_id=a.repo, repo_type=a.type,
                      commit_message=f"arch-exp: {a.path}", ignore_patterns=["*.log.tmp"])
print("[push] done", a.local, "->", f"{a.repo}:{a.path}")
