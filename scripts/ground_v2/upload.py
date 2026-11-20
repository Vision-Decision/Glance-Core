"""Upload a built folder to a PRIVATE HF dataset repo (records + stats + dataset card; no images).

  python scripts/ground_v2/upload.py --repo Vision-Decision/glance-ground-v2 --dir ~/gv2/gd_conv --card card.md
HF_TOKEN comes from the environment (scripts/ground_v2/remote_run.py --env HF_TOKEN).
"""
import argparse

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--dir", required=True)
ap.add_argument("--card", default="")
ap.add_argument("--allow", default="*.jsonl,*.json,full/*,single/*")
a = ap.parse_args()
api = HfApi()
api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
assert api.repo_info(a.repo, repo_type="dataset").private, "repo must be private"
api.upload_folder(folder_path=a.dir, repo_id=a.repo, repo_type="dataset", allow_patterns=a.allow.split(","),
                  commit_message="glance-ground-v2: converted Glance-Data + pos_pos_sole")
if a.card:
    api.upload_file(path_or_fileobj=a.card, path_in_repo="README.md", repo_id=a.repo, repo_type="dataset")
print("uploaded https://huggingface.co/datasets/" + a.repo, sorted(f for f in api.list_repo_files(a.repo, repo_type="dataset")))
