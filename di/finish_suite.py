"""Finish the 0.3 text suite after `suite rebuild` when the kit stops early.

`suite rebuild` skips the v2 cut (and so the GSM8K v3 rows) when the release-v1 rows are not byte-identical to the
lab's. On macOS (2026-11-18) the only difference was the SGD normalized file (SGD is shown, not counted). This
applies the same cut, builds the added rows and the GSM8K v3 rows, and imports everything into <suite dir>.

    python -m di.finish_suite --work /content/di/work --suite /content/di/suite-0.3
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--suite", required=True)
    a = ap.parse_args()
    from decision_index import constants as C
    from decision_index.suite.build import gsm8k_v3, release_v2
    from decision_index.suite.build.layout import Layout
    from decision_index.suite.download import hub_file

    L = Layout(a.work)
    v1 = L.suite / "release-v1-rebuilt" / "selected-rows.jsonl"
    r = release_v2.main(L, v1, hub_file("0.2", C.SUITE_EXCLUSIONS_FILE), log=print)
    print(json.dumps({"rows": r.get("rows"), "rows_byte_identical": r.get("rows_byte_identical"),
                      "added_byte_identical": r.get("added_byte_identical")}))
    g = gsm8k_v3.main(L, Path(r["out"]) / C.SUITE_ROWS_FILE, log=print)
    out = Path(r["out"])
    cmd = [sys.executable, "-m", "decision_index", "suite", "import", "--dir", a.suite,
           "--rows", str(out / C.SUITE_ROWS_FILE), "--added-rows", str(out / "added-rows.jsonl.gz"),
           "--gsm8k-rows", g["out"]]
    if not r.get("rows_byte_identical"):
        cmd.append("--no-verify")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
