#!/usr/bin/env bash
# Download small result files from a pooled VM:  pull.sh SESSION REMOTE_DIR LOCAL_DIR file1 [file2 ...]
S=$1; RD=$2; LD=$3; shift 3
mkdir -p "$LD"
for f in "$@"; do
  timeout 600 colab-pool download "$S" "$RD/$f" "$LD/$f" 2>&1 | grep -v '^\[pool\]' | tail -1
done
