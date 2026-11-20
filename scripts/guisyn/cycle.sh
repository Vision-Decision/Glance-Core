#!/bin/bash
# Every 30 min (cron): build new shards from closed raw jobs, then upload new complete shards to the private HF repo.
# Each step has a hard timeout; failures are logged and retried on the next cycle; generation is never touched.
exec 9>/data/guisyn/cycle.lock
flock -n 9 || { echo "$(date -u +%FT%T) previous cycle still running" >> /data/guisyn/logs/cycle.log; exit 0; }
cd /data/guisyn/code
PY=/data/guisyn/venv/bin/python
echo "$(date -u +%FT%T) build start" >> /data/guisyn/logs/cycle.log
timeout -k 60 1500 nice -n 15 $PY build_shards.py --procs 2 >> /data/guisyn/logs/build.log 2>&1; echo "$(date -u +%FT%T) build rc=$?" >> /data/guisyn/logs/cycle.log
timeout -k 60 1500 $PY stats.py >> /data/guisyn/logs/stats.log 2>&1
timeout -k 60 1500 $PY upload.py >> /data/guisyn/logs/upload.out 2>&1; echo "$(date -u +%FT%T) upload rc=$?" >> /data/guisyn/logs/cycle.log
