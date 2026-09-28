#!/bin/bash
# usage: start.sh [port]   (stop: kill $(cat dumpbrowse.pid))
D=$(cd "$(dirname "$0")" && pwd); PORT=${1:-8790}
if [ -f "$D/dumpbrowse.pid" ] && kill -0 "$(cat "$D/dumpbrowse.pid")" 2>/dev/null; then echo "already running (pid $(cat "$D/dumpbrowse.pid"))"; exit 0; fi
setsid nohup python3 "$D/server.py" --port "$PORT" >> "$D/dumpbrowse.log" 2>&1 &
echo $! > "$D/dumpbrowse.pid"; sleep 1; tail -1 "$D/dumpbrowse.log"
echo "open http://localhost:$PORT  (from your laptop: ssh -L $PORT:localhost:$PORT $(hostname))"
