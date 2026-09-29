#!/usr/bin/env python3
import sys, os, time

def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import server
    run, fname = sys.argv[1], sys.argv[2]
    path = os.path.join(server.DEFAULT_ROOT, run, "dumps", "rollouts", fname)
    os.makedirs(server.DEFAULT_CACHE, exist_ok=True)
    idx = server.Index(path, os.path.join(server.DEFAULT_CACHE, f"{run}__{fname}.idx.json"))
    if idx.offsets and idx.current() and not idx.stale():
        print("up to date", run, fname, len(idx.offsets), "rollouts", flush=True); return
    t = time.time(); idx._run_locked()
    print(idx.state, run, fname, len(idx.offsets), "rollouts", f"{time.time() - t:.0f}s", idx.error or "", flush=True)

if __name__ == "__main__": main()
