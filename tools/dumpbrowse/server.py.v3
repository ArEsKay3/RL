#!/usr/bin/env python3
import argparse, json, os, re, sys, threading, time, zlib, glob, urllib.parse, multiprocessing, csv, math
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_ROOT = "/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs"
DEFAULT_CACHE = os.path.join(WS, "analysis", "dumpbrowse_cache")
DEFAULT_METRICS = os.path.join(WS, "analysis", "datadiff", "browser")
ARM_DESC = {"A": "run A: vLLM from scratch", "B": "run B: MINF from MINF step 10", "C": "chain C: MINF from MINF step 10, no prefix cache",
            "D": "chain D: vLLM from MINF step 10", "F": "chain F: MINF from vLLM step 10", "G": "chain G: MINF from scratch, no prefix cache",
            "I": "chain I: MINF from scratch, prefix cache on, replica 1", "J": "chain J: MINF from scratch, prefix cache on, replica 2",
            "K": "chain K: vLLM from scratch, seed 1234", "L": "chain L: MINF from chain K step 10, prefix cache on, seed 1234",
            "M": "chain M: vLLM from scratch, replica 1 (chain J argument set)", "N": "chain N: vLLM from scratch, replica 2 (chain J argument set)",
            "P": "chain P: MINF engine, chain M rollouts replayed for steps 1-20, live from 21", "Q": "chain Q: vLLM engine, chain M rollouts replayed for steps 1-20, live from 21",
            "R": "chain R: vLLM engine, chain G rollouts replayed for steps 1-10, live from 11", "Rp": "chain R′: vLLM engine, chain G rollouts replayed for steps 1-16, live from 17"}
SECRET_KEY = re.compile(r"(api_key|token|secret|password)", re.I)
SECRET_YAML = re.compile(r"(?im)^(\s*[\w\-]*(?:api_key|secret|password)[\w\-]*\s*:\s*)(\S.*)$")
FILE_RE = re.compile(r"^[\w.\-]+\.jsonl$")
ELIDE = 20000
CACHE_V = 2
NPROC = 4
HEAD_KEYS = ["sample_id", "group_id", "completion_index", "target_step", "start_weight_version", "end_weight_version", "committed_at", "prompt_idx", "environment", "reward", "truncated", "num_assistant_turns", "generation_length", "total_tokens"]

ROLL_COLS = ["arm", "train_step", "adv", "mask_metric_X", "mask_metric_Z", "r_deep_contrib", "r_deep_contrib_per1k_step", "absadv_x_live", "live_deep_1mp", "mean_1mp_deep", "n_deep_turns", "deep_think_tokens",
             "in_X_mask_rank", "in_Y_mask_rank", "in_Z_mask_rank", "rank_X_metric_all_steps", "rank_Z_metric_all_steps", "group_net_contrib", "group_pos_contrib", "group_neg_contrib", "group_cancel_ratio",
             "group_n_deep_rollouts", "group_rank_net_steps1_10", "group_mean_reward", "mask_before", "mask_after", "seq_mult_prob_error"]
TURN_COLS = ["think_tok", "deep_tokens", "live_deep_1mp", "mean_1mp_deep", "turn_contrib", "cls", "close_p_tr", "s1mp_think_tr", "s1mp_entry_tr", "mean_ptr_think"]
GROUP_COLS = ["group_net_contrib", "group_pos_contrib", "group_neg_contrib", "group_cancel_ratio", "group_n_deep_rollouts", "group_rank_net_steps1_10", "group_mean_reward"]

def conv(row, cols):
    out = {}
    for k in cols:
        v = row.get(k)
        if v is None or v == "": out[k] = None; continue
        if k in ("arm", "cls"): out[k] = v; continue
        try:
            f = float(v)
            if math.isnan(f) or math.isinf(f): out[k] = None
            else: out[k] = int(f) if f.is_integer() and abs(f) < 1e15 and k in ("train_step", "n_deep_turns", "deep_think_tokens", "in_X_mask_rank", "in_Y_mask_rank", "in_Z_mask_rank", "rank_X_metric_all_steps", "rank_Z_metric_all_steps", "group_n_deep_rollouts", "group_rank_net_steps1_10", "think_tok", "deep_tokens", "mask_before", "mask_after") else f
        except ValueError: out[k] = v
    return out

class Metrics:
    def __init__(self, d):
        self.dir = d; self.lock = threading.Lock(); self.roll = {}; self.turns = {}; self.steps = {}; self.sig = None; self.checked = 0; self.loading = False; self.loaded_at = None; self.n_roll = 0; self.n_turn = 0
    def _sig(self):
        if not os.path.isdir(self.dir): return ()
        return tuple(sorted((f, os.path.getmtime(os.path.join(self.dir, f))) for f in os.listdir(self.dir) if f.endswith(".csv") and "all_arms" not in f))
    def maybe_refresh(self, force=False):
        now = time.time()
        if not force and now - self.checked < 30: return
        self.checked = now
        try: sig = self._sig()
        except OSError: return
        if sig == self.sig or self.loading: return
        self.loading = True
        threading.Thread(target=self._load, args=(sig,), daemon=True).start()
    def _load(self, sig):
        try:
            roll = {}; turns = {}; steps = {}; nr = nt = 0
            for f, _ in sig:
                p = os.path.join(self.dir, f)
                if f.startswith("rollout_metrics_"):
                    with open(p, newline="") as fh:
                        for r in csv.DictReader(fh):
                            ts = int(float(r["target_step"])); run = r["run"]
                            roll.setdefault(run, {}).setdefault(ts, {})[r["sample_id"]] = conv(r, ROLL_COLS); steps.setdefault(run, set()).add(ts); nr += 1
                elif f.startswith("long_turns_"):
                    with open(p, newline="") as fh:
                        for r in csv.DictReader(fh):
                            turns.setdefault(r["run"], {}).setdefault((int(float(r["target_step"])), r["sample_id"]), {})[int(float(r["turn"]))] = conv(r, TURN_COLS); nt += 1
            with self.lock: self.roll, self.turns, self.steps, self.sig, self.n_roll, self.n_turn, self.loaded_at = roll, turns, steps, sig, nr, nt, time.time()
        except Exception as e:
            print("metrics load failed:", repr(e), flush=True)
        finally: self.loading = False
    def has_step(self, run, ts): return ts in self.steps.get(run, ())
    def rollout(self, run, ts, sid): return self.roll.get(run, {}).get(ts, {}).get(sid)
    def turn_rows(self, run, ts, sid): return self.turns.get(run, {}).get((ts, sid), {})
    def status(self): return {"dir": self.dir, "runs": {r: sorted(v) for r, v in self.steps.items()}, "n_rollout_rows": self.n_roll, "n_turn_rows": self.n_turn, "loaded_at": self.loaded_at, "loading": self.loading}

def load_labels():
    lab = {}
    jt = os.path.join(WS, "analysis", "loopfeedback", "jobs.txt")
    if os.path.exists(jt):
        for line in open(jt):
            p = line.split()
            if len(p) >= 2: lab[p[1]] = p[0]
    sa = os.path.join(WS, "analysis", "loopfeedback", "splice_arms.json")
    if os.path.exists(sa):
        for a, c in json.load(open(sa)).items(): lab[c["exp"]] = a
    return lab

class Index:
    def __init__(self, path, cache_path):
        self.path, self.cache_path = path, cache_path
        self.offsets, self.lengths, self.summaries = [], [], []
        self.size_indexed = 0; self.version = CACHE_V
        self.state = "idle"; self.phase = ""; self.done = 0; self.total = 0; self.error = None
        self.lock = threading.Lock()
        if os.path.exists(cache_path):
            try:
                d = json.load(open(cache_path))
                self.offsets, self.lengths, self.summaries, self.size_indexed = d["offsets"], d["lengths"], d["summaries"], d["size_indexed"]
                self.version = d.get("v", 1)
            except Exception: pass

    def current(self): return self.version == CACHE_V and len(self.summaries) == len(self.offsets) and all(x is not None for x in self.summaries)

    def stale(self):
        try: return os.path.getsize(self.path) > self.size_indexed
        except OSError: return False

    def status(self):
        return {"state": self.state, "phase": self.phase, "done": self.done, "total": self.total, "n_records": len(self.offsets), "n_summarized": sum(1 for x in self.summaries if x is not None),
                "size_indexed": self.size_indexed, "size_now": os.path.getsize(self.path) if os.path.exists(self.path) else 0, "error": self.error, "version": self.version}

    def start(self):
        with self.lock:
            if self.state == "running": return
            self.state = "running"; self.error = None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            size = os.path.getsize(self.path)
            start = (self.offsets[-1] + self.lengths[-1] + 1) if self.offsets else 0
            self.phase = "scan"; self.total = size; self.done = start
            CH = 64 << 20
            with open(self.path, "rb") as f:
                f.seek(start); pos = start; buf = b""
                while True:
                    chunk = f.read(CH)
                    if not chunk: break
                    buf += chunk; s = 0
                    while True:
                        nl = buf.find(b"\n", s)
                        if nl < 0: break
                        if buf[s:nl].strip():
                            self.offsets.append(pos + s); self.lengths.append(nl - s)
                        s = nl + 1
                    buf = buf[s:]; pos += s; self.done = pos
            self.size_indexed = pos
            if self.version != CACHE_V: self.summaries = [None] * len(self.offsets)
            else: self.summaries = list(self.summaries) + [None] * (len(self.offsets) - len(self.summaries))
            todo = [j for j, x in enumerate(self.summaries) if x is None]
            self.phase = "summarize"; self.total = len(todo); self.done = 0
            if todo:
                ctx = multiprocessing.get_context("spawn")
                with ctx.Pool(min(NPROC, len(todo))) as pool:
                    for j, summ in zip(todo, pool.imap(_summarize_at, [(self.path, self.offsets[j], self.lengths[j], j) for j in todo], chunksize=2)):
                        self.summaries[j] = summ; self.done += 1
            self.version = CACHE_V
            tmp = self.cache_path + ".tmp"
            json.dump({"v": CACHE_V, "offsets": self.offsets, "lengths": self.lengths, "summaries": self.summaries, "size_indexed": self.size_indexed}, open(tmp, "w"))
            os.replace(tmp, self.cache_path)
            self.state = "done"
        except Exception as e:
            self.state = "error"; self.error = repr(e)

    def read(self, i):
        with open(self.path, "rb") as f:
            f.seek(self.offsets[i]); return f.read(self.lengths[i])

TAIL_KEYS = ["resolved", "patch_exists", "agent_error_kind", "agent_timed_out", "eval_timed_out", "oom_killed", "eval_oom_killed", "openhands_run_time", "total_model_call_time"]
def loop_stats(rec):
    out = ((rec.get("full_result") or {}).get("response") or {}).get("output") or []
    msgs = rec.get("messages") or []
    turns = []; cur = None
    for o in out:
        t = o.get("type")
        if t == "reasoning":
            if cur is not None: turns.append(cur)
            cur = ["".join(x.get("text", "") for x in (o.get("summary") or []) if isinstance(x, dict)), False]
        elif t == "function_call":
            if cur is None: cur = ["", False]
            cur[1] = True
        elif t == "function_call_output":
            if cur is not None: turns.append(cur); cur = None
    if cur is not None: turns.append(cur)
    n_loop = 0; loop_tok = 0; min_z = None; max_tok = 0; long_turns = 0
    for k, (text, has_call) in enumerate(turns):
        am = msgs[2 * k + 1] if 2 * k + 1 < len(msgs) else {}
        ntok = am.get("n_tokens") if isinstance(am, dict) and am.get("n_tokens") else int(len(text) / 3.5)
        max_tok = max(max_tok, ntok)
        if ntok >= 1000:
            long_turns += 1; z = zratio(text)
            if z is not None:
                min_z = z if min_z is None else min(min_z, z)
                if z < 0.10: n_loop += 1; loop_tok += ntok
    gen = rec.get("generation_length") or 0
    return {"loop_turns": n_loop, "loop_tokens": loop_tok, "loop_pct": round(100 * loop_tok / gen, 1) if gen else None, "min_zlib": min_z,
            "runaway": bool(rec.get("truncated")) and bool(turns) and not turns[-1][1], "max_turn_tokens": max_tok, "long_turns": long_turns}

def summarize(line, i):
    s = {"i": i, "bytes": len(line)}
    try: rec = json.loads(line)
    except Exception as e:
        s["error"] = repr(e); return s
    for k in HEAD_KEYS:
        if k in rec: s[k] = rec[k]
    pm = rec.get("prompt_metadata") or {}
    s["instance_id"] = pm.get("instance_id"); s["repo"] = pm.get("repo")
    fr = rec.get("full_result") or {}
    for k in TAIL_KEYS:
        if k in fr: s[k] = fr[k]
    try: s.update(loop_stats(rec))
    except Exception as e: s["loop_error"] = repr(e)
    return s

def _summarize_at(args):
    path, off, ln, i = args
    with open(path, "rb") as f:
        f.seek(off); line = f.read(ln)
    return summarize(line, i)

def redact(obj, key=None):
    if isinstance(obj, dict): return {k: redact(v, k) for k, v in obj.items()}
    if isinstance(obj, list): return [redact(v, key) for v in obj]
    if isinstance(obj, str):
        if key and SECRET_KEY.search(key) and not key.startswith("return_") and not key.startswith("request_"): return "«redacted»"
        if "api_key" in obj or "password" in obj or "secret" in obj: return SECRET_YAML.sub(r"\1«redacted»", obj)
    return obj

def elide(obj, path=""):
    if isinstance(obj, dict): return {k: elide(v, f"{path}.{k}" if path else k) for k, v in obj.items()}
    if isinstance(obj, list): return [elide(v, f"{path}.{i}") for i, v in enumerate(obj)]
    if isinstance(obj, str) and len(obj) > ELIDE: return {"__elided__": True, "len": len(obj), "path": path, "head": obj[:2000]}
    return obj

def get_path(obj, path):
    for p in path.split("."):
        if p == "": continue
        obj = obj[int(p)] if isinstance(obj, list) else obj[p]
    return obj

def zratio(text):
    b = text.encode("utf-8", "replace")
    return round(len(zlib.compress(b, 6)) / len(b), 3) if len(b) >= 64 else None

def turns_of(rec):
    out = ((rec.get("full_result") or {}).get("response") or {}).get("output") or []
    msgs = rec.get("messages") or []
    ptm = (rec.get("full_result") or {}).get("per_turn_metrics") or {}
    usages = ptm.get("token_usages") or []; lats = ptm.get("response_latencies") or []
    turns = []; cur = None
    def new():
        return {"k": len(turns), "reasoning": "", "message": "", "calls": [], "outputs": [], "generation_str": None, "prompt_str_len": None}
    for o in out:
        t = o.get("type")
        if t == "reasoning":
            if cur is not None and (cur["reasoning"] or cur["calls"] or cur["message"]): turns.append(cur)
            cur = new(); cur["reasoning"] = "".join(x.get("text", "") for x in (o.get("summary") or []))
            if o.get("content"): cur["reasoning"] += "".join(x.get("text", "") for x in o["content"] if isinstance(x, dict))
        elif t == "message":
            if cur is None: cur = new()
            cur["message"] += "".join(x.get("text", "") for x in (o.get("content") or []) if isinstance(x, dict))
        elif t == "function_call":
            if cur is None: cur = new()
            cur["calls"].append({"name": o.get("name"), "arguments": o.get("arguments"), "call_id": o.get("call_id"), "status": o.get("status")})
            if o.get("generation_str") is not None: cur["generation_str"] = o["generation_str"]
            if o.get("prompt_str") is not None: cur["prompt_str_len"] = len(o["prompt_str"])
        elif t == "function_call_output":
            if cur is None: cur = new()
            cur["outputs"].append({"call_id": o.get("call_id"), "output": o.get("output"), "status": o.get("status")})
            turns.append(cur); cur = None
    if cur is not None: turns.append(cur)
    res = []
    for k, t in enumerate(turns):
        am = msgs[2 * k + 1] if 2 * k + 1 < len(msgs) else {}
        um = msgs[2 * k + 2] if 2 * k + 2 < len(msgs) else {}
        u = usages[k] if k < len(usages) else {}; l = lats[k] if k < len(lats) else {}
        gen = t["generation_str"]; think_closed = None if gen is None else ("</think>" in gen)
        res.append({"k": k, "reasoning_len": len(t["reasoning"]), "reasoning_zlib": zratio(t["reasoning"]), "reasoning": t["reasoning"][:4000],
                    "message": t["message"][:4000], "message_len": len(t["message"]),
                    "calls": [{"name": c["name"], "arguments": (c["arguments"] or "")[:4000], "arguments_len": len(c["arguments"] or ""), "status": c["status"]} for c in t["calls"]],
                    "outputs": [{"output": (o["output"] or "")[:4000], "output_len": len(o["output"] or ""), "status": o["status"]} for o in t["outputs"]],
                    "generation_len": None if gen is None else len(gen), "generation_zlib": None if gen is None else zratio(gen), "think_closed": think_closed,
                    "prompt_str_len": t["prompt_str_len"], "assistant_n_tokens": am.get("n_tokens"), "is_invalid_tool_call": am.get("is_invalid_tool_call"),
                    "has_malformed_thinking": am.get("has_malformed_thinking"), "user_n_tokens": um.get("n_tokens"),
                    "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"), "cache_read_tokens": u.get("cache_read_tokens"),
                    "latency": l.get("latency"), "timestamp": l.get("timestamp")})
    return res

def turn_paths(rec):
    out = ((rec.get("full_result") or {}).get("response") or {}).get("output") or []
    paths = {}; k = -1
    for j, o in enumerate(out):
        t = o.get("type")
        if t == "reasoning": k += 1; paths.setdefault(k, {})["reasoning"] = f"full_result.response.output.{j}.summary.0.text"
        elif t == "function_call":
            if k < 0: k = 0
            paths.setdefault(k, {})["generation_str"] = f"full_result.response.output.{j}.generation_str"; paths[k]["prompt_str"] = f"full_result.response.output.{j}.prompt_str"; paths[k]["arguments"] = f"full_result.response.output.{j}.arguments"
        elif t == "function_call_output":
            if k < 0: k = 0
            paths.setdefault(k, {})["output"] = f"full_result.response.output.{j}.output"
        elif t == "message":
            if k < 0: k = 0
            paths.setdefault(k, {})["message"] = f"full_result.response.output.{j}.content.0.text"
    return paths

class App:
    def __init__(self, root, cache, metrics_dir):
        self.root, self.cache = root, cache
        self.metrics = Metrics(metrics_dir); self.metrics.maybe_refresh(force=True)
        self.labels = load_labels(); self.indexes = {}; self.lock = threading.Lock(); self._runs = (0, None)
        self.sem = threading.BoundedSemaphore(2)

    def runs(self):
        if time.time() - self._runs[0] < 20 and self._runs[1] is not None: return self._runs[1]
        res = []
        for d in sorted(glob.glob(os.path.join(self.root, "*", "dumps", "rollouts"))):
            files = [f for f in os.listdir(d) if FILE_RE.match(f)]
            if not files: continue
            name = os.path.basename(os.path.dirname(os.path.dirname(d)))
            tot = 0; latest = 0
            for f in files:
                try: st = os.stat(os.path.join(d, f)); tot += st.st_size; latest = max(latest, st.st_mtime)
                except OSError: pass
            a = self.labels.get(name)
            res.append({"name": name, "arm": a, "label": ARM_DESC.get(a) if a else None, "n_files": len(files), "total_bytes": tot, "latest_mtime": latest, "metrics_steps": len(self.metrics.steps.get(name, ()))})
        res.sort(key=lambda r: -r["latest_mtime"])
        self._runs = (time.time(), res); return res

    def run_dir(self, run):
        if not re.match(r"^[\w.\-]+$", run): raise KeyError(run)
        d = os.path.join(self.root, run, "dumps", "rollouts")
        if not os.path.isdir(d): raise KeyError(run)
        return d

    def files(self, run):
        d = self.run_dir(run); res = []
        for f in sorted(os.listdir(d)):
            if not FILE_RE.match(f): continue
            p = os.path.join(d, f); st = os.stat(p)
            m = re.search(r"(\d+)", f); step0 = int(m.group(1)) if m else None
            idx = self.indexes.get((run, f)); cache_ok = os.path.exists(self._cache_path(run, f))
            res.append({"name": f, "target_step": step0, "train_step": (step0 + 1) if step0 is not None else None, "size": st.st_size, "mtime": st.st_mtime,
                        "indexed": bool(idx and idx.offsets) or cache_ok, "n_records": len(idx.offsets) if idx else None, "status": idx.status() if idx else None, "has_metrics": self.metrics.has_step(run, step0)})
        return res

    def _cache_path(self, run, f): return os.path.join(self.cache, f"{run}__{f}.idx.json")

    def index(self, run, f, start=True):
        if not FILE_RE.match(f): raise KeyError(f)
        p = os.path.join(self.run_dir(run), f)
        if not os.path.exists(p): raise KeyError(f)
        with self.lock:
            idx = self.indexes.get((run, f))
            if idx is None: idx = self.indexes[(run, f)] = Index(p, self._cache_path(run, f))
        if start and idx.state != "running" and (not idx.offsets or idx.stale() or not idx.current()): idx.start()
        return idx

    def record(self, run, f, i):
        idx = self.index(run, f, start=False)
        if i < 0 or i >= len(idx.offsets): raise IndexError(i)
        return json.loads(idx.read(i))

    def file_step(self, f):
        m = re.search(r"(\d+)", f); return int(m.group(1)) if m else None

    def with_metrics(self, run, f, rows):
        ts = self.file_step(f); has = self.metrics.has_step(run, ts)
        if not has: return rows, False
        out = []
        for r in rows:
            m = self.metrics.rollout(run, r.get("target_step", ts), r.get("sample_id")); r2 = dict(r); r2["trained"] = m is not None
            if m: r2.update(m)
            out.append(r2)
        return out, True

def fmt_error(code, msg): return code, {"error": msg}

class Handler(BaseHTTPRequestHandler):
    app = None
    def log_message(self, fmt, *args): pass
    def send_json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def send_bytes(self, b, ctype, code=200, fname=None):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        if fname: self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self.send_bytes(open(os.path.join(HERE, "app.html"), "rb").read(), "text/html; charset=utf-8")
            self.app.metrics.maybe_refresh()
            if u.path == "/api/runs": return self.send_json(self.app.runs())
            if u.path == "/api/metrics": return self.send_json(self.app.metrics.status())
            if u.path == "/api/files": return self.send_json(self.app.files(q["run"]))
            if u.path == "/api/index":
                idx = self.app.index(q["run"], q["file"], start=q.get("start", "1") == "1"); return self.send_json(idx.status())
            if u.path == "/api/records":
                idx = self.app.index(q["run"], q["file"], start=True)
                rows = [r for r in idx.summaries if r is not None]
                if not rows: return self.send_json({"status": idx.status(), "rows": [], "n": 0})
                lf = q.get("loop", "any")
                if lf == "loop": rows = [r for r in rows if r.get("loop_turns")]
                elif lf == "runaway": rows = [r for r in rows if r.get("runaway")]
                elif lf == "none": rows = [r for r in rows if not r.get("loop_turns") and not r.get("runaway")]
                rows, has_metrics = self.app.with_metrics(q["run"], q["file"], rows)
                mf = q.get("mask", "any")
                if has_metrics and mf != "any":
                    if mf in ("X", "Y", "Z"): rows = [r for r in rows if r.get(f"in_{mf}_mask_rank") is not None]
                    elif mf == "deep": rows = [r for r in rows if r.get("n_deep_turns")]
                    elif mf == "untrained": rows = [r for r in rows if r.get("trained") is False]
                    elif mf == "trained": rows = [r for r in rows if r.get("trained")]
                if q.get("filter"):
                    fl = q["filter"].lower(); rows = [r for r in rows if fl in json.dumps({k: v for k, v in r.items() if k != "bytes"}).lower()]
                for k in ("reward", "truncated", "agent_error_kind", "resolved"):
                    if q.get(k) not in (None, "", "any"):
                        want = q[k]; rows = [r for r in rows if str(r.get(k)).lower() == want.lower()]
                sort = q.get("sort", "i"); desc = q.get("desc", "0") == "1"
                have = [r for r in rows if r.get(sort) is not None]; miss = [r for r in rows if r.get(sort) is None]
                if all(isinstance(r.get(sort), (int, float)) for r in have): have.sort(key=lambda r: r.get(sort), reverse=desc)
                else: have.sort(key=lambda r: str(r.get(sort)), reverse=desc)
                rows = have + miss
                off = int(q.get("offset", 0)); lim = int(q.get("limit", 100))
                return self.send_json({"status": idx.status(), "n": len(rows), "rows": rows[off:off + lim], "has_metrics": has_metrics, "arm": self.app.labels.get(q["run"])})
            if u.path == "/api/record":
                rec = self.app.record(q["run"], q["file"], int(q["i"]))
                ts = self.app.file_step(q["file"]); has = self.app.metrics.has_step(q["run"], ts); m = self.app.metrics.rollout(q["run"], rec.get("target_step", ts), rec.get("sample_id")) if has else None
                return self.send_json({"i": int(q["i"]), "n": len(self.app.index(q["run"], q["file"], start=False).offsets), "record": elide(redact(rec)), "turn_paths": turn_paths(rec), "loop": loop_stats(rec), "has_metrics": has, "trained": (m is not None) if has else None, "metrics": m})
            if u.path == "/api/turns":
                rec = self.app.record(q["run"], q["file"], int(q["i"]))
                T = turns_of(rec); ts = self.app.file_step(q["file"]); has = self.app.metrics.has_step(q["run"], ts)
                tm = self.app.metrics.turn_rows(q["run"], rec.get("target_step", ts), rec.get("sample_id")) if has else {}
                for t in T:
                    mt = tm.get(t["k"])
                    if mt: t.update(mt)
                m = self.app.metrics.rollout(q["run"], rec.get("target_step", ts), rec.get("sample_id")) if has else None
                return self.send_json({"turns": T, "paths": turn_paths(rec), "has_metrics": has, "adv": m.get("adv") if m else None})
            if u.path == "/api/value":
                rec = self.app.record(q["run"], q["file"], int(q["i"]))
                v = get_path(redact(rec), q["path"])
                return self.send_json({"path": q["path"], "value": v})
            if u.path == "/api/raw":
                idx = self.app.index(q["run"], q["file"], start=False); i = int(q["i"])
                return self.send_bytes(json.dumps(redact(json.loads(idx.read(i)))).encode(), "application/json", fname=f"{q['file']}.{i}.json")
            if u.path == "/api/group":
                idx = self.app.index(q["run"], q["file"], start=False)
                rows, has_metrics = self.app.with_metrics(q["run"], q["file"], [r for r in idx.summaries if r is not None and r.get("group_id") == q["group_id"]])
                agg = next(({k: r.get(k) for k in GROUP_COLS} for r in rows if r.get("trained") and r.get("group_net_contrib") is not None), None)
                return self.send_json({"rows": rows, "has_metrics": has_metrics, "group": agg})
            if u.path == "/api/search":
                ql = q.get("q", "").lower(); run = q.get("run"); res = []
                for cp in glob.glob(os.path.join(self.app.cache, "*.idx.json")):
                    base = os.path.basename(cp)[:-len(".idx.json")]; r, _, f = base.partition("__")
                    if run and r != run: continue
                    try: d = json.load(open(cp))
                    except Exception: continue
                    for s in d["summaries"]:
                        if s is None: continue
                        if ql in str(s.get("instance_id", "")).lower() or ql in str(s.get("sample_id", "")).lower() or ql in str(s.get("group_id", "")).lower() or ql == str(s.get("prompt_idx")):
                            res.append({"run": r, "file": f, **s})
                    if len(res) > 2000: break
                return self.send_json(res)
            self.send_json({"error": "not found"}, 404)
        except (KeyError, IndexError, ValueError) as e:
            self.send_json({"error": f"bad request: {e!r}"}, 400)
        except Exception as e:
            self.send_json({"error": repr(e)}, 500)

def main():
    ap = argparse.ArgumentParser(description="Browse SWE dump rollout jsonl files")
    ap.add_argument("--root", default=DEFAULT_ROOT); ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--metrics", default=DEFAULT_METRICS); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    Handler.app = App(a.root, a.cache, a.metrics)
    srv = ThreadingHTTPServer((a.host, a.port), Handler); srv.daemon_threads = True
    print(f"dumpbrowse serving {a.root} on http://{a.host}:{a.port}  (cache {a.cache})", flush=True)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass

if __name__ == "__main__": main()
