"""Re-parse every assistant turn in the rollout dumps with both engines' Qwen3-coder tool parsers.

Usage: parser_ab.py OUT_DIR FORK_ML_DIR VLLM_SRC_DIR TOKENIZER_JSON FILE [FILE ...]
Env: LIMIT_ROWS (test), NPROC (default 8).
Per file: OUT_DIR/<run>__<file>.turns.csv and OUT_DIR/<run>__<file>.ab.json
"""
from __future__ import annotations

import csv
import importlib.util
import json
import logging
import os
import re
import sys
import types
from collections import Counter, defaultdict
from multiprocessing import Pool

try:
    import orjson

    loads = orjson.loads
except Exception:  # noqa: BLE001
    loads = json.loads

logging.basicConfig(level=logging.ERROR)
FORK = None
VLLM = None
VTOK = None


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _stub(name, cls=types.ModuleType, **attrs):
    m = cls(name)
    m.__path__ = []
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def load_fork(ml_dir):
    class BaseParser:
        @staticmethod
        def parse(text, **kw):
            raise NotImplementedError

    for n in ["megatron", "megatron.core", "megatron.core.tokenizers", "megatron.core.tokenizers.text", "megatron.core.tokenizers.text.parsers"]:
        if n not in sys.modules:
            _stub(n)
    _stub("megatron.core.tokenizers.text.parsers.base_parser", BaseParser=BaseParser)
    p = os.path.join(ml_dir, "megatron/core/tokenizers/text/parsers/qwen3_coder_tool_parser.py")
    spec = importlib.util.spec_from_file_location("fork_qwen3", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_vllm(src_dir, vocab):
    if "regex" not in sys.modules:
        try:
            import regex  # noqa: F401
        except Exception:  # noqa: BLE001
            sys.modules["regex"] = re

    class ToolParser:
        def __init__(self, tokenizer, tools=None):
            self.model_tokenizer = tokenizer
            self.tools = tools

        @property
        def vocab(self):
            return self.model_tokenizer.get_vocab()

    class ToolParserManager:
        @staticmethod
        def register_module(*a, **k):
            def deco(cls):
                return cls
            return deco

    def find_tool_properties(tools, tool_name):
        for t in tools or []:
            fn = getattr(t, "function", None)
            if fn is not None and getattr(fn, "name", None) == tool_name:
                return (getattr(fn, "parameters", None) or {}).get("properties", {})
        return {}

    class Dummy:
        def __init__(self, *a, **k):
            self.__dict__.update(k)

    class StubModule(types.ModuleType):
        def __getattr__(self, item):
            if item.startswith("__"):
                raise AttributeError(item)
            return Dummy

    for n in ["vllm", "vllm.entrypoints", "vllm.entrypoints.openai", "vllm.entrypoints.openai.chat_completion", "vllm.entrypoints.openai.engine", "vllm.tool_parsers"]:
        _stub(n, StubModule)
    _stub("vllm.entrypoints.openai.chat_completion.protocol", StubModule)
    _stub("vllm.entrypoints.openai.engine.protocol", StubModule, ExtractedToolCallInformation=_Obj, FunctionCall=_Obj, ToolCall=_Obj)
    _stub("vllm.logger", StubModule, init_logger=logging.getLogger)
    _stub("vllm.tokenizers", StubModule, TokenizerLike=object)
    _stub("vllm.tool_parsers.abstract_tool_parser", StubModule, Tool=object, ToolParser=ToolParser, ToolParserManager=ToolParserManager)
    _stub("vllm.tool_parsers.utils", StubModule, find_tool_properties=find_tool_properties)
    p = os.path.join(src_dir, "tool_parsers/qwen3coder_tool_parser.py")
    spec = importlib.util.spec_from_file_location("vllm_qwen3", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    class Tok:
        def get_vocab(self):
            return vocab

    return m, Tok()


def vocab_from_tokenizer(tok_path):
    j = json.load(open(tok_path))
    v = dict(j["model"]["vocab"])
    for t in j.get("added_tokens", []):
        v[t["content"]] = t["id"]
    return v


def norm_calls(calls):
    out = []
    for tc in calls or []:
        if isinstance(tc, dict):
            fn = tc.get("function", {}) or {}
            name, args = fn.get("name"), fn.get("arguments")
        else:
            fn = getattr(tc, "function", None)
            name, args = getattr(fn, "name", None), getattr(fn, "arguments", None)
        try:
            a = json.loads(args) if isinstance(args, str) else args
        except Exception:  # noqa: BLE001
            a = {"__unparsed__": str(args)[:200]}
        out.append((name, a))
    return out


def _loose(d):
    if not isinstance(d, dict):
        return json.dumps(d, sort_keys=True, default=str)
    return {k: (v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)) for k, v in d.items()}


def same(a, b, loose=False):
    if len(a) != len(b):
        return False
    for (n1, x), (n2, y) in zip(a, b):
        if n1 != n2:
            return False
        if loose:
            if _loose(x) != _loose(y):
                return False
        elif x != y:
            return False
    return True


ERR_PATTERNS = [
    ("validation", re.compile(r"ValidationFailure|Missing required|validation error|Invalid arguments|is not a valid|Input should be|Field required|Extra inputs are not permitted", re.I)),
    ("not_exists", re.compile(r"not registered|Unknown tool|does not exist|not found in the tool", re.I)),
    ("json", re.compile(r"Failed to parse|JSONDecodeError|Invalid JSON|Expecting value", re.I)),
    ("timeout", re.compile(r"timed out|timeout", re.I)),
    ("other_error", re.compile(r"^\s*(ERROR|Error)\b|\bERROR:\s|\berror:\s", re.I)),
]


def classify_output(text):
    head = text[:400]
    for name, pat in ERR_PATTERNS:
        if pat.search(head):
            return name
    return "ok"


def text_of(o):
    c = o.get("content")
    if isinstance(c, list):
        return "".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
    return str(c or "")


def reasoning_text(o):
    return "".join(str(s.get("text", "")) for s in (o.get("summary") or []) if isinstance(s, dict))


def convert_tools(resp_tools):
    chat, vobj = [], []
    for t in resp_tools or []:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        name = t.get("name") or (t.get("function") or {}).get("name")
        params = t.get("parameters") if "parameters" in t else (t.get("function") or {}).get("parameters")
        desc = t.get("description") or (t.get("function") or {}).get("description")
        chat.append({"type": "function", "function": {"name": name, "description": desc, "parameters": params}})
        vobj.append(_Obj(type="function", function=_Obj(name=name, parameters=params, description=desc)))
    return chat, vobj


def process(args):
    global FORK, VLLM, VTOK
    out_dir, ml_dir, vsrc, tok_path, path = args
    if FORK is None:
        FORK = load_fork(ml_dir)
        VLLM, VTOK = load_vllm(vsrc, vocab_from_tokenizer(tok_path))
    run = path.split("/runs/")[1].split("/")[0]
    name = os.path.basename(path).replace(".jsonl", "")
    eng = "vllm" if "vllm_dump" in run else "minf"
    limit = int(os.environ.get("LIMIT_ROWS", "0") or 0)
    rows = []
    longturns = []
    rrows = []
    agg = Counter()
    err_heads = Counter()
    groups = defaultdict(list)
    tools_cache = {}
    step = None
    with open(path, "rb") as fh:
        for li, line in enumerate(fh):
            if limit and li >= limit:
                break
            row = loads(line)
            step = row.get("target_step")
            fr = row.get("full_result") or {}
            out = (fr.get("response") or {}).get("output") or []
            rt = (fr.get("responses_create_params") or {}).get("tools") or []
            key = json.dumps(rt, sort_keys=True)[:2000]
            if key not in tools_cache:
                tools_cache[key] = convert_tools(rt)
            tools_chat, tools_v = tools_cache[key]
            vparser = VLLM.Qwen3CoderToolParser(VTOK, tools=tools_v)
            vreq = _Obj(tools=tools_v)
            fparser = FORK._Qwen3CoderToolParser()
            sid = row.get("sample_id")
            gid = row.get("group_id")
            i, n, turn = 0, len(out), 0
            r_nudges = r_errs = r_invalid = 0
            r_maxturn_chars = 0
            while i < n:
                o = out[i]
                if o.get("type") == "message" and o.get("role") == "user":
                    agg["nudges"] += 1
                    r_nudges += 1
                    i += 1
                    continue
                if o.get("type") == "function_call_output":
                    i += 1
                    continue
                items = []
                while i < n and out[i].get("type") in ("reasoning", "message", "function_call") and not (out[i].get("type") == "message" and out[i].get("role") == "user"):
                    items.append(out[i])
                    i += 1
                outputs = []
                while i < n and out[i].get("type") == "function_call_output":
                    outputs.append(out[i])
                    i += 1
                nudge_after = i < n and out[i].get("type") == "message" and out[i].get("role") == "user"
                turn += 1
                fcs = [x for x in items if x.get("type") == "function_call"]
                msgs = [x for x in items if x.get("type") == "message"]
                rsn = [x for x in items if x.get("type") == "reasoning"]
                gen = next((x.get("generation_str") for x in reversed(fcs) if x.get("generation_str")), None)
                recorded = []
                for x in fcs:
                    try:
                        a = json.loads(x.get("arguments") or "{}")
                    except Exception:  # noqa: BLE001
                        a = {"__unparsed__": str(x.get("arguments"))[:200]}
                    recorded.append((x.get("name"), a))
                think_close = None
                tc_in_think = False
                ends_im_end = None
                think_chars = sum(len(reasoning_text(x)) for x in rsn)
                if gen is not None:
                    raw = gen
                    ends_im_end = raw.rstrip().endswith("<|im_end|>")
                    raw = raw.rstrip()
                    if raw.endswith("<|im_end|>"):
                        raw = raw[: -len("<|im_end|>")]
                    think_close = "</think>" in raw
                    if think_close:
                        think_part, content = raw.split("</think>", 1)
                    else:
                        think_part, content = raw, ""
                    tc_in_think = "<tool_call>" in think_part
                    msg_only = False
                else:
                    content = "".join(text_of(x) for x in msgs)
                    msg_only = True
                try:
                    finfo = fparser.extract_tool_calls(content, tools=tools_chat)
                    fcalls = norm_calls(finfo.get("tool_calls") if isinstance(finfo, dict) else getattr(finfo, "tool_calls", None))
                except Exception as e:  # noqa: BLE001
                    fcalls = [("__error__", {"e": str(e)[:100]})]
                try:
                    vinfo = vparser.extract_tool_calls(content, vreq)
                    vcalls = norm_calls(getattr(vinfo, "tool_calls", None))
                except Exception as e:  # noqa: BLE001
                    vcalls = [("__error__", {"e": str(e)[:100]})]
                fco = "none"
                if outputs:
                    cats = [classify_output(text_of(x) if isinstance(x.get("output"), (list, dict)) else str(x.get("output") or "")) for x in outputs]
                    fco = next((c for c in cats if c != "ok"), "ok")
                    for x, c in zip(outputs, cats):
                        if c != "ok":
                            t = str(x.get("output") or "")
                            err_heads[c + " | " + t[:70].replace("\n", " ")] += 1
                invalid_flag = (not fcs) and any(("<tool_call>" in text_of(m) or "<function=" in text_of(m)) for m in msgs)
                rec = {
                    "engine": eng, "run": run, "step": step, "sample_id": sid, "group_id": gid, "turn": turn,
                    "n_rec": len(recorded), "n_fork": len(fcalls), "n_vllm": len(vcalls),
                    "fork_eq_rec": same(fcalls, recorded), "fork_eq_rec_loose": same(fcalls, recorded, True),
                    "vllm_eq_rec": same(vcalls, recorded), "vllm_eq_rec_loose": same(vcalls, recorded, True),
                    "fork_eq_vllm": same(fcalls, vcalls), "fork_eq_vllm_loose": same(fcalls, vcalls, True),
                    "msg_only": msg_only, "has_reasoning": bool(rsn), "think_chars": think_chars, "think_close": think_close,
                    "tc_in_think": tc_in_think, "ends_im_end": ends_im_end, "n_tc_open": content.count("<tool_call>"),
                    "n_tc_close": content.count("</tool_call>"), "n_fn": content.count("<function="), "content_chars": len(content),
                    "invalid_flag": invalid_flag, "fco": fco, "nudge_after": nudge_after,
                    "names": "|".join(str(nm) for nm, _ in recorded)[:120],
                }
                rows.append(rec)
                r_errs += (fco not in ("ok", "none"))
                r_invalid += invalid_flag
                if gen is not None:
                    r_maxturn_chars = max(r_maxturn_chars, len(gen))
                if gen is not None and len(gen) > 30000:
                    import zlib
                    tail = gen[-20000:].encode()
                    longturns.append({"engine": eng, "step": step, "sample_id": sid, "turn": turn, "chars": len(gen), "think_chars": len(think_part), "content_chars": len(content),
                                      "think_close": think_close, "ends_im_end": ends_im_end, "n_rec": len(recorded), "fco": fco,
                                      "zlib_ratio_tail": round(len(zlib.compress(tail, 6)) / max(1, len(tail)), 4),
                                      "zlib_ratio_all": round(len(zlib.compress(gen.encode(), 6)) / max(1, len(gen)), 4),
                                      "tail": gen[-400:]})
                agg["turns"] += 1
                agg[f"fco_{fco}"] += 1
                agg["msg_only"] += msg_only
                agg["invalid_flag"] += invalid_flag
                agg["tc_in_think"] += tc_in_think
                agg["no_think_close"] += (think_close is False)
                agg["no_reasoning_item"] += (not rsn)
                agg["nudge_after"] += nudge_after
                agg["fork_ne_rec"] += (not rec["fork_eq_rec"])
                agg["fork_ne_rec_loose"] += (not rec["fork_eq_rec_loose"])
                agg["vllm_ne_rec"] += (not rec["vllm_eq_rec"])
                agg["vllm_ne_rec_loose"] += (not rec["vllm_eq_rec_loose"])
                agg["fork_ne_vllm"] += (not rec["fork_eq_vllm"])
                agg["fork_ne_vllm_loose"] += (not rec["fork_eq_vllm_loose"])
                agg["count_diff_fork_vllm"] += (len(fcalls) != len(vcalls))
                if turn == 1:
                    key1 = (content if content else "".join(reasoning_text(x) for x in rsn))[:200]
                    groups[gid].append(key1)
            agg["rows"] += 1
            pm = row.get("prompt_metadata") or {}
            rrows.append({"engine": eng, "run": run, "step": step, "sample_id": sid, "group_id": gid, "instance_id": pm.get("instance_id"), "prompt_idx": row.get("prompt_idx"),
                          "start_wv": row.get("start_weight_version"), "end_wv": row.get("end_weight_version"), "reward": row.get("reward"), "truncated": row.get("truncated"),
                          "turns": row.get("num_assistant_turns"), "gen_len": row.get("generation_length"), "total_tokens": row.get("total_tokens"),
                          "agent_error_kind": fr.get("agent_error_kind"), "resolved": fr.get("resolved"), "n_out_items": len(out), "n_turns_seen": turn,
                          "n_nudges": r_nudges, "n_fco_errors": r_errs, "n_invalid": r_invalid, "maxturn_chars": r_maxturn_chars})
    # group diversity
    gstats = Counter()
    for gid, keys in groups.items():
        gstats["groups"] += 1
        gstats["completions"] += len(keys)
        gstats["distinct_first200"] += len(set(keys))
        gstats["groups_with_dup"] += (len(set(keys)) < len(keys))
    os.makedirs(out_dir, exist_ok=True)
    if rows:
        with open(os.path.join(out_dir, f"{run}__{name}.turns.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if rrows:
        with open(os.path.join(out_dir, f"{run}__{name}.rollouts.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rrows[0].keys()))
            w.writeheader()
            w.writerows(rrows)
    with open(os.path.join(out_dir, f"{run}__{name}.longturns.jsonl"), "w") as fh:
        for lt in longturns:
            fh.write(json.dumps(lt) + "\n")
    json.dump({"run": run, "file": name, "engine": eng, "step": step, "agg": dict(agg), "err_heads": err_heads.most_common(40), "group_diversity": dict(gstats)},
              open(os.path.join(out_dir, f"{run}__{name}.ab.json"), "w"), indent=1)
    return path, dict(agg)


def main():
    out_dir, ml_dir, vsrc, tok_path, files = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
    nproc = min(int(os.environ.get("NPROC", "8")), len(files))
    with Pool(nproc) as pool:
        for path, agg in pool.imap_unordered(process, [(out_dir, ml_dir, vsrc, tok_path, f) for f in files]):
            print(f"done {path.rsplit('/', 3)[-3]}/{os.path.basename(path)}: turns={agg.get('turns')} fork_ne_vllm={agg.get('fork_ne_vllm')} vllm_ne_rec={agg.get('vllm_ne_rec')} fork_ne_rec={agg.get('fork_ne_rec')} nudges={agg.get('nudges')}", flush=True)


if __name__ == "__main__":
    main()
