"""In-container check of the masked-replay overlay.

usage: test_mask_replay.py <patches_dir> <mask_csv> <source_run> <target_step>

1. the mounted modules are the patched copies (md5);
2. the mask CSV loads and the per-step expected counts match its target_step column;
3. for every listed sample of <target_step>, ForeignRolloutSource.lookup packs its
   group with sample_mask 0 exactly on the listed rows, all rows kept, every other
   tensor identical to an unmasked pack; the step's hit count equals the CSV count;
4. grpo.compute_and_apply_seq_logprob_error_masking ANDs a pre-zeroed sample_mask
   (the zero survives the threshold rewrite);
5. GRPOAdvantageEstimator's baseline ignores the mask (siblings' leave-one-out
   advantages are unchanged by masking a member).
"""
import hashlib
import inspect
import sys
from types import SimpleNamespace

import torch

patches, mask_csv, source_run, target_step = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
fails = []


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


import nemo_rl.experience.foreign_rollout_source as frs
import nemo_rl.algorithms.single_controller as scmod

for mod, rel in ((frs, "nemo_rl/nemo_rl/experience/foreign_rollout_source.py"), (scmod, "nemo_rl/nemo_rl/algorithms/single_controller.py")):
    same = md5(mod.__file__) == md5(f"{patches}/{rel}")
    print(f"{'patched' if same else 'NOT PATCHED'}: {mod.__file__}")
    if not same:
        fails.append(f"overlay missing for {rel}")
if "mask_file=foreign_rollout_cfg.get" not in inspect.getsource(scmod):
    fails.append("single_controller does not forward mask_file")

ids, expected = frs.ForeignRolloutSource._load_mask_file(mask_csv)
print(f"mask csv: {len(ids)} sample_ids, expected per target_step {dict(sorted(expected.items()))}")
import csv
rows_csv = [r for r in csv.DictReader(open(mask_csv))]
by_step = {}
for r in rows_csv:
    by_step.setdefault(int(r["target_step"]), []).append(r["sample_id"])
if sorted(expected.items()) != sorted((k, len(v)) for k, v in by_step.items()):
    fails.append("expected-per-step disagrees with the CSV")

src = frs.ForeignRolloutSource(source_run, range(target_step + 1, target_step + 2), pad_token_id=0, mask_file=mask_csv)
inv = {gid: pidx for (t, pidx), gid in src._index.items() if t == target_step}
rows = src._get_step_rows(target_step)
print(f"target_step {target_step}: {len(inv)} groups indexed, {len(rows)} rows loaded")
listed = by_step.get(target_step, [])
seen_groups = set()
for sid in listed:
    gid = sid.rsplit("_g", 1)[0]
    if sid not in rows:
        fails.append(f"{sid} not in step rows"); continue
    if rows[sid]["group_id"] != gid:
        fails.append(f"{sid} group_id mismatch {rows[sid]['group_id']}"); continue
    if gid in seen_groups:
        continue
    seen_groups.add(gid)
    pidx = inv.get(gid)
    if pidx is None:
        fails.append(f"group {gid} not in index"); continue
    members = [(s, r) for s, r in rows.items() if r["group_id"] == gid]
    want = torch.tensor([0.0 if s in ids else 1.0 for s, _ in members])
    tb, tags = src.lookup(target_step, pidx)
    got = tb["sample_mask"]
    ok_mask = got.shape == want.shape and torch.equal(got, want) and got.dtype == torch.float32
    saved = src._mask_ids
    src._mask_ids = set()
    tb0, tags0 = src.lookup(target_step, pidx)
    src._mask_ids = saved
    same_rest = all(torch.equal(tb[k], tb0[k]) for k in tb if k != "sample_mask") and tags == tags0 and bool(torch.all(tb0["sample_mask"] == 1.0))
    n_masked = int((got == 0).sum())
    print(f"  group {gid[:8]} prompt_idx={pidx}: rows={len(members)} masked={n_masked} mask_ok={ok_mask} others_identical={same_rest} rewards_of_masked={[round(float(tb['total_reward'][i]),3) for i in range(len(members)) if got[i]==0]}")
    if not ok_mask:
        fails.append(f"sample_mask wrong for group {gid}")
    if not same_rest:
        fails.append(f"masking changed other fields for group {gid}")
    if len(members) != len(tags):
        fails.append(f"row/tag count mismatch for group {gid}")
hits = len(src._mask_hits_by_step.get(target_step, set()))
print(f"step {target_step}: masked hits {hits} expected {len(listed)}")
if hits != len(listed):
    fails.append(f"hit count {hits} != {len(listed)}")

from nemo_rl.algorithms.grpo import compute_and_apply_seq_logprob_error_masking
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
n, L = 4, 6
token_mask = torch.ones((n, L)); token_mask[:, 0] = 0
gen = torch.zeros((n, L)); prev = torch.zeros((n, L)); prev[3, 1:] = -3.0
md = BatchedDataDict({"token_mask": token_mask, "sample_mask": torch.tensor([1.0, 0.0, 1.0, 1.0]), "prev_logprobs": prev, "generation_logprobs": gen})
m = compute_and_apply_seq_logprob_error_masking(train_data=md, rewards=torch.tensor([1.0, 1.0, 0.0, 1.0]), seq_logprob_error_threshold=2.0, step=0)
after = md["sample_mask"]
print(f"seq-error masking: before [1,0,1,1] -> after {after.tolist()} num_masked_seqs={m.get('num_masked_seqs')}")
if after.tolist() != [1.0, 0.0, 1.0, 0.0]:
    fails.append(f"seq-error masking did not preserve the replayed zero: {after.tolist()}")

from nemo_rl.algorithms.advantage_estimator import GRPOAdvantageEstimator
est = GRPOAdvantageEstimator(SimpleNamespace(use_leave_one_out_baseline=True, normalize_rewards=True), SimpleNamespace())
prompt_ids = torch.full((16, 8), 7, dtype=torch.long); rewards = torch.tensor([1.0] * 5 + [0.0] * 11)
mask_a = torch.ones((16, 4)); mask_b = mask_a.clone(); mask_b[2] = 0
adv_a = est.compute_advantage(prompt_ids=prompt_ids, rewards=rewards, mask=mask_a, repeated_batch={})
adv_b = est.compute_advantage(prompt_ids=prompt_ids, rewards=rewards, mask=mask_b, repeated_batch={})
same_adv = torch.allclose(adv_a, adv_b)
print(f"estimator: advantages identical with a member masked: {same_adv}; baseline mask hard-coded ones: {'torch.ones_like(rewards)' in inspect.getsource(GRPOAdvantageEstimator.compute_advantage)}")
if not same_adv:
    fails.append("advantage estimator depends on sample_mask")

import nemo_rl.experience.rollout_dump as rd
for mod, rel in ((rd, "nemo_rl/nemo_rl/experience/rollout_dump.py"),):
    same = md5(mod.__file__) == md5(f"{patches}/{rel}")
    print(f"{'patched' if same else 'NOT PATCHED'}: {mod.__file__}")
    if not same:
        fails.append(f"overlay missing for {rel}")
import nemo_rl.algorithms.single_controller_utils.config as scu_cfg, nemo_rl.algorithms.single_controller_utils.setup as scu_setup
for mod, rel in ((scu_cfg, "nemo_rl/nemo_rl/algorithms/single_controller_utils/config.py"), (scu_setup, "nemo_rl/nemo_rl/algorithms/single_controller_utils/setup.py")):
    same = md5(mod.__file__) == md5(f"{patches}/{rel}")
    print(f"{'patched' if same else 'NOT PATCHED'}: {mod.__file__}")
    if not same:
        fails.append(f"overlay missing for {rel}")
msgs = [
    {"role": "user", "token_ids": torch.tensor([1, 2, 3]), "content": "u"},
    {"role": "assistant", "token_ids": torch.tensor([4, 5]), "generation_token_ids": [4, 5], "prompt_token_ids": [1, 2, 3], "content": "a1", "is_invalid_tool_call": False},
    {"role": "tool", "token_ids": [6, 7, 8, 9], "content": "t"},
    {"role": "assistant", "token_ids": [10], "generation_token_ids": torch.tensor([10]), "prompt_token_ids": list(range(1, 10)), "compact_prompt_token_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9], "content": "a2"},
]
off_new = [rd._message_summary(m, False, "off") for m in msgs]
off_legacy = [rd._message_summary(m, False) for m in msgs]
legacy_keys = {"role", "n_tokens", "is_invalid_tool_call"}
off_ok = off_new == off_legacy and all(set(o).issubset(legacy_keys) for o in off_new) and [o["n_tokens"] for o in off_new] == [3, 2, 4, 1]
print(f"rollout_dump off mode == legacy: {off_ok} keys={[sorted(o) for o in off_new]}")
if not off_ok:
    fails.append("token_ids off mode changed the legacy summary")
dig = [rd._message_summary(m, False, "digest") for m in msgs]
d_ok = (dig[1]["token_ids"] == [4, 5] and dig[1]["generation_token_ids"] == [4, 5] and dig[1]["prompt_token_ids_digest"]["len"] == 3
        and dig[3]["compact_prompt_token_ids_digest"]["sha1"] == rd._id_digest([1, 2, 3, 4, 5, 6, 7, 8, 9])["sha1"] and "prompt_token_ids" not in dig[1])
run = []
for m in dig:
    if m["role"] == "assistant" and "prompt_token_ids_digest" in m:
        if m["prompt_token_ids_digest"]["sha1"] != rd._id_digest(run)["sha1"]:
            d_ok = False
    run += m.get("token_ids", [])
full = [rd._message_summary(m, False, "full") for m in msgs]
f_ok = full[3]["prompt_token_ids"] == list(range(1, 10)) and full[3]["compact_prompt_token_ids"] == [1, 2, 3, 4, 5, 6, 7, 8, 9] and full[1]["token_ids"] == [4, 5]
print(f"rollout_dump digest ok (verbatim token/generation ids, digests, prefix recipe): {d_ok}; full ok: {f_ok}")
if not d_ok:
    fails.append("digest mode output wrong")
if not f_ok:
    fails.append("full mode output wrong")
print("RolloutDumpWriter accepts token_ids_mode:", "token_ids_mode" in inspect.signature(rd.RolloutDumpWriter.__init__).parameters)
if "token_ids_mode" not in inspect.signature(rd.RolloutDumpWriter.__init__).parameters:
    fails.append("RolloutDumpWriter lacks token_ids_mode")

for f in fails:
    print("FAIL:", f)
print("RESULT", "PASS" if not fails else "FAIL")
