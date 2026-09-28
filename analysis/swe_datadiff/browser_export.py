import glob, os, sys, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__))
RUNS = {}
for fn in glob.glob(f'{D}/jobs_*.txt'):
    for line in open(fn):
        p = line.split()
        if len(p) >= 2: RUNS[p[0]] = p[1]
arms = sys.argv[1].split(',') if len(sys.argv) > 1 else sorted({f.split('/')[-1].split('__')[0] for f in glob.glob(f'{D}/parts/*.eff.csv')})
masks = {'X': ('poison_G_rewarded_deep_steps1-10.csv', 'G'), 'Y': ('control_G_random_rewarded_steps1-10.csv', 'G'), 'Z': ('poison_M_punished_deep_steps1-10.csv', 'M')}
mask_rank = {}
for tag, (fn, src) in masks.items():
    m = pd.read_csv(f'{D}/{fn}')
    mask_rank[tag] = (src, dict(zip(m.sample_id, m['rank'])))
allr, allt = [], []
for a in arms:
    fs = sorted(glob.glob(f'{D}/parts/{a}__s*_c*.eff.csv'))
    if not fs: continue
    eff = pd.concat([pd.read_csv(f).assign(train_step=int(f.split('__s')[1][:3])) for f in fs])
    smp = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{D}/parts/{a}__s*_c*.samples.csv'))]).rename(columns={'step': 'train_step'})
    smp = smp[smp.train_step.isin(eff.train_step.unique())]
    eff['deep_tokens'] = np.maximum(0, eff.think_tok - 4096)
    eff['live_deep_1mp'] = np.where(eff.deep_tokens > 0, eff.s1mp_deep_tr, 0.0)
    eff['turn_contrib'] = eff.adv * eff.live_deep_1mp
    step_tok = eff.groupby('train_step').n_tok.sum()
    g = eff.groupby(['train_step', 'sample_id']).agg(think_tok_total=('think_tok', 'sum'), max_think_turn=('think_tok', 'max'), n_deep_turns=('deep_tokens', lambda x: int((x > 0).sum())), deep_think_tokens=('deep_tokens', 'sum'), live_deep_1mp=('live_deep_1mp', 'sum'), n_long_turns=('think_tok', lambda x: int((x >= 1000).sum()))).reset_index()
    r = smp.merge(g, on=['train_step', 'sample_id'], how='left').fillna({'think_tok_total': 0, 'max_think_turn': 0, 'n_deep_turns': 0, 'deep_think_tokens': 0, 'live_deep_1mp': 0.0, 'n_long_turns': 0})
    r['mean_1mp_deep'] = np.where(r.deep_think_tokens > 0, r.live_deep_1mp / r.deep_think_tokens.clip(lower=1), np.nan)
    r['r_deep_contrib'] = r.adv * r.live_deep_1mp
    r['r_deep_contrib_per1k_step'] = 1e3 * r.r_deep_contrib / r.train_step.map(step_tok)
    r['absadv_x_live'] = r.adv.abs() * r.live_deep_1mp
    r['mask_metric_X'] = np.where((r.adv > 0) & (r.deep_think_tokens > 0), r.absadv_x_live, np.nan)
    r['mask_metric_Z'] = np.where((r.adv < 0) & (r.deep_think_tokens > 0), r.absadv_x_live, np.nan)
    early = r.train_step.between(1, 10)
    r['rank_X_metric_steps1_10'] = r.mask_metric_X.where(early).rank(ascending=False, method='first')
    r['rank_Z_metric_steps1_10'] = r.mask_metric_Z.where(early).rank(ascending=False, method='first')
    r['rank_X_metric_all_steps'] = r.mask_metric_X.rank(ascending=False, method='first')
    r['rank_Z_metric_all_steps'] = r.mask_metric_Z.rank(ascending=False, method='first')
    for tag, (src, mp) in mask_rank.items():
        r[f'in_{tag}_mask_rank'] = r.sample_id.map(mp) if a == src else np.nan
    grp = r.groupby(['train_step', 'group_id']).agg(group_net_contrib=('r_deep_contrib', 'sum'), group_pos_contrib=('r_deep_contrib', lambda x: x[x > 0].sum()), group_neg_contrib=('r_deep_contrib', lambda x: x[x < 0].sum()), group_n_deep_rollouts=('deep_think_tokens', lambda x: int((x > 0).sum())), group_mean_reward=('reward', 'mean')).reset_index()
    grp['group_cancel_ratio'] = np.where(grp.group_pos_contrib > 0, -grp.group_neg_contrib / grp.group_pos_contrib, np.nan)
    grp['group_rank_net_steps1_10'] = grp.group_net_contrib.where(grp.train_step.between(1, 10)).rank(ascending=False, method='first')
    r = r.merge(grp, on=['train_step', 'group_id'], how='left')
    r['arm'] = a; r['run'] = RUNS.get(a, '')
    r['target_step'] = r.train_step - 1
    r['loop_share_pct'] = 100 * r.loop_tok / r.n_gen.clip(lower=1)
    cols = ['arm', 'run', 'train_step', 'target_step', 'sample_id', 'group_id', 'reward', 'adv', 'n_gen', 'n_turns', 'max_turn', 'think_tok_total', 'max_think_turn', 'n_long_turns', 'n_deep_turns', 'deep_think_tokens', 'live_deep_1mp', 'mean_1mp_deep', 'r_deep_contrib', 'r_deep_contrib_per1k_step', 'absadv_x_live', 'mask_metric_X', 'mask_metric_Z', 'rank_X_metric_steps1_10', 'rank_Z_metric_steps1_10', 'rank_X_metric_all_steps', 'rank_Z_metric_all_steps', 'in_X_mask_rank', 'in_Y_mask_rank', 'in_Z_mask_rank', 'group_net_contrib', 'group_pos_contrib', 'group_neg_contrib', 'group_cancel_ratio', 'group_n_deep_rollouts', 'group_rank_net_steps1_10', 'group_mean_reward', 'loop_turns', 'loop_tok', 'loop_share_pct', 'near_loop_turns', 'near_loop_tok', 'truncated', 'ends_seq_noclose', 'mask_before', 'mask_after', 'seq_mult_prob_error']
    r = r[cols].sort_values(['train_step', 'sample_id'])
    r.to_csv(f'{D}/browser/rollout_metrics_{a}.csv', index=False, float_format='%.5g')
    t = eff[eff.think_tok >= 1000].copy()
    t.insert(0, 'run', RUNS.get(a, '')); t.insert(0, 'arm', a); t['target_step'] = t.train_step - 1; t['group_id'] = t.sample_id.str.rsplit('_g', n=1).str[0]
    t['mean_1mp_deep'] = np.where(t.deep_tokens > 0, t.live_deep_1mp / t.deep_tokens.clip(lower=1), np.nan)
    t = t[['arm', 'run', 'train_step', 'target_step', 'sample_id', 'group_id', 'turn', 'reward', 'adv', 'n_tok', 'think_tok', 'deep_tokens', 'live_deep_1mp', 'mean_1mp_deep', 'turn_contrib', 'zlib', 'cls', 's1mp_think_tr', 's1mp_entry_tr', 'close_p_tr', 'mean_ptr_think']].sort_values(['train_step', 'sample_id', 'turn'])
    t.to_csv(f'{D}/browser/long_turns_{a}.csv', index=False, float_format='%.5g')
    allr.append(r); allt.append(t)
    print(a, RUNS.get(a, ''), 'steps', r.train_step.min(), '-', r.train_step.max(), 'rollouts', len(r), 'deep rollouts', int((r.deep_think_tokens > 0).sum()), 'long turns', len(t))
pd.concat(allr).to_csv(f'{D}/browser/rollout_metrics_all_arms.csv', index=False, float_format='%.5g')
pd.concat(allt).to_csv(f'{D}/browser/long_turns_all_arms.csv', index=False, float_format='%.5g')
