import glob, sys, numpy as np, pandas as pd
arms = sys.argv[1].split(',')
lo, hi = int(sys.argv[2]), int(sys.argv[3])
rows = []
for a in arms:
    fs = [f for f in sorted(glob.glob(f'parts/{a}__s*_c*.eff.csv')) if lo <= int(f.split('__s')[1][:3]) <= hi]
    if not fs: continue
    df = pd.concat([pd.read_csv(f).assign(step=int(f.split('__s')[1][:3])) for f in fs])
    for s, d in list(df.groupby('step')) + [('all', df)]:
        tot = d.n_tok.sum()
        sel = d.think_tok >= 1000
        pos, neg = sel & (d.adv > 0), sel & (d.adv < 0)
        e = d[(d.think_tok >= 4096) & (d.think_tok < 16384)]
        rows.append(dict(arm=a, step=s, R_deep=1e3 * (d.adv[sel] * d.s1mp_deep_tr[sel]).sum() / tot,
                         Rp=1e3 * (d.adv[pos] * d.s1mp_deep_tr[pos]).sum() / tot, Rn=1e3 * (d.adv[neg] * d.s1mp_deep_tr[neg]).sum() / tot,
                         Sp=1e3 * d.s1mp_deep_tr[pos].sum() / tot, Sn=1e3 * d.s1mp_deep_tr[neg].sum() / tot,
                         E4k16k=1e3 * (e.adv * e.n_tok).sum() / tot, deep_share=100 * np.maximum(0, d.think_tok - 4096).sum() / tot))
t = pd.DataFrame(rows)
pd.set_option('display.width', 250)
order = [a for a in arms if a in t.arm.unique()]
for m, lab in [('R_deep', 'R_deep net (per 1k tokens)'), ('Rp', 'R_deep+ rewarded'), ('Rn', 'R_deep- punished'), ('Sp', 'S+ live rewarded deep tokens per 1k'), ('Sn', 'S- live punished deep tokens per 1k'), ('E4k16k', 'equal-weight E(think 4k-16k)'), ('deep_share', 'deep think tokens (beyond 4096) as % of all tokens')]:
    print(f'\n== {lab}, steps {lo}-{hi}')
    print(t.pivot(index='step', columns='arm', values=m).reindex(columns=order).round(3).to_string())
