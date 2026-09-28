import glob, sys, pandas as pd, numpy as np
arms = sys.argv[1].split(',') if len(sys.argv) > 1 else ['X', 'R', 'A', 'K', 'M', 'N', 'J', 'G', 'I']
lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (11, 18)
rows = []
for a in arms:
    fs = sorted(glob.glob(f'parts/{a}__s*.samples.csv'))
    if not fs: continue
    df = pd.concat([pd.read_csv(f) for f in fs])
    df = df[(df.step >= lo) & (df.step <= hi)]
    for s, g in df.groupby('step'):
        gen = g.n_gen.sum()
        rows.append(dict(arm=a, step=s, n=len(g), loop=100*g.loop_tok.sum()/gen, near=100*g.near_loop_tok.sum()/gen,
                         runaway=100*g.ends_seq_noclose.mean(), trunc=100*g.truncated.mean(), gen_k=g.n_gen.mean()/1000,
                         max_turn_k=g.max_turn.mean()/1000, reward=g.reward.mean(), lp=g.loop_turns.gt(0).mean()*100))
t = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
for m in ['loop', 'near', 'runaway', 'trunc', 'gen_k', 'max_turn_k', 'reward', 'lp']:
    print(f'\n== {m}  (loop/near = % of generated tokens; runaway/trunc/lp = % of rollouts; gen_k/max_turn_k = mean k tokens)')
    print(t.pivot(index='step', columns='arm', values=m).reindex(columns=[a for a in arms if a in t.arm.unique()]).round(2).to_string())
w = t.groupby('arm').apply(lambda g: pd.Series(dict(steps=len(g), loop=g.loop.mean(), near=g.near.mean(), runaway=g.runaway.mean(), trunc=g.trunc.mean(), gen_k=g.gen_k.mean(), reward=g.reward.mean()))).reindex([a for a in arms if a in t.arm.unique()])
print(f'\n== window {lo}-{hi} means'); print(w.round(2).to_string())
