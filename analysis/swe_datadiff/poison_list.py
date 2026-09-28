import glob,csv,collections,sys,random
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
arm=sys.argv[1]; sign=sys.argv[2] if len(sys.argv)>2 else "rewarded"; lo,hi=1,10
def f(x,d=0.0):
    try: return float(x)
    except: return d
per=collections.defaultdict(lambda:[0.0,0,0.0,0,0.0]); allrew=set()
for fn in sorted(glob.glob(f"{D}/parts/{arm}__s*_c*.eff.csv")):
    s=int(fn.split("__s")[1][:3])
    if not lo<=s<=hi: continue
    with open(fn) as fh:
        for r in csv.DictReader(fh):
            adv=f(r["adv"]); deep=max(0,int(f(r["think_tok"]))-4096)
            if adv>0: allrew.add((r["sample_id"],s,adv))
            if (sign=="rewarded" and adv<=0) or (sign=="punished" and adv>=0) or deep<=0: continue
            p=per[r["sample_id"]]; p[0]+=abs(adv)*f(r["s1mp_deep_tr"]); p[1]+=deep; p[2]+=f(r["s1mp_deep_tr"]); p[3]=s; p[4]=adv
items=sorted(per.items(),key=lambda kv:-kv[1][0]); total=sum(v[0] for _,v in items) or 1.0
out=f"{D}/poison_{arm}_{sign}_deep_steps{lo}-{hi}.csv"
with open(out,"w") as fh:
    fh.write("rank,sample_id,group_id,target_step,train_step,adv,deep_think_tokens,live_deep_sum_1mp,absadv_x_live,cum_share\n"); cum=0.0
    for k,(sid,v) in enumerate(items,1):
        cum+=v[0]; fh.write(f"{k},{sid},{sid.rsplit('_g',1)[0]},{v[3]-1},{v[3]},{v[4]:.4f},{v[1]},{v[2]:.1f},{v[0]:.2f},{cum/total:.3f}\n")
print(f"{arm} {sign}: {len(items)} rollouts with a >4096-token think turn -> {out}")
if sign=="rewarded":
    random.seed(1234); pool=sorted(x for x in allrew if x[0] not in per); ctrl=random.sample(pool,len(items))
    outc=f"{D}/control_{arm}_random_rewarded_steps{lo}-{hi}.csv"
    with open(outc,"w") as fh:
        fh.write("rank,sample_id,group_id,target_step,train_step,adv\n")
        for k,(sid,s,adv) in enumerate(sorted(ctrl,key=lambda x:x[1]),1): fh.write(f"{k},{sid},{sid.rsplit('_g',1)[0]},{s-1},{s},{adv:.4f}\n")
    print(f"{arm} control: {len(ctrl)} random rewarded rollouts WITHOUT deep think turns (seed 1234) -> {outc}")
