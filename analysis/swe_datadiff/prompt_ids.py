import sys,os,re,hashlib,glob
import numpy as np
sys.path.insert(0,"/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/tools")
from pt_numpy import load
arm,exp,chunk,outdir=sys.argv[1:5]
m=re.search(r"step_(\d+)_chunk_(\d+)",chunk); step,ck=int(m.group(1)),int(m.group(2))
base=f"{outdir}/{arm}__s{step:03d}_c{ck:03d}.prompts.csv"
if os.path.exists(base): sys.exit(0)
obj=load(chunk); L=np.asarray(obj["input_lengths"]).astype(np.int64); off=np.concatenate([[0],np.cumsum(L)]); ids=np.asarray(obj["input_ids"]).astype(np.int32); tm=np.asarray(obj["token_mask"]).astype(bool)
with open(base+".tmp","w") as fh:
    fh.write("arm,step,chunk,sample_id,prompt_len,prompt_hash\n")
    for i,sid in enumerate(obj["sample_ids"]):
        row=ids[off[i]:off[i+1]]; mrow=tm[off[i]:off[i+1]]; nz=np.where(mrow)[0]; pl=int(nz[0]) if len(nz) else len(row)
        fh.write(f"{arm},{step},{ck},{sid},{pl},{hashlib.md5(row[:pl].tobytes()).hexdigest()[:16]}\n")
os.replace(base+".tmp",base)
