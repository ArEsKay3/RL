# HEADLINE (2026-09-22)

**Question.** Do the three MINF-from-scratch lineages — chain G (MINF from scratch, no prefix cache), chain I (MINF from scratch, prefix cache on), main MINF chain (MINF from scratch, original) — share a deviation from run A (vLLM from scratch) in weight space at step 5 or 10, while their rollouts were statistically identical?

**Answer: no shared MINF deviation is detectable; the four runs are exchangeable in weight space at steps 5 and 10.**

```
statistic (whole model unless noted)                          step 5                     step 10
sign agreement of co-changed elements, MINF-MINF pairs        49.89 / 50.03 / 49.89 %    49.90 / 50.06 / 49.98 %
sign agreement of co-changed elements, MINF-A pairs           50.16 / 50.09 / 50.14 %    50.27 / 49.90 / 50.02 %
  (50 % = independent update directions; binomial SE ~0.005 %)
cos(delta_X, delta_Y) per trained weight group                0.00 +- 0.01 all pairs     0.00 +- 0.01 all pairs
d_LOO(run A) / mean d_LOO(MINF runs), trained groups          0.99                       1.00 - 1.01
cos(G-A, I-A), cos(G-A, M-A), cos(I-A, M-A)                   +0.515 +0.487 +0.496       +0.505 +0.507 +0.506
  within-MINF reference contrasts (the null is +0.50)         +0.494 +0.483 +0.522       +0.499 +0.502 +0.498
fp32 router load-balancing bias (exact): odd-one-out layers   -                          A 6 / G 3 / I 6 / M 8 of 23
elements changed vs base (A / G / I / M)                      1.12 / 1.14 / 1.15 / 1.15 %   2.94 / 2.92 / 2.94 / 2.92 %
||delta|| / ||W_base||  (all four runs)                       6.1e-5                     1.21e-4
```
Two MINF runs compared with each other (chain G vs main chain) are just as independent: 50.0-50.3 % sign agreement on every trained group through step 15. So at the resolution of the bf16 exports the visible weight change is rollout-specific noise; neither the common RL direction nor any engine fingerprint rises above it. **What this excludes:** a MINF-lineage weight deviation comparable to the run-to-run noise at steps 5-10. **What it cannot exclude:** a small systematic bias that is below the bf16 flip threshold on all but ~1-3 % of elements (see caveat below); that requires the fp32 master weights (Megatron optimizer shards, torch needed) or the lr = 0 fixed-weights experiment.

**Sanity / caveats.**
- At step 5 all four runs had trained on statistically identical rollouts (established in the session), so any consistent MINF-common direction here would have been an update-path fingerprint; none appears. main / chain G / chain I differ in prefix caching and overlap_param_gather; run A is a single vLLM replicate.
- bf16 quantization: the exports round fp32 master weights to bf16; after 5-10 steps at lr 3e-6 only ~1-3 % of elements (the small-magnitude ones) show a change at all, always by one bf16 ulp (max |delta| 0.005 / 0.010 / 0.015 at steps 5 / 10 / 15 = ulp steps of the largest weights). Norms and cosines describe this thresholded flip pattern; the sign-agreement statistic is robust to it.
- The whole-model cos(delta_X, delta_Y) of +0.97 (step 5) / +0.86 (step 10) is NOT evidence of shared direction: 93 % of ||delta||^2 sits in the 2,944 fp32 router-bias values (updated deterministically from expert load, identical in all runs) and 4 % in the MTP head (a few elements moving identically); the trained weight groups contribute < 3 %.
- Integrity: identical tensor names / dtypes / shapes in all checkpoints (6,513 tensors, 32.91 B params), no NaN/Inf; ~390 tensors bit-identical to base in every run (MTP head, router weights, most norms, A_log, dt_bias).
- Steps 15 and 20 with all four runs (run A and chain I exported 2026-09-22 13:5x): computed by Slurm job 3932926, which regenerates this report when it finishes; until then only the step-5 and step-10 sections are present below. The step-15 chain G vs main chain numbers quoted above came from the first pass (job 3932397) before the CSV was replaced.

Files: `st_reader.py` (pure-numpy safetensors reader), `compare.py` (per-tensor Gram matrix of deltas + flip/sign counts), `aggregate.py` (this report), `compare.sbatch`/`compare2.sbatch`, `per_tensor_step{5,10,15,20}.csv`.

# Weight-space comparison of early checkpoints: run A (vLLM from scratch) vs chain G (MINF from scratch, no prefix cache) vs chain I (MINF from scratch, prefix cache on) vs main MINF chain (MINF from scratch, original)

Base = the shared starting checkpoint (akamehra/swe_e2e_corrected/base_model/step_18/hf). All comparisons use the HF exports of the Megatron checkpoints (bf16 weights; 24 router-bias tensors are fp32). delta_X = W_X - W_base per tensor, in float64.

IMPORTANT CAVEAT - bf16 quantization: the exports round the fp32 master weights to bf16. After 5-10 steps at lr 3e-6 the accumulated update on most elements (~1e-5 relative) is below half a bf16 ulp (0.4 % relative), so the visible delta is non-zero only on the ~1 % of elements (small-magnitude weights) whose update crossed a rounding boundary. Norm- and cosine-based numbers therefore describe this thresholded 'flip pattern', not the full update. The sign-agreement statistic (among elements that flipped in BOTH runs, how often in the same direction; 50 % = independent) is robust to this and is the primary test of a shared direction.

## Step 5: checkpoints present: A = run A (vLLM from scratch), G = chain G (MINF from scratch, no prefix cache), I = chain I (MINF from scratch, prefix cache on), M = main MINF chain (MINF from scratch, original)

Tensors: 6,513  elements: 32.913 B  ||W_base|| = 4181.6

### 1. Integrity and update size per run (whole model)
Columns: run | tensors bit-identical to base | tensors with NaN/Inf | elements changed (share) | ||delta|| | ||delta||/||W_base|| | max |delta| over all elements
```
run                                              identical tensors  NaN/Inf changed elems  ||delta||   rel norm max|delta|
run A (vLLM from scratch)                                      393        0       1.1216%     0.2545   6.09e-05     0.0050
chain G (MINF from scratch, no prefix cache)                   394        0       1.1411%     0.2556   6.11e-05     0.0050
chain I (MINF from scratch, prefix cache on)                   395        0       1.1509%     0.2556   6.11e-05     0.0050
main MINF chain (MINF from scratch, original)                  394        0       1.1537%     0.2554   6.11e-05     0.0050
```

### 2. Pairwise geometry of the deltas (whole model)
Columns: pair | ||W_X - W_Y|| | cos(delta_X, delta_Y) | elements flipped in both | sign agreement among those (50 % = independent updates)
```
pair        ||W_X-W_Y||  cos(dX,dY)  both flipped  sign agree
A vs G           0.0640     +0.9685   234,788,965      50.16%
A vs I           0.0642     +0.9683   237,710,251      50.09%
A vs M           0.0640     +0.9685   236,899,763      50.14%
G vs I           0.0631     +0.9695   239,896,519      49.89%
G vs M           0.0648     +0.9678   239,507,054      50.03%
I vs M           0.0644     +0.9683   242,058,390      49.89%
```

### 3. Leave-one-out exchangeability test (whole model)
d_LOO(X) = ||delta_X - mean(delta of the other runs)||. If the four runs were exchangeable draws around a common RL direction, all four values would be similar; if the three MINF lineages form a cluster that run A is not part of, run A's value stands out.
```
run                                                   d_LOO  d_LOO / ||delta_X||
run A (vLLM from scratch)                            0.0523                0.205
chain G (MINF from scratch, no prefix cache)         0.0522                0.204
chain I (MINF from scratch, prefix cache on)         0.0520                0.204
main MINF chain (MINF from scratch, original)        0.0528                0.207
```

### 4. Contrast cosines (whole model): do the MINF lineages deviate from run A in a common direction?
cos(G-A, I-A) etc. compare two MINF-minus-A deviations; positive and consistently above the within-MINF reference contrasts (cos(G-I, M-I) etc.) means a shared MINF deviation from run A. Under pure independent noise, every contrast of the form cos(X-R, Y-R) has expectation +0.5 (the shared -R term), so the REFERENCE contrasts, not zero, are the null.
```
contrast                          cos
cos(G-A, I-A)                 +0.5151
cos(G-A, M-A)                 +0.4871
cos(I-A, M-A)                 +0.4957
cos(I-G, M-G)  [ref]          +0.4941
cos(G-I, M-I)  [ref]          +0.4832
cos(G-M, I-M)  [ref]          +0.5224
cos(A-G, I-G)  [ref]          +0.4902
cos(A-I, G-I)  [ref]          +0.4946
cos(A-M, G-M)  [ref]          +0.5061
```

### 5. By parameter group
Columns: group | tensors | elements | changed share per run | cos(dX,dY) for MINF-MINF pairs and MINF-A pairs (mean) | sign agreement MINF-MINF (mean) vs MINF-A (mean) | d_LOO(A) / mean d_LOO(MINF)
```
group                  tensors     elements    chg A    chg G    chg I    chg M   cos MM   cos MA  agree MM  agree MA LOO A/MINF
moe_experts_down          2944 14,687,404,032   1.035%   1.054%   1.061%   1.066%   -0.001   +0.002     49.9%     50.1%      0.987
moe_experts_up            2944 14,687,404,032   1.201%   1.222%   1.232%   1.236%   -0.001   +0.002     49.9%     50.1%      0.988
mtp_head                   270 1,335,325,952   0.000%   0.000%   0.000%   0.000%   +0.996   +0.995     99.2%     98.9%      1.103
mamba_in_proj               23  637,034,496   2.214%   2.237%   2.280%   2.261%   -0.002   +0.003     49.9%     50.2%      0.990
embeddings                   1  352,321,536   0.476%   0.500%   0.483%   0.493%   +0.000   -0.000     50.0%     50.0%      0.986
lm_head                      1  352,321,536   0.260%   0.271%   0.261%   0.270%   +0.000   +0.002     50.2%     50.3%      0.988
mamba_out_proj              23  253,231,104   2.577%   2.608%   2.667%   2.638%   -0.003   +0.005     49.8%     50.4%      0.987
moe_shared_down             23  229,490,688   2.867%   2.910%   2.965%   2.936%   -0.004   +0.002     49.8%     50.2%      0.988
moe_shared_up               23  229,490,688   3.288%   3.335%   3.386%   3.359%   -0.001   +0.003     49.9%     50.2%      0.991
attn_o                       6   66,060,288   2.591%   2.618%   2.675%   2.650%   -0.003   +0.004     49.8%     50.3%      0.990
attn_q                       6   66,060,288   1.989%   2.017%   2.051%   2.036%   -0.003   +0.001     49.8%     50.0%      0.990
moe_router                  23    7,913,472   0.000%   0.000%   0.000%   0.000%     +nan     +nan      nan%      nan%        nan
attn_k                       6    4,128,768   1.789%   1.817%   1.851%   1.832%   -0.001   -0.000     49.9%     49.8%      0.991
attn_v                       6    4,128,768   4.374%   4.406%   4.526%   4.486%   -0.010   -0.001     49.3%     49.8%      0.990
mamba_conv1d                46      706,560   9.005%   9.105%   9.263%   9.160%   -0.004   +0.003     49.6%     50.3%      0.988
layer_norm                  52      139,776   0.054%   0.054%   0.054%   0.054%   +0.065   -0.013     47.7%     50.3%      1.044
mamba_norm                  23       94,208   0.362%   0.366%   0.361%   0.366%   -0.045   +0.029     51.0%     51.1%      0.935
moe_router_bias_f32         23        2,944  99.287%  99.185%  99.389%  98.947%   +0.995   +0.995     99.5%     99.5%      1.061
final_norm                   1        2,688   0.000%   0.000%   0.000%   0.000%     +nan     +nan      nan%      nan%        nan
mamba_A_log                 23        1,472   0.068%   0.068%   0.000%   0.000%     +nan   +1.000      nan%    100.0%      1.000
mamba_D                     23        1,472   2.514%   2.378%   2.514%   2.717%   -0.038   -0.057     47.9%     44.5%      0.955
mamba_dt_bias               23        1,472   0.000%   0.000%   0.000%   0.000%     +nan     +nan      nan%      nan%        nan
```

### 5b. Update norm by parameter group: ||delta_X|| per run and the group's share of the whole-model ||delta||^2 (run A). Groups with a near-1 cosine but no changed elements (mtp_head) or deterministic bias updates (moe_router_bias_f32) dominate the whole-model cosine in section 2.
```
group                      ||dA||     ||dG||     ||dI||     ||dM|| share of ||dA||^2
moe_router_bias_f32        0.2459     0.2469     0.2469     0.2467            93.31%
mtp_head                   0.0513     0.0514     0.0511     0.0512             4.06%
moe_experts_up             0.0274     0.0278     0.0277     0.0280             1.16%
moe_experts_down           0.0251     0.0255     0.0254     0.0256             0.97%
mamba_in_proj              0.0105     0.0106     0.0107     0.0107             0.17%
moe_shared_up              0.0078     0.0078     0.0078     0.0078             0.09%
mamba_out_proj             0.0074     0.0074     0.0075     0.0075             0.08%
moe_shared_down            0.0071     0.0072     0.0072     0.0072             0.08%
attn_o                     0.0039     0.0039     0.0040     0.0040             0.02%
attn_q                     0.0033     0.0033     0.0033     0.0034             0.02%
embeddings                 0.0030     0.0031     0.0031     0.0031             0.01%
lm_head                    0.0023     0.0024     0.0023     0.0024             0.01%
attn_v                     0.0013     0.0013     0.0013     0.0013             0.00%
attn_k                     0.0008     0.0008     0.0008     0.0008             0.00%
mamba_conv1d               0.0007     0.0007     0.0007     0.0007             0.00%
mamba_norm                 0.0000     0.0000     0.0000     0.0000             0.00%
layer_norm                 0.0000     0.0000     0.0000     0.0000             0.00%
mamba_D                    0.0000     0.0000     0.0000     0.0000             0.00%
mamba_A_log                0.0000     0.0000     0.0000     0.0000             0.00%
mamba_dt_bias              0.0000     0.0000     0.0000     0.0000             0.00%
moe_router                 0.0000     0.0000     0.0000     0.0000             0.00%
final_norm                 0.0000     0.0000     0.0000     0.0000             0.00%
```

### 5c. The only unquantized signal: MoE router load-balancing bias (fp32, 23 layers x 128 experts). This bias is updated from expert-load statistics of the training tokens, so it records the routing history of each run exactly.
Columns: pair | ||bias_X - bias_Y|| | cos(delta_X, delta_Y);  then per run d_LOO = ||delta_X - mean(delta of the other runs)||
```
pair        ||bX-bY||  cos(dX,dY)
A vs G        0.02499    +0.99487
A vs I        0.02573    +0.99456
A vs M        0.02470    +0.99497
G vs I        0.02153    +0.99620
G vs M        0.02572    +0.99457
I vs M        0.02464    +0.99502
run                                               ||delta||      d_LOO
run A (vLLM from scratch)                           0.24588    0.02097
chain G (MINF from scratch, no prefix cache)        0.24689    0.01934
chain I (MINF from scratch, prefix cache on)        0.24693    0.01916
main MINF chain (MINF from scratch, original)       0.24671    0.02078
```
Per layer, d_LOO of each run's router-bias delta (which run's routing history is the odd one out, layer by layer):
```
layer    d_LOO A    d_LOO G    d_LOO I    d_LOO M  odd one out
    1    0.00687    0.00509    0.00550    0.00604            A
    3    0.00479    0.00515    0.00470    0.00638            M
    6    0.00487    0.00496    0.00478    0.00429            G
    8    0.00398    0.00451    0.00325    0.00351            G
   10    0.00368    0.00380    0.00424    0.00316            I
   13    0.00531    0.00505    0.00408    0.00547            M
   15    0.00511    0.00425    0.00369    0.00560            M
   17    0.00436    0.00357    0.00456    0.00393            I
   20    0.00429    0.00374    0.00386    0.00419            A
   22    0.00387    0.00431    0.00351    0.00409            G
   24    0.00389    0.00353    0.00283    0.00377            A
   27    0.00433    0.00604    0.00443    0.00390            G
   29    0.00365    0.00353    0.00377    0.00313            I
   31    0.00413    0.00403    0.00510    0.00501            I
   34    0.00441    0.00375    0.00409    0.00398            A
   36    0.00356    0.00343    0.00330    0.00413            M
   38    0.00238    0.00197    0.00304    0.00256            I
   40    0.00487    0.00279    0.00309    0.00309            A
   43    0.00466    0.00359    0.00416    0.00416            A
   45    0.00296    0.00338    0.00296    0.00296            G
   47    0.00476    0.00346    0.00346    0.00545            M
   49    0.00394    0.00290    0.00371    0.00320            A
   51    0.00393    0.00369    0.00436    0.00484            M
```

### 6. By layer (backbone layers only): sign agreement MINF-MINF vs MINF-A, and d_LOO(A) / mean d_LOO(MINF)
```
layer     elements  agree MM  agree MA   cos MM   cos MA LOO A/MINF
    0   76,600,000     50.1%     49.2%   +0.001   -0.008      1.016
    1 2,594,939,008     49.9%     49.9%   +0.983   +0.981      1.075
    2   38,744,896     50.2%     49.4%   +0.003   -0.007      1.002
    3 1,297,468,160     50.1%     50.0%   +0.963   +0.965      0.964
    4   38,744,896     50.0%     49.8%   +0.000   -0.001      0.994
    5   23,399,040     49.9%     49.9%   -0.003   +0.001      0.992
    6 1,297,468,160     50.0%     50.1%   +0.964   +0.964      0.997
    7   38,744,896     49.8%     49.9%   -0.003   -0.001      0.990
    8 1,297,468,160     50.0%     50.1%   +0.972   +0.972      0.997
    9   38,744,896     49.8%     50.0%   -0.003   +0.001      0.989
   10 1,297,468,160     49.9%     50.2%   +0.972   +0.973      0.985
   11   38,744,896     49.7%     50.3%   -0.004   +0.003      0.988
   12   23,399,040     49.8%     50.1%   -0.003   +0.002      0.991
   13 1,297,468,160     49.9%     50.1%   +0.965   +0.964      1.007
   14   38,744,896     49.7%     50.2%   -0.004   +0.003      0.991
   15 1,297,468,160     49.9%     50.1%   +0.966   +0.966      1.012
   16   38,744,896     49.9%     50.2%   -0.002   +0.003      0.993
   17 1,297,468,160     49.9%     50.1%   +0.956   +0.956      0.999
   18   38,744,896     49.7%     50.1%   -0.005   +0.002      0.993
   19   23,399,040     49.7%     50.0%   -0.004   +0.000      0.997
   20 1,297,468,160     49.9%     50.1%   +0.963   +0.962      1.001
   21   38,744,896     49.8%     50.1%   -0.003   +0.001      0.994
   22 1,297,468,160     49.9%     50.1%   +0.950   +0.951      0.987
   23   38,744,896     49.9%     50.2%   -0.002   +0.002      0.991
   24 1,297,468,160     49.9%     50.1%   +0.967   +0.967      1.001
   25   38,744,896     49.9%     50.2%   -0.002   +0.002      0.989
   26   23,399,040     49.6%     50.1%   -0.006   +0.001      0.989
   27 1,297,468,160     49.9%     50.1%   +0.956   +0.958      0.975
   28   38,744,896     49.6%     50.1%   -0.006   +0.001      0.990
   29 1,297,468,160     50.0%     50.1%   +0.961   +0.961      0.993
   30   38,744,896     49.9%     50.2%   -0.002   +0.003      0.990
   31 1,297,468,160     50.0%     50.2%   +0.966   +0.968      0.969
   32   38,744,896     49.8%     50.4%   -0.005   +0.006      0.985
   33   23,399,040     49.8%     50.4%   -0.004   +0.005      0.985
   34 1,297,468,160     49.9%     50.1%   +0.974   +0.974      1.008
   35   38,744,896     50.0%     50.5%   -0.000   +0.006      0.988
   36 1,297,468,160     49.9%     50.1%   +0.979   +0.979      0.985
   37   38,744,896     50.0%     50.4%   -0.001   +0.005      0.987
   38 1,297,468,160     49.8%     50.2%   +0.987   +0.987      0.981
   39   38,744,896     49.8%     50.6%   -0.004   +0.008      0.978
   40 1,297,468,160     49.9%     50.2%   +0.986   +0.982      1.134
   41   38,744,896     50.0%     50.4%   -0.002   +0.005      0.983
   42   23,399,040     49.9%     50.6%   -0.000   +0.008      0.984
   43 1,297,468,160     50.0%     50.1%   +0.985   +0.984      1.046
   44   38,744,896     49.9%     50.8%   -0.002   +0.010      0.983
   45 1,297,468,160     49.9%     50.1%   +0.985   +0.985      0.979
   46   38,744,896     49.8%     50.8%   -0.004   +0.010      0.982
   47 1,297,468,160     50.0%     50.3%   +0.983   +0.983      1.028
   48   38,744,896     49.9%     51.0%   -0.003   +0.012      0.980
   49 1,297,468,160     49.9%     50.2%   +0.987   +0.986      1.038
   50   38,744,896     50.1%     51.1%   +0.000   +0.015      0.980
   51 1,297,468,160     50.0%     50.2%   +0.978   +0.979      0.970
```

## Step 10: checkpoints present: A = run A (vLLM from scratch), G = chain G (MINF from scratch, no prefix cache), I = chain I (MINF from scratch, prefix cache on), M = main MINF chain (MINF from scratch, original)

Tensors: 6,513  elements: 32.913 B  ||W_base|| = 4181.6

### 1. Integrity and update size per run (whole model)
Columns: run | tensors bit-identical to base | tensors with NaN/Inf | elements changed (share) | ||delta|| | ||delta||/||W_base|| | max |delta| over all elements
```
run                                              identical tensors  NaN/Inf changed elems  ||delta||   rel norm max|delta|
run A (vLLM from scratch)                                      388        0       2.9414%     0.5065   1.21e-04     0.0100
chain G (MINF from scratch, no prefix cache)                   391        0       2.9200%     0.5059   1.21e-04     0.0100
chain I (MINF from scratch, prefix cache on)                   391        0       2.9389%     0.5055   1.21e-04     0.0100
main MINF chain (MINF from scratch, original)                  391        0       2.9215%     0.5054   1.21e-04     0.0100
```

### 2. Pairwise geometry of the deltas (whole model)
Columns: pair | ||W_X - W_Y|| | cos(delta_X, delta_Y) | elements flipped in both | sign agreement among those (50 % = independent updates)
```
pair        ||W_X-W_Y||  cos(dX,dY)  both flipped  sign agree
A vs G           0.2649     +0.8631   587,019,915      50.27%
A vs I           0.2656     +0.8622   590,963,170      49.90%
A vs M           0.2649     +0.8629   588,397,884      50.02%
G vs I           0.2639     +0.8638   588,199,570      49.90%
G vs M           0.2632     +0.8645   586,034,764      50.06%
I vs M           0.2637     +0.8639   589,722,565      49.98%
```

### 3. Leave-one-out exchangeability test (whole model)
d_LOO(X) = ||delta_X - mean(delta of the other runs)||. If the four runs were exchangeable draws around a common RL direction, all four values would be similar; if the three MINF lineages form a cluster that run A is not part of, run A's value stands out.
```
run                                                   d_LOO  d_LOO / ||delta_X||
run A (vLLM from scratch)                            0.2171                0.429
chain G (MINF from scratch, no prefix cache)         0.2152                0.425
chain I (MINF from scratch, prefix cache on)         0.2159                0.427
main MINF chain (MINF from scratch, original)        0.2152                0.426
```

### 4. Contrast cosines (whole model): do the MINF lineages deviate from run A in a common direction?
cos(G-A, I-A) etc. compare two MINF-minus-A deviations; positive and consistently above the within-MINF reference contrasts (cos(G-I, M-I) etc.) means a shared MINF deviation from run A. Under pure independent noise, every contrast of the form cos(X-R, Y-R) has expectation +0.5 (the shared -R term), so the REFERENCE contrasts, not zero, are the null.
```
contrast                          cos
cos(G-A, I-A)                 +0.5051
cos(G-A, M-A)                 +0.5065
cos(I-A, M-A)                 +0.5058
cos(I-G, M-G)  [ref]          +0.4993
cos(G-I, M-I)  [ref]          +0.5024
cos(G-M, I-M)  [ref]          +0.4983
cos(A-G, I-G)  [ref]          +0.4954
cos(A-I, G-I)  [ref]          +0.4995
cos(A-M, G-M)  [ref]          +0.4968
```

### 5. By parameter group
Columns: group | tensors | elements | changed share per run | cos(dX,dY) for MINF-MINF pairs and MINF-A pairs (mean) | sign agreement MINF-MINF (mean) vs MINF-A (mean) | d_LOO(A) / mean d_LOO(MINF)
```
group                  tensors     elements    chg A    chg G    chg I    chg M   cos MM   cos MA  agree MM  agree MA LOO A/MINF
moe_experts_down          2944 14,687,404,032   2.724%   2.705%   2.721%   2.704%   -0.000   +0.001     50.0%     50.0%      1.008
moe_experts_up            2944 14,687,404,032   3.144%   3.120%   3.141%   3.122%   -0.000   +0.001     50.0%     50.0%      1.008
mtp_head                   270 1,335,325,952   0.000%   0.000%   0.000%   0.000%   +0.996   +0.997     99.5%    100.0%      0.834
mamba_in_proj               23  637,034,496   5.717%   5.662%   5.725%   5.687%   -0.000   +0.002     50.0%     50.2%      1.005
embeddings                   1  352,321,536   1.387%   1.415%   1.387%   1.392%   +0.000   +0.000     50.0%     50.0%      1.001
lm_head                      1  352,321,536   0.784%   0.801%   0.785%   0.793%   +0.001   -0.000     50.2%     50.1%      1.001
mamba_out_proj              23  253,231,104   6.688%   6.616%   6.711%   6.662%   -0.002   +0.003     49.9%     50.3%      1.002
moe_shared_down             23  229,490,688   7.450%   7.387%   7.465%   7.419%   -0.003   +0.002     49.8%     50.2%      1.003
moe_shared_up               23  229,490,688   8.500%   8.444%   8.504%   8.473%   -0.001   +0.002     49.9%     50.2%      1.004
attn_o                       6   66,060,288   6.744%   6.668%   6.777%   6.724%   -0.001   +0.003     49.9%     50.2%      1.002
attn_q                       6   66,060,288   5.177%   5.128%   5.189%   5.177%   -0.001   -0.000     49.9%     50.0%      1.004
moe_router                  23    7,913,472   0.000%   0.000%   0.000%   0.000%     +nan     +nan      nan%      nan%        nan
attn_k                       6    4,128,768   4.623%   4.591%   4.662%   4.648%   +0.001   +0.002     50.1%     50.0%      1.000
attn_v                       6    4,128,768  11.039%  10.879%  11.107%  11.011%   -0.006   -0.002     49.5%     49.8%      1.003
mamba_conv1d                46      706,560  17.666%  17.491%  17.709%  17.576%   -0.005   +0.002     49.8%     50.2%      1.000
layer_norm                  52      139,776   0.065%   0.066%   0.062%   0.064%   -0.023   +0.003     47.6%     52.7%      0.964
mamba_norm                  23       94,208   0.373%   0.377%   0.391%   0.378%   -0.041   +0.007     48.9%     51.0%      0.919
moe_router_bias_f32         23        2,944  97.113%  97.147%  96.637%  96.977%   +0.998   +0.997     99.9%     99.9%      1.046
final_norm                   1        2,688   0.037%   0.000%   0.037%   0.000%     +nan   -1.000      nan%      0.0%      3.000
mamba_A_log                 23        1,472   0.068%   0.068%   0.068%   0.068%   -0.333   +0.333     33.3%     66.7%      0.000
mamba_D                     23        1,472   5.163%   5.231%   5.231%   5.163%   -0.025   -0.016     49.1%     47.1%      0.921
mamba_dt_bias               23        1,472   0.000%   0.068%   0.000%   0.000%     +nan     +nan      nan%      nan%      0.600
```

### 5b. Update norm by parameter group: ||delta_X|| per run and the group's share of the whole-model ||delta||^2 (run A). Groups with a near-1 cosine but no changed elements (mtp_head) or deterministic bias updates (moe_router_bias_f32) dominate the whole-model cosine in section 2.
```
group                      ||dA||     ||dG||     ||dI||     ||dM|| share of ||dA||^2
moe_router_bias_f32        0.4610     0.4613     0.4609     0.4609            82.86%
moe_experts_up             0.1235     0.1221     0.1220     0.1215             5.94%
moe_experts_down           0.1136     0.1123     0.1122     0.1118             5.03%
mtp_head                   0.0946     0.0943     0.0938     0.0946             3.49%
mamba_in_proj              0.0489     0.0484     0.0487     0.0484             0.93%
moe_shared_up              0.0357     0.0354     0.0354     0.0353             0.50%
mamba_out_proj             0.0343     0.0339     0.0342     0.0340             0.46%
moe_shared_down            0.0328     0.0324     0.0326     0.0324             0.42%
attn_o                     0.0181     0.0179     0.0181     0.0179             0.13%
attn_q                     0.0154     0.0153     0.0154     0.0154             0.09%
embeddings                 0.0140     0.0141     0.0139     0.0140             0.08%
lm_head                    0.0109     0.0110     0.0108     0.0109             0.05%
attn_v                     0.0058     0.0057     0.0058     0.0057             0.01%
attn_k                     0.0038     0.0037     0.0038     0.0038             0.01%
mamba_conv1d               0.0027     0.0026     0.0027     0.0026             0.00%
mamba_norm                 0.0001     0.0001     0.0001     0.0001             0.00%
mamba_D                    0.0001     0.0001     0.0001     0.0001             0.00%
layer_norm                 0.0001     0.0001     0.0001     0.0001             0.00%
final_norm                 0.0000     0.0000     0.0000     0.0000             0.00%
mamba_A_log                0.0000     0.0000     0.0000     0.0000             0.00%
mamba_dt_bias              0.0000     0.0000     0.0000     0.0000             0.00%
moe_router                 0.0000     0.0000     0.0000     0.0000             0.00%
```

### 5c. The only unquantized signal: MoE router load-balancing bias (fp32, 23 layers x 128 experts). This bias is updated from expert-load statistics of the training tokens, so it records the routing history of each run exactly.
Columns: pair | ||bias_X - bias_Y|| | cos(delta_X, delta_Y);  then per run d_LOO = ||delta_X - mean(delta of the other runs)||
```
pair        ||bX-bY||  cos(dX,dY)
A vs G        0.03330    +0.99739
A vs I        0.03383    +0.99731
A vs M        0.03403    +0.99728
G vs I        0.03126    +0.99770
G vs M        0.03215    +0.99757
I vs M        0.03429    +0.99723
run                                               ||delta||      d_LOO
run A (vLLM from scratch)                           0.46103    0.02798
chain G (MINF from scratch, no prefix cache)        0.46129    0.02556
chain I (MINF from scratch, prefix cache on)        0.46091    0.02706
main MINF chain (MINF from scratch, original)       0.46093    0.02763
```
Per layer, d_LOO of each run's router-bias delta (which run's routing history is the odd one out, layer by layer):
```
layer    d_LOO A    d_LOO G    d_LOO I    d_LOO M  odd one out
    1    0.00681    0.00725    0.00700    0.00731            M
    3    0.00721    0.00696    0.00739    0.00727            I
    6    0.00545    0.00529    0.00577    0.00529            I
    8    0.00557    0.00660    0.00632    0.00490            G
   10    0.00604    0.00619    0.00619    0.00550            I
   13    0.00705    0.00499    0.00686    0.00588            A
   15    0.00597    0.00482    0.00558    0.00626            M
   17    0.00634    0.00641    0.00567    0.00527            G
   20    0.00620    0.00483    0.00464    0.00510            A
   22    0.00496    0.00571    0.00594    0.00571            I
   24    0.00580    0.00515    0.00602    0.00610            M
   27    0.00547    0.00487    0.00563    0.00571            M
   29    0.00558    0.00472    0.00482    0.00412            A
   31    0.00588    0.00685    0.00617    0.00595            G
   34    0.00500    0.00412    0.00482    0.00566            M
   36    0.00641    0.00454    0.00492    0.00668            M
   38    0.00565    0.00421    0.00611    0.00581            I
   40    0.00592    0.00503    0.00585    0.00676            M
   43    0.00510    0.00403    0.00535    0.00527            I
   45    0.00473    0.00424    0.00403    0.00454            A
   47    0.00433    0.00433    0.00378    0.00526            M
   49    0.00608    0.00504    0.00438    0.00562            A
   51    0.00568    0.00436    0.00484    0.00536            A
```

### 6. By layer (backbone layers only): sign agreement MINF-MINF vs MINF-A, and d_LOO(A) / mean d_LOO(MINF)
```
layer     elements  agree MM  agree MA   cos MM   cos MA LOO A/MINF
    0   76,600,000     50.9%     51.1%   +0.011   +0.013      0.999
    1 2,594,939,008     50.1%     50.2%   +0.936   +0.937      0.993
    2   38,744,896     50.4%     50.1%   +0.003   +0.001      1.001
    3 1,297,468,160     50.1%     50.0%   +0.860   +0.858      1.008
    4   38,744,896     50.2%     50.1%   +0.003   +0.001      1.007
    5   23,399,040     50.4%     50.1%   +0.005   +0.002      1.001
    6 1,297,468,160     50.0%     50.0%   +0.856   +0.853      1.009
    7   38,744,896     49.9%     49.9%   -0.001   -0.001      1.005
    8 1,297,468,160     50.0%     50.0%   +0.877   +0.875      1.005
    9   38,744,896     49.8%     50.0%   -0.002   +0.000      1.002
   10 1,297,468,160     50.0%     50.1%   +0.880   +0.879      1.007
   11   38,744,896     49.9%     50.1%   -0.002   -0.000      1.001
   12   23,399,040     49.8%     50.0%   -0.003   +0.001      1.000
   13 1,297,468,160     50.0%     50.0%   +0.865   +0.862      1.012
   14   38,744,896     49.8%     50.0%   -0.002   -0.000      1.004
   15 1,297,468,160     50.0%     50.0%   +0.859   +0.857      1.010
   16   38,744,896     50.0%     50.1%   +0.000   +0.001      1.007
   17 1,297,468,160     50.0%     50.1%   +0.818   +0.816      1.011
   18   38,744,896     49.9%     50.1%   -0.001   +0.001      1.008
   19   23,399,040     49.8%     50.2%   -0.002   +0.002      1.009
   20 1,297,468,160     50.0%     50.1%   +0.842   +0.839      1.012
   21   38,744,896     50.0%     50.2%   -0.001   +0.002      1.007
   22 1,297,468,160     50.0%     50.1%   +0.794   +0.794      1.007
   23   38,744,896     50.0%     50.2%   -0.000   +0.002      1.005
   24 1,297,468,160     49.9%     50.0%   +0.847   +0.846      1.009
   25   38,744,896     49.9%     50.2%   -0.001   +0.002      1.005
   26   23,399,040     49.6%     50.1%   -0.004   +0.001      1.004
   27 1,297,468,160     49.9%     50.0%   +0.824   +0.822      1.008
   28   38,744,896     49.7%     50.0%   -0.004   +0.000      1.006
   29 1,297,468,160     49.9%     50.1%   +0.835   +0.833      1.010
   30   38,744,896     49.9%     50.0%   -0.001   +0.001      1.005
   31 1,297,468,160     50.0%     50.1%   +0.863   +0.861      1.008
   32   38,744,896     49.6%     50.1%   -0.005   +0.002      1.000
   33   23,399,040     49.7%     50.1%   -0.004   +0.001      1.001
   34 1,297,468,160     49.9%     50.0%   +0.894   +0.893      1.007
   35   38,744,896     50.0%     50.4%   -0.001   +0.004      1.006
   36 1,297,468,160     49.9%     50.0%   +0.909   +0.908      1.013
   37   38,744,896     50.1%     50.2%   +0.000   +0.002      1.006
   38 1,297,468,160     49.9%     50.1%   +0.934   +0.933      1.009
   39   38,744,896     49.8%     50.4%   -0.002   +0.004      1.002
   40 1,297,468,160     49.9%     50.1%   +0.926   +0.926      1.004
   41   38,744,896     49.9%     50.3%   -0.002   +0.004      1.002
   42   23,399,040     49.9%     50.2%   -0.001   +0.002      1.001
   43 1,297,468,160     49.9%     50.0%   +0.945   +0.944      1.008
   44   38,744,896     49.9%     50.4%   -0.001   +0.004      1.006
   45 1,297,468,160     49.9%     50.0%   +0.931   +0.931      1.007
   46   38,744,896     49.8%     50.3%   -0.002   +0.003      1.005
   47 1,297,468,160     49.9%     50.1%   +0.938   +0.938      1.003
   48   38,744,896     49.6%     50.5%   -0.004   +0.005      1.001
   49 1,297,468,160     49.9%     50.1%   +0.943   +0.942      1.015
   50   38,744,896     49.8%     50.6%   -0.003   +0.006      0.999
   51 1,297,468,160     50.0%     50.0%   +0.914   +0.913      1.012
```

## Step 15: checkpoints present: A = run A (vLLM from scratch), G = chain G (MINF from scratch, no prefix cache), I = chain I (MINF from scratch, prefix cache on), M = main MINF chain (MINF from scratch, original)

Tensors: 6,513  elements: 32.913 B  ||W_base|| = 4181.6

### 1. Integrity and update size per run (whole model)
Columns: run | tensors bit-identical to base | tensors with NaN/Inf | elements changed (share) | ||delta|| | ||delta||/||W_base|| | max |delta| over all elements
```
run                                              identical tensors  NaN/Inf changed elems  ||delta||   rel norm max|delta|
run A (vLLM from scratch)                                      387        0       4.7908%     0.7701   1.84e-04     0.0150
chain G (MINF from scratch, no prefix cache)                   386        0       4.7809%     0.7675   1.84e-04     0.0150
chain I (MINF from scratch, prefix cache on)                   388        0       4.7993%     0.7690   1.84e-04     0.0150
main MINF chain (MINF from scratch, original)                  389        0       4.7376%     0.7662   1.83e-04     0.0150
```

### 2. Pairwise geometry of the deltas (whole model)
Columns: pair | ||W_X - W_Y|| | cos(delta_X, delta_Y) | elements flipped in both | sign agreement among those (50 % = independent updates)
```
pair        ||W_X-W_Y||  cos(dX,dY)  both flipped  sign agree
A vs G           0.5531     +0.7412   952,685,956      50.23%
A vs I           0.5555     +0.7394   956,253,960      49.93%
A vs M           0.5508     +0.7429   949,221,353      50.01%
G vs I           0.5544     +0.7396   954,552,958      49.97%
G vs M           0.5499     +0.7429   947,598,492      50.08%
I vs M           0.5509     +0.7425   951,185,681      50.02%
```

### 3. Leave-one-out exchangeability test (whole model)
d_LOO(X) = ||delta_X - mean(delta of the other runs)||. If the four runs were exchangeable draws around a common RL direction, all four values would be similar; if the three MINF lineages form a cluster that run A is not part of, run A's value stands out.
```
run                                                   d_LOO  d_LOO / ||delta_X||
run A (vLLM from scratch)                            0.4522                0.587
chain G (MINF from scratch, no prefix cache)         0.4511                0.588
chain I (MINF from scratch, prefix cache on)         0.4530                0.589
main MINF chain (MINF from scratch, original)        0.4479                0.585
```

### 4. Contrast cosines (whole model): do the MINF lineages deviate from run A in a common direction?
cos(G-A, I-A) etc. compare two MINF-minus-A deviations; positive and consistently above the within-MINF reference contrasts (cos(G-I, M-I) etc.) means a shared MINF deviation from run A. Under pure independent noise, every contrast of the form cos(X-R, Y-R) has expectation +0.5 (the shared -R term), so the REFERENCE contrasts, not zero, are the null.
```
contrast                          cos
cos(G-A, I-A)                 +0.4998
cos(G-A, M-A)                 +0.5038
cos(I-A, M-A)                 +0.5041
cos(I-G, M-G)  [ref]          +0.5022
cos(G-I, M-I)  [ref]          +0.5051
cos(G-M, I-M)  [ref]          +0.4927
cos(A-G, I-G)  [ref]          +0.4968
cos(A-I, G-I)  [ref]          +0.5034
cos(A-M, G-M)  [ref]          +0.4950
```

### 5. By parameter group
Columns: group | tensors | elements | changed share per run | cos(dX,dY) for MINF-MINF pairs and MINF-A pairs (mean) | sign agreement MINF-MINF (mean) vs MINF-A (mean) | d_LOO(A) / mean d_LOO(MINF)
```
group                  tensors     elements    chg A    chg G    chg I    chg M   cos MM   cos MA  agree MM  agree MA LOO A/MINF
moe_experts_down          2944 14,687,404,032   4.450%   4.443%   4.457%   4.398%   +0.000   +0.000     50.0%     50.0%      1.005
moe_experts_up            2944 14,687,404,032   5.128%   5.115%   5.135%   5.067%   +0.000   +0.001     50.0%     50.0%      1.005
mtp_head                   270 1,335,325,952   0.000%   0.000%   0.000%   0.000%   +0.995   +0.995     98.7%     99.5%      0.960
mamba_in_proj               23  637,034,496   9.110%   9.079%   9.155%   9.063%   +0.001   +0.001     50.1%     50.2%      1.002
embeddings                   1  352,321,536   2.346%   2.391%   2.357%   2.348%   +0.001   +0.000     50.0%     50.0%      0.999
lm_head                      1  352,321,536   1.349%   1.379%   1.361%   1.360%   +0.001   -0.000     50.2%     50.1%      0.999
mamba_out_proj              23  253,231,104  10.621%  10.584%  10.680%  10.584%   -0.001   +0.002     49.9%     50.2%      0.999
moe_shared_down             23  229,490,688  11.889%  11.845%  11.916%  11.807%   -0.001   +0.001     49.9%     50.1%      1.002
moe_shared_up               23  229,490,688  13.518%  13.481%  13.542%  13.436%   +0.000   +0.002     50.0%     50.2%      1.002
attn_o                       6   66,060,288  10.742%  10.703%  10.809%  10.661%   -0.000   +0.002     50.0%     50.2%      1.001
attn_q                       6   66,060,288   8.302%   8.251%   8.315%   8.267%   -0.000   -0.001     49.9%     49.9%      1.004
moe_router                  23    7,913,472   0.000%   0.000%   0.000%   0.000%     +nan     +nan      nan%      nan%        nan
attn_k                       6    4,128,768   7.408%   7.390%   7.474%   7.423%   +0.000   +0.001     50.0%     50.0%      0.998
attn_v                       6    4,128,768  16.990%  16.920%  17.053%  16.872%   -0.004   -0.001     49.7%     49.9%      1.000
mamba_conv1d                46      706,560  23.633%  23.577%  23.760%  23.520%   -0.003   +0.001     49.9%     50.0%      1.000
layer_norm                  52      139,776   0.072%   0.075%   0.073%   0.072%   -0.045   -0.013     46.4%     52.2%      0.917
mamba_norm                  23       94,208   0.402%   0.414%   0.420%   0.413%   -0.034   +0.019     50.9%     50.8%      0.940
moe_router_bias_f32         23        2,944  98.981%  98.913%  98.947%  98.947%   +0.997   +0.997     99.2%     99.2%      0.879
final_norm                   1        2,688   0.037%   0.037%   0.037%   0.000%   -1.000   +0.000      0.0%     50.0%      1.000
mamba_A_log                 23        1,472   0.068%   0.136%   0.068%   0.068%   -0.333   +0.200     33.3%     66.7%      0.667
mamba_D                     23        1,472   6.861%   7.745%   6.929%   7.269%   -0.043   -0.057     46.2%     44.3%      0.968
mamba_dt_bias               23        1,472   0.068%   0.068%   0.000%   0.000%     +nan   -1.000      nan%      0.0%      3.000
```

### 5b. Update norm by parameter group: ||delta_X|| per run and the group's share of the whole-model ||delta||^2 (run A). Groups with a near-1 cosine but no changed elements (mtp_head) or deterministic bias updates (moe_router_bias_f32) dominate the whole-model cosine in section 2.
```
group                      ||dA||     ||dG||     ||dI||     ||dM|| share of ||dA||^2
moe_router_bias_f32        0.6497     0.6477     0.6493     0.6488            71.18%
moe_experts_up             0.2587     0.2576     0.2578     0.2543            11.28%
moe_experts_down           0.2380     0.2370     0.2371     0.2340             9.55%
mtp_head                   0.1341     0.1333     0.1328     0.1339             3.03%
mamba_in_proj              0.1013     0.1011     0.1016     0.1003             1.73%
moe_shared_up              0.0734     0.0733     0.0733     0.0726             0.91%
mamba_out_proj             0.0711     0.0709     0.0713     0.0706             0.85%
moe_shared_down            0.0676     0.0674     0.0676     0.0668             0.77%
attn_o                     0.0377     0.0376     0.0378     0.0372             0.24%
attn_q                     0.0324     0.0322     0.0323     0.0321             0.18%
embeddings                 0.0295     0.0298     0.0295     0.0294             0.15%
lm_head                    0.0234     0.0236     0.0234     0.0234             0.09%
attn_v                     0.0115     0.0115     0.0115     0.0114             0.02%
attn_k                     0.0079     0.0079     0.0080     0.0079             0.01%
mamba_conv1d               0.0049     0.0049     0.0049     0.0049             0.00%
mamba_norm                 0.0002     0.0002     0.0002     0.0002             0.00%
mamba_D                    0.0001     0.0001     0.0001     0.0001             0.00%
layer_norm                 0.0001     0.0001     0.0001     0.0001             0.00%
mamba_dt_bias              0.0000     0.0000     0.0000     0.0000             0.00%
final_norm                 0.0000     0.0000     0.0000     0.0000             0.00%
mamba_A_log                0.0000     0.0000     0.0000     0.0000             0.00%
moe_router                 0.0000     0.0000     0.0000     0.0000             0.00%
```

### 5c. The only unquantized signal: MoE router load-balancing bias (fp32, 23 layers x 128 experts). This bias is updated from expert-load statistics of the training tokens, so it records the routing history of each run exactly.
Columns: pair | ||bias_X - bias_Y|| | cos(delta_X, delta_Y);  then per run d_LOO = ||delta_X - mean(delta of the other runs)||
```
pair        ||bX-bY||  cos(dX,dY)
A vs G        0.04483    +0.99762
A vs I        0.05425    +0.99651
A vs M        0.04555    +0.99754
G vs I        0.05624    +0.99624
G vs M        0.05012    +0.99701
I vs M        0.05276    +0.99670
run                                               ||delta||      d_LOO
run A (vLLM from scratch)                           0.64970    0.03745
chain G (MINF from scratch, no prefix cache)        0.64768    0.04117
chain I (MINF from scratch, prefix cache on)        0.64931    0.04722
main MINF chain (MINF from scratch, original)       0.64875    0.03944
```
Per layer, d_LOO of each run's router-bias delta (which run's routing history is the odd one out, layer by layer):
```
layer    d_LOO A    d_LOO G    d_LOO I    d_LOO M  odd one out
    1    0.00763    0.00791    0.00991    0.00840            I
    3    0.00899    0.00952    0.00884    0.00993            M
    6    0.00886    0.00939    0.00891    0.00976            M
    8    0.00707    0.00812    0.01033    0.00844            I
   10    0.00808    0.00916    0.01167    0.00780            I
   13    0.00935    0.00991    0.01321    0.00926            I
   15    0.00733    0.00886    0.00991    0.00745            I
   17    0.00773    0.00840    0.01112    0.00834            I
   20    0.00975    0.00938    0.00975    0.01010            M
   22    0.00737    0.01080    0.01147    0.00767            I
   24    0.00982    0.00877    0.01168    0.00936            I
   27    0.00847    0.00820    0.01153    0.00815            I
   29    0.00721    0.00917    0.00797    0.00814            G
   31    0.00744    0.00985    0.01008    0.00896            I
   34    0.00748    0.00777    0.00923    0.00760            I
   36    0.00823    0.00860    0.01136    0.00860            I
   38    0.00628    0.00738    0.00813    0.00732            I
   40    0.00612    0.00700    0.00760    0.00749            I
   43    0.00684    0.00826    0.00857    0.00710            I
   45    0.00660    0.00625    0.00653    0.00646            A
   47    0.00668    0.00817    0.00895    0.00725            I
   49    0.00760    0.00821    0.00816    0.00705            G
   51    0.00712    0.00687    0.00854    0.00712            I
```

### 6. By layer (backbone layers only): sign agreement MINF-MINF vs MINF-A, and d_LOO(A) / mean d_LOO(MINF)
```
layer     elements  agree MM  agree MA   cos MM   cos MA LOO A/MINF
    0   76,600,000     51.9%     51.4%   +0.022   +0.015      0.985
    1 2,594,939,008     50.3%     50.2%   +0.866   +0.867      0.997
    2   38,744,896     50.7%     50.4%   +0.008   +0.003      0.997
    3 1,297,468,160     50.1%     50.0%   +0.732   +0.730      1.006
    4   38,744,896     50.4%     50.2%   +0.004   +0.002      1.006
    5   23,399,040     50.6%     50.1%   +0.007   +0.002      0.998
    6 1,297,468,160     50.1%     50.0%   +0.729   +0.727      1.007
    7   38,744,896     50.0%     49.8%   +0.000   -0.002      1.004
    8 1,297,468,160     50.0%     50.0%   +0.756   +0.757      1.003
    9   38,744,896     49.9%     49.8%   -0.001   -0.002      1.000
   10 1,297,468,160     50.0%     50.0%   +0.765   +0.765      1.004
   11   38,744,896     50.0%     49.9%   -0.001   -0.001      0.999
   12   23,399,040     49.9%     50.0%   -0.002   -0.000      1.002
   13 1,297,468,160     50.0%     50.0%   +0.735   +0.734      1.005
   14   38,744,896     50.0%     49.9%   -0.001   -0.001      1.003
   15 1,297,468,160     50.0%     50.0%   +0.737   +0.737      1.004
   16   38,744,896     50.1%     49.9%   +0.001   -0.002      1.006
   17 1,297,468,160     50.0%     50.0%   +0.680   +0.679      1.007
   18   38,744,896     49.9%     49.9%   -0.001   -0.002      1.007
   19   23,399,040     49.8%     50.0%   -0.002   +0.000      1.008
   20 1,297,468,160     50.0%     50.1%   +0.716   +0.715      1.006
   21   38,744,896     49.9%     50.2%   -0.002   +0.002      1.003
   22 1,297,468,160     50.0%     50.1%   +0.649   +0.648      1.004
   23   38,744,896     50.0%     50.2%   -0.000   +0.001      1.004
   24 1,297,468,160     50.0%     50.1%   +0.714   +0.713      1.006
   25   38,744,896     49.9%     50.2%   -0.001   +0.002      1.003
   26   23,399,040     49.6%     50.1%   -0.004   +0.001      1.005
   27 1,297,468,160     50.0%     50.0%   +0.688   +0.687      1.005
   28   38,744,896     49.8%     50.0%   -0.002   +0.000      1.005
   29 1,297,468,160     50.0%     50.1%   +0.704   +0.703      1.003
   30   38,744,896     49.9%     50.1%   -0.001   +0.001      1.001
   31 1,297,468,160     50.0%     50.1%   +0.750   +0.750      1.003
   32   38,744,896     49.8%     50.1%   -0.003   +0.002      0.997
   33   23,399,040     49.8%     50.1%   -0.002   +0.000      1.000
   34 1,297,468,160     50.0%     50.0%   +0.803   +0.802      1.002
   35   38,744,896     50.1%     50.3%   +0.001   +0.003      1.003
   36 1,297,468,160     50.0%     50.0%   +0.820   +0.821      1.003
   37   38,744,896     50.1%     50.0%   +0.001   +0.000      1.004
   38 1,297,468,160     49.9%     50.1%   +0.849   +0.849      1.002
   39   38,744,896     49.9%     50.3%   -0.001   +0.003      1.000
   40 1,297,468,160     50.0%     50.1%   +0.838   +0.839      0.999
   41   38,744,896     49.8%     50.2%   -0.002   +0.002      1.001
   42   23,399,040     49.9%     50.1%   -0.001   +0.001      0.999
   43 1,297,468,160     49.9%     50.1%   +0.884   +0.884      0.999
   44   38,744,896     49.8%     50.3%   -0.002   +0.003      0.999
   45 1,297,468,160     49.9%     50.1%   +0.853   +0.853      1.000
   46   38,744,896     49.7%     50.3%   -0.003   +0.003      0.997
   47 1,297,468,160     50.0%     50.2%   +0.870   +0.870      1.000
   48   38,744,896     49.6%     50.5%   -0.004   +0.005      0.995
   49 1,297,468,160     49.9%     50.1%   +0.871   +0.869      1.004
   50   38,744,896     49.7%     50.4%   -0.004   +0.004      0.993
   51 1,297,468,160     50.0%     50.0%   +0.834   +0.833      1.004
```

## Step 20: checkpoints present: A = run A (vLLM from scratch), G = chain G (MINF from scratch, no prefix cache), I = chain I (MINF from scratch, prefix cache on), M = main MINF chain (MINF from scratch, original)

Tensors: 6,513  elements: 32.913 B  ||W_base|| = 4181.6

### 1. Integrity and update size per run (whole model)
Columns: run | tensors bit-identical to base | tensors with NaN/Inf | elements changed (share) | ||delta|| | ||delta||/||W_base|| | max |delta| over all elements
```
run                                              identical tensors  NaN/Inf changed elems  ||delta||   rel norm max|delta|
run A (vLLM from scratch)                                      387        0       6.2534%     1.0176   2.43e-04     0.0200
chain G (MINF from scratch, no prefix cache)                   385        0       6.2539%     1.0140   2.42e-04     0.0200
chain I (MINF from scratch, prefix cache on)                   388        0       6.2359%     1.0166   2.43e-04     0.0200
main MINF chain (MINF from scratch, original)                  387        0       6.1274%     1.0087   2.41e-04     0.0200
```

### 2. Pairwise geometry of the deltas (whole model)
Columns: pair | ||W_X - W_Y|| | cos(delta_X, delta_Y) | elements flipped in both | sign agreement among those (50 % = independent updates)
```
pair        ||W_X-W_Y||  cos(dX,dY)  both flipped  sign agree
A vs G           0.8311     +0.6653 1,237,649,508      50.21%
A vs I           0.8339     +0.6639 1,237,408,254      49.96%
A vs M           0.8242     +0.6692 1,225,890,824      49.98%
G vs I           0.8325     +0.6639 1,237,046,745      50.07%
G vs M           0.8236     +0.6685 1,225,368,765      50.09%
I vs M           0.8227     +0.6700 1,225,248,803      50.07%
```

### 3. Leave-one-out exchangeability test (whole model)
d_LOO(X) = ||delta_X - mean(delta of the other runs)||. If the four runs were exchangeable draws around a common RL direction, all four values would be similar; if the three MINF lineages form a cluster that run A is not part of, run A's value stands out.
```
run                                                   d_LOO  d_LOO / ||delta_X||
run A (vLLM from scratch)                            0.6789                0.667
chain G (MINF from scratch, no prefix cache)         0.6778                0.668
chain I (MINF from scratch, prefix cache on)         0.6788                0.668
main MINF chain (MINF from scratch, original)        0.6687                0.663
```

### 4. Contrast cosines (whole model): do the MINF lineages deviate from run A in a common direction?
cos(G-A, I-A) etc. compare two MINF-minus-A deviations; positive and consistently above the within-MINF reference contrasts (cos(G-I, M-I) etc.) means a shared MINF deviation from run A. Under pure independent noise, every contrast of the form cos(X-R, Y-R) has expectation +0.5 (the shared -R term), so the REFERENCE contrasts, not zero, are the null.
```
contrast                          cos
cos(G-A, I-A)                 +0.5000
cos(G-A, M-A)                 +0.5049
cos(I-A, M-A)                 +0.5076
cos(I-G, M-G)  [ref]          +0.5064
cos(G-I, M-I)  [ref]          +0.5049
cos(G-M, I-M)  [ref]          +0.4886
cos(A-G, I-G)  [ref]          +0.4975
cos(A-I, G-I)  [ref]          +0.5025
cos(A-M, G-M)  [ref]          +0.4912
```

### 5. By parameter group
Columns: group | tensors | elements | changed share per run | cos(dX,dY) for MINF-MINF pairs and MINF-A pairs (mean) | sign agreement MINF-MINF (mean) vs MINF-A (mean) | d_LOO(A) / mean d_LOO(MINF)
```
group                  tensors     elements    chg A    chg G    chg I    chg M   cos MM   cos MA  agree MM  agree MA LOO A/MINF
moe_experts_down          2944 14,687,404,032   5.822%   5.824%   5.803%   5.694%   +0.001   +0.000     50.1%     50.0%      1.008
moe_experts_up            2944 14,687,404,032   6.696%   6.693%   6.674%   6.551%   +0.001   +0.000     50.1%     50.0%      1.008
mtp_head                   270 1,335,325,952   0.000%   0.000%   0.000%   0.000%   +0.994   +0.995     98.4%     99.5%      0.877
mamba_in_proj               23  637,034,496  11.744%  11.755%  11.774%  11.678%   +0.002   +0.001     50.2%     50.1%      1.002
embeddings                   1  352,321,536   3.148%   3.202%   3.154%   3.121%   +0.001   +0.000     50.0%     50.0%      1.002
lm_head                      1  352,321,536   1.855%   1.891%   1.870%   1.853%   +0.000   -0.001     50.1%     50.1%      1.000
mamba_out_proj              23  253,231,104  13.649%  13.667%  13.686%  13.613%   +0.001   +0.002     50.1%     50.2%      1.000
moe_shared_down             23  229,490,688  15.290%  15.287%  15.272%  15.157%   +0.001   +0.001     50.1%     50.1%      1.003
moe_shared_up               23  229,490,688  17.336%  17.335%  17.327%  17.193%   +0.001   +0.002     50.1%     50.2%      1.003
attn_o                       6   66,060,288  13.807%  13.821%  13.834%  13.698%   +0.001   +0.002     50.1%     50.2%      1.002
attn_q                       6   66,060,288  10.750%  10.702%  10.709%  10.669%   -0.000   -0.001     50.0%     49.9%      1.006
moe_router                  23    7,913,472   0.000%   0.000%   0.000%   0.000%     +nan     +nan      nan%      nan%        nan
attn_k                       6    4,128,768   9.569%   9.577%   9.603%   9.592%   +0.002   +0.001     50.2%     50.0%      0.999
attn_v                       6    4,128,768  21.216%  21.206%  21.231%  21.027%   -0.002   -0.002     49.9%     49.9%      1.003
mamba_conv1d                46      706,560  27.353%  27.400%  27.422%  27.259%   +0.003   -0.000     50.2%     50.0%      1.003
layer_norm                  52      139,776   0.072%   0.077%   0.078%   0.074%   -0.035   +0.010     46.6%     50.0%      0.893
mamba_norm                  23       94,208   0.430%   0.438%   0.451%   0.450%   -0.002   +0.002     52.0%     52.0%      0.967
moe_router_bias_f32         23        2,944  97.860%  97.792%  97.622%  97.792%   +0.994   +0.995     98.7%     98.8%      0.870
final_norm                   1        2,688   0.037%   0.037%   0.037%   0.037%   -0.333   -0.333     33.3%     33.3%      1.000
mamba_A_log                 23        1,472   0.068%   0.136%   0.068%   0.068%   -0.333   +0.149     33.3%     66.7%      1.145
mamba_D                     23        1,472   8.356%   8.084%   8.356%   8.288%   -0.036   -0.044     45.7%     43.1%      0.975
mamba_dt_bias               23        1,472   0.068%   0.068%   0.000%   0.000%     +nan   -1.000      nan%      0.0%      3.000
```

### 5b. Update norm by parameter group: ||delta_X|| per run and the group's share of the whole-model ||delta||^2 (run A). Groups with a near-1 cosine but no changed elements (mtp_head) or deterministic bias updates (moe_router_bias_f32) dominate the whole-model cosine in section 2.
```
group                      ||dA||     ||dG||     ||dI||     ||dM|| share of ||dA||^2
moe_router_bias_f32        0.8135     0.8097     0.8153     0.8120            63.91%
moe_experts_up             0.3906     0.3898     0.3876     0.3802            14.73%
moe_experts_down           0.3592     0.3585     0.3564     0.3499            12.46%
mtp_head                   0.1707     0.1697     0.1690     0.1712             2.81%
mamba_in_proj              0.1487     0.1491     0.1487     0.1472             2.13%
moe_shared_up              0.1069     0.1070     0.1065     0.1056             1.10%
mamba_out_proj             0.1035     0.1039     0.1036     0.1030             1.03%
moe_shared_down            0.0988     0.0989     0.0984     0.0976             0.94%
attn_o                     0.0548     0.0549     0.0547     0.0540             0.29%
attn_q                     0.0475     0.0473     0.0471     0.0469             0.22%
embeddings                 0.0449     0.0452     0.0448     0.0444             0.19%
lm_head                    0.0361     0.0363     0.0362     0.0360             0.13%
attn_v                     0.0164     0.0164     0.0163     0.0161             0.03%
attn_k                     0.0115     0.0116     0.0116     0.0116             0.01%
mamba_conv1d               0.0067     0.0067     0.0067     0.0067             0.00%
mamba_norm                 0.0003     0.0003     0.0003     0.0003             0.00%
mamba_D                    0.0002     0.0002     0.0002     0.0002             0.00%
layer_norm                 0.0001     0.0001     0.0001     0.0001             0.00%
mamba_A_log                0.0000     0.0000     0.0000     0.0000             0.00%
mamba_dt_bias              0.0000     0.0000     0.0000     0.0000             0.00%
final_norm                 0.0000     0.0000     0.0000     0.0000             0.00%
moe_router                 0.0000     0.0000     0.0000     0.0000             0.00%
```

### 5c. The only unquantized signal: MoE router load-balancing bias (fp32, 23 layers x 128 experts). This bias is updated from expert-load statistics of the training tokens, so it records the routing history of each run exactly.
Columns: pair | ||bias_X - bias_Y|| | cos(delta_X, delta_Y);  then per run d_LOO = ||delta_X - mean(delta of the other runs)||
```
pair        ||bX-bY||  cos(dX,dY)
A vs G        0.06551    +0.99675
A vs I        0.09648    +0.99299
A vs M        0.07288    +0.99598
G vs I        0.09652    +0.99297
G vs M        0.07862    +0.99530
I vs M        0.08816    +0.99414
run                                               ||delta||      d_LOO
run A (vLLM from scratch)                           0.81353    0.06098
chain G (MINF from scratch, no prefix cache)        0.80970    0.06409
chain I (MINF from scratch, prefix cache on)        0.81532    0.08393
main MINF chain (MINF from scratch, original)       0.81201    0.06225
```
Per layer, d_LOO of each run's router-bias delta (which run's routing history is the odd one out, layer by layer):
```
layer    d_LOO A    d_LOO G    d_LOO I    d_LOO M  odd one out
    1    0.00993    0.01067    0.01343    0.01095            I
    3    0.01308    0.01191    0.01539    0.01533            I
    6    0.01392    0.01223    0.01304    0.01515            M
    8    0.01075    0.01353    0.01675    0.01254            I
   10    0.01116    0.01520    0.01932    0.01124            I
   13    0.01503    0.01854    0.02341    0.01611            I
   15    0.01266    0.01600    0.02168    0.01327            I
   17    0.01295    0.01478    0.02226    0.01316            I
   20    0.01370    0.01351    0.01666    0.01437            I
   22    0.01390    0.01898    0.01973    0.01690            I
   24    0.01503    0.01448    0.02238    0.01448            I
   27    0.01549    0.01373    0.01961    0.01552            I
   29    0.01354    0.01461    0.01589    0.01310            I
   31    0.01575    0.01464    0.01888    0.01652            I
   34    0.01258    0.01193    0.02091    0.01222            I
   36    0.01356    0.01169    0.01554    0.01095            I
   38    0.01167    0.01104    0.01751    0.00944            I
   40    0.00939    0.00855    0.01170    0.00920            I
   43    0.01260    0.01253    0.01626    0.01291            I
   45    0.01025    0.00855    0.01204    0.00900            I
   47    0.01030    0.01251    0.01455    0.00896            I
   49    0.00967    0.00990    0.01376    0.00958            I
   51    0.01238    0.01197    0.01399    0.01220            I
```

### 6. By layer (backbone layers only): sign agreement MINF-MINF vs MINF-A, and d_LOO(A) / mean d_LOO(MINF)
```
layer     elements  agree MM  agree MA   cos MM   cos MA LOO A/MINF
    0   76,600,000     52.5%     51.6%   +0.030   +0.018      0.990
    1 2,594,939,008     50.4%     50.2%   +0.813   +0.814      1.001
    2   38,744,896     50.9%     50.5%   +0.010   +0.005      0.998
    3 1,297,468,160     50.1%     50.0%   +0.649   +0.648      1.009
    4   38,744,896     50.3%     50.2%   +0.003   +0.002      1.006
    5   23,399,040     50.6%     50.0%   +0.007   +0.001      1.005
    6 1,297,468,160     50.1%     50.0%   +0.655   +0.652      1.011
    7   38,744,896     50.1%     49.7%   +0.001   -0.003      1.005
    8 1,297,468,160     50.1%     50.0%   +0.677   +0.677      1.006
    9   38,744,896     49.9%     49.8%   -0.001   -0.002      1.001
   10 1,297,468,160     50.0%     50.0%   +0.693   +0.693      1.005
   11   38,744,896     50.0%     49.9%   -0.001   -0.001      0.999
   12   23,399,040     50.0%     50.0%   -0.000   -0.001      1.005
   13 1,297,468,160     50.1%     50.0%   +0.651   +0.650      1.006
   14   38,744,896     50.1%     49.8%   +0.001   -0.002      1.005
   15 1,297,468,160     50.0%     50.0%   +0.658   +0.657      1.006
   16   38,744,896     50.2%     49.7%   +0.002   -0.003      1.009
   17 1,297,468,160     50.1%     50.0%   +0.604   +0.602      1.008
   18   38,744,896     50.1%     49.7%   +0.001   -0.004      1.009
   19   23,399,040     49.9%     50.0%   -0.002   -0.000      1.008
   20 1,297,468,160     50.1%     50.1%   +0.650   +0.648      1.008
   21   38,744,896     49.9%     50.2%   -0.002   +0.002      1.003
   22 1,297,468,160     50.1%     50.0%   +0.565   +0.565      1.005
   23   38,744,896     50.0%     50.1%   -0.000   +0.000      1.004
   24 1,297,468,160     50.0%     50.1%   +0.634   +0.632      1.006
   25   38,744,896     49.9%     50.2%   -0.001   +0.002      1.002
   26   23,399,040     49.7%     50.2%   -0.004   +0.002      1.004
   27 1,297,468,160     50.0%     50.0%   +0.614   +0.614      1.007
   28   38,744,896     49.9%     50.0%   -0.000   +0.000      1.006
   29 1,297,468,160     50.1%     50.1%   +0.635   +0.633      1.005
   30   38,744,896     50.1%     50.1%   +0.001   +0.001      1.002
   31 1,297,468,160     50.1%     50.1%   +0.684   +0.682      1.006
   32   38,744,896     49.9%     50.2%   -0.001   +0.002      0.999
   33   23,399,040     50.0%     50.1%   +0.000   +0.001      1.000
   34 1,297,468,160     50.0%     50.0%   +0.752   +0.752      1.003
   35   38,744,896     50.3%     50.3%   +0.003   +0.003      1.002
   36 1,297,468,160     50.0%     50.0%   +0.763   +0.761      1.008
   37   38,744,896     50.3%     49.9%   +0.003   -0.001      1.005
   38 1,297,468,160     50.0%     50.0%   +0.782   +0.781      1.005
   39   38,744,896     50.2%     50.3%   +0.002   +0.003      1.000
   40 1,297,468,160     50.1%     50.1%   +0.772   +0.771      1.006
   41   38,744,896     50.0%     50.1%   +0.000   +0.001      1.002
   42   23,399,040     50.2%     50.2%   +0.001   +0.002      1.000
   43 1,297,468,160     50.0%     50.1%   +0.838   +0.837      1.003
   44   38,744,896     50.0%     50.3%   +0.001   +0.003      0.999
   45 1,297,468,160     50.0%     50.1%   +0.782   +0.782      1.004
   46   38,744,896     49.9%     50.3%   -0.000   +0.004      0.996
   47 1,297,468,160     50.0%     50.1%   +0.818   +0.817      1.003
   48   38,744,896     49.9%     50.5%   -0.001   +0.005      0.994
   49 1,297,468,160     50.0%     50.1%   +0.815   +0.812      1.005
   50   38,744,896     49.8%     50.5%   -0.001   +0.006      0.989
   51 1,297,468,160     50.1%     50.0%   +0.783   +0.782      1.005
```

