# Dump aggregate 2026-09-19 11:02

## Coverage
        rows      steps  groups  instances     wv
engine                                           
minf    4096  10-17 (8)     256        256  10-16
vllm    4176   0-9 (10)     261        261    0-8

rows per target_step:
target_step   0    1    2    3    4    5    6    7   8   9    10   11   12   13   14   15   16   17
engine                                                                                             
minf           0    0    0    0    0    0    0    0   0   0  512  512  512  512  512  512  512  512
vllm         512  512  512  512  512  512  512  512  64  16    0    0    0    0    0    0    0    0

Overlap of instance_ids between engines: 0

## Per-engine rollout stats
engine                                  minf            vllm
rows                                    4096            4176
reward_mean                         0.279053        0.363985
resolved_rate                       0.279053        0.363985
patch_exists_rate                   0.954834        0.961446
patch_len_p50                         1510.5          1287.0
truncated_rate                      0.016602         0.01341
at_max_turns_rate                   0.049316        0.051964
agent_max_turns                          200             200
turns_mean                         99.247803       94.342193
turns_p50                               91.0            84.0
turns_p99                              200.0           200.0
turns_max                                200             200
gen_len_mean                    29112.057373    26952.984435
gen_len_p99                         88756.05         82224.0
gen_len_max                           163172          172600
total_tokens_p50                     89237.5         83826.5
total_tokens_max                      196608          196608
prompt_tokens_mean               6098.058594     6119.793103
max_asst_tokens_p99                  24003.4         15612.0
max_asst_tokens_max                   141356          152727
invalid_tool_call_rows              0.007324        0.006705
invalid_tool_call_sum                     32              29
malformed_thinking_rows                  0.0             0.0
malformed_thinking_sum                     0               0
finish_called_rate                  0.910156        0.909004
last_gen_has_tool_call              0.996094        0.997126
last_gen_has_im_end                 0.997803        0.997845
rows_gen_without_im_end             0.009766        0.008381
sum_gen_without_im_end                    50              50
rows_gen_without_think_close        0.007568        0.006226
sum_gen_without_think_close               41              41
max_gen_chars_p99                   55620.45        47319.25
max_gen_chars_max                     351893          405024
n_message_mean                     99.251953        5.467193
n_message_nonblank_mean             4.526611        5.466715
n_reasoning_mean                   99.247803       94.342193
n_function_call_mean               99.231934       94.330939
cache_read_tokens_mean        5611180.212485             0.0
usage_prompt_tokens_mean      5833066.161567  5555694.765117
usage_completion_tokens_mean    29181.168176    27044.224524
latency_max_p50                    23.246707       15.710212
latency_max_p99                   225.986742       115.28749
latency_max_max                   819.938258      664.551265
openhands_run_time_p50            553.560539      558.570946
openhands_run_time_p99           1372.150898     1513.558318
model_call_time_p50               368.762318      356.980126
cmd_exec_time_p50                  79.488225        78.00348
final_eval_time_p50                21.143492        19.72537
agent_timed_out                          0.0             0.0
eval_timed_out                          27.0            38.0
oom_killed                               3.0            12.0
eval_oom_killed                          0.0             0.0
resp_error_rows                            0               0
resp_incomplete_rows                       0               0

### agent_error_kind counts
engine                
minf    None              3731
        max_iteration      202
        context_window     150
        stuck_in_loop        6
        other                4
        oom                  3
vllm    None              3801
        max_iteration      218
        context_window     123
        stuck_in_loop       13
        oom                 12
        other                9

### mask_sample counts
engine  minf 
minf    False    4029
        True       67
vllm    False    4097
        True       79

### resp_status counts
engine  minf
minf    None    4096
vllm    None    4176

### last_item_type counts
engine  minf                
minf    function_call           3946
        function_call_output      91
        message                   59
vllm    function_call           4049
        function_call_output      80
        message                   47

### last_item_name counts
engine  minf              
minf    finish                3728
        execute_bash           193
        None                   150
        str_replace_editor      23
        task_tracker             2
vllm    finish                3796
        execute_bash           221
        None                   127
        str_replace_editor      31
        task_tracker             1

### resp_error counts
engine  minf
minf    None    4096
vllm    None    4176

### resp_incomplete counts
engine  minf
minf    None    4096
vllm    None    4176

### item_statuses (summed)
minf {'completed': 1215392}
vllm {'completed': 806546}

### fc_names (summed, top 12)
minf {'execute_bash': 316805, 'str_replace_editor': 78445, 'task_tracker': 6211, 'finish': 3728, 'think': 1265}
vllm {'execute_bash': 306424, 'str_replace_editor': 77073, 'task_tracker': 5176, 'finish': 3796, 'think': 1457}

## Reward attribution cross-tabs (reward mean / n)

### reward by error_kind
                 mean         size      
engine           minf   vllm  minf  vllm
error_kind                              
context_window  0.107  0.057   150   123
max_iteration   0.104  0.101   202   218
none            0.296  0.392  3731  3801
oom             0.000  0.000     3    12
other           0.000  0.111     4     9
stuck_in_loop   0.000  0.077     6    13

### reward by ctx_hit
          mean         size      
engine    minf   vllm  minf  vllm
ctx_hit                          
False    0.283  0.369  4028  4120
True     0.044  0.018    68    56

### reward by at_max_turns
               mean         size      
engine         minf   vllm  minf  vllm
at_max_turns                          
False         0.288  0.378  3894  3959
True          0.104  0.101   202   217

### reward by mask_sample
              mean         size      
engine        minf   vllm  minf  vllm
mask_sample                          
False        0.275  0.364  4029  4097
True         0.552  0.367    67    79

### reward by last_item_type
                       mean         size      
engine                 minf   vllm  minf  vllm
last_item_type                                
function_call         0.286  0.374  3946  4049
function_call_output  0.143  0.088    91    80
message               0.051  0.000    59    47

### reward by invalid_tc
             mean         size      
engine       minf   vllm  minf  vllm
invalid_tc                          
False       0.280  0.365  4066  4148
True        0.133  0.143    30    28

### reward by malformed
            mean         size      
engine      minf   vllm  minf  vllm
malformed                          
False      0.279  0.364  4096  4176

### reward by finish
         mean         size      
engine   minf   vllm  minf  vllm
finish                          
False   0.101  0.082   368   380
True    0.297  0.392  3728  3796

### reward by gen_no_im_end
               mean         size      
engine         minf   vllm  minf  vllm
gen_no_im_end                         
False          0.28  0.364  4056  4141
True           0.15  0.371    40    35

### reward by gen_no_think_close
                     mean         size      
engine               minf   vllm  minf  vllm
gen_no_think_close                          
False               0.280  0.363  4065  4150
True                0.194  0.462    31    26

### resolved==True but reward != 1, or resolved False but reward != 0
none

### reward when resolved but error kind set (Gym wants these masked; SC keeps them)
engine  error_kind    
minf    context_window    16
        max_iteration     21
vllm    context_window     7
        max_iteration     22
        other              1
        stuck_in_loop      1

### truncated / max-turn rows: what ended them
                                              n  reward    turns    gen_len  total_tokens  last_no_im_end
engine ctx_hit at_max_turns error_kind                                                                   
minf   False   True         max_iteration   194   0.103  200.000  44915.577    137864.902           0.000
                            none              8   0.125  200.000  56229.125    152412.375           0.000
       True    False        context_window   66   0.045  126.379  89575.167    196608.000           0.106
                            none              2   0.000  171.500  67362.500    196608.000           1.000
vllm   False   True         max_iteration   212   0.104  200.000  45909.212    138808.057           0.000
                            none              5   0.000  200.000  49732.000    149136.200           0.000
       True    False        context_window   49   0.020  126.755  84978.837    196608.000           0.061
                            none              2   0.000  104.000  28512.000    196608.000           0.500
                            other             4   0.000  102.750  78595.500    196608.000           1.000
                            stuck_in_loop     1   0.000  117.000  90750.000    196608.000           1.000

## Group-level (GRPO) reward structure
                 groups  rows
engine kind                  
minf   all_one       17   272
       all_zero     121  1936
       mixed        118  1888
vllm   all_one       31   496
       all_zero     107  1712
       mixed        123  1968

group sizes: {('minf', 16): 256, ('vllm', 16): 261}
groups spanning >1 target_step: {'minf': 0, 'vllm': 0}

## Per-step trend
                      n  reward  trunc  max_turns    turns    gen_len  invalid_tc  finish  lat_p99
engine target_step                                                                                
minf   10           512   0.277  0.018      0.049   99.781  28965.002       0.004   0.910  190.053
       11           512   0.336  0.010      0.047   94.750  28071.199       0.012   0.912  145.114
       12           512   0.270  0.010      0.102  113.682  31448.207       0.004   0.867  118.845
       13           512   0.271  0.018      0.018   88.771  26364.662       0.010   0.938  381.001
       14           512   0.229  0.021      0.064  105.318  31886.795       0.010   0.902  349.927
       15           512   0.316  0.023      0.045  107.322  31750.594       0.002   0.895  385.943
       16           512   0.234  0.010      0.029   89.547  25623.611       0.012   0.949  248.334
       17           512   0.299  0.023      0.041   94.811  28786.389       0.006   0.908  124.801
vllm   0            512   0.344  0.014      0.078  108.076  26493.076       0.006   0.896  184.158
       1            512   0.541  0.006      0.039   81.221  20568.309       0.004   0.945   96.898
       2            512   0.256  0.000      0.041   92.523  26381.305       0.004   0.947   98.133
       3            512   0.402  0.018      0.053   96.021  27127.457       0.014   0.896  108.125
       4            512   0.254  0.021      0.104  110.236  33049.572       0.006   0.836   88.355
       5            512   0.371  0.012      0.053   91.709  25578.457       0.002   0.898   81.039
       6            512   0.432  0.012      0.023   88.803  26319.373       0.006   0.930   75.509
       7            512   0.289  0.027      0.033   95.207  33225.416       0.014   0.908  418.623
       8             64   0.391  0.000      0.000   37.906   7418.641       0.000   1.000   28.691
       9             16   1.000  0.000      0.000   30.188   5279.500       0.000   1.000   10.567

## Token-level coverage
        rows  files                                            steps  gen_tokens
engine                                                                          
minf    4096     19  00011,00012,00013,00014,00015,00016,00017,00018   119242987
vllm    4096     20  00001,00002,00003,00004,00005,00006,00007,00008   111996398

## Per-engine token-level stats
engine                          minf          vllm
rows                    4.096000e+03  4.096000e+03
reward_mean             2.790527e-01  3.610840e-01
n_gen_tokens_mean       2.911206e+04  2.734287e+04
n_gen_tokens_max        1.631720e+05  1.726000e+05
input_length_max        1.966080e+05  1.966080e+05
rollout_truncated_rate  1.660156e-02  1.367188e-02
gen_len_tag_eq_n_gen    1.000000e+00  1.000000e+00
mask_before_mean        1.000000e+00  1.000000e+00
mask_after_mean         9.990234e-01  9.985352e-01
masked_rows             4.000000e+00  6.000000e+00
seq_mult_err_p50        1.017515e+00  1.015764e+00
seq_mult_err_p90        1.022402e+00  1.020878e+00
seq_mult_err_p99        1.027459e+00  1.025846e+00
seq_mult_err_max        5.379415e+03  1.591688e+01
seq_mult_err_gt2        4.000000e+00  6.000000e+00
lp_err_mean_mean        1.545250e-02  1.404525e-02
lp_err_mean_p99         2.317859e-02  2.150249e-02
lp_err_p99_p50          2.503458e-01  2.433033e-01
lp_err_max_p50          1.270030e+00  1.265269e+00
lp_err_max_p99          4.816381e+00  4.282741e+00
lp_err_max_max          1.938344e+01  1.350646e+01
tokens_err_gt1          1.103900e+04  1.056300e+04
tokens_err_gt5          3.800000e+01  3.000000e+01
rows_err_gt1            7.575684e-01  7.407227e-01
tokens_genlp_lt_m10     6.272000e+03  5.441000e+03
gen_lp_mean_mean       -1.767488e-01 -1.627421e-01
prev_lp_mean_mean      -1.783096e-01 -1.641420e-01
gen_lp_min_min         -1.841165e+01 -1.846929e+01
exp_err_mean_mean       2.411986e+00  1.026777e+00
adv_zero_rate           5.390625e-01  5.234375e-01
adv_min                -3.614770e+00 -3.614770e+00
adv_max                 3.614770e+00  3.614770e+00
adv_std_within_row_max  9.536743e-07  9.536743e-07

### last generated token id (top 6)
minf {'11=<|im_end|>': 4028, '1278=?': 7, '1046=?': 3, '1010=?': 2, '1261=?': 2, '3636=?': 2}
vllm {'11=<|im_end|>': 4040, '1278=?': 7, '9458=?': 3, '1046=?': 2, '9816=?': 2, '1561=?': 2}

### advantage consistency within groups (adv sign vs reward - group mean)
engine
minf    1.0
vllm    1.0
rows with adv==0 but reward != group mean: {}
rows with adv!=0 but reward == group mean: {}
group size seen in token chunks: {('minf', 16): 4096, ('vllm', 16): 4096}

### masked rows (sample_mask_after == 0)
  engine                                 sample_id  reward  n_gen_tokens  seq_mult_prob_error  lp_err_max  n_err_gt1 agent_error_kind  truncated  turns  resolved
0   minf  798c5fe9-d1a8-4334-b1d4-c5cb501bd35e_g12     0.0         48693          5379.414551   19.383436          4    max_iteration      False    200     False
1   minf   223511d7-c9a0-4ac5-b787-898f76c911ab_g2     0.0         39337            31.680656   14.003157          3    max_iteration      False    200     False
2   minf   bbf71fd1-fe74-4383-9f76-256550d96ced_g7     0.0         49007           160.979126   15.874703          4             None      False    197     False
3   minf  3a8ab125-f5a9-4700-a31f-51cffbd84cd7_g14     1.0         13804           142.038895   14.481648          1             None      False     90      True
4   vllm  505b832a-8435-41a4-b643-53eacd7bfa85_g14     0.0         17468             7.362216   11.616625          1             None      False     80     False
5   vllm  f544a353-abf5-4f2d-b351-cc81c397082a_g15     0.0         33926             3.548233   11.360438          3             None      False     77     False
6   vllm  693e1ad5-b8b3-4303-bf27-803c78a80cff_g10     0.0         47587            10.551061   13.025301          7             None      False    126     False
7   vllm  41306e67-ffda-4045-a2ec-c52513b7c7a4_g14     0.0         49256            15.916878   13.506462          4    max_iteration      False    200     False
8   vllm  eb28d4a7-a3b0-4e7c-989b-a9db79ec741f_g12     0.0         13795             7.917688   11.463479          3             None      False     85     False
9   vllm   084b8ca1-2113-4259-91ba-40c880ed6c17_g0     0.0         16482             3.280268   10.524735          3             None      False     76     False

### rows with seq_mult_prob_error > 2 or lp_err_max > 5 (top 15 by seq_mult_prob_error)
   engine                  file  reward  n_gen_tokens  seq_mult_prob_error  lp_err_mean  lp_err_max  n_err_gt1  n_err_gt5  sample_mask_after  truncated  turns  n_invalid_tool_call
0    minf  step_00013_chunk_002     0.0         48693          5379.414551     0.016448   19.383436          4          1                0.0      False    200                    0
1    minf  step_00014_chunk_001     0.0         49007           160.979126     0.008732   15.874703          4          1                0.0      False    197                    0
2    minf  step_00017_chunk_002     1.0         13804           142.038895     0.014185   14.481648          1          1                0.0      False     90                    0
3    minf  step_00013_chunk_003     0.0         39337            31.680656     0.011320   14.003157          3          1                0.0      False    200                    0
4    vllm  step_00005_chunk_003     0.0         49256            15.916878     0.011219   13.506462          4          1                0.0      False    200                    0
5    vllm  step_00004_chunk_001     0.0         47587            10.551061     0.014271   13.025301          7          1                0.0      False    126                    0
6    vllm  step_00007_chunk_003     0.0         13795             7.917688     0.017470   11.463479          3          1                0.0      False     85                    0
7    vllm  step_00002_chunk_001     0.0         17468             7.362216     0.011619   11.616625          1          1                0.0      False     80                    0
8    vllm  step_00003_chunk_003     0.0         33926             3.548233     0.015959   11.360438          3          1                0.0      False     77                    0
9    vllm  step_00008_chunk_001     0.0         16482             3.280268     0.020281   10.524735          3          1                0.0      False     76                    0
10   minf  step_00017_chunk_001     0.0         27324             1.813547     0.017311    9.985800          2          1                1.0      False     37                    0
11   vllm  step_00001_chunk_001     1.0         12171             1.673722     0.013305    8.991268          1          1                1.0      False     78                    0
12   minf  step_00012_chunk_001     0.0         20572             1.558933     0.015308    9.320006          3          1                1.0      False     58                    0
13   vllm  step_00007_chunk_001     1.0         11756             1.548218     0.013731    8.744454          1          1                1.0      False     33                    0
14   vllm  step_00008_chunk_001     0.0         33476             1.426131     0.012505    9.532432          5          1                1.0      False    101                    0

### join check: token rows with a rollout summary
        joined  reward_match
engine                      
minf       1.0           1.0
vllm       1.0           1.0

## lp_error histogram (fraction of generated tokens)

minf: |gen_lp - prev_lp| over 119242987 tokens
  [     0, 0.0001)   0.675973  80605045
  [0.0001,  0.001)   0.072889  8691517
  [ 0.001,   0.01)   0.087994  10492695
  [  0.01,   0.05)   0.075408  8991911
  [  0.05,    0.1)   0.036217  4318621
  [   0.1,   0.25)   0.040760  4860303
  [  0.25,    0.5)   0.009238  1101542
  [   0.5,      1)   0.001428  170311
  [     1,      2)   0.000088  10495
  [     2,      5)   0.000004  509
  [     5,     10)   0.000000  33
  [    10,     20)   0.000000  5
  [    20,     50)   0.000000  0
  [    50,    inf)   0.000000  0
minf: generation logprob distribution
  [  -inf,    -20)   0.000000  0
  [   -20,    -15)   0.000002  265
  [   -15,    -10)   0.000050  6007
  [   -10,     -8)   0.000143  17077
  [    -8,     -6)   0.000643  76684
  [    -6,     -4)   0.003167  377584
  [    -4,     -2)   0.017004  2027588
  [    -2,     -1)   0.031517  3758227
  [    -1,   -0.5)   0.038945  4643904
  [  -0.5,   -0.1)   0.089745  10701499
  [  -0.1,      0)   0.818783  97634152

vllm: |gen_lp - prev_lp| over 111996398 tokens
  [     0, 0.0001)   0.703543  78794294
  [0.0001,  0.001)   0.068945  7721627
  [ 0.001,   0.01)   0.080621  9029258
  [  0.01,   0.05)   0.067936  7608608
  [  0.05,    0.1)   0.032595  3650575
  [   0.1,   0.25)   0.036635  4102981
  [  0.25,    0.5)   0.008290  928429
  [   0.5,      1)   0.001340  150061
  [     1,      2)   0.000089  10005
  [     2,      5)   0.000005  530
  [     5,     10)   0.000000  24
  [    10,     20)   0.000000  6
  [    20,     50)   0.000000  0
  [    50,    inf)   0.000000  0
vllm: generation logprob distribution
  [  -inf,    -20)   0.000000  0
  [   -20,    -15)   0.000002  230
  [   -15,    -10)   0.000047  5211
  [   -10,     -8)   0.000135  15106
  [    -8,     -6)   0.000598  67028
  [    -6,     -4)   0.002907  325599
  [    -4,     -2)   0.015489  1734721
  [    -2,     -1)   0.028673  3211258
  [    -1,   -0.5)   0.035508  3976777
  [  -0.5,   -0.1)   0.082277  9214681
  [  -0.1,      0)   0.834364  93445787
