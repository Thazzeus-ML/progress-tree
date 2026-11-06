# Main comparison: 6 arms x 10 seeds, 2,400,000 env steps, merge noise 0.0, task seeds 5000/6000, update mode fixed_minibatch

## At equal environment steps (2,400,000)

| arm | IQM | mean | CI95 | n |
|---|---|---|---|---|
| GRPO | 0.260 | 0.255 | (0.217, 0.297) | 10 |
| GraphGPO | 0.240 | 0.251 | (0.216, 0.284) | 10 |
| GraphGPO(2xG) | 0.270 | 0.286 | (0.223, 0.346) | 10 |
| Tree+GraphGPO | 0.343 | 0.333 | (0.294, 0.379) | 10 |
| Progress+Tree | 0.426 | 0.430 | (0.398, 0.462) | 10 |
| Progress+Tree(LOO) | 0.464 | 0.461 | (0.441, 0.483) | 10 |

- `Progress+Tree` vs `GraphGPO`: +0.186 -> **PASS** (CIs separated: True; permutation p = 0.0001)
- `Progress+Tree` vs `GraphGPO(2xG)`: +0.156 -> **PASS** (CIs separated: True; permutation p = 0.0004)
- `Progress+Tree(LOO)` vs `GraphGPO`: +0.224 -> **PASS** (CIs separated: True; permutation p = 0.0001)
- `Progress+Tree(LOO)` vs `GraphGPO(2xG)`: +0.194 -> **PASS** (CIs separated: True; permutation p = 0.0001)

## At equal iterations (226)

| arm | IQM | mean | CI95 | n |
|---|---|---|---|---|
| GRPO | 0.087 | 0.089 | (0.079, 0.098) | 10 |
| GraphGPO | 0.147 | 0.149 | (0.134, 0.165) | 10 |
| GraphGPO(2xG) | 0.236 | 0.249 | (0.203, 0.297) | 10 |
| Tree+GraphGPO | 0.320 | 0.311 | (0.280, 0.347) | 10 |
| Progress+Tree | 0.415 | 0.414 | (0.378, 0.450) | 10 |
| Progress+Tree(LOO) | 0.457 | 0.457 | (0.441, 0.471) | 10 |

- `Progress+Tree` vs `GraphGPO`: +0.268 -> **PASS** (CIs separated: True; permutation p = 0.0001)
- `Progress+Tree` vs `GraphGPO(2xG)`: +0.180 -> **PASS** (CIs separated: True; permutation p = 0.0001)
- `Progress+Tree(LOO)` vs `GraphGPO`: +0.310 -> **PASS** (CIs separated: True; permutation p = 0.0001)
- `Progress+Tree(LOO)` vs `GraphGPO(2xG)`: +0.221 -> **PASS** (CIs separated: True; permutation p = 0.0001)

Wall time: 2067s on 14 workers.
