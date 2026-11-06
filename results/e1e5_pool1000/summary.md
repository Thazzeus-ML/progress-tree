# Main comparison: 10 arms x 10 seeds, 2,400,000 env steps, merge noise 0.0, task seeds 1000/2000, update mode fixed_minibatch

## At equal environment steps (2,400,000)

| arm | IQM | mean | CI95 | n |
|---|---|---|---|---|
| GraphGPO | 0.368 | 0.352 | (0.316, 0.397) | 10 |
| GraphGPO@gamma=0.5 | 0.223 | 0.233 | (0.179, 0.284) | 10 |
| GraphGPO@gamma=0.8 | 0.244 | 0.239 | (0.198, 0.279) | 10 |
| GraphGPO@graph_relative=True | 0.262 | 0.270 | (0.212, 0.325) | 10 |
| GiGPO | 0.403 | 0.404 | (0.381, 0.424) | 10 |
| Tree+GraphGPO | 0.369 | 0.376 | (0.356, 0.392) | 10 |
| Progress+Tree(LOO) | 0.535 | 0.535 | (0.519, 0.552) | 10 |
| Tree+GraphGPO@branch_mode=consistent | 0.364 | 0.365 | (0.346, 0.383) | 10 |
| Tree+GiGPO@branch_mode=consistent | 0.381 | 0.384 | (0.367, 0.400) | 10 |
| Progress+Tree(LOO)@branch_mode=consistent | 0.495 | 0.490 | (0.469, 0.508) | 10 |

- `Progress+Tree(LOO)` vs `GraphGPO`: +0.167 -> **PASS** (CIs separated: True; permutation p = 0.0001)

## At equal iterations (177)

| arm | IQM | mean | CI95 | n |
|---|---|---|---|---|
| GraphGPO | 0.201 | 0.201 | (0.178, 0.221) | 10 |
| GraphGPO@gamma=0.5 | 0.157 | 0.160 | (0.127, 0.189) | 10 |
| GraphGPO@gamma=0.8 | 0.156 | 0.158 | (0.124, 0.194) | 10 |
| GraphGPO@graph_relative=True | 0.181 | 0.181 | (0.150, 0.210) | 10 |
| GiGPO | 0.152 | 0.153 | (0.132, 0.172) | 10 |
| Tree+GraphGPO | 0.309 | 0.310 | (0.290, 0.330) | 10 |
| Progress+Tree(LOO) | 0.493 | 0.489 | (0.477, 0.504) | 10 |
| Tree+GraphGPO@branch_mode=consistent | 0.339 | 0.339 | (0.325, 0.350) | 10 |
| Tree+GiGPO@branch_mode=consistent | 0.373 | 0.374 | (0.353, 0.395) | 10 |
| Progress+Tree(LOO)@branch_mode=consistent | 0.480 | 0.472 | (0.454, 0.491) | 10 |

- `Progress+Tree(LOO)` vs `GraphGPO`: +0.292 -> **PASS** (CIs separated: True; permutation p = 0.0001)

Wall time: 2725s on 14 workers.
