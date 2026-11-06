# Main comparison: 5 arms x 10 seeds, 2,400,000 env steps, merge noise 0.0, task seeds 3000/4000, update mode fixed_minibatch

## At equal environment steps (2,400,000)

| arm | IQM | mean | CI95 | n |
|---|---|---|---|---|
| GraphGPO | 0.256 | 0.274 | (0.211, 0.331) | 10 |
| GraphGPO(2xG) | 0.216 | 0.232 | (0.190, 0.265) | 10 |
| Tree+GraphGPO | 0.361 | 0.343 | (0.294, 0.396) | 10 |
| Progress+Tree | 0.489 | 0.486 | (0.461, 0.516) | 10 |
| Progress+Tree(LOO) | 0.511 | 0.510 | (0.491, 0.529) | 10 |

- `Progress+Tree` vs `GraphGPO`: +0.233 -> **PASS** (CIs separated: True; permutation p = 0.0001)
- `Progress+Tree` vs `GraphGPO(2xG)`: +0.274 -> **PASS** (CIs separated: True; permutation p = 0.0001)
- `Progress+Tree(LOO)` vs `GraphGPO`: +0.255 -> **PASS** (CIs separated: True; permutation p = 0.0001)
- `Progress+Tree(LOO)` vs `GraphGPO(2xG)`: +0.295 -> **PASS** (CIs separated: True; permutation p = 0.0001)

## At equal iterations (224)

| arm | IQM | mean | CI95 | n |
|---|---|---|---|---|
| GraphGPO | 0.174 | 0.179 | (0.151, 0.205) | 10 |
| GraphGPO(2xG) | 0.201 | 0.212 | (0.181, 0.237) | 10 |
| Tree+GraphGPO | 0.337 | 0.324 | (0.283, 0.365) | 10 |
| Progress+Tree | 0.477 | 0.470 | (0.443, 0.499) | 10 |
| Progress+Tree(LOO) | 0.509 | 0.508 | (0.491, 0.524) | 10 |

- `Progress+Tree` vs `GraphGPO`: +0.303 -> **PASS** (CIs separated: True; permutation p = 0.0001)
- `Progress+Tree` vs `GraphGPO(2xG)`: +0.276 -> **PASS** (CIs separated: True; permutation p = 0.0001)
- `Progress+Tree(LOO)` vs `GraphGPO`: +0.335 -> **PASS** (CIs separated: True; permutation p = 0.0001)
- `Progress+Tree(LOO)` vs `GraphGPO(2xG)`: +0.309 -> **PASS** (CIs separated: True; permutation p = 0.0001)

Wall time: 1923s on 14 workers.
