# Main comparison: 10 arms x 10 seeds, 1,200,000 env steps, merge noise 0.0, task seeds 1000/2000, update mode fixed_minibatch

## At equal environment steps (1,200,000)

| arm | IQM | mean | CI95 | n |
|---|---|---|---|---|
| GraphGPO@merge_noise=0.05 | 0.272 | 0.270 | (0.256, 0.287) | 10 |
| Progress+Tree(LOO)@branch_mode=consistent,absorb_phys=True,merge_noise=0.05 | 0.277 | 0.278 | (0.257, 0.298) | 10 |
| GraphGPO@merge_noise=0.1 | 0.276 | 0.275 | (0.248, 0.302) | 10 |
| Progress+Tree(LOO)@branch_mode=consistent,absorb_phys=True,merge_noise=0.1 | 0.280 | 0.274 | (0.252, 0.297) | 10 |
| GraphGPO@merge_noise=0.15 | 0.287 | 0.284 | (0.267, 0.301) | 10 |
| Progress+Tree(LOO)@branch_mode=consistent,absorb_phys=True,merge_noise=0.15 | 0.254 | 0.253 | (0.241, 0.267) | 10 |
| GraphGPO@merge_noise=0.2 | 0.301 | 0.303 | (0.286, 0.317) | 10 |
| Progress+Tree(LOO)@branch_mode=consistent,absorb_phys=True,merge_noise=0.2 | 0.243 | 0.244 | (0.230, 0.259) | 10 |
| GraphGPO@merge_noise=0.3 | 0.284 | 0.281 | (0.268, 0.293) | 10 |
| Progress+Tree(LOO)@branch_mode=consistent,absorb_phys=True,merge_noise=0.3 | 0.203 | 0.201 | (0.191, 0.211) | 10 |


## At equal iterations (65)

| arm | IQM | mean | CI95 | n |
|---|---|---|---|---|
| GraphGPO@merge_noise=0.05 | 0.072 | 0.073 | (0.063, 0.081) | 10 |
| Progress+Tree(LOO)@branch_mode=consistent,absorb_phys=True,merge_noise=0.05 | 0.257 | 0.258 | (0.244, 0.273) | 10 |
| GraphGPO@merge_noise=0.1 | 0.071 | 0.071 | (0.061, 0.082) | 10 |
| Progress+Tree(LOO)@branch_mode=consistent,absorb_phys=True,merge_noise=0.1 | 0.257 | 0.254 | (0.236, 0.275) | 10 |
| GraphGPO@merge_noise=0.15 | 0.069 | 0.070 | (0.059, 0.082) | 10 |
| Progress+Tree(LOO)@branch_mode=consistent,absorb_phys=True,merge_noise=0.15 | 0.246 | 0.247 | (0.233, 0.259) | 10 |
| GraphGPO@merge_noise=0.2 | 0.072 | 0.072 | (0.064, 0.080) | 10 |
| Progress+Tree(LOO)@branch_mode=consistent,absorb_phys=True,merge_noise=0.2 | 0.237 | 0.236 | (0.224, 0.248) | 10 |
| GraphGPO@merge_noise=0.3 | 0.060 | 0.060 | (0.054, 0.066) | 10 |
| Progress+Tree(LOO)@branch_mode=consistent,absorb_phys=True,merge_noise=0.3 | 0.203 | 0.202 | (0.192, 0.211) | 10 |


Wall time: 1289s on 14 workers.
