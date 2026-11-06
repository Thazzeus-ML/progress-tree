# Main comparison: 6 arms x 10 seeds, 2,400,000 env steps, merge noise 0.0, task seeds 3000/4000, update mode fixed_minibatch

## At equal environment steps (2,400,000)

| arm | IQM | mean | CI95 | n |
|---|---|---|---|---|
| GraphGPO@gamma=0.5 | 0.197 | 0.208 | (0.174, 0.237) | 10 |
| GraphGPO@gamma=0.8 | 0.182 | 0.182 | (0.162, 0.205) | 10 |
| GiGPO | 0.192 | 0.195 | (0.178, 0.209) | 10 |
| Tree+GraphGPO@branch_mode=consistent | 0.374 | 0.364 | (0.314, 0.410) | 10 |
| Tree+GiGPO@branch_mode=consistent | 0.342 | 0.337 | (0.275, 0.389) | 10 |
| Progress+Tree(LOO)@branch_mode=consistent | 0.461 | 0.465 | (0.438, 0.495) | 10 |


## At equal iterations (161)

| arm | IQM | mean | CI95 | n |
|---|---|---|---|---|
| GraphGPO@gamma=0.5 | 0.135 | 0.138 | (0.123, 0.156) | 10 |
| GraphGPO@gamma=0.8 | 0.145 | 0.141 | (0.130, 0.155) | 10 |
| GiGPO | 0.100 | 0.099 | (0.092, 0.106) | 10 |
| Tree+GraphGPO@branch_mode=consistent | 0.313 | 0.309 | (0.271, 0.346) | 10 |
| Tree+GiGPO@branch_mode=consistent | 0.319 | 0.315 | (0.263, 0.367) | 10 |
| Progress+Tree(LOO)@branch_mode=consistent | 0.459 | 0.456 | (0.435, 0.480) | 10 |


Wall time: 1561s on 14 workers.
