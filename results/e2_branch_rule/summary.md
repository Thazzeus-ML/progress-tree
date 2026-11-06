# Main comparison: 2 arms x 10 seeds, 2,400,000 env steps, merge noise 0.0, task seeds 1000/2000, update mode fixed_minibatch

## At equal environment steps (2,400,000)

| arm | IQM | mean | CI95 | n |
|---|---|---|---|---|
| Progress+Tree(LOO,random)@branch_mode=consistent,branch_rng=separate | 0.452 | 0.442 | (0.421, 0.467) | 10 |
| Progress+Tree(LOO,random_filtered)@branch_mode=consistent,branch_rng=separate | 0.461 | 0.460 | (0.442, 0.474) | 10 |


## At equal iterations (198)

| arm | IQM | mean | CI95 | n |
|---|---|---|---|---|
| Progress+Tree(LOO,random)@branch_mode=consistent,branch_rng=separate | 0.446 | 0.441 | (0.431, 0.454) | 10 |
| Progress+Tree(LOO,random_filtered)@branch_mode=consistent,branch_rng=separate | 0.473 | 0.475 | (0.465, 0.486) | 10 |


Wall time: 557s on 14 workers.
