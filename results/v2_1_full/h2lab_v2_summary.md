# Findings summary: v2 (progress-credited tree rollouts)

## 1. How we got here
- **Survey (mid-2025 to Nov 2026).** About 40 RL concepts reviewed; agentic step-level credit assignment was the clearest open gap.
- **Hybrid review.** Several cited foundations of the proposed hybrids were miscited or unverifiable (e.g. "Sony US 12,276,511 B2" is an Aptiv routing patent; SHARPO could not be found). Hybrid II (segment → state graph → Monte Carlo at the graph's frontier) was chosen and merged with H (exploration from restored states).
- **v1 result (full mode, 5 seeds).** Frontier MC improved credit on frontier steps (Spearman +0.17 vs GraphGPO) but not learning: at equal environment steps every arm that spent extra steps per iteration lost to plain GraphGPO (0.225), including frontier MC (0.168) and GraphGPO with 2× group size (0.178).
- **Diagnosis.** (1) On frontier steps GraphGPO's step advantage is zero by construction, leaving GRPO's whole-episode blame. (2) Re-linking frontier states through shortest paths discarded the success rates the probes measured: even MC at every state barely moved GraphGPO's gradient alignment (cosine 0.365 → 0.393), while the same probes through progress credit reached 0.547. (3) Probe steps were thrown away instead of trained on.
- **v2 design.** Branch the group into a tree from selected visited states, train on every branch, pool outcomes over merged states, and use progress credit V(s′) − V(s), which redistributes the GRPO advantage to the steps where success probability changed. Closest prior art: Tree-GRPO (random expansion, no cross-branch state merging), VinePPO (per-step MC values), PAV (progress as step reward), GraphGPO (merged state graph).


## 2. Kill test and branch diagnostics
- GraphGPO moving steps that start in a frontier state: **49%** on average; above 5% in **100%** of iterations → **PASS**.
- `Progress+Tree(random)`: 368 branches per iteration (63% of its env steps); 6% of candidate states skipped as absorbing, 1% as cached dead ends; branch success rate 26%.
- `Tree+GRPO`: 411 branches per iteration (64% of its env steps); 7% of candidate states skipped as absorbing, 1% as cached dead ends; branch success rate 16%.
- `Progress+Tree`: 335 branches per iteration (60% of its env steps); 5% of candidate states skipped as absorbing, 2% as cached dead ends; branch success rate 47%.
- `Progress+Tree(LOO)`: 340 branches per iteration (59% of its env steps); 5% of candidate states skipped as absorbing, 2% as cached dead ends; branch success rate 47%.
- `Tree+GraphGPO`: 294 branches per iteration (56% of its env steps); 5% of candidate states skipped as absorbing, 2% as cached dead ends; branch success rate 41%.

## 3. Main result (IQM eval success across seeds, 95% bootstrap CI)

**At equal environment steps (2,400,000), same minibatch size for every arm — primary test:**

| arm | IQM | CI95 | n |
|---|---|---|---|
| GRPO | 0.365 | (0.347, 0.395) | 5 |
| GraphGPO | 0.392 | (0.330, 0.432) | 5 |
| GraphGPO(2xG) | 0.374 | (0.342, 0.388) | 5 |
| GraphGPO+FrontierMC | 0.323 | (0.286, 0.380) | 5 |
| Tree+GRPO | 0.267 | (0.229, 0.328) | 5 |
| Tree+GraphGPO | 0.384 | (0.355, 0.423) | 5 |
| Progress+Tree | 0.502 | (0.480, 0.522) | 5 |
| Progress+Tree(LOO) | 0.505 | (0.470, 0.534) | 5 |
| Progress+Tree(random) | 0.411 | (0.384, 0.452) | 5 |

- `Progress+Tree` vs `GraphGPO`: +0.110 → **PASS**

- `Progress+Tree` vs `GraphGPO(2xG)`: +0.128 → **PASS**

- `Progress+Tree(LOO)` vs `GraphGPO`: +0.114 → **PASS**

- `Progress+Tree(LOO)` vs `GraphGPO(2xG)`: +0.131 → **PASS**


**At equal iterations (192)** (arms with more samples per iteration also take more optimizer steps here):

| arm | IQM |
|---|---|
| GRPO | 0.096 |
| GraphGPO | 0.230 |
| GraphGPO(2xG) | 0.300 |
| GraphGPO+FrontierMC | 0.228 |
| Tree+GRPO | 0.261 |
| Tree+GraphGPO | 0.330 |
| Progress+Tree | 0.467 |
| Progress+Tree(LOO) | 0.509 |
| Progress+Tree(random) | 0.407 |


**At equal total cost = env steps + per-iteration overhead:**

| arm | overhead=0 | overhead=1,000 | overhead=5,000 | overhead=20,000 |
|---|---|---|---|---|
| GRPO | 0.365 | 0.313 | 0.204 | 0.141 |
| GraphGPO | 0.392 | 0.369 | 0.312 | 0.262 |
| GraphGPO(2xG) | 0.374 | 0.348 | 0.330 | 0.325 |
| GraphGPO+FrontierMC | 0.323 | 0.318 | 0.285 | 0.247 |
| Tree+GRPO | 0.267 | 0.267 | 0.265 | 0.263 |
| Tree+GraphGPO | 0.384 | 0.397 | 0.371 | 0.364 |
| Progress+Tree | 0.502 | 0.487 | 0.479 | 0.478 |
| Progress+Tree(LOO) | 0.506 | 0.503 | 0.520 | 0.524 |
| Progress+Tree(random) | 0.411 | 0.410 | 0.414 | 0.410 |

## 4. Attribution

| effect (equal env steps) | comparison | delta | CIs_separated |
|---|---|---|---|
| Data effect (graph credit): tree vs flat G | Tree+GraphGPO − GraphGPO | -0.008 | no |
| Data effect (graph credit): tree vs flat 2G | Tree+GraphGPO − GraphGPO(2xG) | +0.010 | no |
| Credit effect on the tree: progress vs graph | Progress+Tree − Tree+GraphGPO | +0.118 | yes |
| Credit effect on the tree: progress vs outcome-only | Progress+Tree − Tree+GRPO | +0.235 | yes |
| Branch-point effect: smart vs random | Progress+Tree − Progress+Tree(random) | +0.091 | yes |
| Baseline effect: leave-one-out vs telescoping | Progress+Tree(LOO) − Progress+Tree | +0.003 | no |

## 5. Update-setting check (equal optimizer steps per iteration)
- Progress+Tree, equal env steps (2,400,036): 0.474 vs GraphGPO 0.373 → +0.101 → **PASS**
- Progress+Tree, equal iterations (225): 0.463 vs GraphGPO 0.211 → +0.251 → **PASS**
- Progress+Tree(LOO), equal env steps (2,400,036): 0.485 vs GraphGPO 0.373 → +0.112 → **PASS**
- Progress+Tree(LOO), equal iterations (225): 0.471 vs GraphGPO 0.211 → +0.260 → **PASS**

## 6. Credit accuracy (gradient alignment with ground truth)
Averaged over snapshots [60, 120, 200]:

| estimator | cosine all | cosine frontier | cosine linked | extra env steps / batch |
|---|---|---|---|---|
| GRPO | 0.179 | 0.103 | 0.193 | 0 |
| GraphGPO | 0.282 | 0.108 | 0.307 | 0 |
| GraphGPO+FrontierMC | 0.286 | 0.163 | 0.307 | 14,390 |
| GraphGPO+VineAll | 0.290 | 0.175 | 0.311 | 52,812 |
| Progress | 0.430 | 0.000 | 0.469 | 0 |
| Progress+VineAll | 0.559 | 0.533 | 0.623 | 52,813 |
| Tree+GRPO | 0.183 | 0.176 | 0.188 | 11,569 |
| Tree+GraphGPO | 0.298 | 0.264 | 0.313 | 11,578 |
| Progress+Tree | 0.465 | 0.271 | 0.526 | 11,560 |
| Progress+Tree(LOO) | 0.449 | 0.432 | 0.488 | 11,567 |
- Progress+Tree vs GraphGPO, all steps: **+0.183** cosine.
- Progress+Tree(LOO) vs GraphGPO, all steps: **+0.167** cosine.
- Same probes, different channel: Progress+VineAll 0.559 vs GraphGPO+VineAll 0.290.

## 7. Sparse-graph stress test (IQM success at 1,200,000 env steps)

| arm | noise=0.0 | noise=0.5 | noise=0.9 |
|---|---|---|---|
| GraphGPO | 0.286 | 0.235 | 0.154 |
| Progress+Tree | 0.359 | 0.050 | 0.005 |
| Progress+Tree(LOO) | 0.363 | 0.126 | 0.069 |
| Tree+GraphGPO | 0.259 | 0.183 | 0.085 |

- Progress+Tree − GraphGPO: noise=0.0: +0.073, noise=0.5: -0.186, noise=0.9: -0.149

- Progress+Tree(LOO) − GraphGPO: noise=0.0: +0.077, noise=0.5: -0.110, noise=0.9: -0.085

## Bottom line
- **`Progress+Tree` passes the pre-registered test at equal environment steps** (same minibatch size for all arms), against GraphGPO and the budget-matched GraphGPO(2xG).
- **`Progress+Tree(LOO)` passes the pre-registered test at equal environment steps** (same minibatch size for all arms), against GraphGPO and the budget-matched GraphGPO(2xG).
- **Minibatching matters (Progress+Tree).** With equal optimizer steps per iteration: +0.101 vs GraphGPO at equal env steps, +0.251 at equal iterations. In a real trainer, size minibatches in samples so branches add updates in proportion to the data.
- **Minibatching matters (Progress+Tree(LOO)).** With equal optimizer steps per iteration: +0.112 vs GraphGPO at equal env steps, +0.260 at equal iterations. In a real trainer, size minibatches in samples so branches add updates in proportion to the data.
- **Sparse graphs (Progress+Tree):** -0.149 vs GraphGPO at merge noise 0.9 — the gains depend on states merging; measure the merge rate on the target benchmark first.
- **Sparse graphs (Progress+Tree(LOO)):** -0.085 vs GraphGPO at merge noise 0.9 — the gains depend on states merging; measure the merge rate on the target benchmark first.
- **Credit quality:** Progress+Tree's advantages align better with ground truth than GraphGPO's (+0.183 cosine).
- **Next steps:**
  1. Confirm with `full` mode (5 seeds); GraphGPO's seed-to-seed spread is large, so 3 seeds can mislead.
  2. Measure how often states merge in the target environment (exactly, and with similarity-based merging such as GraphGPO's `enable_similarity`). v2's gains depend on it.
  3. Tune the branch budget and branch-point rule (section 6 ablations), especially for sparse graphs, where each branch informs only its own branch point.
  4. Port to a real agent benchmark (section 8): Tree-GRPO or verl-agent GraphGPO on ALFWorld/WebShop, minibatches sized in samples, compared at equal GPU-hours against Tree-GRPO and budget-matched GraphGPO.
- **Caveat:** a small grid-world with exact state restoration and cheap environment steps. Real agents merge fewer states, restore states by replaying action prefixes, and pay for every branch in LLM tokens.