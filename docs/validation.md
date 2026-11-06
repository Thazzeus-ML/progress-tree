# Validation report

This report covers everything done to test the toy results before publishing. It has four parts: an independent re-run, an adversarial code audit, fixes, and the experiments that settle the audit's open questions.

All numbers are eval success on 128 held-out KeyDoor maps × 4 rollouts, read at an equal environment-step budget, with branch steps counted. Comparisons are **paired by seed**: arms with the same seed share initial weights and the task sequence. Each comparison reports the mean paired difference with a t-based 95% CI and the number of seeds won.

Raw logs are in `results/*/runs.csv`. `scripts/analyze.py` reproduces every table in this report.

## 1. Re-running the claim on fresh task pools

The author's v2.1 run used the same maps the method was developed on: 5 seeds, train/eval pools 1000/2000. Its data is archived in `results/v2_1_full/`. The pre-registered `replicate` check, 10 seeds on maps never used before, had not been run. It was run here, plus a second replicate on another fresh pool with GRPO added. Both used the toy exactly as shipped.

| run | maps | Progress+Tree(LOO) | Progress+Tree | Tree+GraphGPO | GraphGPO | GraphGPO(2×G) | GRPO |
|---|---|---|---|---|---|---|---|
| v2.1 (author, 5 seeds, IQM) | 1000/2000 | 0.505 | 0.502 | 0.384 | 0.392 | 0.374 | 0.365 |
| `results/replicate_a` (10 seeds, IQM) | 3000/4000 | **0.511** | 0.489 | 0.361 | 0.256 | 0.216 | — |
| `results/replicate_b` (10 seeds, IQM) | 5000/6000 | **0.464** | 0.426 | 0.343 | 0.240 | 0.270 | 0.260 |

Paired against GraphGPO, Progress+Tree(LOO) is ahead by:

- +0.237 on replicate A, CI [+0.18, +0.30], 10/10 seeds;
- +0.211 on replicate B, CI [+0.16, +0.26], 10/10 seeds;
- +0.206 against GRPO, CI [+0.15, +0.26], 10/10 seeds.

The headline replicates, with larger margins than the original run.

## 2. Audit

The code went through an adversarial review: six independent reviewers, each finding challenged by one or two skeptics. The full report is in [`audit.md`](audit.md). No finding reversed the headline. The issues that mattered:

| # | issue | severity | resolution |
|---|---|---|---|
| 1 | Toy branches restart the episode clock at t=0 with a 20-step cap. The verl-agent port runs a branch for the remaining `horizon - t` steps. | major | Added `branch_mode='consistent'` (restore the clock, run to the episode's own horizon) and re-ran (§3). |
| 2 | Restoring a branch state is free in the toy; the port replays the parent's actions. | minor | Replay steps are now logged (`replay_steps`). Results are reported with replay charged at w ∈ {0, 0.25, 1} (§3). |
| 3 | GraphGPO might be under-tuned: γ=0.1, and γ^d underflows the normalizer's eps. GiGPO, the closest relative, was missing. | major (fairness) | Added a γ sweep, an underflow-free `graph_relative` variant and a `GiGPO` arm (§3). |
| 4 | "Smart branch points +0.09 over random" includes the random arm wasting branches on frozen trap states. | major (attribution) | Added `random_filtered` and a separate branch-point RNG so task sequences stay paired (§4). |
| 5 | Under merge noise, salted keys make trapped agents look like they move, so smart selection branches from them. | major (stress test) | Added `absorb_phys=True` (decide "absorbing" on the physical state) and re-ran the crossover with 10 seeds (§5). |
| 6 | Port: the wall-clock guard timed only the rollout and could run past the budget; `compare_runs.py` scored slower arms on older checkpoints. | major (GPU protocol) | Fixed: full-step timing, a 3× guard that provably stops within budget, and interpolated scoring (`recipe/ProgressTree/tree_rollout_loop.py`, `compare_runs.py`). |
| 7 | "Branching alone adds ~0" is an equivalence claim 5 seeds cannot support. | major (attribution) | Re-measured with paired CIs (§3). |
| 8 | The credit-study cosine gap (0.47 vs 0.28) gives the tree estimators extra data, and GraphGPO's underflow lowers its cosine. | minor | Not re-run. The audit's estimate: the gap shrinks by roughly half, and the direction holds. |

All new options default to off. With defaults, the library reproduces the original code **bit for bit** on every arm, with and without merge noise.

## 3. Port-faithful branches, stronger baselines, charged restore (10 seeds, 2.4M steps)

`results/e1e5_pool1000` uses the original maps. `results/e1e5_pool3000` uses replicate A's maps and reuses that replicate's runs of the toy-default arms. "consistent" means `branch_mode='consistent'`.

| arm | maps 1000/2000 (mean) | maps 3000/4000 (mean) |
|---|---|---|
| Progress+Tree(LOO), toy branches (v2) | 0.535 | 0.510 |
| **Progress+Tree(LOO), consistent** | **0.490** | **0.465** |
| Tree+GiGPO, consistent | 0.384 | 0.337 |
| Tree+GraphGPO, consistent | 0.365 | 0.364 |
| GiGPO | 0.404 | 0.195 |
| GraphGPO (γ=0.1, upstream default) | 0.352 | 0.274 |
| GraphGPO γ=0.5 | 0.233 | 0.208 |
| GraphGPO γ=0.8 | 0.239 | 0.182 |
| GraphGPO, relative distances (no underflow) | 0.270 | — |

**Paired differences** (Progress+Tree(LOO), consistent, minus the baseline), shown as diff [95% CI], seeds won:

| baseline | maps 1000 | maps 3000 |
|---|---|---|
| strongest flat baseline (GiGPO on 1000, GraphGPO on 3000) | **+0.085** [+0.044, +0.126], 9/10 | **+0.192** [+0.111, +0.273], 9/10 |
| same tree, GiGPO credit | +0.106 [+0.076, +0.136], 10/10 | +0.129 [+0.050, +0.208], 9/10 |
| same tree, GraphGPO credit | +0.125 [+0.096, +0.154], 10/10 | +0.101 [+0.032, +0.171], 9/10 |
| replay charged 1:1, vs GiGPO | +0.041 [−0.004, +0.086], 8/10 | +0.241 [+0.215, +0.266], 10/10 |
| replay charged 1:1, vs GraphGPO | +0.094 [+0.037, +0.150], 10/10 | +0.162 [+0.091, +0.233], 9/10 |

What this shows:

- **The toy's branch shortcut inflated the result by 0.045.** The gap is consistent across pools (CI [+0.02, +0.07], 10/10 seeds on both). The method still beats every baseline with port-faithful branches.
- **GraphGPO was not handicapped by its γ.** Larger γ and the underflow-free variant are both worse. GiGPO is the stronger baseline on one pool and the weaker on the other, so baseline rankings vary from pool to pool. Progress+Tree(LOO) beats all of them on both pools.
- **Charging replay narrows the lead.** At the worst-case 1:1 charge, the margin over GiGPO on the original maps is +0.04, and its t-CI includes 0. The port budgets in GPU-hours, where replay costs no LLM generation, so the w=0 to 0.25 rows are closer to that setting.
- **Attribution.** Progress credit beats graph and GiGPO credit on the same tree by +0.10 to +0.13 in every comparison: the robust part of the gain. The tree's effect on its own depends on the pool and the credit:

  | tree vs flat | maps 1000 | maps 3000 | earlier replicates |
  |---|---|---|---|
  | Tree+GraphGPO vs GraphGPO | +0.01 | +0.09 | +0.07, +0.08 |
  | Tree+GiGPO vs GiGPO | −0.02 | +0.14 | — |

  "Branching alone adds nothing" is not supported in general. It holds on the original maps only.

## 4. Branch-point rule (`results/e2_branch_rule`, maps 1000, consistent branches, separate branch RNG)

| comparison | diff [95% CI] | seeds won | branch success |
|---|---|---|---|
| smart vs random (Tree-GRPO-style uniform) | +0.048 [+0.005, +0.091] | 8/10 | 49% vs 27% |
| smart vs random_filtered (same filters, no ranking) | +0.030 [+0.005, +0.055] | 9/10 | 49% vs 42% |
| random_filtered vs random (filters only) | +0.018 [−0.025, +0.061] | 4/10 | 42% vs 27% |

Smart selection helps, but by +0.03 to +0.05, not the reported +0.09. Most of the branch-success gap (47% vs 26% in v2.1) comes from not branching out of absorbing states.

## 5. Merge-rate crossover (`results/e3_merge_noise`, maps 1000, 10 seeds, 1.2M steps)

These runs use consistent branches and `absorb_phys=True`. Noise 0 is read from the 2.4M runs at 1.2M steps.

| merge noise | shared-step fraction | Progress+Tree(LOO) − GraphGPO | seeds won |
|---|---|---|---|
| 0 | 0.88 | **+0.061** [+0.019, +0.103] | 8/10 |
| 0.05 | 0.83 | +0.008 [−0.020, +0.036] | 5/10 |
| 0.1 | 0.79 | −0.000 [−0.037, +0.037] | 7/10 |
| 0.15 | 0.74 | −0.031 [−0.055, −0.007] | 1/10 |
| 0.2 | 0.69 | −0.059 [−0.084, −0.033] | 0/10 |
| 0.3 | 0.59 | −0.080 [−0.098, −0.062] | 0/10 |

The original limitation claim holds, and is if anything slightly tighter. The method wins only when about 85% or more of steps are on shared states, ties at 0.79–0.83, and loses at 0.74 and below.

The ALFWorld merge probes (`results/port/`) put real agents at:

- 0.68–0.79 for weak-to-good policies, with similarity merging;
- 0.93 for a strong policy.

So the toy predicts GraphGPO ahead for most of an ALFWorld run. The toy's merge noise is uniform; ALFWorld's non-merging is structured, so this prediction is uncertain.

## 6. Estimator checks (`tests/test_estimators.py`)

- On a 4-state MDP with known advantages, progress credit with leave-one-out baselines converges to the exact Q(s,a) − V(s) at every state-action pair, within 0.01 over 96k rollouts.
- The plain version is shrunk toward zero at decisive states, as theory predicts.
- With no merging, LOO progress credit reduces exactly to RLOO on every step.
- The verl-agent port's estimator equals the toy's, and branch steps are charged exactly.

The new options have tests too:

- consistent branches respect the episode clock;
- the absorbing fix skips trapped states;
- the filtered random rule;
- relative distances leave GraphGPO's advantages unchanged;
- GiGPO's step term matches a hand computation.

## 7. Not validated

- **No LLM or GPU training run.** The LLM track (`toy/h2lab_llm.py`) and the ALFWorld recipe have only been exercised on CPU with stand-in models.
- **No comparison with the closest published methods**: G2PO, PGPO, GRAFT, TSBPO and BPO (see [`prior_work.md`](prior_work.md)).
- **The credit-alignment study was not re-run.** Its headline gap is likely overstated by about half (audit issue 8).
- **The merge-noise crossover was measured at 1.2M steps only.**
