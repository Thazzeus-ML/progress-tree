# Progress+Tree for verl-agent (ALFWorld)

A drop-in recipe for [verl-agent](https://github.com/langfengQ/verl-agent) that adds the h2lab v2
concept — **progress credit with leave-one-out baselines + tree rollouts branched from visited
states** — next to the upstream GraphGPO recipe, so the arms can be compared at equal GPU-hours.
Upstream files are not edited: `main_progress_tree.py` patches the GraphGPO modules inside the
Ray task-runner process before the upstream trainer runs.

## Arms (`algorithm.progress_tree.arm`)

| arm | rollouts | credit |
|---|---|---|
| `graphgpo` | G per task (upstream) | GraphGPO (upstream) |
| `progress` | G per task | progress credit, leave-one-out baselines (drop-in credit change) |
| `progress_tree` | G per task + branches from the most-visited states | progress credit (LOO) — **the concept** |
| `progress_tree_spread` | + branches spread evenly along each trajectory | progress credit (LOO) |
| `tree_grpo` | + branches from random (trajectory, step) positions | Tree-GRPO-style: intra-tree + inter-tree z-scores (proxy, see below) |
| `tree_graphgpo` | same branches as `progress_tree` | GraphGPO — attribution: tree vs credit |

Budget-matched GraphGPO is `graphgpo` with a larger group (`GROUP=12 bash run_alfworld_1gpu.sh graphgpo ...`);
under the equal-GPU-hours protocol every arm already gets the same compute, so this is optional.

## What was verified (CPU, no GPU available where this was written)

* `tests/test_core_progress.py` — the port's estimator reproduces the toy library's
  `progress_loo_advantages`/`progress_advantages` **exactly** (max |diff| 0) on 24 toy batches
  (merge noise 0/0.5/0.9, with and without branches, ~40% of rows carrying non-zero credit);
  progress credit telescopes to R − V(s₀); branch-point selection is identical to the toy's;
  the similarity canonicaliser forms the same clusters as GraphGPO's `StateCanonicalizer`.
* `tests/test_tree_collector_alfworld.py` — real ALFWorld through verl-agent's own collector,
  env manager, Ray workers, reward manager and invalid-action penalty, with a mock LLM:
  main + branch rows concatenate into one `DataProto`; **every restore matched the logged state
  and every branch's first prompt was token-for-token identical to its parent's prompt at the
  branch step**; after branching, groups still play one game each; the invalid-action penalty
  enters exactly as in GraphGPO's episode-level term (checked against `episode_norm_reward`).
* Entry point: the Hydra config composes on top of verl's `ppo_trainer`, and all patches apply.
* State restore by action replay was also checked on 96 random mid-episode states (4 policies × 24): 100% match.
* The logged merge statistic equals the toy's `shared_step_frac` on toy batches.

**Not verified here:** a GPU training run. Run `preflight_gpu.py` and the smoke run below first.

## ALFWorld merge rates, and what the toy says about them

`tools/alfworld_merge_probe.py`, 16 games × group of 8, GraphGPO's anchor strings
(observation + location/holding + moved items + sorted admissible commands). The statistic is the
share of steps (t ≥ 1) on a state another rollout of the same group also visited; the toy records
the same statistic for every batch (`shared_step_frac`), so ALFWorld can be placed on the toy's
merge-noise scale (pooled calibration curve from 106 toy runs):

| policy | success | shared steps: exact | ≈ toy noise | with similarity merging (0.95) | ≈ toy noise |
|---|---|---|---|---|---|
| uniform random | 0.00 | 0.44 | 0.46 | 0.68 | 0.21 |
| 50% expert | 0.61 | 0.59 | 0.29 | 0.77 | 0.11 |
| 80% expert | 0.75 | 0.71 | 0.15 | 0.79 | 0.09 |
| 95% expert | 0.83 | 0.91 | 0.00 | 0.93 | 0.00 |

And the toy's verdict by noise level (IQM eval success at 1.2M env steps, 3 seeds; the v2.1 `full`
run at 2.4M had Progress+Tree(LOO) 0.51 vs GraphGPO 0.39 at noise 0):

| merge noise (shared steps) | 0 (≈0.85) | 0.15 (≈0.71) | 0.3 (≈0.58) | 0.5 (≈0.39) | 0.9 (≈0.06) |
|---|---|---|---|---|---|
| GraphGPO | 0.308 | 0.294 | 0.297 | 0.236 | 0.161 |
| Progress+Tree(LOO), 32 branch points/group | **0.351** | 0.234 | 0.201 | 0.128 | 0.067 |
| same, 8 branch points/group | 0.227 | 0.180 | 0.153 | 0.114 | 0.091 |
| Progress(LOO), no tree | 0.133 | 0.115 | 0.107 | 0.101 | 0.137 |

> **Update (10-seed validation, `docs/validation.md` at the repo root).** With branches that run to the
> episode's own horizon (as this recipe does), 10 seeds per arm at 1.2M steps: Progress+Tree(LOO) beats
> GraphGPO at 0.88 shared steps (+0.061, CI [+0.019, +0.103]), ties at 0.79-0.83, and loses at 0.74 and
> below (-0.031 at 0.74, -0.080 at 0.59). The tables in this section are the earlier 3-seed pilot. The toy
> numbers here also used a 20-step branch cap that inflated Progress+Tree by about 0.045.

**Read this before spending GPU-hours.** In the toy, v2 wins only when states merge almost as
often as they can (≈ 0.85 of steps shared); at 0.71 it is already behind GraphGPO, which is
remarkably robust to partial merging. Sparse-graph remedies tested in the toy (branch points spread
along trajectories, a learned critic for unmerged states, smaller or K=4 branch budgets) did not
close the gap. ALFWorld with similarity merging sits at 0.68–0.79 for weak-to-good policies and
reaches ≈ 0.93 only for a strong one, so **the toy predicts GraphGPO ahead for most of training**.
The toy's noise is uniform random while ALFWorld's non-merging is structured (rollouts diverge and
rarely re-merge), and an LLM policy is less random than the uniform policy probed here, so the
prediction can be wrong — but test it cheaply first (below). Every run logs the live statistic
(`main_shared_step_frac`, groups of 8), and `compare_runs.py` reports it.

## Setup (Colab G4 / A100 / H100, one GPU)

```bash
bash ProgressTree/setup_colab.sh ProgressTree      # clones verl-agent, installs, copies this recipe, preflight
```

Colab's G4 is an RTX PRO 6000 Blackwell (sm_120, 96 GB; GCP g4-standard-48 ≈ 48 vCPUs, 180 GB RAM).
verl's actor hard-codes `flash_attention_2`; if the preflight reports that flash-attn kernels do
not run on the GPU, run with `ATTN=sdpa` (the setup script lets verl read `VERL_ATTN_IMPL`; this
also switches off `use_remove_padding`). If vLLM fails, try `VLLM_ATTENTION_BACKEND=FLASHINFER`
or `TRITON_ATTN`; if it still fails, use an A100/H100 runtime, where verl-agent's pinned stack is
known to work.

Smoke run (≈15–30 min; checks the whole loop, not a result):
```bash
cd ~/verl-agent
MODEL=Qwen/Qwen2.5-0.5B-Instruct TRAIN_TASKS=2 GROUP=4 VAL_TASKS=8 MAX_STEPS=20 \
  bash recipe/ProgressTree/run_alfworld_1gpu.sh progress_tree 0 0.25
```
Check `~/progress_tree_runs/progress_tree_s0/progress_tree_log.csv`: `restore_mismatch` must be 0,
`branch_points` > 0, and `rollout_seconds` tells you the cost per iteration.

## Protocol: pilot first, then equal GPU-hours

Pilot (6 GPU-h): one seed each of `graphgpo` and `progress_tree` for 3 h with similarity merging.
Go on to the full comparison if `main_shared_step_frac` is ≳ 0.85 or `progress_tree` is ahead at
3 h; otherwise the toy's prediction holds and the next step is a merge-robust estimator, not more seeds.

```bash
for ARM in graphgpo progress_tree; do
  bash recipe/ProgressTree/run_alfworld_1gpu.sh $ARM 0 3 algorithm.graphgpo.enable_similarity=True
done
python recipe/ProgressTree/compare_runs.py ~/progress_tree_runs --baseline graphgpo
```

Full comparison:

```bash
for SEED in 0 1 2; do for ARM in graphgpo progress_tree tree_grpo; do
  bash recipe/ProgressTree/run_alfworld_1gpu.sh $ARM $SEED 6 algorithm.graphgpo.enable_similarity=True
done; done
python recipe/ProgressTree/compare_runs.py ~/progress_tree_runs --baseline graphgpo
```

* Each run trains until its wall-clock budget (`max_hours`, validation time excluded), then does a
  final validation. The same `SEED` gives every arm the same training-game sequence.
* `compare_runs.py` reads each run's validation success at the common budget (last validation at or
  before it) and the area under the success-vs-hours curve, reports IQM over seeds and the paired
  difference to the baseline with a bootstrap CI.
* Cost: 3 arms × 3 seeds × 6 h = 54 GPU-hours. Run `graphgpo` vs `progress_tree` first (36 GPU-h);
  add `tree_grpo` and `progress` if the first comparison is promising. With 3 seeds per arm the
  result is still a pilot: RL run-to-run spread is large (in the toy, 3-seed verdicts were unreliable),
  and 6 h of LLM training covers early training only.
* Branch budget: `branch_max_nodes=12`, `branch_K=1` by default (up to 12 branches per task group of 8
  per iteration, each running to the end of the episode; estimated at 55–60% of environment steps on
  branches early in training — check `branch_steps / (main_steps + branch_steps)` in the log). In the
  toy the winning setting spent 61% of env steps on branches; with 33% (8 branch points per group)
  it lost to GraphGPO even without merge noise (0.227 vs 0.308).

## How it works

* **Restore by replay.** ALFWorld (TextWorld, `domain_randomization: False`) is deterministic: a
  branch replays its parent's first *t* environment actions on a private single-game env inside
  the same Ray worker, leaving the worker's main env (and the group-seeded game sequence) untouched.
  The env manager rebuilds the prompt state (history memory, task text, last observation) from a
  snapshot taken after the main rollouts. Every restore is checked against the logged anchor;
  mismatches are dropped and counted (`restore_mismatch`).
* **Waves.** Branches run on the same workers after the main rollouts, in waves of at most
  `G × tasks` branches; only active rows are tokenised and generated.
* **Rows.** Branch rows keep the task's `uid`, get a new `traj_uid`, and carry `step_idx` (absolute
  step), `is_branch`, `parent_traj_uid`, `branch_step`. Their episode reward is the branch's own outcome.
* **Credit** (`core_progress.py`): A_t = Q_t − b_t, Q_t = pooled first-visit value of the next
  state (the trajectory's own outcome on its last step), b_t = leave-one-out value of the current
  state (falling back to the last available baseline), divided by the group's outcome std (floor
  `sigma_min × reward_scale`). The invalid-action penalty is added exactly as it enters GraphGPO's
  episode-level term, so the format signal is the same across arms.
* **Tree-GRPO proxy.** Ji et al. (ICLR 2026) expand trees at random agent-step nodes and score each
  leaf by intra-tree + inter-tree group-relative advantages. Here: one expansion round from the G
  main chains, random (trajectory, step) branch points with the same budget as `progress_tree`,
  and each step row gets the sum of the advantages of all leaves whose path contains it (what
  duplicating shared prefixes per leaf gives under a token-mean loss). It is a proxy, not their code.

## Limits

* Branching needs a restorable environment; it is implemented for ALFWorld. `progress` (no tree)
  works for any verl-agent environment. WebShop could use the same replay mechanism (its sessions
  are seeded) but is not implemented or tested.
* Dynamic sampling (`algorithm.filter_groups.enable`) is not supported with these arms.
* Resuming from a checkpoint restarts the wall-clock accounting.

## Files

`core_progress.py` estimators and branch selection · `branch_envs.py` restorable workers/env
manager · `tree_rollout_loop.py` tree collector, CSV log, budget guard · `main_progress_tree.py`
entry point and patches · `config/progress_tree_trainer.yaml` · `run_alfworld_1gpu.sh` ·
`setup_colab.sh` · `preflight_gpu.py` · `compare_runs.py` · `tests/` (CPU tests; `toy_core.py` is
the h2lab v2.2 toy library they check against) · `tools/alfworld_merge_probe.py` ·
`results/` (merge-probe JSON and the toy sweep's per-run CSV behind the tables above).
