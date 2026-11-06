# Progress+Tree

**Step-level credit for multi-turn agent RL. Progress credit computes each step's advantage as the change in estimated success probability on a pooled state graph, and tree branching grows that graph where outcomes are uncertain.**

Progress+Tree is a GRPO-family credit-assignment method for agents with sparse success rewards. It joins two published lines of work:

- progress credit on a graph of merged states, `A_t = V(s_{t+1}) − V(s_t)` (cf. G2PO, PGPO, GRAFT);
- tree rollouts branched from revisited intermediate states (cf. TSBPO, BPO, TriGRPO).

To our knowledge, what it adds is using branching to *create* the shared states that progress credit depends on. Branch points are the graph's most-visited states that are not absorbing and not cached dead ends (states whose pooled success rate has stayed at zero), every branch is pooled into one graph, and a per-state leave-one-out baseline keeps the step advantages unbiased. See [`docs/prior_work.md`](docs/prior_work.md) for the component-by-component comparison.

> **Status.** The method is validated in a toy environment only. It has not been trained with an LLM or on a GPU. The verl-agent/ALFWorld recipe is CPU-tested and ready for a pilot run.

## The method

For each task, a group of G rollouts is collected. Then:

1. **Merge states.** Identical states (exactly, or by similarity in text environments) across all rollouts of the group become one node. Each node's value is its first-visit Monte Carlo success rate: `V(s)` = the fraction of the group's rollouts through `s` that succeeded.
2. **Branch.** From selected visited states, the episode is restored (in the port, by replaying the parent's actions) and continued with the current policy. The selected states are the most-visited non-absorbing states, minus cached dead ends. Every branch is training data and is pooled into the same graph.
3. **Credit.** Step `t` of trajectory `i` gets `A_t = (Q_t − b_t) / σ_group`, where:
   - `Q_t = V(s_{t+1})`, or the trajectory's own outcome on its last step;
   - `b_t = V(s_t)` estimated **without trajectory i** (the leave-one-out, LOO, variant), falling back to the last available baseline when no other rollout visited `s_t`.

   With no merging this reduces exactly to RLOO on every step. With merging, credit concentrates on the steps where success probability actually changed.
4. **Update.** PPO-clip on all main and branch steps.

## Results (toy: KeyDoor-Traps 9×9, MLP policy, 10 seeds per arm)

All arms train to the same environment-step budget (2.4M), with branch steps counted. Arms are compared seed by seed on 128 held-out maps. The full tables, CIs and raw logs are in [`docs/validation.md`](docs/validation.md).

| | original maps | fresh maps A | fresh maps B |
|---|---|---|---|
| **Progress+Tree(LOO)**, port-faithful branches | **0.490** | **0.465** | — |
| Progress+Tree(LOO), toy branches (v2 design) | 0.535 | 0.510 | 0.461 |
| GiGPO | 0.404 | 0.195 | — |
| GraphGPO | 0.352 | 0.274 | 0.251 |
| GRPO | — | — | 0.255 |
| Tree + GiGPO credit (same branches) | 0.384 | 0.337 | — |
| Tree + GraphGPO credit (same branches) | 0.365 | 0.364 | 0.333† |

*Mean final eval success. Port-faithful branches run to the episode's own horizon, as in the verl-agent port. The v2 toy restarted each branch with a fresh 20-step clock, which inflated its numbers by 0.045. † toy branches.*

### What holds up

- **It beats every baseline we tried** when states merge, by +0.085 to +0.19 over the strongest flat baseline (9/10 seeds on each map pool).
- **The baselines weren't left weak.** We also tried GraphGPO γ ∈ {0.1, 0.5, 0.8}, an underflow-free GraphGPO, and GiGPO.
- **Progress credit is the robust part of the gain.** It beats GraphGPO and GiGPO credit *on the same tree* by +0.10 to +0.13 with port-faithful branches (+0.10 to +0.17 with the toy's branches) in every comparison.
- **The estimator converges to the true advantage.** On an MDP with known advantages, LOO progress credit converges to the exact Q(s,a) − V(s) (`tests/test_estimators.py`).

### What doesn't, or needs qualifying

- **It only wins when states merge heavily.** Against GraphGPO it wins when about 85% or more of steps land on states other rollouts also visited. It ties at 79–83% and loses at 74% and below (10-seed merge-noise sweep).
  - ALFWorld probes measure 68–79% for weak-to-good policies and 93% for a strong one. The toy therefore predicts GraphGPO ahead for most of an ALFWorld run, though ALFWorld's non-merging is structured and the toy's noise is uniform.
- **Restoring states costs something.** If replaying the parent's prefix is charged as environment steps (worst case), the lead over GiGPO on the original maps drops to +0.04 (CI includes 0). Under a GPU-hours budget, replay is cheap: no generation.
- **Branching on its own** adds anywhere from −0.02 to +0.14 depending on the map pool and the credit used. It is not reliably zero, and not reliably positive.
- **Smart branch-point selection** adds +0.03 to +0.05 over random selection, not the +0.09 first reported.
- **No LLM training run, and no comparison with the closest published methods** (G2PO, PGPO, GRAFT, TSBPO, BPO).

## Repository

| path | what |
|---|---|
| `toy/h2lab_core.py` | Toy lab: KeyDoor environment, GRPO / GraphGPO / GiGPO / progress / progress-LOO credit, tree branching, PPO, training loop, credit study. |
| `toy/h2lab_llm.py` | Optional LLM policy (Qwen + LoRA) for the toy (GPU, untested at scale). |
| `notebooks/Hybrid_II_v2_2_ProgressTree.ipynb` | Original research notebook (Colab); its library cell is `toy/h2lab_core.py` before the validation options were added. |
| `scripts/run_main.py` | Headless, multi-core runner for the main comparison; arms accept config overrides, e.g. `"Progress+Tree(LOO)@branch_mode=consistent"`. |
| `scripts/analyze.py` | Seed-paired comparisons with CIs, optionally charging replay cost. |
| `recipe/ProgressTree/` | Drop-in [verl-agent](https://github.com/langfengQ/verl-agent) recipe for ALFWorld (copy to `verl-agent/recipe/ProgressTree`); see its [README](recipe/ProgressTree/README.md). |
| `tests/` | Independent estimator tests plus a wrapper for the recipe's CPU tests. |
| `results/` | Every run behind the numbers above: `v2_1_full/` is the author's original 5-seed run; the others are the validation runs. |
| `docs/` | [`validation.md`](docs/validation.md) (what was tested and found), [`audit.md`](docs/audit.md) (adversarial code review), [`prior_work.md`](docs/prior_work.md) (related work and novelty). |

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest tests -q                       # ~15 s

# one 10-seed comparison, ~30 min on 14 cores
python scripts/run_main.py --out results/my_run --seeds 0 1 2 3 4 5 6 7 8 9 --budget 2400000 --eval-points 24 \
  --arms GraphGPO GiGPO "Progress+Tree(LOO)@branch_mode=consistent"
python scripts/analyze.py results/my_run --pair "Progress+Tree(LOO)@branch_mode=consistent" GiGPO
```

To run the ALFWorld pilot (GPU), follow [`recipe/ProgressTree/README.md`](recipe/ProgressTree/README.md). Its protocol is a 6 GPU-hour pilot first, then a multi-seed comparison only if the live merge rate is ≳ 0.85 or Progress+Tree is ahead.

## Next steps

1. **Run the ALFWorld pilot** and read the live merge rate (`main_shared_step_frac`).
2. **Compare against G2PO or PGPO**, the closest credit methods, and a tree variant of one of them.
3. **Build a credit estimator that holds up when states rarely merge.** This is the method's main weakness.

---

By Thazzeus, developed with Claude.
