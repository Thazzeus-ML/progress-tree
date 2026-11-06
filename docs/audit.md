# Progress+Tree audit synthesis

> Adversarial code review of the repository as imported (6 reviewer lenses, each finding challenged by 1-2 skeptics). Line numbers refer to the code before the validation options were added; the fixes and the experiments that settle the open questions are in [validation.md](validation.md). Scratch paths below refer to the reviewers' machine.

Scratch scripts and logs cited below (e.g. `guard_sim.py`) were in the reviewers' scratch directory and are not included. Line numbers in `toy/h2lab_core.py` were re-checked against the repo copy.

## 1. Bottom line

No finding reverses the headline. Within the toy's own accounting, Progress+Tree(LOO) and Progress+Tree beat GraphGPO, GraphGPO(2xG) and GRPO at equal environment steps. Two 10-seed replications on fresh task pools, run during this audit (`results/replicate_a`, pools 3000/4000; `results/replicate_b`, pools 5000/6000; 2.4M steps), both PASS with larger margins: Progress+Tree(LOO) 0.511 and 0.464 vs GraphGPO 0.256 and 0.240, all CIs separated, permutation p ≤ 0.0004.

The size of the margin depends on two toy-only modelling choices. Both favour the tree arms compared with the method as described and ported:
- branches restart at t=0 with a 20-step cap;
- restoring a branch state costs nothing.

Each one alone cuts the original-pool margin from +0.11 to roughly +0.04 to +0.07. Their combined effect is untested.

The attribution needs rewriting:
- "Branching alone adds ~0" does not replicate. Tree+GraphGPO beats GraphGPO by +0.10 on both fresh pools.
- "Progress credit adds +0.12 on the same tree" holds (+0.13 and +0.08).
- Roughly half or more of "smart +0.09 over random" comes from the random arm branching out of frozen trap states.

The credit study points the right way, but its 0.47 vs 0.28 gap is overstated by roughly half or more. The limitation claim is too strong: the loss at noise 0.15 is mostly a toy artifact and becomes a tie once fixed. GraphGPO is reliably ahead only at noise ≥ 0.3.

| Claim | Status | Best current estimate |
|---|---|---|
| P+T(LOO) 0.505 / P+T 0.502 vs GraphGPO 0.392 / 2xG 0.374 / GRPO 0.365, PASS | **Holds** (direction). Absolute baseline values depend on the task pool. | v2.1: +0.11 (data not shipped). Fresh pools: +0.22 to +0.26. Original pool with port-like branches or charged restore: about +0.04 to +0.07 each; jointly untested. |
| Branching alone adds ~0 (0.384 vs 0.392) | **Not supported** | +0.105 (replicate_a), +0.103 (replicate_b) |
| Progress credit adds +0.12 on the same tree | **Holds** | +0.128 / +0.083 |
| Smart branch points +0.09 over random | **Overstated** | Ranking alone is about +0.03 to +0.05. The trap filter is about +0.03 to +0.07 (1.2M steps, 2-5 seeds, noisy). |
| Credit cosine 0.47 vs 0.28 | **Direction holds, size overstated** | About +0.06 to +0.10 once data is matched and the GraphGPO underflow is removed |
| Wins only at ~85% shared steps; loses at noise ≥ 0.15 | **Overstated** | Tie at 0.15 after the fix. Loss at ≥ 0.3 holds. Crossover is around 0.15-0.3 (0.71-0.58 shared). |

Caveat: the replications use the same toy code, so they inherit issues 1, 4, 5 and 7. They show the result is robust to the task pool, not to those modelling choices.

## 2. Confirmed issues (ranked)

None is critical. "Survives" means at least half of the skeptics upheld it. Where the skeptics disagreed on severity, both grades are shown.

### 1. Toy branches restart at t=0 with a 20-step cap, unlike the port (major; skeptics split major/minor)
- **Location:**
  - `toy/h2lab_core.py:704`: `cfg['mc_horizon']` is passed to the branches.
  - `:536-537`: `rollout` is called without `init_t`.
  - `:94`: so t starts at 0.
  - `:801`: `mc_horizon=20`; the main episode horizon is 40.
  - `:399` and `:427`: a timed-out final step is valued 0.
  - `:709-714`: branch and main rows are pooled into one V.
  - `:563-573`: the dead-state cache is fed the same truncated outcomes.
  - The port does it differently: `recipe/ProgressTree/tree_rollout_loop.py:211` sets `steps_left = max_steps - t`, and `branch_horizon` is null.
- **What is wrong:**
  - Pooled V̂ at branch points mixes 20-step and (40−t)-step success rates. It comes out 17-28% low (for example 0.390 vs 0.471 true; 0.154 vs 0.195 true).
  - The notebook says progress advantages "add up exactly to the GRPO advantage" (raw line 51). That fails in tree mode.
  - The toy's evidence was produced by a different branch process from the one the port runs, although the estimator function is identical.
- **Effect on P+T vs GraphGPO:**
  - The estimator bias hurts the progress arms. But truncated branches are cheaper: 15-19 steps vs 22-28, which gives about 30% more iterations. At equal env steps the net effect favours the tree arms.
  - P+T(LOO), toy branches vs horizon-consistent branches: 0.347 vs 0.297 at 1.2M (toy ahead in 4 of 4 paired seeds); 0.530 vs 0.464 at 2.4M (2 seeds).
  - On the original pool that leaves about +0.07 over GraphGPO (0.392), with CI separation not established.
  - In the credit study, consistent branches raise P+T(LOO)'s cosine (+0.09 to +0.15 in one reviewer's check, about 0 in the other's).
- **Fix:** have `select_branch_nodes` return each state's earliest visit time, and restore branches at that time with the full horizon:
  ```python
  # select_branch_nodes (loop at 606-612): first_t[s] = min(first_t.get(s, t), t)
  # out += [(g, task, s, first_t[s]) ...]; random mode keeps the sampled position's own t
  # mc_continuations (536-537):
  rb = rollout(tasks, ids, policy, cfg['horizon'], device,
               init_keys=keys, init_t=np.repeat([n[3] for n in nodes], K), merge_noise=merge_noise)
  ```
  Keep `mc_horizon` for probes only. Add `cfg['branch_mode'] in {'toy20','consistent'}` and report both.
- **Needs a re-run:** yes (E1).

### 2. "Branching alone adds ~0" is an equivalence claim 5 seeds cannot support, and it does not replicate (major; 2/2 major)
- **Location:**
  - `notebooks/Hybrid_II_v2_2_ProgressTree.ipynb`, cell 16 `verdict()` (raw ~1281): returns only the point difference and whether the CIs are separated.
  - Cell 19 (raw 1349-1356) and cell 38 (raw ~2115-2130) use that output.
  - The prose is at raw line 69.
- **What is wrong:**
  - There is no interval on the difference and no equivalence test.
  - Observed seed SDs are 0.054-0.093, which puts the 80%-power minimum detectable effect at about 0.12-0.14 for n = 5.
  - Both fresh pools contradict the claim:
    - replicate_a: Tree+GraphGPO 0.361 vs GraphGPO 0.256 (paired t95 [−0.012, +0.151]);
    - replicate_b: 0.343 vs 0.240 (9-seed paired t95 [+0.018, +0.139]).
- **Effect:** none on the headline. It moves roughly 40% of the gain from progress credit to tree sampling. The progress-credit-on-tree effect (+0.13 / +0.08) survives.
- **Fix:**
  - Report paired per-seed differences with t or bootstrap CIs. Arms share initial weights and task sequences per seed (`h2lab_core.py:817, 822, 838`).
  - Write "no detectable effect (95% CI [x, y])" rather than "adds nothing".
  - Publish the replicate attribution.
- **Needs a re-run:** no for the tree effect (already measured). Smart vs random on fresh pools is still missing (E2).

### 3. Merge noise defeats the absorbing-state filter, so smart selection branches from trap states (major; 2/2 major)
- **Location:**
  - `toy/h2lab_core.py:119`: a trapped agent never moves.
  - `:210-212`: every next key is salted, frozen agents included.
  - `:611-615`: "absorbing" is decided on the salted key (`if sn != s`, then `moves[s] > 0`).
  - `:618`: candidates are ranked by visit count.
  - `:558-560`: the dead-set backstop needs the start state to succeed at least 25% of the time, and needs `dead_min_n` tries. Each salted copy is unique, so it never reaches that count.
- **What is wrong:**
  - At q ≥ 0.15 a trapped key looks like it "moves". It is the most-visited state, so it is the top-ranked branch point in 72-82% of groups.
  - Trap branches take 16-24% of branch steps, never succeed and get A = 0 (LOO, `:422-428`).
- **Effect:** biases against the tree arms at noise > 0 only.
  - Paired reruns with the fix at noise 0.15: +0.037 (5 seeds, IQM 0.243 → 0.276) and +0.062 (3 seeds, 0.251 → 0.313). GraphGPO is about 0.29-0.31, so the published loss becomes a tie.
  - At noise 0.3 the gain is about +0.02 (noise), and the method stays behind.
  - The crossover moves from below 0.15 to about 0.15-0.3, which weakens the README's ALFWorld prediction (toy noise ≈ 0.09-0.21).
  - The Tree+GraphGPO cells at noise > 0 are biased the same way.
- **Fix:** decide absorption on the physical state. This is a toy-only correction; ALFWorld has no salt, though `core_progress.py:313` uses the same key-based test.
  ```python
  phys = lambda k: k if k == SUCCESS else k % SALT_BASE
  if phys(sn) != phys(s): moves[phys(s)] += 1          # :611-612
  cand = [s for s in cand if moves[phys(s)] > 0]        # :615
  ```
- **Needs a re-run:** yes, more seeds and a finer noise grid (E3).

### 4. The random branch-point ablation draws from frozen trap states (major for the attribution claim only; merged from three findings)
Verdicts: two findings upheld 2/2 (major, major); one upheld 2/2 at minor. A fourth finding framed as "unfaithful to Tree-GRPO" was refuted (section 3).
- **Location:**
  - `toy/h2lab_core.py:605-610`: `positions` gets one entry per active step, including every step of a frozen agent.
  - `:613-618`: smart mode's filters.
  - `:634-635`: random mode samples `positions` with no filter.
  - `:119` and `:138-139`: trapped agents stay live until the horizon.
  - `:536-537`, `:704`, `:843`: each branch runs 20 steps, all charged.
- **What is wrong:**
  - 34-42% of random branch points, and 35-44% of random branch steps, start in trap states. That is 22-26% of the arm's total env budget, with zero successes and zero advantage.
  - On a matched policy, random's non-trapped points succeed as often as smart's (31% vs 28.5% at the 990k checkpoint). So the published "branch success 47% vs 26%" is mostly the filter, not the ranking.
  - Training at 1.2M steps (2-5 seeds per cell, noisy, one pair reversed):
    - random as shipped: about 0.24;
    - random with smart's filters: about 0.30;
    - smart: about 0.33-0.35.
  - Correction to the finding: wrong-key states are not an asymmetry. Smart picks more of them (19-23%) than random does (8-12%).
- **Effect:** none on the headline; no headline arm uses random mode. The "+0.09" is a correct comparison of the two documented rules, but it overstates what the most-visited ranking contributes. `docs/prior_work.md:42` sets it beside TSBPO's 53% vs 28%, and `:71`/`:77` describe the selection as uncertainty-based, while the code ranks by visit count (`:618`).
- **Fix:**
  - Add `mode='random_filtered'`, drawing from `[s for s in positions if s != SUCCESS and moves[s] > 0 and s not in dead]`, plus an ARMS entry.
  - Give branch selection its own RNG (`np.random.default_rng(seed + 10**6)`). Random draws at `:635` currently shift the task sequence at `:838`, which breaks pairing with the other arms.
  - Relabel the result as "full smart rule (filters + ranking) vs Tree-GRPO-style uniform expansion".
- **Needs a re-run:** yes (E2).

### 5. GraphGPO's step credit underflows against eps=1e-6 in KeyDoor (major for the credit study, minor for training; skeptics split)
- **Location:**
  - `toy/h2lab_core.py:325`: step return is `10*gamma**d(s')`.
  - `:296`: z-score uses `std(ddof=1)+1e-6`.
  - `:336`: grouped by (task, anchor state).
  - `:800`: `gamma=0.1`.
  - The port inherits the same default: `recipe/ProgressTree/config/progress_tree_trainer.yaml:24, 31`.
- **What is wrong:**
  - Within an anchor group the returns are about 10^(1−d). For d ≥ 8 the std is far below 1e-6, so the step advantage collapses toward 0.
  - 48-80% of rows that carry step signal lose more than 95% of it. The step channel's norm is 0.30-0.65 of its exact value.
  - This copies upstream exactly (`core_graph.py:825` with `normalize_distance=False`; `core_gigpo` eps=1e-6). Upstream itself uses gamma = 0.8 for its Sokoban gridworld.
- **Effect:**
  - Credit study: GraphGPO's cosine rises from 0.27-0.29 (faithful) to 0.35-0.37 (eps-free). The gap shrinks by 30-55%. The ordering holds: P+T(LOO) beats eps-free GraphGPO in 7 of 9 cells.
  - Training: eps-free GraphGPO showed no gain at 1.2M steps (0.146 / 0.199 / 0.256 vs 0.160 / 0.285 faithful). A gridworld-appropriate gamma is untested in training.
- **Fix:**
  ```python
  # graph_step_returns :325, normalize_distance variant (identical z-score inside an anchor group, no underflow)
  ret[t, b] = 10.0 * gamma ** (dfill[int(batch['keys'][t+1, b])] - dfill[int(batch['keys'][t, b])] + 1)
  ```
  Report both forms, and sweep gamma ∈ {0.1, 0.5, 0.8} for KeyDoor.
- **Needs a re-run:** the credit study is quantified; training with gamma 0.8 needs E5.

### 6. The port's wall-clock guard times only the rollout and has an off-by-one; compare_runs scores slower arms on older checkpoints (port protocol only; skeptics split major/minor; biases against the tree arm)
- **Location:**
  - `recipe/ProgressTree/tree_rollout_loop.py:114` and `:143-144`: `dt` covers rollout and branches only.
  - `:79-82`: guard fires when `train_seconds + 2*avg > budget`.
  - Upstream `graphgpo_ray_trainer.py:1090`: `is_last_step` is computed before the rollout.
  - `compare_runs.py:82`: H = minimum over runs of each run's last validation time.
  - `compare_runs.py:56-58`: each run is scored at its last validation at or before H.
  - `run_alfworld_1gpu.sh:90`: `test_freq=5`.
- **What is wrong:**
  - Every run overshoots the budget (simulated: 13 min for graphgpo vs 30 min for the tree arm at 6 h).
  - Even with full-step timing, the 2× rule still overshoots by up to one step, because steps k and k+1 both run.
  - About half of the scoring bias comes from validating every 5 steps rather than on a time schedule.
  - Simulated scoring times, graphgpo vs progress_tree: 5.95 h vs 5.75 h at 6 h; 3.21 h vs 1.92 h in the 3-h pilot (assuming 6+4 vs 14+9 min per step).
- **Effect:** only false negatives for the method. It could flip the pilot's go/no-go rule ("progress_tree ahead at 3 h", README:102-104). The toy claims are unaffected.
- **Fix:**
  - Time the whole step: from one `multi_turn_loop` start to the next, minus validation time.
  - Guard on `train_seconds + 3*avg_step > budget`. The finding's suggested `+ avg_step` would make the overshoot worse.
  - In compare_runs, default H to `max_hours`, read validations on a time grid (or interpolate), and flag runs whose last validation is before H. That last check also covers the no-pipefail crash case (`run_alfworld_1gpu.sh:14, 93`).
- **Needs a re-run:** needs measured per-step times (GPU), then the existing `guard_sim.py`.

### 7. State restore is free in the toy; the port replays prefixes (minor; 2/2 downgraded from major)
- **Location:**
  - Toy: `toy/h2lab_core.py:92-94` (restore by decoding the key), `:537`, `:545` (only continuation steps are charged), `:843`.
  - Port: `branch_envs.py:65-73` and `:159` (replays `memory[p][:t]`); `core_progress.py:316, 334` (branch realised at its earliest visit).
- **What is wrong:** replay would add 31-36% to the tree arms' charged steps (10.9 → 9.0 steps per branch over training). The cost is disclosed only qualitatively (notebook cell 38).
- **Effect:** favours the tree arms under an env-step cost model. Charging replay 1:1 (worst case):
  - original pool: +0.11 → about +0.03 to +0.07;
  - replicate_a: +0.25 → +0.21;
  - partial replicate_b (7 seeds): +0.22 → +0.14, with CI separation against 2xG becoming marginal.

  Under the port's GPU-hours protocol, replay needs no LLM generation and is already charged at its true cost. Tree+GraphGPO pays the same cost, so the credit attribution is unchanged; "branching alone" drops by about 0.07.
- **Fix:** in `train_arm`, log `replay_steps += sum(t_first)` over branch nodes each iteration. Report `at_budget` on the axis `env_steps + w*replay_steps` for w ∈ {0, 0.25, 1}.
- **Needs a re-run:** only to log replay depth. It can ride along with E1.

### 8. The credit-study headline gives tree estimators extra branch data (minor; 2/2 downgraded)
- **Location:**
  - `toy/h2lab_core.py:701-714`: tree estimators compute credit on main + branch rows.
  - `:740-741`: GraphGPO uses the main batch only.
  - `:905`: every estimator scores the same main-batch pairs.
  - Notebook raw 70, 2155-2159 (`cos_gain = max(...)` over the two flagships), and 2251.
- **What is wrong:**
  - The extra data explains only 10-25% of the gap (Tree+GraphGPO − GraphGPO = +0.017 to +0.046). The data-matched gap is +0.13 to +0.15. Progress+VineAll vs GraphGPO+VineAll is +0.22.
  - The data-matched comparator is computed and tabulated (raw 1413-1414 and the "extra env steps" column at 1447/2151); only the prose leaves it out.
- **Effect:** the credit gap shrinks by about a quarter; the direction holds. Combined with issue 5, the data-matched, eps-free gap is not yet measured (E4).
- **Fix:** headline P+T vs Tree+GraphGPO and the VineAll pair, drop the `max()`, and report paired CIs over (seed, snapshot) cells.
- **Needs a re-run:** a cheap credit-study rerun (E4).

### 9. The crossover rests on 3-seed comparisons that fail the notebook's own PASS rule (minor; 2/2 minor)
- **Location:** `recipe/ProgressTree/README.md:57-72, 134-135`; notebook cell 0 (raw 16, 30-33); `results/port/toy_sweep_runs.csv`. Cell 31 (raw 1634, 1668-1670) never actually tests smart32 against GraphGPO.
- **What is wrong:**
  - Noise 0: +0.043, CIs overlap, permutation p = 0.40 (per-seed −0.057 / +0.100 / +0.086).
  - Noise 0.15: −0.060, CIs overlap, p = 0.20.
  - smart8 vs GraphGPO at noise 0: −0.081, CIs overlap.
  - CIs separate only at noise ≥ 0.3.
  - With n = 3, "IQM" is the plain mean, the bootstrap CI is just the seed range, and the smallest possible permutation p is 0.10.
- **Effect:** none on the headline. The crossover is undetermined between noise 0 and about 0.25, so the ALFWorld prediction is untested rather than supported. Issue 3 compounds this.
- **Fix:** call these "3-seed pilot trends" with CIs, and rerun (E3).

### 10. None of the headline v2.1 numbers ship with the repo (minor/note; skeptics minor and note)
- **Location:**
  - The notebook's 41 cells all have 0 outputs.
  - The v2.1 figures appear only as prose: raw 10 and 66-70, a hardcoded string at raw 2071, and README.md:57-58.
  - No `h2lab_v2_*.csv` exists anywhere in the repo.
- **What is wrong:** none of the headline CIs or PASS verdicts can be checked from shipped data.
  - Correction to the finding: "every (tag, noise) cell has n=3" is false. GraphGPO's calibration cells have n=1, and n=4 at noise 0.3.
  - `results/replicate_a` and `results/replicate_b` were written today (07:53 and 08:28), after the repo was copied in at 07:20, by the replication job running alongside the audit. The author did not ship them.
- **Effect:** none; this is about verifiability. Rerunning the shipped code reproduces the result. GraphGPO on the original pool at 2.4M gave 0.301 / 0.426 for seeds 0/1, consistent with 0.392.
- **Fix:**
  - Ship the per-(arm, seed) eval CSV, the credit CSV and the executed notebook.
  - Replace the v2.1 prose with the replicate numbers.
  - Update "Not run yet" (cell 0, raw 14).

### 2b. Unverified notes (one auditor each, no skeptic check)

| Note | Location | Point | Likely direction |
|---|---|---|---|
| No GiGPO baseline | `h2lab_core.py:643-662` | GiGPO (exact-anchor MC returns) is the closest relative and would be the most direct test of pooled first-visit V | Unknown |
| GraphGPO sweep values don't reproduce | `toy_sweep_runs.csv:14-16` | Reruns at 1.2M gave 0.160 / 0.285 vs 0.344 / 0.324 in the CSV | The CSV favours the baseline |
| 2xG not iteration-matched | `h2lab_core.py:647, 827`; nb raw 1155; README:20 | A tree iteration costs about 2.6-2.8× a G=8 iteration; 3xG would match | Label only |
| ITER_OVERHEAD cost model | nb cell 17 | A tree iteration has two sequential rollout phases but is charged one overhead | Favours tree arms |
| Random arm shifts the task RNG | `h2lab_core.py:822, 838, 842, 635` | The random arm trains on a different task sequence | Unpaired variance only |
| Credit-study step subset | `h2lab_core.py:897-899` | Only steps that change the de-salted state are scored; not documented. M=32 noise lowers all cosines. | Conservative |
| Time-unaware / non-Markov V | `h2lab_core.py:64-66, 358-380`; `core_progress.py:76-96, 183-191` | Time-wasting self-loops get 0 credit; similarity clustering can merge different success rates | Shared with GraphGPO |
| Unpaired tests, mislabelled permutation test | `run_main.py:6-7, 96-106, 120-125` | Tests ignore seed pairing; the permutation test runs on at-budget means, 8 tests uncorrected; P+T and LOO PASSes are not independent | Conservative / label |
| "IQM" of 3 is the mean; README uses final evals | `run_main.py:55-61`; README:57 | Label is inaccurate; overshoot ≤ 1.1% | Negligible |
| Mixed-seed stress table | nb cell 25, raw 1480-1486 | The noise-0 row is 5 seeds (interpolated from 2.4M runs); other rows are 3 seeds at 1.2M | Inconsistent with cell 31 |
| Port has no dead-state memory | `core_progress.py:286-335`; test `:120` | The equivalence test uses memory=None; the memory skips about 0.1 nodes per iteration | Small |
| Canonicalizer test nearly vacuous | `tests/test_core_progress.py:171-198` | 40 clusters from 41 states; equivalence does hold on realistic anchors | Test quality |
| Implementation overhead counted in the budget | `tree_rollout_loop.py:129, 191-221, 239-280`; `branch_envs.py:47-63` | Synchronous waves, env re-creation, about 7 s per group for canonicalization | Unclear |
| Calibration not reproducible | README:48, 60; `compare_runs.py:129` | The CSV has no shared-step column; README says 0.71, compare_runs says 0.75 | Documentation |
| Protocol details | `setup_colab.sh:10`; `run_alfworld_1gpu.sh:27` | Upstream clone not pinned (works at 20bd331); `GROUP=12` graphgpo run collides with the plain graphgpo run name | Reproducibility |

## 3. Refuted or downgraded findings

- **random-branch-baseline-wastes-budget, refuted (0/2):** Tree-GRPO samples non-leaf nodes uniformly and cannot know that a state is absorbing, so the random arm is a faithful Tree-GRPO adaptation, and +0.09 correctly compares the two documented rules. The decomposition concern survives as issue 4.
- **same-task-pools-dev-and-claim, refuted (0/2):** the confirmatory runs on fresh pools now exist and PASS with larger margins. What remains is the stale "Not run yet" text and pool-specific absolute numbers.
- **restore-is-free, major → minor (2/2):** restoring by decoding the key is a disclosed modelling assumption. The port budgets in GPU-hours, where a replay step costs no LLM generation, and the margin stays positive under the 1:1 worst case.
- **credit-study-unequal-data, major → minor (2/2):** the extra data explains 10-25% of the gap, not the claimed 40% (which came from one snapshot), and the comparator is already in the tables.
- **crossover-not-significant, major → minor (2/2):** it is about wording and certainty, not bias. All 3 paired seeds are negative at 0.15, and the noise ≥ 0.3 losses are separated.
- **headline-not-in-shipped-data, major → minor/note:** a verifiability gap only; independent reruns confirm the headline.
- **random-ablation-unfiltered, major → minor (2/2):** not leakage, since smart uses only observed transitions. Merged into issue 4.
- **graphgpo-step-credit-eps-underflow, kept major / minor:** it matches upstream code, eps-free GraphGPO trained no better, and the credit ranking survives.
- **toy-branch-horizon-truncation, minor / major:** the minor verdict had no training runs. The major verdict measured the equal-env-step penalty of consistent branches (4 of 4 paired seeds), so it is ranked major here.
- **wall-clock-guard-rollout-only, minor / major:** the bias is conservative and affects only the port's planned runs. Partly mis-attributed: the off-by-one and step-scheduled validation each contribute.

## 4. Checked and fine

- **GraphGPO fidelity:**
  - Step returns `10*gamma^d(s')` match upstream with `normalize_distance=False`.
  - Distances use reverse BFS from SUCCESS, with unreachable set to max+1.
  - SUCCESS sentinel, exact anchor grouping, and mean/std normalisation (ddof=1, eps 1e-6, singleton groups → 0) all match.
  - The episode term is normalised across steps; w_step = w_ep = 1; gamma = 0.1.
  - The port's graphgpo arm is unmodified upstream credit and config.
- **Code identity:** the notebook's `%%writefile` library and `recipe/ProgressTree/tests/toy_core.py` are byte-identical to `toy/h2lab_core.py`.
- **Budget accounting:**
  - Main, branch and probe steps are all charged at `:843`, and each charged step is exactly one training sample.
  - Evaluation is never charged.
  - Each run stops at the first iteration past the budget, and `at_budget` interpolation removes the overshoot.
  - `fixed_minibatch` gives about equal Adam steps per env step (2.02-2.04 vs 1.97-2.3 per 1k samples).
  - `fixed_steps` is conservative for the tree arms. With no LR schedule, a mid-run read at 1.2M is valid.
- **Information parity:**
  - The dead-state memory is per run and built only from charged data.
  - Tree+GraphGPO and Tree+GRPO use the same smart selection and memory as Progress+Tree.
  - No cross-group or cross-iteration leakage into credit.
  - Train and eval pools are disjoint (0 shared layouts).
  - Every arm gets an identical stochastic evaluation (128 maps × 4 reps).
  - Merge noise is applied identically to every arm; start states and SUCCESS are never salted.
  - For a given seed, initial weights and task sequence are the same across arms (except the random arm).
- **Estimator math:**
  - LOO subtracts the trajectory's own outcome exactly once, and its fallback is a valid baseline.
  - Telescoping holds; first-visit deduplication is correct.
  - `group_scale` uses main rows only; `light_concat` pads correctly.
  - LOO is close to unbiased on main rows with horizon-consistent branches (cosine 0.983-0.990 against the exact gradient).
  - On expected gradients P+T still beats GraphGPO by a wide margin.
- **Credit-study ground truth:** forced first action, restore at the true `init_t`, full state key, the same batch and pairs for every estimator. The step filter does not favour progress (0.355 vs 0.357 for P+T with and without it).
- **Port:**
  - Estimator equals the toy on 24 batch/mode combinations (max |diff| 0); `tests/test_core_progress.py` passes (6/6).
  - Branches are horizon-consistent.
  - The tree_grpo proxy matches Ji et al. Eq. 6-7.
  - Every patched upstream symbol resolves at verl-agent 20bd331.
  - Branch rows have consistent keys, uids and step indices; restore replays projected actions and anchors match.
  - Validation time is excluded symmetrically for all arms; Hydra overrides are valid.
- **Reported numbers that reproduce:**
  - All 26 values in the README noise table, and the branch shares (0.612 / 0.330).
  - "Spreading costs 0.10", and the critic figures.
  - The ALFWorld merge-probe tables.
  - Shared-step header values (rerun: 0.853 / 0.715 / 0.571).
  - `iqm()` equals `trim_mean(x, 0.25)`.
- **Statistics:** at 5 seeds the PASS rule has a false-PASS rate of 0.5-1.5% per comparison, and 67-99% power for a true +0.11.

## 5. Recommended experiments (CPU)

Cost basis: replicate_b ran 60 runs at 2.4M on 14 workers in 2067 s, about 8 core-minutes per run on average (tree arms take longer). Use `torch.set_num_threads(1)`.

*(Since addressed: `scripts/run_main.py` now takes per-arm config overrides, `ARM@key=value`.)* At the time, `scripts/run_main.py` had no hooks for new modes. Put the patches in a scratch wrapper that applies them inside the worker's `_job`, and use `Pool(maxtasksperchild=1)`. Patches leaking through reused pool workers contaminated two audit runs (`e2_random.log`, `ab_round2`).

**E0 (no compute):** from `results/replicate_a/runs.csv` and `results/replicate_b/runs.csv`, report paired per-seed differences with t and bootstrap 95% CIs for:
- P+T(LOO) − GraphGPO;
- P+T(LOO) − GraphGPO(2xG);
- Tree+GraphGPO − GraphGPO;
- P+T − Tree+GraphGPO.

**E1, port-faithful branch process (settles the headline margin for the method as ported):**
- Arms: GraphGPO; Tree+GraphGPO and Progress+Tree(LOO) under `branch_mode ∈ {toy20, consistent}`. Consistent means `init_t = t_first` and `horizon = 40`, mirroring `tree_rollout_loop.py:211`.
- Log `replay_steps = Σ t_first` per iteration, then read `at_budget` on `env_steps + w·replay_steps` for w ∈ {0, 0.25, 1} after the fact. Charged restore needs no extra runs.
- Seeds 0-9; pools 1000/2000, then 3000/4000; budget 2.4M; eval_points 24.
- Cost: 5 arm-variants × 10 seeds × 2 pools = 100 runs, about 15 core-hours.
- Decision: is P+T(LOO, consistent, w=1) − GraphGPO ≥ 0.03 with a paired CI excluding 0?

**E2, filtered random ablation (settles "smart +0.09"):**
- Arms: Progress+Tree (smart); Progress+Tree(random); random_filtered (`s != SUCCESS and moves[s] > 0 and s not in dead`); random_nonabsorbing (`moves[s] > 0` only, which separates out the dead cache).
- All random variants use a separate branch RNG, `default_rng(seed + 10**6)`, so task sequences stay paired.
- Seeds 0-9, pools 1000/2000, 2.4M, eval_points 24.
- Log per iteration the trapped-node share and branch success split by trapped start.
- Cost: 40 runs, about 6 core-hours.

**E3, merge-noise crossover with the absorbing fix (settles the "~85% shared" claim):**
- Arms: GraphGPO; Progress+Tree(LOO) with the physical-key absorbing test (issue 3).
- `--merge-noise ∈ {0, 0.05, 0.1, 0.15, 0.2, 0.3}`; seeds 0-9; budget 1.2M (comparable to the sweep).
- Then repeat at 2.4M for {0.1, 0.15, 0.2}.
- Log `shared_step_frac` so the calibration curve can be rebuilt.
- Cost: about 120 short runs plus 60 long, about 16 core-hours.
- Report the paired difference with a t CI for each noise level, and the noise at which the CI crosses 0.

**E4, data-matched, eps-free credit study (settles the cosine claim):**
- The notebook's `_cs_job` protocol: CS_CFG, GraphGPO-trained snapshots at 60/120/200 iterations, 32 tasks × 8, 400 pairs, M = 64 (up from 32, to reduce ground-truth noise). Seeds 0-4.
- Estimators: GraphGPO, Tree+GraphGPO and GraphGPO+VineAll, each with {faithful, normalize_distance} × gamma {0.1, 0.8}; Progress+Tree and Progress+Tree(LOO), each with {toy20, consistent}; Progress+VineAll.
- Headline: P+T(LOO) − Tree+GraphGPO (eps-free), with a paired CI over 15 cells.
- Cost: 5 GraphGPO trainings to about 1M steps plus the evaluations, about 2 core-hours.

**E5, GraphGPO tuned for the gridworld (settles whether the baseline is under-tuned):**
- Arms: GraphGPO with gamma 0.1 faithful, gamma 0.1 normalize_distance, gamma 0.5, gamma 0.8.
- Optionally GiGPO with exact anchors, gamma 0.95, mean/std normalisation.
- Seeds 0-9, pools 1000/2000, 2.4M. Compare against P+T(LOO) from E1.
- Cost: 40-50 runs, about 5 core-hours.

E0-E5 total about 45 core-hours: about 3.5 h on 14 cores, or about 22 h on 2.

Non-CPU follow-up for issue 6: time 10 steps each of graphgpo and progress_tree (rollout vs update) on the target GPU, feed the times into `guard_sim.py`, and confirm the fixed guard (full-step timing, 3× rule, H = max_hours) stops within budget for both arms.
