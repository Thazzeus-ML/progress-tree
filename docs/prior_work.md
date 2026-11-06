# Progress+Tree: prior work and novelty assessment

> Written by an automated literature search (8 papers read and checked against their method sections, Nov 2026). Corrected afterwards where it described the branch-point rule, and the numbers in section 4 were updated after the 10-seed validation ([validation.md](validation.md)). The field moves fast: re-run the search before any public claim.

## 1. Verdict

Each of the five components already has a published precedent. Seven of the eight verified papers use the same setting: GRPO-family RL for multi-turn agents with a sparse success reward, evaluated on ALFWorld and/or WebShop.

- **C2 + C3 (state pooling and progress credit).** Merging identical or similar states across a group's rollouts and crediting each step with `V(s_{t+1}) - V(s_t)` is published in G2PO (Eq. 5), PGPO (Eq. 7) and GRAFT (Eq. 8). TreeRL does the same on token prefix trees (Eq. 3).
- **C4 (leave-one-out value).** A leave-one-out version of the pooled per-state value appears as an ablation in PGPO (App. D.2). BPO uses a sibling leave-one-out baseline at branch states (Eq. 3) and proves it unbiased.
- **C1 (tree rollouts).** Branching from revisited intermediate states and training on every branch appears in TSBPO, BPO, TriGRPO and 3SPO. TreeRL does it for single-turn generation.
- **C5 (branch-point selection).** TSBPO chooses branch points by outcome uncertainty and reports a comparison against random branching: on WebShop, 53.16% of its selected groups turned mixed-outcome, against 27.68% for random. BPO and TreeRL pick branch points by policy entropy and also beat random choice.

None of the components can honestly be called new on its own. What the eight verified papers do not contain is the specific combination:

- branching on purpose to add shared states to a pooled state graph that covers the current group, with every branch pooled in;
- computing raw progress credit on that graph;
- choosing branch points from the graph's own empirical success rates.

The nearest misses:

- G2PO lists tree sampling as future work.
- GRAFT argues that resampling from intermediate states costs too much.
- 3SPO comes closest. However, it pools counts over the whole training history, its sibling normalization cancels the `V(s_t)` baseline, and it sends extra rollouts to low-success states rather than uncertain ones.
- TSBPO branches by outcome uncertainty but computes no progress credit on a pooled graph.

A second contribution is plausibly new: the empirical attribution. The project reports a sample-controlled result that the gain comes from C3 + C5 rather than from extra rollouts. It also reports an operating range: the method wins only when about 85% or more of steps land on shared states, and fails on sparse graphs. Related observations exist:

- G2PO's group-size statistics and its smaller margin on AppWorld;
- GRAFT's argument about graph saturation;
- PGPO's warning that estimates get noisy when groups are small;
- TSBPO's group-density measurements.

None of these papers reports a threshold or a failure case. The findings so far come from the KeyDoor toy environment. A README should describe the combination with "to our knowledge" rather than as novel, and present the findings as results in that setting.

## 2. Component table

| Component | Closest prior work | Overlap | Main difference from Progress+Tree |
|---|---|---|---|
| C1 Tree rollouts from replayed states, every branch trained on | TSBPO (Sec. 3.3, 4.2, Alg. 1); also BPO, TriGRPO, 3SPO; TreeRL at token level | Substantial | TSBPO branches only from root (factual) rollouts and restores state from a snapshot plus history. BPO's tree is one level deep. Progress+Tree also branches off earlier branches and restores state by replaying actions. That is an implementation difference, not a novelty claim. |
| C2 State pooling; V(s) = success rate of all rollouts through s | G2PO (Sec. 4.1-4.2, App. G); also PGPO, GRAFT, 3SPO | Substantial | G2PO merges exact and similar observations (SequenceMatcher > 0.95) over N linear rollouts. Its node value is a gamma=0.95 discounted return averaged per visit. Progress+Tree uses an undiscounted per-rollout success rate over a tree that includes branches. GRAFT gets V from Bellman value iteration. 3SPO accumulates counts over the whole training history. |
| C3 Progress credit `A_t = V(s_{t+1}) - V(s_t)` | G2PO (Eq. 5), PGPO (Eq. 7), GRAFT (Eq. 8); TreeRL (Eq. 3) | Substantial | Prior work z-normalizes the term and/or adds it to GRPO/GiGPO advantages or Graph GAE. Progress+Tree uses it raw as the only advantage, with terminal V = outcome. PGPO also substitutes the final reward at the last step. |
| C4 Leave-one-out per-state V | PGPO (App. D.2, Table 11); BPO (Eq. 3, Thm 4.1, 4.3) | Substantial | PGPO drops the scored visit from the current and successor groups, as an ablation only. On ALFWorld it scored 92.32/88.80, against 93.03/88.73 without LOO. Progress+Tree drops the whole trajectory. At a branch step whose child has a single rollout, the Progress+Tree C3+C4 estimate equals BPO's sibling LOO. |
| C5 Rule-chosen branch points (frontier / uncertain outcome) | TSBPO (Sec. 4.3, App. E.1); also BPO and TreeRL (entropy), 3SPO, TriGRPO | Substantial | TSBPO estimates the chance that a new suffix flips the outcome with a fitted Beta-mixture model, then picks branch points by integer DP. Progress+Tree ranks by visit count after filtering absorbing states and zero-success dead ends. TSBPO measures mixed-outcome conversion (53% vs 28% for random). Progress+Tree measures branch success (47% vs 26% in v2.1; 49% vs 27% in the 10-seed validation, most of it from the filters). |

## 3. Related work (paste-ready)

*Citations for the baselines (GRPO, GraphGPO, Tree-GRPO) are not included below because this search did not verify their links. Add them separately.*

### Related work

**Step credit on pooled state graphs.** Several recent methods merge recurring states across a GRPO group's rollouts and credit each step with the change in estimated value between consecutive states.

- [G2PO](https://arxiv.org/abs/2606.22995) (Wang et al., 2026) groups identical observations into graph nodes (similar ones too, in an appendix). Each node's value is the mean return over its visits. The step term `V(next) - V(current)` (Eq. 5) is added to a GRPO episode advantage, on ALFWorld, WebShop and AppWorld. G2PO names tree sampling as future work.
- [PGPO](https://arxiv.org/abs/2609.02236) (Zheng et al., 2026) computes a Monte Carlo potential for each exact-match anchor state. It adds `Phi(s_{t+1}) - Phi(s_t)`, with the realized reward at the final step, to GiGPO's step advantage. Its App. D.2 tests a leave-one-out version of the potential.
- [GRAFT](https://arxiv.org/abs/2609.28963) (Yao et al., 2026) merges states by hash or embedding similarity into a trajectory graph and solves for node values by value iteration over empirical edge counts. It defines `A_t = gamma*V(s_{t+1}) - V(s_t)` (Eq. 8) and adds a graph GAE. GRAFT argues that resampling from intermediate states is too expensive for long-horizon training.

Progress+Tree uses the same progress signal, raw and as its only advantage. It differs in creating shared states on purpose through branching.

**Tree and branched rollouts.**

- [TSBPO](https://arxiv.org/abs/2610.41697) (Yu et al., 2026) replays visited states of root rollouts to add new suffixes and trains on every new turn. It picks branch points by the predicted chance that a new suffix flips the outcome. Against random branching on WebShop, 53% of its selected groups became mixed-outcome, against 28%.
- [BPO](https://arxiv.org/abs/2607.14171) (He et al., 2026) snapshot-restores a sandbox at the highest-entropy steps of one backbone trajectory and forks siblings to termination. Its sibling leave-one-out advantage at each branch point is proved unbiased. In a WebShop ablation, the entropy schedule beats random branch points.
- [TriGRPO](https://arxiv.org/abs/2611.02706) (Zhao et al., 2026) clones the environment when a rollout reaches a state from a bank of successful trajectories but takes a different action. It credits branches with trajectory-level outcomes and with paired parent-vs-sibling outcome differences.
- [3SPO](https://arxiv.org/abs/2606.09961) (Han et al., 2026) runs a backtracking search per task and keeps exact-match success counts per state across training. It rewards transitions by the change in a saturating state score and gives more rollouts to low-success states.
- [TreeRL](https://arxiv.org/abs/2506.11902) (Hou et al., 2025) works on single-turn reasoning. It forks earlier responses at high-uncertainty tokens and values each prefix-tree node by the fraction of correct leaves below it. Its step reward includes `V(s_n) - V(parent)`.

**Position of Progress+Tree.** Progress+Tree joins these two lines of work:

- it branches from replayed states, as TSBPO, BPO and TriGRPO do;
- it pools every rollout and branch of a group into one state graph, as G2PO, PGPO and GRAFT do for linear rollouts;
- it computes raw progress credit on that graph, optionally with a per-state leave-one-out value (cf. PGPO App. D.2, BPO Eq. 3);
- it branches from the graph's most-visited states, skipping absorbing states and states whose pooled success rate has stayed at zero (a simpler rule than TSBPO's outcome-flip criterion).

To our knowledge, neither this combination nor an analysis of when it helps has been reported before.

## 4. Suggested README framing

Progress+Tree is a GRPO-family credit-assignment method for multi-turn agents. It combines two published ideas: progress credit on a pooled state graph, `A_t = V(s_{t+1}) - V(s_t)` (G2PO, PGPO, GRAFT), and tree rollouts branched from revisited intermediate states (TSBPO, BPO, TriGRPO). To our knowledge, what it adds is using branching to create the shared states that progress credit depends on: branch points are the graph's most-visited non-absorbing states, and all branches are pooled into one graph. In our KeyDoor experiments the robust part of the gain came from progress credit (+0.10 to +0.13 over other credit on the same tree), and it appeared only when about 85% or more of steps landed on shared states; on sparse graphs the method failed.

## 5. Gaps

- **Coverage.** Only eight papers were verified, so other prior work may exist. These candidates are possibly related but were not checked:
  - Closest by title: RTMC (https://arxiv.org/abs/2604.11037), Belief-Shift Branching (https://arxiv.org/html/2609.11061), SPARK (https://aclanthology.org/2026.acl-long.1100.pdf), TRACE (https://arxiv.org/pdf/2606.11119), ProGPO (https://arxiv.org/abs/2607.04242), RewardFlow (https://arxiv.org/abs/2603.18859), TEMPO (https://arxiv.org/abs/2509.18314), VinePPO (https://arxiv.org/abs/2410.01679).
  - Others: SPO, CARL, GraphPO, MCGPO, PAV, LVRL, "The Trace Is the State", and verl issue #7244.
  - STEP (Chen et al., 2025), which 3SPO cites for per-state smoothed success-rate records and adaptive resampling. No URL; not checked.
- **Baseline papers not assessed.** GraphGPO, GiGPO and Tree-GRPO were not checked. If GraphGPO already computes progress-style credit over merged states, that is another precedent for C3.
- **Details that could not be read:**
  - GRAFT's App. D.2 group-size numbers were missing from the extracted text, so it is unknown whether GRAFT reports an overlap threshold.
  - 3SPO gives no smoothing formula and does not say how it restores states.
  - G2PO does not define δ for the terminal step.
  - BPO's oracle "value-disparity" schedule is the one closest to an outcome-based rule (68.4 vs 67.8 for entropy), but it is barely described.
  - TSBPO's random-branching run has no final-success evaluation.
- **No comparison against the closest work.** Progress+Tree is compared only to GRPO, GraphGPO, Tree-GRPO and Tree+GraphGPO, never to G2PO, PGPO, GRAFT, TSBPO or BPO. It is unknown whether it beats them. The most informative additions would be:
  - PGPO or G2PO, the closest credit methods;
  - a tree variant of one of them, which G2PO itself proposes;
  - TSBPO's selection rule in place of the Progress+Tree rule, to isolate C5.
- **Attribution beyond the toy setting:**
  - All reported findings come from KeyDoor.
  - TreeRL found that training on 16 instead of 30 responses at the same reward cost 3.2 points. More training sequences helped there.
  - PGPO found its progress term alone weaker than GiGPO's step advantage: 80.33/74.41 vs 89.71/83.65 on ALFWorld.
  - The "not extra samples" claim and the 85% threshold therefore need re-checking on ALFWorld/WebShop with matched sample counts.
  - G2PO reports 88.1% (WebShop) and 91.9% (ALFWorld) of steps in shared groups. If the definitions match, both benchmarks sit just above the threshold.
- **Number coincidence.** The Progress+Tree figure of 47% vs 26% is numerically very close to TSBPO's 47.32% vs 25.93%. TSBPO's figure is on ALFWorld and measures the predicted mixed-outcome rate for selected groups vs eligible candidates, which is a different quantity. The README should define its own metric precisely so readers do not mix the two up.
- **Recency.** TSBPO (v2, 5 Nov 2026) and TriGRPO (2 Nov 2026) appeared in the past week. Whether they count as concurrent work depends on when Progress+Tree was developed, which this search cannot establish. The area is moving fast, so the search should be rerun before any public claim.
