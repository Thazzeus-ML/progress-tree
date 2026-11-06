"""Progress+Tree credit assignment for verl-agent (port of h2lab v2.2, `Progress+Tree(LOO)`).

Everything here works on plain per-row arrays (one row = one agent step), so it can be
unit-tested without verl. `compute_progress_tree_advantage` is the adapter that reads a
verl `DataProto` and returns `(advantages, returns)` shaped like GraphGPO's estimator.

Row fields (all length N):
  uid          task group id (GRPO/GraphGPO group; every rollout of one game shares it)
  traj_uid     trajectory id (main rollouts and branches each have their own)
  step_idx     absolute step index inside the episode (a branch from step t starts at t)
  anchor, nxt  GraphGPO anchor strings before/after the step; nxt == 'success' on the
               last step of a successful episode (set by rollout_loop.gather_rollout_data)
  outcome      the trajectory's episode score (reward-manager score before any
               invalid-action penalty); identical for all rows of a trajectory
  penalty      per-row invalid-action penalty (<= 0), already in reward units; it enters
               every estimator exactly as it enters GraphGPO's episode-level term
               (graphgpo_penalty_term), so the format signal is held fixed across arms
  is_main      True for the G main rollouts, False for branch continuations
  parent       parent traj_uid for branches ('' for main rollouts)
  branch_step  step index at which a branch left its parent (-1 for main rollouts)

Estimators (`mode`):
  progress_loo  A_t = Q_t - b_t.  Q_t = pooled value V(s_{t+1}) over every trajectory of
                the group that visited the (canonicalised) state, or the trajectory's own
                outcome on its last step.  b_t = V(s_t) computed WITHOUT this trajectory's
                own sample; if nobody else visited s_t, the last available baseline along
                the trajectory is reused (the group's leave-one-out mean at the start state
                for main rollouts, the branch point's value for branches).  Divided by the
                group's outcome std (floored at sigma_min * reward_scale).
  progress      the same with b_t = V(s_t) including the trajectory itself (telescopes to
                R - V(s_0) exactly; noisier where states never merge).
  tree_grpo     Tree-GRPO-style proxy (Ji et al., ICLR 2026): every trajectory is a leaf;
                leaf advantage = intra-tree z-score + inter-tree z-score; a row's advantage
                is the sum over all leaves whose root-to-leaf path contains that row (what
                duplicating shared prefixes per leaf would give under a token-mean loss).
"""
from collections import defaultdict

import numpy as np

SUCCESS = 'success'


# ------------------------------------------------------------------------------------------
# Canonicalisation (GraphGPO's StateCanonicalizer when available)
# ------------------------------------------------------------------------------------------
def to_hashable(x):
    if isinstance(x, (int, float, str, bool)):
        return x
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return tuple(x.flatten())
    if isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    if isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    raise TypeError(f"Unsupported type: {type(x)}")


class FastCanonicalizer:
    """Same clustering rule as GraphGPO's StateCanonicalizer (greedy, first cluster whose
    representative has difflib ratio >= threshold), with exact cheap pre-filters
    (rapidfuzz Indel similarity >= ratio, real_quick_ratio, quick_ratio) so it gives the same
    clusters much faster on long anchors. enable=False -> exact matching."""

    def __init__(self, enable=False, threshold=0.95):
        self.enable, self.threshold = enable, threshold
        self._clusters, self._cache = [], {}
        try:
            from rapidfuzz.distance import Indel
            self._indel = Indel.normalized_similarity
        except Exception:                                    # pragma: no cover
            self._indel = None

    def canonicalize(self, state):
        h = to_hashable(state)
        if not self.enable or h == SUCCESS:
            return h
        if h in self._cache:
            return self._cache[h]
        from difflib import SequenceMatcher
        s = state if isinstance(state, str) else str(state)
        for ch, cs in self._clusters:
            # Indel normalized similarity == 2*LCS/(len_a+len_b) >= difflib ratio (difflib's
            # matching blocks are a common subsequence), so it is a safe rejection test.
            if self._indel is not None and self._indel(s, cs) < self.threshold:
                continue
            sm = SequenceMatcher(None, s, cs)
            if sm.real_quick_ratio() >= self.threshold and sm.quick_ratio() >= self.threshold \
                    and sm.ratio() >= self.threshold:
                self._cache[h] = ch
                return ch
        self._clusters.append((h, s))
        self._cache[h] = h
        return h


def make_canonicalizer_factory(enable_similarity=False, threshold=0.95):
    return lambda: FastCanonicalizer(enable_similarity, threshold)


# ------------------------------------------------------------------------------------------
# Trajectory bookkeeping
# ------------------------------------------------------------------------------------------
def _group_trajectories(uid, traj_uid, step_idx):
    """{uid: {traj: [row indices sorted by step]}} preserving first-appearance order."""
    groups = defaultdict(dict)
    for i in range(len(uid)):
        groups[uid[i]].setdefault(traj_uid[i], []).append(i)
    for trajs in groups.values():
        for t, rows in trajs.items():
            rows.sort(key=lambda r: step_idx[r])
    return groups


def _keys(rows, anchor, nxt, canon):
    """Canonical state before each row and after the last one."""
    ks = [canon.canonicalize(anchor[r]) for r in rows]
    last = nxt[rows[-1]]
    ks.append(SUCCESS if to_hashable(last) == SUCCESS else canon.canonicalize(last))
    return ks


def _value_stats(traj_keys, traj_R):
    """First-visit Monte Carlo statistics: {state: [sum of outcomes, count]}."""
    stats = {}
    for tr, ks in traj_keys.items():
        seen = set()
        for s in ks:
            if s == SUCCESS or s in seen:
                continue
            seen.add(s)
            st = stats.setdefault(s, [0.0, 0])
            st[0] += traj_R[tr]
            st[1] += 1
    return stats


def graphgpo_penalty_term(uid, outcome, penalty, eps=1e-6):
    """The invalid-action penalty's share of GraphGPO's episode-level advantage: GraphGPO
    z-scores score_i = outcome_i + penalty_i over all rows of the uid group
    (gigpo.episode_norm_reward, unbiased std); its penalty part is
    (penalty_i - mean_g penalty) / (std_g score + eps). Groups of one row get 0."""
    N = len(uid)
    out = np.zeros(N)
    if penalty is None or not np.any(penalty):
        return out
    idx_of = defaultdict(list)
    for i in range(N):
        idx_of[uid[i]].append(i)
    for idx in idx_of.values():
        if len(idx) < 2:
            continue
        idx = np.array(idx)
        score = outcome[idx] + penalty[idx]
        out[idx] = (penalty[idx] - penalty[idx].mean()) / (score.std(ddof=1) + eps)
    return out


def _group_scale(main_R, sigma_min, reward_scale):
    sd = float(np.std(main_R, ddof=1)) if len(main_R) > 1 else 0.0
    return max(sd, sigma_min * reward_scale, 1e-8)


# ------------------------------------------------------------------------------------------
# Estimators
# ------------------------------------------------------------------------------------------
def progress_advantages(uid, traj_uid, step_idx, anchor, nxt, outcome, penalty, is_main,
                        mode='progress_loo', canon_factory=None, sigma_min=0.2, reward_scale=None,
                        penalty_weight=1.0):
    """Per-row advantages (N,) and diagnostics. See module docstring."""
    N = len(uid)
    canon_factory = canon_factory or make_canonicalizer_factory(False)
    outcome = np.asarray(outcome, dtype=np.float64)
    penalty = np.zeros(N) if penalty is None else np.asarray(penalty, dtype=np.float64)
    if reward_scale is None:
        reward_scale = float(np.max(np.abs(outcome))) if N and np.max(np.abs(outcome)) > 0 else 1.0
    adv = np.zeros(N)
    n_shared = n_after0 = n_nonzero = 0
    for g, trajs in _group_trajectories(uid, traj_uid, step_idx).items():
        canon = canon_factory()
        traj_keys = {tr: _keys(rows, anchor, nxt, canon) for tr, rows in trajs.items()}
        traj_R = {tr: float(outcome[rows[0]]) for tr, rows in trajs.items()}
        stats = _value_stats(traj_keys, traj_R)
        main_R = [traj_R[tr] for tr, rows in trajs.items() if is_main[rows[0]]]
        scale = _group_scale(main_R, sigma_min, reward_scale)

        def V(s):
            st = stats.get(s)
            return st[0] / st[1] if st else 0.0

        for tr, rows in trajs.items():
            ks, R = traj_keys[tr], traj_R[tr]
            base = None
            for j, r in enumerate(rows):
                s, sn = ks[j], ks[j + 1]
                q = R if j == len(rows) - 1 else V(sn)
                if mode == 'progress':
                    b = V(s)
                else:
                    sm, n = stats.get(s, (0.0, 0))
                    if n > 1:
                        base = (sm - R) / (n - 1)
                    elif base is None:
                        base = 0.0
                    b = base
                adv[r] = (q - b) / scale
                n_nonzero += int(abs(q - b) > 1e-9)
                if step_idx[r] > 0 or not is_main[r]:
                    n_after0 += 1
                    n_shared += int(stats.get(s, (0, 0))[1] >= 2)
    adv += penalty_weight * graphgpo_penalty_term(uid, outcome, penalty)
    diag = dict(nonzero_credit_frac=n_nonzero / max(N, 1), shared_state_frac=n_shared / max(n_after0, 1))
    return adv, diag


def tree_grpo_advantages(uid, traj_uid, step_idx, outcome, penalty, is_main, parent, branch_step,
                         eps=1e-6, sigma_min=0.0, reward_scale=1.0, penalty_weight=1.0):
    """Tree-GRPO-style proxy (see module docstring). Trees are rooted at main rollouts."""
    N = len(uid)
    outcome = np.asarray(outcome, dtype=np.float64)
    penalty = np.zeros(N) if penalty is None else np.asarray(penalty, dtype=np.float64)
    adv = np.zeros(N)
    for g, trajs in _group_trajectories(uid, traj_uid, step_idx).items():
        R = {tr: float(outcome[rows[0]]) for tr, rows in trajs.items()}
        par = {tr: (parent[rows[0]] if not is_main[rows[0]] else '') for tr, rows in trajs.items()}
        bstep = {tr: int(branch_step[rows[0]]) for tr, rows in trajs.items()}

        def root(tr):
            seen = set()
            while par.get(tr) and par[tr] in R and tr not in seen:
                seen.add(tr)
                tr = par[tr]
            return tr

        roots = {tr: root(tr) for tr in trajs}
        allR = np.array(list(R.values()))
        mu, sd = allR.mean(), (allR.std(ddof=1) if len(allR) > 1 else 0.0)
        sd = max(sd, sigma_min * reward_scale)
        inter = {tr: (R[tr] - mu) / (sd + eps) for tr in trajs}
        intra = {}
        by_tree = defaultdict(list)
        for tr in trajs:
            by_tree[roots[tr]].append(tr)
        for members in by_tree.values():
            r = np.array([R[m] for m in members])
            if len(members) > 1:
                m_, s_ = r.mean(), max(r.std(ddof=1), sigma_min * reward_scale)
                for m in members:
                    intra[m] = (R[m] - m_) / (s_ + eps)
            else:
                intra[members[0]] = 0.0
        leaf_adv = {tr: intra[tr] + inter[tr] for tr in trajs}
        # children[tr] = [(branch_step, child)]
        children = defaultdict(list)
        for tr in trajs:
            if par[tr] and par[tr] in R:
                children[par[tr]].append((bstep[tr], tr))

        def through(tr, t):
            """Sum of leaf advantages over leaves whose path contains row (tr, step t)."""
            tot = leaf_adv[tr]
            for bs, ch in children.get(tr, ()):
                if bs > t:
                    tot += subtree_sum(ch)
            return tot

        memo = {}

        def subtree_sum(tr):
            if tr not in memo:
                memo[tr] = leaf_adv[tr] + sum(subtree_sum(ch) for _, ch in children.get(tr, ()))
            return memo[tr]

        for tr, rows in trajs.items():
            for r in rows:
                adv[r] = through(tr, int(step_idx[r]))
    adv += penalty_weight * graphgpo_penalty_term(uid, outcome, penalty)
    return adv, {}


# ------------------------------------------------------------------------------------------
# Branch-point selection (runs in the rollout collector after the main rollouts)
# ------------------------------------------------------------------------------------------
def select_branch_points(groups, mode='smart', max_nodes=8, K=1, canon_factory=None, rng=None,
                         min_remaining=1, max_steps=None):
    """groups: {uid: [(env_index, [anchor_0..anchor_{L-1}], [next_0..next_{L-1}]), ...]}
    Returns a list of (uid, env_index, t, state_key): restore env `env_index`'s episode to
    step t (the state before its t-th action) and continue it with the current policy.
      smart:  distinct visited states that are not absorbing (some visit moved elsewhere),
              most visited first; each realised at its earliest-step visit.
      spread: evenly spaced points along each trajectory's distinct-state sequence.
      random: positions drawn uniformly from all (trajectory, step) positions
              (Tree-GRPO-style random expansion).
    Each point is used K times (K continuations). States after which fewer than
    `min_remaining` steps would remain (max_steps - t) are skipped."""
    rng = rng if rng is not None else np.random.default_rng(0)
    canon_factory = canon_factory or make_canonicalizer_factory(False)
    out = []
    for g, trajs in groups.items():
        canon = canon_factory()
        visits, moves, first = defaultdict(int), defaultdict(int), {}
        positions, seqs = [], []
        for env_i, anchors, nexts in trajs:
            seq = []
            for t, (a, n) in enumerate(zip(anchors, nexts)):
                if max_steps is not None and max_steps - t < min_remaining:
                    break
                s = canon.canonicalize(a)
                sn = SUCCESS if to_hashable(n) == SUCCESS else canon.canonicalize(n)
                visits[s] += 1
                moves[s] += int(sn != s)
                positions.append((env_i, t, s))
                if s not in first or t < first[s][1]:
                    first[s] = (env_i, t)
                seq.append(s)
            seqs.append(seq)
        cand = sorted([s for s in visits if moves[s] > 0], key=lambda s: -visits[s])[:max_nodes]
        if mode == 'spread':
            budget = len(cand)
            per = max(1, int(np.ceil(budget / max(len(seqs), 1))))
            picked = []
            for seq in seqs:
                seq = [s for s in dict.fromkeys(seq) if moves[s] > 0]
                if seq:
                    pos = np.unique(np.linspace(0, len(seq) - 1, min(per, len(seq))).round().astype(int))
                    picked += [seq[p] for p in pos]
            cand = list(dict.fromkeys(picked))[:budget]
        if mode == 'random':
            pts = [positions[i] for i in rng.integers(len(positions), size=len(cand))] if positions else []
            out += [(g, e, t, s) for e, t, s in pts for _ in range(K)]
            continue
        out += [(g, first[s][0], first[s][1], s) for s in cand for _ in range(K)]
    return out


def group_shared_step_frac(groups, canon_factory=None):
    """The merge-rate statistic of tools/alfworld_merge_probe.py and the toy's shared_step_frac:
    share of steps t >= 1 whose (canonical) state is visited by >= 2 trajectories of its group.
    groups: {uid: [(env_index, [anchor_0..anchor_{L-1}], [next_0..]), ...]} (main rollouts only)."""
    canon_factory = canon_factory or make_canonicalizer_factory(False)
    shared = tot = 0
    for g, trajs in groups.items():
        canon = canon_factory()
        keyed = [[canon.canonicalize(a) for a in anchors] for _, anchors, _ in trajs]
        vis = defaultdict(set)
        for i, ks in enumerate(keyed):
            for s in ks:
                vis[s].add(i)
        for ks in keyed:
            for s in ks[1:]:
                tot += 1
                shared += int(len(vis[s]) >= 2)
    return shared / max(tot, 1)


# ------------------------------------------------------------------------------------------
# verl adapter
# ------------------------------------------------------------------------------------------
def compute_progress_tree_advantage(data, mode='progress_loo', enable_similarity=False, similarity_thresh=0.95,
                                    sigma_min=0.2, reward_scale=None, invalid_action_penalty_coef=0.0,
                                    use_invalid_action_penalty=True, penalty_weight=1.0):
    """Reads a verl DataProto (after the reward/penalty step) and returns
    (advantages, returns, metrics) with advantages broadcast over response tokens."""
    import torch
    nt = data.non_tensor_batch
    N = len(nt['uid'])
    response_mask = data.batch['response_mask']
    row_score = data.batch['token_level_rewards'].sum(dim=-1).detach().cpu().numpy().astype(np.float64)
    valid = nt['is_action_valid'].astype(np.float64) if 'is_action_valid' in nt else np.ones(N)
    coef = invalid_action_penalty_coef if use_invalid_action_penalty else 0.0
    penalty = -coef * (1.0 - valid)
    outcome_rows = row_score - penalty                       # reward-manager score per trajectory
    traj_uid = nt['traj_uid']
    # one outcome per trajectory (rows of a trajectory share it; take the first row's)
    first = {}
    for i in range(N):
        first.setdefault(traj_uid[i], outcome_rows[i])
    outcome = np.array([first[traj_uid[i]] for i in range(N)])
    step_idx = nt['step_idx'].astype(np.int64) if 'step_idx' in nt else np.arange(N)
    is_main = ~nt['is_branch'].astype(bool) if 'is_branch' in nt else np.ones(N, bool)
    parent = nt['parent_traj_uid'] if 'parent_traj_uid' in nt else np.array([''] * N, dtype=object)
    branch_step = nt['branch_step'].astype(np.int64) if 'branch_step' in nt else -np.ones(N, np.int64)
    if mode == 'tree_grpo':
        adv, diag = tree_grpo_advantages(nt['uid'], traj_uid, step_idx, outcome, penalty, is_main, parent, branch_step,
                                         penalty_weight=penalty_weight)
    else:
        adv, diag = progress_advantages(nt['uid'], traj_uid, step_idx, nt['anchor_obs'], nt['next_obs'], outcome,
                                        penalty, is_main, mode=mode,
                                        canon_factory=make_canonicalizer_factory(enable_similarity, similarity_thresh),
                                        sigma_min=sigma_min, reward_scale=reward_scale, penalty_weight=penalty_weight)
    scores = torch.tensor(adv, dtype=torch.float32, device=response_mask.device).unsqueeze(-1) * response_mask
    metrics = {f'progress_tree/{k}': v for k, v in diag.items()}
    metrics['progress_tree/adv_abs_mean'] = float(np.abs(adv).mean()) if N else 0.0
    metrics['progress_tree/branch_row_frac'] = float((~is_main).mean()) if N else 0.0
    return scores, scores, metrics
