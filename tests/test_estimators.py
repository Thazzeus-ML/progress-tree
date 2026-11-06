"""Independent checks of the credit estimators against exact values (no shared code with the
author's tests in recipe/ProgressTree/tests).

The main check uses a 4-state MDP whose true advantages Q(s,a) - V(s) are known in closed
form, and verifies that progress credit with leave-one-out baselines converges to them, that
the plain (non-LOO) version is shrunk toward zero as expected, and that the verl-agent port
computes the same numbers as the toy library.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'toy'))
sys.path.insert(0, os.path.join(ROOT, 'recipe', 'ProgressTree'))

import h2lab_core as H          # noqa: E402
import core_progress as cp      # noqa: E402

# ---------------------------------------------------------------------------------------
# Tiny MDP, horizon 3, uniform random policy over 2 actions:
#   A --a0--> B    A --a1--> C
#   B --a0--> SUCCESS          B --a1--> D (dead end, self-loop)
#   C --a0--> B    C --a1--> D
# V(B) = 1/2, V(C) = 1/4, V(A) = 3/8 (every B is reached with at least one step left).
# ---------------------------------------------------------------------------------------
A_, B_, C_, D_ = 10, 11, 12, 13
SUCC = H.SUCCESS
NEXT = {(A_, 0): B_, (A_, 1): C_, (B_, 0): SUCC, (B_, 1): D_, (C_, 0): B_, (C_, 1): D_, (D_, 0): D_, (D_, 1): D_}
HORIZON = 3
TRUE_ADV = {(A_, 0): 0.125, (A_, 1): -0.125, (B_, 0): 0.5, (B_, 1): -0.5,
            (C_, 0): 0.25, (C_, 1): -0.25, (D_, 0): 0.0, (D_, 1): 0.0}


def sample_batch(n_groups, G, rng):
    """Batch in the toy library's layout: keys [T+1, B], active/actions [T, B], success [B]."""
    B = n_groups * G
    keys = np.zeros((HORIZON + 1, B), np.int64)
    active = np.zeros((HORIZON, B), bool)
    actions = np.zeros((HORIZON, B), np.int64)
    success = np.zeros(B, bool)
    for b in range(B):
        s = A_
        keys[0, b] = s
        for t in range(HORIZON):
            if s == SUCC:
                keys[t + 1, b] = keys[t, b]
                continue
            a = int(rng.integers(2))
            sn = NEXT[(s, a)]
            active[t, b], actions[t, b] = True, a
            keys[t + 1, b] = sn
            s = sn
        success[b] = s == SUCC
    group = np.repeat(np.arange(n_groups), G)
    return dict(keys=keys, active=active, actions=actions, success=success), group


def mean_adv_by_pair(batch, A):
    sums, cnts = {}, {}
    tt, bb = np.nonzero(batch['active'])
    for t, b in zip(tt, bb):
        k = (int(batch['keys'][t, b]), int(batch['actions'][t, b]))
        sums[k] = sums.get(k, 0.0) + A[t, b]
        cnts[k] = cnts.get(k, 0) + 1
    return {k: sums[k] / cnts[k] for k in sums}


@pytest.fixture(scope='module')
def big_batch():
    return sample_batch(n_groups=3000, G=32, rng=np.random.default_rng(0))


def test_progress_loo_converges_to_true_advantages(big_batch):
    batch, group = big_batch
    unit = {int(g): 1.0 for g in np.unique(group)}
    est = mean_adv_by_pair(batch, H.progress_loo_advantages(batch, group, unit))
    for k, v in TRUE_ADV.items():
        assert est[k] == pytest.approx(v, abs=0.01), (k, est[k], v)


def test_plain_progress_is_shrunk_toward_zero_at_branching_states(big_batch):
    """Including the trajectory's own outcome in V(s_t) biases the baseline toward that outcome,
    so |E[A]| < |Q - V| at states where the action decides the outcome (B). LOO removes this."""
    batch, group = big_batch
    unit = {int(g): 1.0 for g in np.unique(group)}
    plain = mean_adv_by_pair(batch, H.progress_advantages(batch, group, unit))
    for k in [(B_, 0), (B_, 1)]:
        assert abs(plain[k]) < abs(TRUE_ADV[k]) - 0.005
        assert np.sign(plain[k]) == np.sign(TRUE_ADV[k])


def test_grpo_does_not_localise_credit(big_batch):
    """Reference point: GRPO broadcasts R - mean(R) to every step, so its per-step mean at the
    decisive state B is far from the true advantage (it is still an unbiased policy gradient)."""
    batch, group = big_batch
    est = mean_adv_by_pair(batch, H.grpo_advantages(batch, group))
    # GRPO also divides by the group std, so compare after undoing the scale roughly: just check
    # that the dead-end self-loop D gets non-zero credit, which a localised estimator never gives.
    assert abs(est[(D_, 0)]) > 0.1
    loo = mean_adv_by_pair(batch, H.progress_loo_advantages(batch, group, {int(g): 1.0 for g in np.unique(group)}))
    assert abs(loo[(D_, 0)]) < 1e-12


def test_loo_without_merging_is_rloo_on_every_step():
    """If no state except the start is shared, LOO progress credit gives every step of trajectory i
    the RLOO advantage R_i - mean_{j != i} R_j (divided by the group scale)."""
    rng = np.random.default_rng(1)
    G, T = 6, 5
    keys = np.zeros((T + 1, G), np.int64)
    keys[0] = 1                                   # shared start state
    keys[1:] = 1000 + np.arange(T * G).reshape(T, G)   # unique afterwards
    R = rng.integers(0, 2, G).astype(bool)
    R[0], R[1] = True, False
    keys[T, R] = SUCC
    batch = dict(keys=keys, active=np.ones((T, G), bool), success=R)
    group = np.zeros(G, int)
    A = H.progress_loo_advantages(batch, group, {0: 1.0})
    for i in range(G):
        rloo = R[i] - np.delete(R, i).mean()
        np.testing.assert_allclose(A[:-1, i], rloo, atol=1e-12)          # interior steps
        np.testing.assert_allclose(A[-1, i], float(R[i]) - np.delete(R, i).mean(), atol=1e-12)


def test_port_matches_toy_on_the_tabular_mdp():
    """recipe/ProgressTree/core_progress.py (row layout used inside verl) == toy library."""
    batch, group = sample_batch(n_groups=50, G=8, rng=np.random.default_rng(2))
    scale = H.group_scale(batch, group, 0.2)
    for mode, fn in (('progress_loo', H.progress_loo_advantages), ('progress', H.progress_advantages)):
        A_toy = fn(batch, group, scale)
        rows = []
        tt, bb = np.nonzero(batch['active'])
        for t, b in zip(tt, bb):
            sn = int(batch['keys'][t + 1, b])
            rows.append((group[b], f't{b}', t, int(batch['keys'][t, b]), 'success' if sn == SUCC else sn,
                         float(batch['success'][b]), A_toy[t, b]))
        uid, traj, step, anc, nxt, out, want = zip(*rows)    # keep anc/nxt as Python objects ('success' vs ints)
        uid, traj, step, out, want = map(np.array, (uid, traj, step, out, want))
        got, _ = cp.progress_advantages(uid, traj, step, list(anc), list(nxt), out, None,
                                        np.ones(len(uid), bool), mode=mode, sigma_min=0.2, reward_scale=1.0)
        np.testing.assert_allclose(got, want.astype(float), atol=1e-12)


def test_branch_steps_are_charged_to_the_budget():
    """compute_advantages for a tree arm reports exactly the branch rollouts' active steps as cost."""
    cfg = dict(H.DEFAULT_CFG)
    H.set_seed(0)
    tasks = H.KeyDoorTasks(16, cfg['size'], cfg['n_traps'], seed=7)
    pol = H.MLPPolicy(tasks.obs_dim, hidden=32)
    G, n = cfg['group_size'], 4
    ids = np.repeat(np.arange(n), G)
    group = np.repeat(np.arange(n), G)
    batch = H.rollout(tasks, ids, pol, cfg['horizon'], 'cpu')
    for arm in ('Progress+Tree(LOO)', 'Tree+GraphGPO'):
        A, extra, cost, diag = H.compute_advantages(arm, batch, group, tasks, pol, cfg, 'cpu',
                                                    dict(cache={}, start={}), np.random.default_rng(0))
        assert extra is not None and cost > 0
        assert cost == int(extra['active'].sum())
        assert diag['branch_steps'] == cost


def test_bundled_toy_copy_matches_toy_library():
    a = open(os.path.join(ROOT, 'toy', 'h2lab_core.py'), encoding='utf-8').read().strip()
    b = open(os.path.join(ROOT, 'recipe', 'ProgressTree', 'tests', 'toy_core.py'), encoding='utf-8').read().strip()
    assert a == b


# ---------------------------------------------------------------------------------------
# Options added during validation (defaults keep the v2 behaviour; see docs/validation.md)
# ---------------------------------------------------------------------------------------
def _toy_batch(noise=0.0, seed=0, n=4):
    cfg = dict(H.DEFAULT_CFG, merge_noise=noise)
    H.set_seed(seed)
    tasks = H.KeyDoorTasks(16, cfg['size'], cfg['n_traps'], seed=7)
    pol = H.MLPPolicy(tasks.obs_dim, hidden=32)
    ids = np.repeat(np.arange(n), cfg['group_size'])
    group = np.repeat(np.arange(n), cfg['group_size'])
    return cfg, tasks, pol, H.rollout(tasks, ids, pol, cfg['horizon'], 'cpu', merge_noise=noise), group


def test_consistent_branches_respect_the_episode_clock():
    cfg, tasks, pol, batch, group = _toy_batch()
    nodes, sk = H.select_branch_nodes(batch, group, cfg, 'smart', None, np.random.default_rng(0))
    t0 = np.asarray(sk['_t'])
    for (g, task, s), t in zip(nodes, t0):          # each branch point is a real visit at that step
        cols = np.where(group == g)[0]
        assert any(batch['keys'][t, c] == s and batch['active'][t, c] for c in cols)
    _, br, cost = H.mc_continuations(tasks, nodes, pol, 1, cfg['horizon'], 'cpu', init_t=t0)
    lengths = br['active'].sum(0)
    assert np.all(lengths <= cfg['horizon'] - t0)
    assert np.any(lengths > cfg['mc_horizon'])       # early branch points get more than the toy's 20 steps


def test_light_concat_keeps_branches_longer_than_the_main_batch():
    cfg, tasks, pol, batch, group = _toy_batch()
    short = {k: (v[:6] if k in ('active', 'actions', 'logp', 'reward', 'obs') else v) for k, v in batch.items()}
    short['keys'] = batch['keys'][:7]
    comb, gcomb = H.light_concat(short, batch, group, group)
    assert comb['active'].shape[0] == batch['active'].shape[0]
    assert comb['active'][:, len(group):].sum() == batch['active'].sum()


def test_physical_absorbing_test_skips_trapped_states_under_merge_noise():
    for seed in range(6):
        cfg, tasks, pol, batch, group = _toy_batch(noise=0.5, seed=seed, n=8)
        nodes, _ = H.select_branch_nodes(batch, group, dict(cfg, absorb_phys=True), 'smart', None, None)
        trapped = [s for _, _, s in nodes if H.decode_key(s, cfg['size'])[4] == 1]
        assert not trapped
        nodes_v2, _ = H.select_branch_nodes(batch, group, cfg, 'smart', None, None)
        if any(H.decode_key(s, cfg['size'])[4] == 1 for _, _, s in nodes_v2):
            return                                   # the v2 rule does pick trapped states here
    pytest.fail('expected the v2 rule to pick a trapped state in at least one noisy batch')


def test_random_filtered_never_branches_from_absorbing_states():
    cfg, tasks, pol, batch, group = _toy_batch(n=8)
    nodes, _ = H.select_branch_nodes(batch, group, cfg, 'random_filtered', None, np.random.default_rng(0))
    assert nodes and not [s for _, _, s in nodes if H.decode_key(s, cfg['size'])[4] == 1]


def test_graph_relative_returns_give_the_same_step_advantages_without_underflow():
    cfg, tasks, pol, batch, group = _toy_batch(n=8)
    for gamma in (0.5, 0.8):                         # no underflow at these gammas: must agree exactly
        r1, _ = H.graph_step_returns(batch, group, gamma=gamma)
        r2, _ = H.graph_step_returns(batch, group, gamma=gamma, relative=True)
        a1 = H.graphgpo_advantages(batch, group, r1)
        a2 = H.graphgpo_advantages(batch, group, r2)
        np.testing.assert_allclose(a1, a2, atol=1e-6)


def test_gigpo_step_term_matches_hand_computation():
    keys = np.array([[1, 1], [2, 3], [SUCC, 3]])     # traj 0: 1->2->goal ; traj 1: 1->3->3 (fail)
    batch = dict(keys=keys, active=np.ones((2, 2), bool), success=np.array([True, False]))
    A = H.gigpo_advantages(batch, np.zeros(2, int), gamma=0.5, w_step=1.0, w_ep=0.0)
    # anchor 1 at t=0 is shared: returns 0.5 (traj 0) vs 0 (traj 1) -> +-0.7071 after z-scoring
    np.testing.assert_allclose(A[0], [1 / np.sqrt(2), -1 / np.sqrt(2)], atol=1e-5)
    np.testing.assert_allclose(A[1], [0.0, 0.0])     # singleton anchor groups get 0


def test_port_core_tests_pass():
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'recipe', 'ProgressTree', 'tests', 'test_core_progress.py')],
                       capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0, r.stdout + r.stderr
