"""Unit tests for core_progress (no verl needed).

1. The port's progress / progress_loo estimators reproduce the toy library's (h2lab core.py)
   advantages exactly on toy batches, with and without merge noise, with and without branches.
2. Progress credit telescopes: per main trajectory, sum_t A_t * scale = R - V(s_0).
3. select_branch_points('smart') picks the same states as the toy's select_branch_nodes.
4. Tree-GRPO proxy on a hand-built tree.

usage: python tests/test_core_progress.py [/path/to/h2lab]   (default: the bundled tests/toy_core.py)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])
    import core                  # noqa: E402  (toy library from an h2lab folder)
else:
    sys.path.insert(0, HERE)
    import toy_core as core      # noqa: E402  (bundled copy of the h2lab v2.2 toy library)
import core_progress as cp       # noqa: E402


def to_rows(batch, group, B_main):
    """Toy batch columns -> port rows."""
    T, B = batch['active'].shape
    rows = dict(uid=[], traj=[], step=[], anchor=[], nxt=[], outcome=[], is_main=[], col=[], t=[])
    for b in range(B):
        for t in range(T):
            if not batch['active'][t, b]:
                break
            s, sn = int(batch['keys'][t, b]), int(batch['keys'][t + 1, b])
            rows['uid'].append(int(group[b])); rows['traj'].append(b); rows['step'].append(t)
            rows['anchor'].append(s); rows['nxt'].append(cp.SUCCESS if sn == core.SUCCESS else sn)
            rows['outcome'].append(float(batch['success'][b])); rows['is_main'].append(b < B_main)
            rows['col'].append(b); rows['t'].append(t)
    return {k: np.array(v, dtype=object if k in ('anchor', 'nxt') else None) for k, v in rows.items()}


def port_adv(r, mode, sigma_min=0.2):
    adv, _ = cp.progress_advantages(r['uid'], r['traj'], r['step'], r['anchor'], r['nxt'], r['outcome'],
                                    None, r['is_main'], mode=mode, sigma_min=sigma_min, reward_scale=1.0)
    return adv


_TRAINED = {}


def trained_policy():
    """A small policy that succeeds ~50% on 6x6 tasks, so test batches have real credit."""
    if not _TRAINED:
        cfg = dict(core.DEFAULT_CFG)
        cfg.update(size=6, n_traps=0, env_budget=150_000, eval_points=1, hidden=64, n_train_tasks=32, n_eval_tasks=32)
        tasks = core.KeyDoorTasks(32, 6, 0, seed=7)
        policy, _ = core.train_arm('GraphGPO', 0, cfg, train_tasks=tasks, eval_tasks=tasks)
        _TRAINED.update(tasks=tasks, policy=policy)
    return _TRAINED['tasks'], _TRAINED['policy']


def toy_batch(noise, seed, n_tasks=6, G=8, branches=False):
    cfg = dict(core.DEFAULT_CFG)
    cfg.update(merge_noise=noise, size=6, n_traps=0)
    tasks, policy = trained_policy()
    core.set_seed(seed)
    ids = np.repeat(np.arange(n_tasks), G)
    group = np.repeat(np.arange(n_tasks), G)
    batch = core.rollout(tasks, ids, policy, cfg['horizon'], 'cpu', merge_noise=noise)
    scale = core.group_scale(batch, group, cfg['progress_sigma_min'])
    if not branches:
        return batch, group, scale, batch['active'].shape[1], cfg, tasks, policy
    nodes, _ = core.select_branch_nodes(batch, group, cfg, 'smart', None, np.random.default_rng(seed))
    _, br, _ = core.mc_continuations(tasks, nodes, policy, cfg['branch_K'], cfg['mc_horizon'], 'cpu', noise)
    comb, gcomb = core.light_concat(batch, br, group, br['group'])
    return comb, gcomb, scale, batch['active'].shape[1], cfg, tasks, policy


def check_equivalence():
    worst, nonzero, succ = 0.0, [], []
    for noise in (0.0, 0.5, 0.9):
        for seed in (0, 1):
            for branches in (False, True):
                b, g, scale, B_main, *_ = toy_batch(noise, seed, branches=branches)
                r = to_rows(b, g, B_main)
                succ.append(float(b['success'].mean()))
                for mode, fn in (('progress_loo', core.progress_loo_advantages), ('progress', core.progress_advantages)):
                    A = fn(b, g, scale)
                    want = np.array([A[t, c] for t, c in zip(r['t'], r['col'])])
                    got = port_adv(r, mode)
                    worst = max(worst, float(np.abs(want - got).max()))
                    nonzero.append(float(np.mean(np.abs(want) > 1e-12)))
    assert worst < 1e-9, worst
    assert min(nonzero) > 0.05 and min(succ) > 0.1, (nonzero, succ)   # the batches carry real credit
    print(f'[ok] port == toy estimator on 24 batch/mode combos (max |diff| {worst:.1e}; '
          f'success {np.mean(succ):.2f}, nonzero credit {np.mean(nonzero):.2f} of rows)')


def check_telescoping():
    b, g, scale, B_main, *_ = toy_batch(0.0, 3)
    r = to_rows(b, g, B_main)
    adv = port_adv(r, 'progress')
    stats = core.value_stats(b, g)
    worst = 0.0
    for col in range(B_main):
        m = r['col'] == col
        if not m.any():
            continue
        gg = int(g[col]); s0 = int(b['keys'][0, col])
        lhs = adv[m].sum() * scale[gg]
        rhs = float(b['success'][col]) - core._V(stats, gg, s0)
        worst = max(worst, abs(lhs - rhs))
    assert worst < 1e-9, worst
    print(f'[ok] progress credit telescopes to R - V(s0) (max error {worst:.1e})')


def check_branch_selection():
    b, g, scale, B_main, cfg, *_ = toy_batch(0.0, 5)
    toy_nodes, _ = core.select_branch_nodes(b, g, cfg, 'smart', None, np.random.default_rng(0))
    groups = {}
    T = b['active'].shape[0]
    for col in range(B_main):
        act = [t for t in range(T) if b['active'][t, col]]
        anchors = [int(b['keys'][t, col]) for t in act]
        nexts = [cp.SUCCESS if int(b['keys'][t + 1, col]) == core.SUCCESS else int(b['keys'][t + 1, col]) for t in act]
        groups.setdefault(int(g[col]), []).append((col, anchors, nexts))
    pts = cp.select_branch_points(groups, mode='smart', max_nodes=cfg['branch_max_nodes'], K=1)
    port = [(gg, s) for gg, _, _, s in pts]
    toy = [(gg, s) for gg, _, s in toy_nodes]
    assert port == toy, (port[:5], toy[:5])
    # each point restores to a real visit of that state
    for gg, e, t, s in pts:
        assert int(b['keys'][t, e]) == s
    print(f'[ok] smart branch points identical to the toy ({len(pts)} points)')


def check_merge_statistic():
    for noise in (0.0, 0.5):
        b, g, scale, B_main, cfg, *_ = toy_batch(noise, 2)
        groups = {}
        T = b['active'].shape[0]
        for col in range(B_main):
            act = [t for t in range(T) if b['active'][t, col]]
            groups.setdefault(int(g[col]), []).append((col, [int(b['keys'][t, col]) for t in act], [0] * len(act)))
        want = core.shared_step_frac(b, g)
        got = cp.group_shared_step_frac(groups)
        assert abs(want - got) < 1e-12, (noise, want, got)
    print('[ok] merge-rate statistic identical to the toy\'s shared_step_frac')


def check_tree_grpo():
    # group 'g': main a (R=1, 3 steps), main b (R=0, 3 steps), branch c from a at step 1 (R=0, 2 steps)
    uid = np.array(['g'] * 8, dtype=object)
    traj = np.array(['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c'], dtype=object)
    step = np.array([0, 1, 2, 0, 1, 2, 1, 2])
    R = np.array([1, 1, 1, 0, 0, 0, 0, 0], float)
    is_main = np.array([1, 1, 1, 1, 1, 1, 0, 0], bool)
    parent = np.array(['', '', '', '', '', '', 'a', 'a'], dtype=object)
    bstep = np.array([-1, -1, -1, -1, -1, -1, 1, 1])
    adv, _ = cp.tree_grpo_advantages(uid, traj, step, R, None, is_main, parent, bstep, eps=0.0)
    allR = np.array([1.0, 0.0, 0.0]); mu, sd = allR.mean(), allR.std(ddof=1)
    inter = {k: (v - mu) / sd for k, v in dict(a=1.0, b=0.0, c=0.0).items()}
    tr = np.array([1.0, 0.0]); intra_a, intra_c = (1 - tr.mean()) / tr.std(ddof=1), (0 - tr.mean()) / tr.std(ddof=1)
    leaf = dict(a=intra_a + inter['a'], b=inter['b'], c=intra_c + inter['c'])
    want = np.array([leaf['a'] + leaf['c'], leaf['a'], leaf['a'], leaf['b'], leaf['b'], leaf['b'], leaf['c'], leaf['c']])
    assert np.allclose(adv, want), (adv, want)
    print('[ok] Tree-GRPO proxy: shared prefix row gets the sum over its leaves')


def check_similarity_canonicalizer():
    from difflib import SequenceMatcher
    rng = np.random.default_rng(0)
    base = ''.join(rng.choice(list('abcdefgh '), 300))
    states = [base]
    for k in range(40):
        s = list(base)
        for _ in range(int(rng.integers(0, 25))):
            s[int(rng.integers(len(s)))] = str(rng.choice(list('xyz')))
        states.append(''.join(s))

    class Ref:   # GraphGPO's StateCanonicalizer, verbatim logic
        def __init__(self, th):
            self.th, self.cl, self.cache = th, [], {}

        def canonicalize(self, s):
            if s in self.cache:
                return self.cache[s]
            for h, cs in self.cl:
                if SequenceMatcher(None, s, cs).ratio() >= self.th:
                    self.cache[s] = h
                    return h
            self.cl.append((s, s)); self.cache[s] = s
            return s

    ref, fast = Ref(0.95), cp.FastCanonicalizer(True, 0.95)
    assert [ref.canonicalize(s) for s in states] == [fast.canonicalize(s) for s in states]
    print('[ok] FastCanonicalizer gives the same clusters as GraphGPO\'s StateCanonicalizer')


if __name__ == '__main__':
    check_equivalence()
    check_telescoping()
    check_branch_selection()
    check_merge_statistic()
    check_tree_grpo()
    check_similarity_canonicalizer()
    print('all core_progress tests passed')
