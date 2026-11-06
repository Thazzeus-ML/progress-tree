"""Headless runner for the toy lab's main comparison (notebook section 3), parallel over CPU cores.

Every (arm, seed) run trains to the same environment-step budget (branch and probe steps
count toward it). The script then reports IQM eval success at equal environment steps and
at equal iterations, with 95% bootstrap CIs and the notebook's pre-registered verdict
(flagship beats baseline by >= 0.03 IQM with separated CIs). It also prints a two-sided
permutation p-value on per-seed final success as a second check.

    python scripts/run_main.py --preset replicate --out results/replicate
    python scripts/run_main.py --arms GraphGPO Progress+Tree(LOO) --seeds 0 1 2 --budget 300000 --out results/quick

An arm can carry config overrides after '@', e.g. 'Progress+Tree(LOO)@branch_mode=consistent' or
'GraphGPO@gamma=0.8,merge_noise=0.15'; the full label is the arm's name in the outputs.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'toy'))

PRESETS = {
    # same arms/seeds/budgets as the notebook's presets (section 3 only)
    'smoke': dict(budget=150_000, seeds=[0, 1], eval_points=5,
                  arms=['GraphGPO', 'Tree+GraphGPO', 'Progress+Tree', 'Progress+Tree(LOO)']),
    'fast': dict(budget=1_200_000, seeds=[0, 1, 2], eval_points=20,
                 arms=['GraphGPO', 'GraphGPO(2xG)', 'Tree+GRPO', 'Tree+GraphGPO', 'Progress+Tree',
                       'Progress+Tree(LOO)', 'Progress+Tree(random)']),
    'full': dict(budget=2_400_000, seeds=[0, 1, 2, 3, 4], eval_points=24,
                 arms=['GRPO', 'GraphGPO', 'GraphGPO(2xG)', 'GraphGPO+FrontierMC', 'Tree+GRPO', 'Tree+GraphGPO',
                       'Progress+Tree', 'Progress+Tree(LOO)', 'Progress+Tree(random)']),
    'replicate': dict(budget=2_400_000, seeds=list(range(10)), eval_points=24,
                      arms=['GraphGPO', 'GraphGPO(2xG)', 'Tree+GraphGPO', 'Progress+Tree', 'Progress+Tree(LOO)'],
                      task_seed_train=3000, task_seed_eval=4000),
}
FLAGSHIPS = ['Progress+Tree', 'Progress+Tree(LOO)']
BASELINES = ['GraphGPO', 'GraphGPO(2xG)']


def _job(args):
    import warnings
    import torch
    import h2lab_core as H
    warnings.filterwarnings('ignore')
    torch.set_num_threads(1)
    label, seed, cfg = args
    arm, over = parse_arm(label)
    cfg = dict(cfg, **over)
    t = time.time()
    _, log = H.train_arm(arm, seed, cfg)
    for l in log:
        l['arm'] = label
    return log, time.time() - t


def parse_arm(label):
    """'Progress+Tree(LOO)@branch_mode=consistent,absorb_phys=True' -> (arm, {cfg overrides})."""
    arm, _, rest = label.partition('@')
    over = {}
    for kv in filter(None, rest.split(',')):
        k, v = kv.split('=', 1)
        for conv in (int, float):
            try:
                v = conv(v)
                break
            except ValueError:
                pass
        else:
            v = {'True': True, 'False': False}.get(v, v)
        over[k] = v
    return arm, over


def iqm(x):
    x = np.sort(np.asarray(x, float))
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    lo, hi = int(np.floor(0.25 * len(x))), int(np.ceil(0.75 * len(x)))
    return x[lo:hi].mean() if hi > lo else x.mean()


def boot_ci(x, stat=iqm, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    return tuple(np.percentile([stat(rng.choice(x, len(x))) for _ in range(n)], [2.5, 97.5]))


def at_budget(ev, xcol, budget):
    rows = []
    for (arm, seed), d in ev.groupby(['arm', 'seed']):
        d = d.sort_values(xcol)
        rows.append(dict(arm=arm, seed=seed, success=np.interp(budget, d[xcol], d.eval_success, right=np.nan)))
    return pd.DataFrame(rows)


def table(ev, xcol, budget, arms):
    d = at_budget(ev, xcol, budget)
    out = d.groupby('arm').success.agg(IQM=iqm, mean='mean', n='count')
    out['CI95'] = [tuple(np.round(boot_ci(d[d.arm == a].success.values), 3)) for a in out.index]
    return out.loc[[a for a in arms if a in out.index]], d


def verdict(t, a, b):
    if a not in t.index or b not in t.index:
        return None
    diff = t.loc[a, 'IQM'] - t.loc[b, 'IQM']
    sep = t.loc[a, 'CI95'][0] > t.loc[b, 'CI95'][1]
    return diff, sep, ('PASS' if diff >= 0.03 and sep else 'not demonstrated')


def perm_p(x, y, n=20000, seed=0):
    """Two-sided permutation test on the difference of means of per-seed scores."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = abs(x.mean() - y.mean())
    z = np.concatenate([x, y])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        rng.shuffle(z)
        hits += abs(z[:len(x)].mean() - z[len(x):].mean()) >= obs - 1e-12
    return (hits + 1) / (n + 1)


def report(ev, arms, budget):
    lines = []
    B_steps = min(budget, ev.groupby(['arm', 'seed']).env_steps.max().min())
    B_iter = int(ev.groupby(['arm', 'seed']).iter.max().min())
    for name, xcol, B in (('environment steps', 'env_steps', B_steps), ('iterations', 'iter', B_iter)):
        t, d = table(ev, xcol, B, arms)
        lines.append(f"\n## At equal {name} ({B:,.0f})\n")
        lines.append('| arm | IQM | mean | CI95 | n |\n|---|---|---|---|---|')
        for a, r in t.iterrows():
            lines.append(f"| {a} | {r.IQM:.3f} | {r['mean']:.3f} | ({r.CI95[0]:.3f}, {r.CI95[1]:.3f}) | {int(r.n)} |")
        lines.append('')
        for f in FLAGSHIPS:
            for b in BASELINES:
                v = verdict(t, f, b)
                if v:
                    p = perm_p(d[d.arm == f].success.values, d[d.arm == b].success.values)
                    lines.append(f"- `{f}` vs `{b}`: {v[0]:+.3f} -> **{v[2]}** (CIs separated: {v[1]}; permutation p = {p:.4f})")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', choices=sorted(PRESETS), default=None)
    ap.add_argument('--arms', nargs='+')
    ap.add_argument('--seeds', nargs='+', type=int)
    ap.add_argument('--budget', type=int)
    ap.add_argument('--eval-points', type=int)
    ap.add_argument('--task-seed-train', type=int)
    ap.add_argument('--task-seed-eval', type=int)
    ap.add_argument('--merge-noise', type=float, default=0.0)
    ap.add_argument('--update-mode', choices=['fixed_minibatch', 'fixed_steps'], default=None)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    import h2lab_core as H
    P = dict(PRESETS[a.preset]) if a.preset else dict(budget=300_000, seeds=[0, 1, 2], eval_points=10,
                                                      arms=['GraphGPO', 'Progress+Tree(LOO)'])
    for k in ('arms', 'seeds', 'budget', 'eval_points', 'task_seed_train', 'task_seed_eval'):
        v = getattr(a, k)
        if v is not None:
            P[k] = v
    cfg = dict(H.DEFAULT_CFG)
    cfg.update(env_budget=P['budget'], eval_points=P['eval_points'], merge_noise=a.merge_noise,
               task_seed_train=P.get('task_seed_train', cfg['task_seed_train']),
               task_seed_eval=P.get('task_seed_eval', cfg['task_seed_eval']))
    if a.update_mode:
        cfg['update_mode'] = a.update_mode
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'config.json'), 'w') as f:
        json.dump(dict(preset=a.preset, arms=P['arms'], seeds=P['seeds'], cfg=cfg), f, indent=1)

    jobs = [(arm, s, cfg) for s in P['seeds'] for arm in P['arms']]
    print(f"{len(jobs)} runs ({len(P['arms'])} arms x {len(P['seeds'])} seeds), budget {P['budget']:,} env steps, "
          f"{a.workers} workers, merge_noise {a.merge_noise}", flush=True)
    logs, t0 = [], time.time()
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for i, (log, dt) in enumerate(pool.imap_unordered(_job, jobs)):
            logs += log
            last = [l for l in log if 'eval_success' in l][-1]
            print(f"[{i + 1}/{len(jobs)}] {log[0]['arm']:22s} seed {log[0]['seed']:2d} {dt:6.0f}s  iters {last['iter']:4d}  "
                  f"env {last['env_steps']:>9,}  final eval {last['eval_success']:.3f}  ({time.time() - t0:.0f}s elapsed)",
                  flush=True)
            pd.DataFrame(logs).to_csv(os.path.join(a.out, 'runs.csv'), index=False)   # checkpoint
    df = pd.DataFrame(logs)
    ev = df[df.eval_success.notna()][['arm', 'seed', 'iter', 'env_steps', 'eval_success']]
    ev.to_csv(os.path.join(a.out, 'eval.csv'), index=False)
    md = (f"# Main comparison: {len(P['arms'])} arms x {len(P['seeds'])} seeds, {P['budget']:,} env steps, "
          f"merge noise {a.merge_noise}, task seeds {cfg['task_seed_train']}/{cfg['task_seed_eval']}, "
          f"update mode {cfg['update_mode']}\n" + report(ev, P['arms'], P['budget']) +
          f"\n\nWall time: {time.time() - t0:.0f}s on {a.workers} workers.\n")
    with open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write(md)
    print(md)


if __name__ == '__main__':
    main()
