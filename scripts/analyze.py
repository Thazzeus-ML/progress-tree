"""Paired comparisons between arms of scripts/run_main.py runs.

Arms that share a seed share initial weights and (except random-branch arms without
branch_rng=separate) the training task sequence, so each comparison is made seed by seed:
mean paired difference with a t-based and a bootstrap 95% CI, and wins/seeds.

Optionally charges the replay cost of restoring branch states (the toy restores for free;
the verl-agent port replays the parent's actions): x = env_steps + w * cumulative replay_steps.

    python scripts/analyze.py results/e1e5_pool1000 results/replicate_a \
        --pair "Progress+Tree(LOO)@branch_mode=consistent" GraphGPO --budget 2400000 --replay-w 0 0.25 1
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats


def load_runs(dirs):
    dfs = []
    for d in dirs:
        df = pd.read_csv(os.path.join(d, 'runs.csv'))
        df['source'] = os.path.basename(os.path.normpath(d))
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    if 'replay_steps' not in df:
        df['replay_steps'] = 0.0
    df['replay_steps'] = df['replay_steps'].fillna(0.0)
    df = df.sort_values(['arm', 'seed', 'iter', 'eval_success'], na_position='first')
    df['replay_cum'] = df.groupby(['arm', 'seed'])['replay_steps'].cumsum()
    return df


def success_at(df, arm, budget, w=0.0):
    """{seed: eval success interpolated at x = budget}, x = env_steps + w * replay."""
    ev = df[(df.arm == arm) & df.eval_success.notna()].copy()
    ev['x'] = ev.env_steps + w * ev.replay_cum
    out = {}
    for s, d in ev.groupby('seed'):
        d = d.sort_values('x')
        out[int(s)] = float(np.interp(budget, d.x, d.eval_success, right=np.nan))
    return out


def paired(a, b, n_boot=20000, seed=0):
    common = sorted(k for k in set(a) & set(b) if not (np.isnan(a[k]) or np.isnan(b[k])))
    d = np.array([a[k] - b[k] for k in common])
    if len(d) < 2:
        return dict(n=len(d))
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    tq = stats.t.ppf(0.975, len(d) - 1)
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, (n_boot, len(d))).mean(1)
    return dict(n=len(d), mean_a=np.mean([a[k] for k in common]), mean_b=np.mean([b[k] for k in common]),
                diff=m, t_lo=m - tq * se, t_hi=m + tq * se, b_lo=np.percentile(bs, 2.5), b_hi=np.percentile(bs, 97.5),
                wins=int((d > 0).sum()), p=float(stats.ttest_1samp(d, 0).pvalue))


def fmt(r, a, b, extra=''):
    if r.get('n', 0) < 2:
        return f"| {a} | {b} |{extra} {r.get('n', 0)} | n/a | | | |"
    return (f"| {a} | {b} |{extra} {r['n']} | {r['mean_a']:.3f} vs {r['mean_b']:.3f} | **{r['diff']:+.3f}** | "
            f"[{r['t_lo']:+.3f}, {r['t_hi']:+.3f}] / [{r['b_lo']:+.3f}, {r['b_hi']:+.3f}] | {r['wins']}/{r['n']} | {r['p']:.4f} |")


HEADER = ('| A | B |{x} seeds | mean A vs B | A - B | 95% CI (t / bootstrap) | A wins | paired t p |\n'
          '|---|---|{y}---|---|---|---|---|---|')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--pair', nargs=2, action='append', required=True, metavar=('A', 'B'))
    ap.add_argument('--budget', type=float, nargs='+', default=[2_400_000])
    ap.add_argument('--replay-w', type=float, nargs='+', default=[0.0])
    a = ap.parse_args()
    df = load_runs(a.dirs)
    multi = len(a.budget) > 1 or len(a.replay_w) > 1
    print(HEADER.format(x=' budget | replay w |' if multi else '', y='---|---|' if multi else ''))
    for A, B in a.pair:
        for bud in a.budget:
            for w in a.replay_w:
                # the baseline arm never branches, so replay charging only moves A (and tree baselines)
                r = paired(success_at(df, A, bud, w), success_at(df, B, bud, w))
                print(fmt(r, A, B, f' {bud:,.0f} | {w:g} |' if multi else ''))


if __name__ == '__main__':
    main()
