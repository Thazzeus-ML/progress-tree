"""Compare arms at equal GPU-hours from the runs' progress_tree_log.csv files.

    python recipe/ProgressTree/compare_runs.py ~/progress_tree_runs [--baseline graphgpo] [--hours H]

Each run directory is <arm>_s<seed> (as run_alfworld_1gpu.sh names them). Per run, the
validation success rate is read at the common budget H (default: the largest budget every
run reached; validation time is not counted), interpolated linearly between the validations
around H (validation is scheduled in steps, so slower arms validate less often in time and
"last validation before H" would score them on older checkpoints), and as
the area under the success-vs-hours step curve over [0, H] (AUC, a learning-speed measure).
Arms are summarised by IQM over seeds and compared with the baseline seed by seed (same
seed = same game sequence), with a bootstrap 95% CI on the mean paired difference.
Writes compare_summary.md and compare_curves.png next to the runs.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


def iqm(x):
    x = np.sort(np.asarray(x, float))
    if len(x) < 4:
        return float(np.mean(x)) if len(x) else float('nan')
    k = len(x) // 4
    return float(np.mean(x[k:len(x) - k]))


def boot_ci(x, fn=np.mean, n=5000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    v = [fn(x[rng.integers(len(x), size=len(x))]) for _ in range(n)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def load(root):
    runs = []
    for d in sorted(glob.glob(os.path.join(root, '*_s*'))):
        f = os.path.join(d, 'progress_tree_log.csv')
        m = re.match(r'(.+)_s(\d+)$', os.path.basename(d))
        if not (m and os.path.exists(f)):
            continue
        df = pd.read_csv(f)
        runs.append(dict(arm=m.group(1), seed=int(m.group(2)), df=df, dir=d))
    return runs


def val_curve(df, key):
    v = df[df['kind'] == 'val'][['train_seconds', key]].dropna().sort_values('train_seconds')
    return v['train_seconds'].values / 3600.0, v[key].values.astype(float)


def at_budget(h, y, H):
    if len(h) == 0 or H > h[-1] + 1e-9:
        return float('nan')
    return float(np.interp(H, h, y))


def auc(h, y, H):
    if len(h) == 0:
        return float('nan')
    grid = np.linspace(0, H, 200)
    idx = np.searchsorted(h, grid, side='right') - 1
    vals = np.where(idx >= 0, y[np.clip(idx, 0, None)], y[0])
    return float(np.trapezoid(vals, grid) / max(H, 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--baseline', default='graphgpo')
    ap.add_argument('--hours', type=float, default=None)
    ap.add_argument('--metric', default='val/success_rate')
    args = ap.parse_args()
    runs = load(args.root)
    if not runs:
        raise SystemExit(f'no runs with progress_tree_log.csv under {args.root}')
    for r in runs:
        r['h'], r['y'] = val_curve(r['df'], args.metric)
    H = args.hours or min(r['h'].max() for r in runs if len(r['h']))
    late = [f"{r['arm']}_s{r['seed']} (last validation at {r['h'].max():.2f} h)" for r in runs
            if len(r['h']) and r['h'].max() < 0.97 * max(rr['h'].max() for rr in runs if len(rr['h']))]
    if late:
        print('WARNING: these runs stopped validating well before the others (crash or budget overrun?); '
              'H is capped by them: ' + '; '.join(late))
    rows = []
    for r in runs:
        tr = r['df'][r['df']['kind'] == 'train']
        tr = tr[tr['train_seconds'] <= H * 3600 + 1e-6]
        rows.append(dict(arm=r['arm'], seed=r['seed'], success_at_H=at_budget(r['h'], r['y'], H), auc=auc(r['h'], r['y'], H),
                         initial=float(r['y'][0]) if len(r['y']) else float('nan'), iterations=len(tr),
                         env_steps=float(tr.get('main_steps', pd.Series(dtype=float)).sum() + tr.get('branch_steps', pd.Series(dtype=float)).fillna(0).sum()),
                         branch_share=float(tr.get('branch_steps', pd.Series([0.0])).fillna(0).sum() /
                                            max(tr.get('main_steps', pd.Series([0.0])).sum() + tr.get('branch_steps', pd.Series([0.0])).fillna(0).sum(), 1)),
                         train_success_last5=float(tr['main_success'].tail(5).mean()) if len(tr) else float('nan'),
                         shared_first5=float(tr['main_shared_step_frac'].head(5).mean()) if 'main_shared_step_frac' in tr else float('nan'),
                         shared_all=float(tr['main_shared_step_frac'].mean()) if 'main_shared_step_frac' in tr else float('nan')))
    per = pd.DataFrame(rows).sort_values(['arm', 'seed'])
    summ = []
    for arm, d in per.groupby('arm'):
        lo, hi = boot_ci(d.success_at_H.values, iqm)
        summ.append(dict(arm=arm, seeds=len(d), IQM_success_at_H=iqm(d.success_at_H), CI95=f'[{lo:.3f}, {hi:.3f}]',
                         mean_AUC=d.auc.mean(), iterations=d.iterations.mean(), env_steps=d.env_steps.mean(),
                         branch_share=d.branch_share.mean(), shared_steps=d.shared_all.mean()))
    summ = pd.DataFrame(summ).sort_values('IQM_success_at_H', ascending=False)
    lines = [f'# Progress+Tree vs baselines at equal GPU-hours ({H:.2f} h of training, validation excluded)', '',
             f'Metric: `{args.metric}` on the validation games; per run, the last validation at or before {H:.2f} h.', '',
             summ.to_markdown(index=False, floatfmt='.3f'), '']
    base = per[per.arm == args.baseline].set_index('seed')
    if len(base):
        lines += [f'## Paired with `{args.baseline}` (same seeds = same game sequence)', '']
        pr = []
        for arm, d in per[per.arm != args.baseline].groupby('arm'):
            d = d.set_index('seed')
            common = sorted(set(d.index) & set(base.index))
            if not common:
                continue
            diff = d.loc[common, 'success_at_H'].values - base.loc[common, 'success_at_H'].values
            dauc = d.loc[common, 'auc'].values - base.loc[common, 'auc'].values
            lo, hi = boot_ci(diff)
            pr.append(dict(arm=arm, pairs=len(common), mean_diff_success=diff.mean(), CI95=f'[{lo:.3f}, {hi:.3f}]',
                           wins=int((diff > 0).sum()), mean_diff_AUC=dauc.mean()))
        if pr:
            lines += [pd.DataFrame(pr).to_markdown(index=False, floatfmt='.3f'), '',
                      'A difference is credible only if its CI excludes 0; with fewer than ~5 seeds per arm, treat any '
                      'result as a pilot.', '']
    sh = per.shared_all.dropna()
    if len(sh):
        lines += ['## Merge rate', '',
                  f'Main rollouts: {sh.mean():.2f} of steps (t >= 1) were on a state another rollout of the group also visited '
                  f'({per.shared_first5.mean():.2f} in the first 5 iterations). In the toy (10 seeds), Progress+Tree beat GraphGPO only when this '
                  f'share was about 0.85 or more; it tied at 0.79-0.83 and was behind at 0.74 and below.', '']
    lines += ['## Runs', '', per.to_markdown(index=False, floatfmt='.3f')]
    out = os.path.join(args.root, 'compare_summary.md')
    open(out, 'w').write('\n'.join(lines))
    print('\n'.join(lines))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for k, arm in enumerate(sorted(per.arm.unique())):
            for j, r in enumerate([r for r in runs if r['arm'] == arm]):
                ax.step(r['h'], r['y'], where='post', color=f'C{k}', alpha=0.8, label=arm if j == 0 else None)
        ax.axvline(H, color='gray', ls=':')
        ax.set_xlabel('training hours (validation excluded)')
        ax.set_ylabel(args.metric)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.root, 'compare_curves.png'), dpi=120)
        print('wrote', out, 'and compare_curves.png')
    except Exception as e:
        print('plot skipped:', e)


if __name__ == '__main__':
    main()
