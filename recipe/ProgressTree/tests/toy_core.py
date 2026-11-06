# ===== h2lab v2: progress-credited tree rollouts on a merged state graph =====
# Environment, policy, rollouts, credit estimators (GRPO, GraphGPO, progress),
# frontier MC probes (v1), tree branching (v2), PPO, training loop, credit study.
import math, time, random
from collections import defaultdict, deque
import numpy as np
import torch
import torch.nn as nn

SUCCESS = -1          # sentinel graph node for the verified goal state
SALT_BASE = 10 ** 6   # graph-identity salt for the sparse-graph stress test; env state = key % SALT_BASE

# ---------------------------------------------------------------------------
# 1. Environment: "KeyDoor-Traps", a multi-turn task with irreversible mistakes
# ---------------------------------------------------------------------------
# A wall column splits a left room from a right room; a locked door sits in that
# wall. The agent starts in the left room and must pick up the correct key K, walk
# through the door and reach the goal G. Irreversible mistakes create dead ends
# (like a destructive tool call): picking up the wrong key X first (inventory is
# taken forever) or stepping on a trap (agent frozen). Episodes run to the horizon.
MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # up, down, left, right
ACTION_NAMES = ["up", "down", "left", "right"]


class KeyDoorTasks:
    def __init__(self, n_tasks, size=9, n_traps=2, seed=0):
        rng = np.random.default_rng(seed)
        S = size
        self.S, self.N = S, n_tasks
        self.wall = np.zeros((n_tasks, S, S), bool)
        self.trap = np.zeros((n_tasks, S, S), bool)
        self.door = np.zeros((n_tasks, 2), int)
        self.key = np.zeros((n_tasks, 2), int)
        self.bad = np.zeros((n_tasks, 2), int)
        self.goal = np.zeros((n_tasks, 2), int)
        self.start = np.zeros((n_tasks, 2), int)
        wc = S - 3
        for i in range(n_tasks):
            w = self.wall[i]
            w[0, :] = w[-1, :] = w[:, 0] = w[:, -1] = True
            w[:, wc] = True
            dr = rng.integers(1, S - 1)
            w[dr, wc] = False
            self.door[i] = (dr, wc)
            left = [(r, c) for r in range(1, S - 1) for c in range(1, wc)]
            right = [(r, c) for r in range(1, S - 1) for c in range(wc + 1, S - 1)]
            left = [p for p in left if p != (dr, wc - 1)]  # keep the cell in front of the door free
            picks = rng.choice(len(left), size=3 + n_traps, replace=False)
            cells = [left[j] for j in picks]
            self.start[i], self.key[i], self.bad[i] = cells[0], cells[1], cells[2]
            for (r, c) in cells[3:]:
                self.trap[i, r, c] = True
            self.goal[i] = right[rng.integers(len(right))]
        st = np.zeros((n_tasks, 4, S, S), np.float32)
        st[:, 0] = self.wall
        st[:, 1] = self.trap
        ar = np.arange(n_tasks)
        st[ar, 2, self.goal[:, 0], self.goal[:, 1]] = 1
        st[ar, 3, self.door[:, 0], self.door[:, 1]] = 1
        self.static_obs = st
        self.obs_dim = 7 * S * S + 4


def encode_key(r, c, inv, door, trapped, S):
    """Exact state hash used for graph-node merging."""
    return ((((r * S) + c) * 3 + inv) * 2 + door) * 2 + trapped


def decode_key(k, S):
    k = np.asarray(k, np.int64) % SALT_BASE  # strip a graph-identity salt, if any
    trapped = k % 2; k = k // 2
    door = k % 2; k = k // 2
    inv = k % 3; k = k // 3
    c = k % S; r = k // S
    return r, c, inv, door, trapped


class VecEnv:
    """Vectorized, state-restorable environment."""

    def __init__(self, tasks: KeyDoorTasks, task_ids, horizon, init_keys=None, init_t=None):
        self.T = tasks
        self.S = tasks.S
        self.task = np.asarray(task_ids, int)
        B = len(self.task)
        self.H = horizon
        if init_keys is None:
            st = tasks.start[self.task]
            self.r, self.c = st[:, 0].copy(), st[:, 1].copy()
            self.inv = np.zeros(B, int); self.door = np.zeros(B, int); self.trapped = np.zeros(B, int)
        else:
            r, c, inv, door, tr = decode_key(init_keys, self.S)
            self.r, self.c, self.inv, self.door, self.trapped = (np.asarray(x, int).copy() for x in (r, c, inv, door, tr))
        self.t = np.zeros(B, int) if init_t is None else np.asarray(init_t, int).copy()
        self.done = self.t >= self.H
        self.success = np.zeros(B, bool)

    def keys(self):
        return encode_key(self.r, self.c, self.inv, self.door, self.trapped, self.S).astype(np.int64)

    def obs(self):
        T, S, B = self.T, self.S, len(self.task)
        o = np.zeros((B, 7, S, S), np.float32)
        o[:, :4] = T.static_obs[self.task]
        ar = np.arange(B)
        o[:, 3] *= (1 - self.door)[:, None, None]
        kp = self.inv != 1; bp = self.inv != 2
        o[ar[kp], 4, T.key[self.task[kp], 0], T.key[self.task[kp], 1]] = 1
        o[ar[bp], 5, T.bad[self.task[bp], 0], T.bad[self.task[bp], 1]] = 1
        o[ar, 6, self.r, self.c] = 1
        extra = np.zeros((B, 4), np.float32)
        extra[ar, self.inv] = 1
        extra[:, 3] = self.trapped
        return np.concatenate([o.reshape(B, -1), extra], 1)

    def step(self, a):
        T = self.T
        live = ~self.done
        mv = live & (self.trapped == 0)
        nr = self.r + MOVES[a, 0] * mv
        nc = self.c + MOVES[a, 1] * mv
        is_door = (nr == T.door[self.task, 0]) & (nc == T.door[self.task, 1])
        blocked = T.wall[self.task, nr, nc]
        need_key = is_door & (self.door == 0)
        can_open = need_key & (self.inv == 1)
        blocked |= need_key & ~can_open
        self.door = np.where(can_open & mv, 1, self.door)
        self.r = np.where(blocked, self.r, nr)
        self.c = np.where(blocked, self.c, nc)
        on_trap = T.trap[self.task, self.r, self.c] & mv
        self.trapped = np.where(on_trap, 1, self.trapped)
        on_k = (self.r == T.key[self.task, 0]) & (self.c == T.key[self.task, 1]) & (self.inv == 0) & mv
        on_x = (self.r == T.bad[self.task, 0]) & (self.c == T.bad[self.task, 1]) & (self.inv == 0) & mv
        self.inv = np.where(on_k, 1, np.where(on_x, 2, self.inv))
        at_goal = (self.r == T.goal[self.task, 0]) & (self.c == T.goal[self.task, 1]) & live
        rew = at_goal.astype(np.float32)
        self.success |= at_goal
        self.t += live
        self.done |= at_goal | (self.t >= self.H)
        return rew, live


# ---------------------------------------------------------------------------
# 2. Policy
# ---------------------------------------------------------------------------
class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, n_actions=4, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def act(self, obs_np, device):
        logits = self.forward(torch.as_tensor(obs_np, device=device))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a.cpu().numpy(), dist.log_prob(a).cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Rollouts (main rollouts, MC probes and tree branches share this)
# ---------------------------------------------------------------------------
def _pad_time(x, L, is_keys):
    if x.shape[0] >= L:
        return x
    fill = np.repeat(x[-1:], L - x.shape[0], 0) if is_keys else np.zeros((L - x.shape[0],) + x.shape[1:], x.dtype)
    return np.concatenate([x, fill], 0)


def rollout(tasks, task_ids, policy, horizon, device, init_keys=None, first_action=None,
            keep_obs=True, chunk=8192, init_t=None, merge_noise=0.0):
    """Run envs to completion; returns [T, B] arrays (keys are [T+1, B]).
    init_keys: restore these states (keys[0] keeps the given, possibly salted, key).
    first_action: forced first action (ground-truth Q). merge_noise: probability that a
    visited state gets a unique graph identity (sparse-graph stress test)."""
    B = len(task_ids)
    if B > chunk:
        parts = [rollout(tasks, task_ids[i:i + chunk], policy, horizon, device,
                         None if init_keys is None else init_keys[i:i + chunk],
                         None if first_action is None else first_action[i:i + chunk],
                         keep_obs, chunk, None if init_t is None else init_t[i:i + chunk], merge_noise)
                 for i in range(0, B, chunk)]
        Tmax = max(p['actions'].shape[0] for p in parts)
        out = {}
        for k in parts[0]:
            if k == 'env_steps':
                out[k] = sum(p[k] for p in parts)
            elif k in ('success', 'task'):
                out[k] = np.concatenate([p[k] for p in parts])
            else:
                out[k] = np.concatenate([_pad_time(p[k], Tmax + 1 if k == 'keys' else Tmax, k == 'keys') for p in parts], 1)
        return out
    env = VecEnv(tasks, task_ids, horizon, init_keys, init_t)
    keys = [np.asarray(init_keys, np.int64).copy() if init_keys is not None else env.keys()]
    acts, logps, actives, rews, obs_l = [], [], [], [], []
    for t in range(horizon):
        o = env.obs()
        a, lp = policy.act(o, device)
        if t == 0 and first_action is not None:
            a = np.asarray(first_action, int)
            with torch.no_grad():
                lg = policy(torch.as_tensor(o, device=device))
                lp = torch.log_softmax(lg, -1).gather(1, torch.as_tensor(a, device=device)[:, None])[:, 0].cpu().numpy()
        r, live = env.step(a)
        k = env.keys()
        if merge_noise > 0:
            salt = (np.random.random(B) < merge_noise) * np.random.randint(1, SALT_BASE, B)
            k = k + SALT_BASE * salt.astype(np.int64)
        k = np.where(r > 0, SUCCESS, k)
        k = np.where(live, k, keys[-1])
        keys.append(k); acts.append(a); logps.append(lp); actives.append(live); rews.append(r)
        if keep_obs:
            obs_l.append(o)
        if env.done.all():
            break
    out = dict(keys=np.stack(keys), actions=np.stack(acts), logp=np.stack(logps),
               active=np.stack(actives), reward=np.stack(rews),
               success=env.success.copy(), task=np.asarray(task_ids), env_steps=int(np.stack(actives).sum()))
    if keep_obs:
        out['obs'] = np.stack(obs_l)
    return out


def obs_concat(main, extra):
    """Observations of main + extra columns, time-padded (needed by the critic credit)."""
    T = max(main['active'].shape[0], extra['active'].shape[0])
    return np.concatenate([_pad_time(main['obs'], T, False), _pad_time(extra['obs'], T, False)[:T]], 1)


def light_concat(main, extra, g_main, g_extra):
    """Columns of main + extra (keys/active/success/task only) for credit computation.
    Both parts are padded to the longer of the two, so branches longer than the main
    batch (possible with branch_mode='consistent') are not truncated."""
    T = max(main['active'].shape[0], extra['active'].shape[0])
    comb = dict(keys=np.concatenate([_pad_time(main['keys'], T + 1, True), _pad_time(extra['keys'], T + 1, True)[:T + 1]], 1),
                active=np.concatenate([_pad_time(main['active'], T, False), _pad_time(extra['active'], T, False)[:T]], 1),
                success=np.concatenate([main['success'], extra['success']]),
                task=np.concatenate([main['task'], extra['task']]))
    return comb, np.concatenate([g_main, g_extra])


# ---------------------------------------------------------------------------
# 4. Graph construction and distances (GraphGPO)
# ---------------------------------------------------------------------------
def build_reverse_graph(transitions):
    rev = defaultdict(set)
    for s, sn in transitions:
        rev[sn].add(s)
    return rev


def distances_to_success(rev, nodes):
    """Unit-cost reverse BFS from SUCCESS (= the official Dijkstra with weight 1)."""
    d = {n: math.inf for n in nodes}
    d[SUCCESS] = 0
    q = deque([SUCCESS])
    while q:
        u = q.popleft()
        for p in rev.get(u, ()):
            if d.get(p, math.inf) == math.inf:
                d[p] = d[u] + 1
                q.append(p)
    return d


def fill_unreachable(d):
    """Official GraphGPO behaviour: unreachable -> (max finite distance) + 1."""
    finite = [v for v in d.values() if v != math.inf]
    far = max(finite) if finite else 0
    return {k: (far + 1 if v == math.inf else v) for k, v in d.items()}, far


def traj_transitions(batch, idx):
    K, act = batch['keys'], batch['active']
    out = []
    for b in idx:
        for t in range(act.shape[0]):
            if act[t, b]:
                out.append((int(K[t, b]), int(K[t + 1, b])))
    return out


# ---------------------------------------------------------------------------
# 5. Credit estimators
# ---------------------------------------------------------------------------
def _norm_groups(values, groups, eps=1e-6):
    """Mean/std-normalize within groups (singleton groups -> 0), as verl-agent does."""
    out = np.zeros_like(values, dtype=np.float64)
    buckets = defaultdict(list)
    for i, g in enumerate(groups):
        buckets[g].append(i)
    for g, ix in buckets.items():
        v = values[ix]
        out[ix] = 0.0 if len(ix) == 1 else (v - v.mean()) / (v.std(ddof=1) + eps)
    return out


def grpo_advantages(batch, group):
    """Outcome advantage (R - mean_group) / std_group, broadcast to every step."""
    A_traj = _norm_groups(batch['success'].astype(np.float64), group)
    return np.broadcast_to(A_traj[None, :], batch['active'].shape) * batch['active']


def graph_step_returns(batch, group, extra_transitions=None, gamma=0.1, relative=False):
    """GraphGPO step return 10*gamma^d(next state) per task group (+ optional
    extra transitions from MC probes = graph completion). Returns returns and the
    raw (pre-fill) distance dict per group.
    relative=True uses 10*gamma^(d(next) - d(current) + 1): the same z-score inside an
    anchor-state group (d(current) is constant there) without the float underflow of
    gamma^d against the normalizer's eps when d is large (audit issue 5)."""
    T, B = batch['active'].shape
    ret = np.zeros((T, B))
    raw_d = {}
    for g in np.unique(group):
        idx = np.where(group == g)[0]
        trans = traj_transitions(batch, idx)
        if extra_transitions and g in extra_transitions:
            trans = trans + extra_transitions[g]
        nodes = {s for s, _ in trans} | {sn for _, sn in trans}
        d = distances_to_success(build_reverse_graph(trans), nodes)
        raw_d[g] = d
        dfill, _ = fill_unreachable(d)
        for b in idx:
            for t in range(T):
                if batch['active'][t, b]:
                    dn = dfill[int(batch['keys'][t + 1, b])]
                    ret[t, b] = 10.0 * gamma ** ((dn - dfill[int(batch['keys'][t, b])] + 1) if relative else dn)
    return ret, raw_d


def graphgpo_advantages(batch, group, step_returns, w_step=1.0, w_ep=1.0):
    """Step advantage normalized within (task, anchor-state) groups + episode
    advantage normalized across the steps of the task group (official recipe)."""
    T, B = batch['active'].shape
    tt, bb = np.nonzero(batch['active'])
    anchor = batch['keys'][tt, bb]
    grp = group[bb]
    a_step = _norm_groups(step_returns[tt, bb], [(int(g), int(s)) for g, s in zip(grp, anchor)])
    a_ep = _norm_groups(batch['success'][bb].astype(np.float64), [int(g) for g in grp])
    A = np.zeros((T, B))
    A[tt, bb] = w_step * a_step + w_ep * a_ep
    return A


def gigpo_advantages(batch, group, gamma=0.95, w_step=1.0, w_ep=1.0):
    """GiGPO (Feng et al., 2025; verl-agent): step advantage = discounted Monte Carlo return
    from the step (reward only at success), normalized within (task, anchor-state) groups,
    plus the same episode-level term as graphgpo_advantages."""
    T, B = batch['active'].shape
    tt, bb = np.nonzero(batch['active'])
    L = batch['active'].sum(0)
    ret = batch['success'][bb].astype(np.float64) * gamma ** (L[bb] - 1 - tt)
    grp = group[bb]
    a_step = _norm_groups(ret, [(int(g), int(s)) for g, s in zip(grp, batch['keys'][tt, bb])])
    a_ep = _norm_groups(batch['success'][bb].astype(np.float64), [int(g) for g in grp])
    A = np.zeros((T, B))
    A[tt, bb] = w_step * a_step + w_ep * a_ep
    return A


# --- progress credit: on-policy value differences on the merged graph ---------
# V(s) = success rate of every observed continuation through s (first visit per
# trajectory, branch or probe), pooled over merged states. Step advantage =
# V(s_{t+1}) - V(s_t) with V(SUCCESS) = 1 and a timed-out final step valued 0.
# Along a trajectory the advantages telescope to R - V(start): for main rollouts,
# the GRPO advantage, redistributed to the steps where success probability changed.
def _traj_states(keys_col, active_col):
    out = [int(keys_col[0])]
    for t in range(active_col.shape[0]):
        if not active_col[t]:
            break
        out.append(int(keys_col[t + 1]))
    return out


def value_stats(batch, group, probes=()):
    stats = defaultdict(dict)
    def add(g, states, outcome):
        seen = set()
        for s in states:
            if s == SUCCESS or s in seen:
                continue
            seen.add(s)
            st = stats[g].setdefault(s, [0.0, 0])
            st[0] += outcome; st[1] += 1
    for b in range(batch['active'].shape[1]):
        add(int(group[b]), _traj_states(batch['keys'][:, b], batch['active'][:, b]), float(batch['success'][b]))
    for rb in probes:
        for c in range(rb['active'].shape[1]):
            add(int(rb['group'][c]), _traj_states(rb['keys'][:, c], rb['active'][:, c]), float(rb['success'][c]))
    return stats


def _V(stats, g, s):
    if s == SUCCESS:
        return 1.0
    st = stats[g].get(s)
    return st[0] / st[1] if st else 0.0


def group_scale(batch, group, sigma_min):
    return {int(g): max(batch['success'][group == g].astype(float).std(ddof=1) if (group == g).sum() > 1 else 0.0, sigma_min)
            for g in np.unique(group)}


def progress_advantages(batch, group, scale, probes=()):
    stats = value_stats(batch, group, probes)
    T, B = batch['active'].shape
    A = np.zeros((T, B))
    for b in range(B):
        g = int(group[b])
        act = np.nonzero(batch['active'][:, b])[0]
        if len(act) == 0:
            continue
        for t in act:
            s = int(batch['keys'][t, b]); sn = int(batch['keys'][t + 1, b])
            q = 1.0 if sn == SUCCESS else (0.0 if t == act[-1] else _V(stats, g, sn))
            A[t, b] = (q - _V(stats, g, s)) / scale[g]
    return A


# --- v1 probes: MC continuations used only for credit ---------------------------
def progress_loo_advantages(batch, group, scale, probes=()):
    """Progress credit with leave-one-out baselines, robust to sparse merging.
    A_t = Q(t) - b(t): Q(t) = pooled V(s_{t+1}) (1 at success, 0 at timeout);
    b(t) = V(s_t) estimated WITHOUT this trajectory's own sample; if no other
    trajectory visited s_t, b(t) falls back to the last available baseline along
    this trajectory (ultimately the leave-one-out group mean at the start state).
    Dense merging -> close to progress credit; no merging -> RLOO spread over the
    unshared segment; inside a branch -> outcome minus the branch point's value."""
    stats = value_stats(batch, group, probes)
    T, B = batch['active'].shape
    A = np.zeros((T, B))
    for b in range(B):
        g = int(group[b]); R = float(batch['success'][b])
        act = np.nonzero(batch['active'][:, b])[0]
        base = None
        for t in act:
            s = int(batch['keys'][t, b]); sn = int(batch['keys'][t + 1, b])
            sm, n = stats[g].get(s, [0.0, 0])
            if n > 1:
                base = (sm - R) / (n - 1)
            elif base is None:
                base = 0.0
            q = 1.0 if sn == SUCCESS else (0.0 if t == act[-1] else _V(stats, g, sn))
            A[t, b] = (q - base) / scale[g]
    return A


class ValueNet(nn.Module):
    """Small learned value baseline V(obs) ~ P(success), trained online on observed outcomes."""
    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x)[:, 0])


def _first_visit_targets(keys, active, success, obs):
    """(obs, outcome) pairs for the first visit of each state in each column."""
    X, y = [], []
    for b in range(active.shape[1]):
        seen = set()
        for t in range(active.shape[0]):
            if not active[t, b]:
                break
            k = int(keys[t, b])
            if k in seen:
                continue
            seen.add(k); X.append(obs[t, b]); y.append(float(success[b]))
    return (np.asarray(X, np.float32), np.asarray(y, np.float32)) if X else (None, None)


def train_critic(memory, obs_dim, keys, active, success, obs, cfg, device='cpu'):
    if memory.get('critic') is None:
        net = ValueNet(obs_dim, cfg['critic_hidden']).to(device)
        memory['critic'] = (net, torch.optim.Adam(net.parameters(), lr=cfg['critic_lr']))
    net, opt = memory['critic']
    X, y = _first_visit_targets(keys, active, success, obs)
    if X is None:
        return net
    Xt, yt = torch.as_tensor(X, device=device), torch.as_tensor(y, device=device)
    for _ in range(cfg['critic_epochs']):
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), 1024):
            ix = perm[i:i + 1024]
            loss = nn.functional.binary_cross_entropy(net(Xt[ix]), yt[ix])
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def progress_hybrid_advantages(batch, group, scale, obs, critic, n_min=2, device='cpu'):
    """Progress credit with hybrid values: V(s) = pooled Monte Carlo success rate where
    at least n_min observed continuations passed through s (merged states), otherwise
    the learned critic's estimate. A_t = V(s_{t+1}) - V(s_t), with V(SUCCESS) = 1 and a
    timed-out final step valued 0, so each trajectory's advantages still telescope to
    R - V(start). Where states never merge, credit follows the critic's value changes
    instead of collapsing onto the first unshared step."""
    stats = value_stats(batch, group)
    T, B = batch['active'].shape
    with torch.no_grad():
        flat = torch.as_tensor(obs.reshape(-1, obs.shape[-1]), device=device)
        vc = torch.cat([critic(flat[i:i + 8192]) for i in range(0, len(flat), 8192)]).cpu().numpy().reshape(T, B)
    A = np.zeros((T, B))
    for b in range(B):
        g = int(group[b])
        act = np.nonzero(batch['active'][:, b])[0]
        def V(t):
            s = int(batch['keys'][t, b])
            st = stats[g].get(s)
            return st[0] / st[1] if (st and st[1] >= n_min) else float(vc[t, b])
        for t in act:
            sn = int(batch['keys'][t + 1, b])
            q = 1.0 if sn == SUCCESS else (0.0 if t == act[-1] else V(t + 1))
            A[t, b] = (q - V(t)) / scale[g]
    return A


def select_mc_nodes(batch, group, raw_d, mode='frontier', max_nodes=8, priority='branching', trigger='always'):
    """v1 stage-3 selection. mode='frontier': anchor states with no path to success;
    'all': every visited state. priority='branching': decision points first,
    self-loop-only states skipped; 'visits': most visited first.
    trigger: 'always' | 'mixed' (groups with >=1 success) | 'no_success'."""
    out = []
    T = batch['active'].shape[0]
    for g in np.unique(group):
        idx = np.where(group == g)[0]
        n_succ = int(batch['success'][idx].sum())
        if (trigger == 'mixed' and n_succ == 0) or (trigger == 'no_success' and n_succ > 0):
            continue
        cnt = defaultdict(int); succ = defaultdict(set)
        for b in idx:
            for t in range(T):
                if batch['active'][t, b]:
                    s = int(batch['keys'][t, b]); sn = int(batch['keys'][t + 1, b])
                    if mode == 'all' or raw_d[g].get(s, math.inf) == math.inf:
                        cnt[s] += 1
                        if sn != s:
                            succ[s].add(sn)
        if priority == 'branching':
            ranked = sorted([s for s in cnt if succ[s]], key=lambda s: (-len(succ[s]), -cnt[s]))[:max_nodes]
        else:
            ranked = sorted(cnt, key=lambda s: -cnt[s])[:max_nodes]
        out += [(int(g), int(batch['task'][idx[0]]), s) for s in ranked]
    return out


def mc_continuations(tasks, nodes, policy, K, horizon, device, merge_noise=0.0, init_t=None):
    """K on-policy continuations from each node (restored state). Returns graph
    transitions per group, the rollout batch (with 'group'), and its env-step cost.
    init_t (one per node): restore the episode clock too, so a continuation gets the
    remaining (horizon - t) steps instead of a fresh `horizon`-step episode."""
    if not nodes:
        return {}, None, 0
    rb = rollout(tasks, np.repeat([n[1] for n in nodes], K), policy, horizon, device,
                 init_keys=np.repeat(np.asarray([n[2] for n in nodes], np.int64), K), merge_noise=merge_noise,
                 init_t=None if init_t is None else np.repeat(np.asarray(init_t, int), K))
    rb['group'] = np.repeat([n[0] for n in nodes], K)
    extra = defaultdict(list)
    for c in range(rb['active'].shape[1]):
        g = int(rb['group'][c])
        for t in range(rb['active'].shape[0]):
            if rb['active'][t, c]:
                extra[g].append((int(rb['keys'][t, c]), int(rb['keys'][t + 1, c])))
    return dict(extra), rb, rb['env_steps']


# --- v2 tree branching: branches are training data -----------------------------
def _dead_set(memory, task, cfg):
    """States that never succeeded in >= dead_min_n pooled tries (decayed), while the
    task's start state succeeds >= dead_task_rate of the time."""
    if memory is None:
        return set()
    cache = memory['cache'].get(task); s0 = memory['start'].get(task)
    if not cache or s0 not in cache:
        return set()
    sm0, n0 = cache[s0]
    if n0 <= 0 or sm0 / n0 < cfg['dead_task_rate']:
        return set()
    return {s for s, (sm, n) in cache.items() if n >= cfg['dead_min_n'] and sm <= 1e-9}


def update_memory(memory, comb, gcomb, cfg):
    stats = value_stats(comb, gcomb)
    for g, d in stats.items():
        idx = np.where(gcomb == g)[0]
        task = int(comb['task'][idx[0]])
        cache = memory['cache'].setdefault(task, {})
        for st in cache.values():
            st[0] *= cfg['dead_decay']; st[1] *= cfg['dead_decay']
        for s, (sm, n) in d.items():
            c = cache.setdefault(s, [0.0, 0.0]); c[0] += sm; c[1] += n
        memory['start'][task] = int(comb['keys'][0, idx[0]])


def shared_step_frac(batch, group):
    """Share of steps t >= 1 whose state is visited by >= 2 trajectories of its group
    (the same statistic as port/alfworld_merge_probe.py)."""
    shared = tot = 0
    for g in np.unique(group):
        idx = np.where(group == g)[0]
        vis = defaultdict(set); steps = []
        for b in idx:
            for t in range(batch['active'].shape[0]):
                if batch['active'][t, b]:
                    s = int(batch['keys'][t, b]); vis[s].add(b)
                    if t >= 1:
                        steps.append(s)
        tot += len(steps); shared += sum(1 for s in steps if len(vis[s]) >= 2)
    return shared / max(tot, 1)


def _phys(k):
    """Physical environment state of a (possibly salted) key; SUCCESS stays SUCCESS."""
    return k if k == SUCCESS else k % SALT_BASE


def select_branch_nodes(batch, group, cfg, mode='smart', memory=None, rng=None):
    """Where to branch the group's tree.
    smart:  distinct visited states that are not absorbing (>=1 observed move out),
            not cached dead ends, most visited first, capped at branch_max_nodes.
    random: the same number of branch points drawn uniformly from the group's
            trajectory positions (Tree-GRPO-style random expansion).
    random_filtered: like random, but only from positions smart mode could pick
            (non-absorbing, not cached dead ends): separates the filters from the ranking.
    cfg['absorb_phys']=True decides "absorbing" on the physical state, so merge-noise
    salts do not make a frozen (trapped) agent look like it moves (audit issue 3).
    The returned diagnostics carry '_t': each branch point's step index (its earliest
    visit for smart/spread, the sampled position for random), used to restore the
    episode clock (branch_mode='consistent') and to count replay cost."""
    out, n_frozen, n_dead, times = [], 0, 0, []
    T = batch['active'].shape[0]
    rng = rng if rng is not None else np.random.default_rng(0)
    phys = _phys if cfg.get('absorb_phys', False) else (lambda k: k)
    for g in np.unique(group):
        idx = np.where(group == g)[0]
        task = int(batch['task'][idx[0]])
        visits = defaultdict(int); moves = defaultdict(int); positions = []; first_t = {}
        for b in idx:
            for t in range(T):
                if batch['active'][t, b]:
                    s = int(batch['keys'][t, b]); sn = int(batch['keys'][t + 1, b])
                    visits[s] += 1; positions.append((s, t))
                    if s not in first_t or t < first_t[s]:
                        first_t[s] = t
                    if phys(sn) != phys(s):
                        moves[s] += 1
        cand = [s for s in visits if s != SUCCESS]
        n_frozen += sum(1 for s in cand if moves[s] == 0)
        cand = [s for s in cand if moves[s] > 0]
        dead = _dead_set(memory, task, cfg)
        n_dead += sum(1 for s in cand if s in dead)
        cand = sorted([s for s in cand if s not in dead], key=lambda s: -visits[s])[:cfg['branch_max_nodes']]
        if mode == 'spread':
            # Evenly spaced points along each trajectory's distinct-state sequence (round-robin
            # over trajectories), skipping absorbing states and cached dead ends: covers the whole
            # horizon even when no state repeats.
            seqs = []
            for b in idx:
                seq = [s for s in _traj_states(batch['keys'][:, b], batch['active'][:, b])
                       if s != SUCCESS and moves[s] > 0 and s not in dead]
                seqs.append(list(dict.fromkeys(seq)))
            budget = len(cand); per = max(1, int(math.ceil(budget / max(len(seqs), 1)))); picked = []
            for seq in seqs:
                if seq:
                    pos = np.unique(np.linspace(0, len(seq) - 1, min(per, len(seq))).round().astype(int))
                    picked += [seq[p] for p in pos]
            cand = list(dict.fromkeys(picked))[:budget]
        if mode in ('random', 'random_filtered'):
            pool = positions if mode == 'random' else [p for p in positions if moves[p[0]] > 0 and p[0] not in dead]
            pts = [pool[i] for i in rng.integers(len(pool), size=len(cand))] if pool else []
            out += [(int(g), task, s) for s, _ in pts]
            times += [t for _, t in pts]
            continue
        out += [(int(g), task, s) for s in cand]
        times += [first_t[s] for s in cand]
    return out, dict(branch_frozen_skipped=n_frozen, branch_dead_skipped=n_dead, _t=times)


# ---------------------------------------------------------------------------
# 6. Arms
# ---------------------------------------------------------------------------
ARMS = {
    'GRPO':                  dict(credit='grpo'),
    'GRPO(2xG)':             dict(credit='grpo', gmult=2),
    'GraphGPO':              dict(credit='graph'),
    'GraphGPO(2xG)':         dict(credit='graph', gmult=2),
    'GraphGPO+FrontierMC':   dict(credit='graph', probes='frontier'),   # v1 concept (probes for credit only)
    'GraphGPO+VineAll':      dict(credit='graph', probes='all'),        # credit study: graph-channel ceiling
    'Progress':              dict(credit='progress'),                   # progress credit, no extra samples
    'Progress+VineAll':      dict(credit='progress', probes='all'),     # credit study: progress-channel ceiling
    'Progress+Tree':         dict(credit='progress', tree='smart'),     # v2 concept
    'Progress+Tree(random)': dict(credit='progress', tree='random'),    # branch-point ablation
    'Progress+Tree(LOO)':    dict(credit='progress_loo', tree='smart'), # v2.1: leave-one-out baselines (sparse-graph robust)
    'Progress+Tree(critic)': dict(credit='progress_hybrid', tree='smart'),   # v2.2: MC values where merged, learned critic elsewhere
    'Progress+Tree(LOO,spread)': dict(credit='progress_loo', tree='spread'),# v2.2: branch points spread along trajectories
    'Progress+Tree(critic,spread)': dict(credit='progress_hybrid', tree='spread'),
    'Progress(LOO)':         dict(credit='progress_loo'),               # no tree: drop-in credit change (first port step)
    'Progress(critic)':      dict(credit='progress_hybrid'),
    'Tree+GraphGPO':         dict(credit='graph', tree='smart'),        # attribution: same tree, graph credit
    'Tree+GRPO':             dict(credit='grpo', tree='smart'),         # attribution: same tree, outcome credit
    # added during validation (see docs/validation.md)
    'GiGPO':                 dict(credit='gigpo'),                      # anchor-state MC returns (Feng et al., 2025)
    'Tree+GiGPO':            dict(credit='gigpo', tree='smart'),
    'Progress+Tree(LOO,random)': dict(credit='progress_loo', tree='random'),
    'Progress+Tree(LOO,random_filtered)': dict(credit='progress_loo', tree='random_filtered'),
}


def arm_spec(arm):
    return ARMS[arm]


def _credit(credit, batch, group, cfg, scale=None, probes=(), extra_transitions=None, obs=None, critic=None):
    if credit == 'progress_hybrid':
        return progress_hybrid_advantages(batch, group, scale, obs, critic, cfg['critic_n_min'])
    if credit == 'grpo':
        return grpo_advantages(batch, group)
    if credit == 'graph':
        ret, _ = graph_step_returns(batch, group, extra_transitions=extra_transitions, gamma=cfg['gamma'],
                                    relative=cfg.get('graph_relative', False))
        return graphgpo_advantages(batch, group, ret, cfg['w_step'], cfg['w_ep'])
    if credit == 'gigpo':
        return gigpo_advantages(batch, group, cfg.get('gigpo_gamma', 0.95), cfg['w_step'], cfg['w_ep'])
    if credit == 'progress':
        return progress_advantages(batch, group, scale, probes)
    if credit == 'progress_loo':
        return progress_loo_advantages(batch, group, scale, probes)
    raise ValueError(credit)


def compute_advantages(arm, batch, group, tasks, policy, cfg, device, memory=None, rng=None):
    """Returns (A for the main batch [T,B], extra training batch or None,
    extra env steps, diagnostics)."""
    spec = arm_spec(arm)
    credit, nm = spec['credit'], cfg.get('merge_noise', 0.0)
    diag = {}
    if credit == 'progress_hybrid' and memory is None:
        memory = dict(cache={}, start={})   # credit study: a critic fit on this batch only
    ret0, raw_d = graph_step_returns(batch, group, gamma=cfg['gamma'], relative=cfg.get('graph_relative', False))
    tt, bb = np.nonzero(batch['active'])
    unl = np.array([raw_d[group[b]].get(int(batch['keys'][t, b]), math.inf) == math.inf for t, b in zip(tt, bb)])
    moving = batch['keys'][tt, bb] != batch['keys'][tt + 1, bb]
    diag['frontier_step_frac'] = float(unl.mean()) if len(unl) else 0.0
    diag['frontier_moving_step_frac'] = float(unl[moving].mean()) if moving.any() else 0.0
    diag['shared_step_frac'] = shared_step_frac(batch, group)
    scale = group_scale(batch, group, cfg['progress_sigma_min'])

    if spec.get('tree'):
        nodes, sk = select_branch_nodes(batch, group, cfg, spec['tree'], memory, rng)
        t_branch = sk.pop('_t')
        diag.update(sk)
        # replay cost of restoring each branch by re-running its parent's prefix (what the
        # verl-agent port does); the toy restores for free, so this is logged, not charged
        diag['replay_steps'] = int(np.sum(t_branch)) * cfg['branch_K']
        if cfg.get('branch_mode', 'toy20') == 'consistent':
            # like the port: restore the episode clock, run to the episode's own horizon
            _, br, cost = mc_continuations(tasks, nodes, policy, cfg['branch_K'], cfg['horizon'], device, nm,
                                           init_t=t_branch)
        else:
            # v2 toy default: a fresh mc_horizon-step continuation from the restored state
            _, br, cost = mc_continuations(tasks, nodes, policy, cfg['branch_K'], cfg['mc_horizon'], device, nm)
        if br is None:
            critic = train_critic(memory, batch['obs'].shape[-1], batch['keys'], batch['active'], batch['success'],
                                  batch['obs'], cfg, device) if credit == 'progress_hybrid' else None
            return _credit(credit, batch, group, cfg, scale, obs=batch['obs'], critic=critic), None, 0, diag
        comb, gcomb = light_concat(batch, br, group, br['group'])
        obs_c = critic = None
        if credit == 'progress_hybrid':
            obs_c = obs_concat(batch, br)
            critic = train_critic(memory, obs_c.shape[-1], comb['keys'], comb['active'], comb['success'], obs_c, cfg, device)
        A_comb = _credit(credit, comb, gcomb, cfg, scale, obs=obs_c, critic=critic)
        B = batch['active'].shape[1]; Tb = br['active'].shape[0]
        extra = dict(obs=br['obs'], actions=br['actions'], logp=br['logp'], active=br['active'],
                     adv=A_comb[:Tb, B:])
        if memory is not None:
            update_memory(memory, comb, gcomb, cfg)
        diag.update(branch_nodes=len(nodes), branch_steps=int(cost), branch_success=float(br['success'].mean()))
        return A_comb[:batch['active'].shape[0], :B], extra, cost, diag

    if spec.get('probes'):
        mode = spec['probes']
        nodes = select_mc_nodes(batch, group, raw_d, mode=mode,
                                max_nodes=cfg['mc_max_nodes'] if mode == 'frontier' else cfg['vine_max_nodes'],
                                priority=cfg['mc_priority'] if mode == 'frontier' else 'visits',
                                trigger=cfg['mc_trigger'] if mode == 'frontier' else 'always')
        extra_t, rb, cost = mc_continuations(tasks, nodes, policy, cfg['mc_K'], cfg['mc_horizon'], device, nm)
        if credit == 'graph':
            ret2, raw2 = graph_step_returns(batch, group, extra_transitions=extra_t, gamma=cfg['gamma'],
                                            relative=cfg.get('graph_relative', False))
            A = graphgpo_advantages(batch, group, ret2, cfg['w_step'], cfg['w_ep'])
            if nodes:
                diag['relink_rate'] = float(np.mean([raw2[g].get(s, math.inf) != math.inf for g, _, s in nodes]))
        else:
            A = progress_advantages(batch, group, scale, [rb] if rb is not None else [])
        diag['mc_nodes'] = len(nodes)
        return A, None, cost, diag

    if credit == 'graph':
        return graphgpo_advantages(batch, group, ret0, cfg['w_step'], cfg['w_ep']), None, 0, diag
    critic = train_critic(memory, batch['obs'].shape[-1], batch['keys'], batch['active'], batch['success'],
                          batch['obs'], cfg, device) if credit == 'progress_hybrid' else None
    return _credit(credit, batch, group, cfg, scale, obs=batch['obs'], critic=critic), None, 0, diag


# ---------------------------------------------------------------------------
# 7. PPO-clip update on per-step advantages
# ---------------------------------------------------------------------------
def flatten_for_update(batch, A):
    tt, bb = np.nonzero(batch['active'])
    return (batch['obs'][tt, bb], batch['actions'][tt, bb], batch['logp'][tt, bb], A[tt, bb])


def ppo_update(policy, opt, data, cfg, device):
    """update_mode='fixed_minibatch' (default): the same minibatch size for every arm.
    Every arm trains on all its samples for ppo_epochs, so at equal environment steps
    all arms get the same training compute AND about the same number of optimizer
    steps: the fair equal-compute comparison.
    'fixed_steps': n_minibatches per epoch whatever the data size, so every arm gets
    the same number of optimizer steps per iteration (arms with more samples per
    iteration get fewer, larger steps at equal env steps). This is what trainers that
    size minibatches per prompt do (e.g. verl scales the PPO minibatch by the number
    of responses per prompt), and it isolates per-update credit quality."""
    obs, act, logp_old, adv = [torch.as_tensor(x, device=device) for x in data]
    adv = adv.float(); logp_old = logp_old.float()
    n = len(obs)
    if n == 0:
        return
    mb = int(math.ceil(n / cfg['n_minibatches'])) if cfg.get('update_mode', 'fixed_steps') == 'fixed_steps' else cfg['minibatch']
    for _ in range(cfg['ppo_epochs']):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, mb):
            ix = perm[i:i + mb]
            logits = policy(obs[ix])
            lp = torch.log_softmax(logits, -1).gather(1, act[ix].long()[:, None])[:, 0]
            ratio = torch.exp(lp - logp_old[ix])
            a = adv[ix]
            pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg['clip'], 1 + cfg['clip']) * a).mean()
            ent = torch.distributions.Categorical(logits=logits).entropy().mean()
            opt.zero_grad(); (pg - cfg['ent_coef'] * ent).backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()


@torch.no_grad()
def evaluate(tasks, task_ids, policy, horizon, device, reps=4):
    rb = rollout(tasks, np.repeat(task_ids, reps), policy, horizon, device, keep_obs=False)
    return float(rb['success'].mean())


# ---------------------------------------------------------------------------
# 8. Training loop for one (arm, seed)
# ---------------------------------------------------------------------------
DEFAULT_CFG = dict(
    size=9, n_traps=2, horizon=40, n_train_tasks=256, n_eval_tasks=128,
    tasks_per_iter=16, group_size=8,
    env_budget=None, max_iters=3000, eval_points=20,   # env-step budget mode (v2 default in the notebook)
    iters=200, eval_every=20,                          # iteration mode (used when env_budget is None)
    gamma=0.1, w_step=1.0, w_ep=1.0,
    mc_K=4, mc_max_nodes=8, mc_horizon=20, mc_priority='branching', mc_trigger='mixed', vine_max_nodes=64,
    branch_K=1, branch_max_nodes=32, dead_min_n=8, dead_task_rate=0.25, dead_decay=0.8,
    progress_sigma_min=0.2, merge_noise=0.0,
    critic_hidden=256, critic_lr=1e-3, critic_epochs=2, critic_n_min=2,
    task_seed_train=1000, task_seed_eval=2000,
    lr=3e-4, ppo_epochs=2, update_mode='fixed_minibatch', n_minibatches=5, minibatch=1024,
    clip=0.2, ent_coef=0.01, hidden=256,
)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def train_arm(arm, seed, cfg, device='cpu', train_tasks=None, eval_tasks=None, verbose=False,
              policy_factory=None):
    set_seed(seed)
    train_tasks = train_tasks or KeyDoorTasks(cfg['n_train_tasks'], cfg['size'], cfg['n_traps'], seed=cfg.get('task_seed_train', 1000))
    eval_tasks = eval_tasks or KeyDoorTasks(cfg['n_eval_tasks'], cfg['size'], cfg['n_traps'], seed=cfg.get('task_seed_eval', 2000))
    policy = policy_factory() if policy_factory else MLPPolicy(train_tasks.obs_dim, hidden=cfg['hidden']).to(device)
    opt = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=cfg['lr'])
    rng = np.random.default_rng(seed)
    # cfg['branch_rng']='separate': branch-point sampling gets its own stream, so the random-branch
    # arms train on the same task sequence as every other arm (default: shared, as in v2)
    branch_rng = np.random.default_rng(seed + 10 ** 6) if cfg.get('branch_rng') == 'separate' else rng
    memory = dict(cache={}, start={})
    budget = cfg.get('env_budget')
    marks = list(np.linspace(0, budget, cfg['eval_points'] + 1)[1:]) if budget else None
    max_iters = cfg['max_iters'] if budget else cfg['iters']
    G = cfg['group_size'] * arm_spec(arm).get('gmult', 1)
    env_steps, it, log, t0 = 0, 0, [], time.time()

    def do_eval():
        ev = evaluate(eval_tasks, np.arange(eval_tasks.N), policy, cfg['horizon'], device)
        log.append(dict(arm=arm, seed=seed, iter=it, env_steps=env_steps, eval_success=ev, wall=time.time() - t0))
        if verbose:
            print(f"[{arm}|s{seed}] it {it:4d} steps {env_steps:9d} eval {ev:.3f}")

    do_eval()
    while it < max_iters and not (budget and env_steps >= budget):
        tids = rng.choice(train_tasks.N, cfg['tasks_per_iter'], replace=False)
        ids = np.repeat(tids, G)
        group = np.repeat(np.arange(cfg['tasks_per_iter']), G)
        batch = rollout(train_tasks, ids, policy, cfg['horizon'], device, merge_noise=cfg.get('merge_noise', 0.0))
        A, extra, cost, diag = compute_advantages(arm, batch, group, train_tasks, policy, cfg, device, memory, branch_rng)
        env_steps += batch['env_steps'] + cost
        data = flatten_for_update(batch, A)
        if extra is not None:
            data = tuple(np.concatenate([a, b]) for a, b in zip(data, flatten_for_update(extra, extra['adv'])))
        ppo_update(policy, opt, data, cfg, device)
        it += 1
        log.append(dict(arm=arm, seed=seed, iter=it, env_steps=env_steps,
                        train_success=float(batch['success'].mean()), **diag))
        if budget:
            if marks and env_steps >= marks[0]:
                while marks and env_steps >= marks[0]:
                    marks.pop(0)
                do_eval()
        elif it % cfg['eval_every'] == 0:
            do_eval()
    if log[-1].get('eval_success') is None:
        do_eval()
    return policy, log


# ---------------------------------------------------------------------------
# 9. Credit-accuracy study: do step advantages match ground-truth advantages?
# ---------------------------------------------------------------------------
@torch.no_grad()
def ground_truth_advantages(tasks, policy, batch, pairs, horizon, device, M=32):
    """A*(s_t, a_t) = Q(s_t, a_t) - sum_a pi(a|s_t) Q(s_t, a); Q = success probability
    within the remaining horizon from M brute-force rollouts per action."""
    tt = np.array([p[0] for p in pairs]); bb = np.array([p[1] for p in pairs])
    keys = batch['keys'][tt, bb]; tids = batch['task'][bb]
    n, nA = len(pairs), len(MOVES)
    rb = rollout(tasks, np.repeat(np.repeat(tids, nA), M), policy, horizon, device,
                 init_keys=np.repeat(np.repeat(keys, nA), M), first_action=np.repeat(np.tile(np.arange(nA), n), M),
                 keep_obs=False, init_t=np.repeat(np.repeat(tt, nA), M))
    Q = rb['success'].reshape(n, nA, M).mean(-1)
    pi = torch.softmax(policy(torch.as_tensor(batch['obs'][tt, bb], device=device)), -1).cpu().numpy()
    V = (pi * Q).sum(-1)
    return Q[np.arange(n), batch['actions'][tt, bb]] - V, Q, V


def credit_study(tasks, policy, cfg, device, estimators=('GRPO', 'GraphGPO', 'GraphGPO+FrontierMC', 'Progress+Tree'),
                 n_pairs=400, M=32, seed=0, tasks_per_iter=None):
    """All estimators' advantages on the SAME batch vs ground truth. Metrics:
    cosine (alignment of the advantage vector with the true one, a proxy for how
    well the policy-gradient direction is estimated), Spearman, sign agreement;
    split by whether the step's start state is linked to success in the graph."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    P = tasks_per_iter or cfg['tasks_per_iter']
    tids = rng.choice(tasks.N, P, replace=False)
    group = np.repeat(np.arange(P), cfg['group_size'])
    batch = rollout(tasks, np.repeat(tids, cfg['group_size']), policy, cfg['horizon'], device,
                    merge_noise=cfg.get('merge_noise', 0.0))
    _, raw_d = graph_step_returns(batch, group, gamma=cfg['gamma'])
    tt, bb = np.nonzero(batch['active'])
    real_move = (batch['keys'][tt, bb] % SALT_BASE) != (batch['keys'][tt + 1, bb] % SALT_BASE)
    cand = np.where(real_move | (batch['keys'][tt + 1, bb] == SUCCESS))[0]
    pick = rng.choice(cand, min(n_pairs, len(cand)), replace=False)
    pairs = list(zip(tt[pick], bb[pick]))
    A_true, _, _ = ground_truth_advantages(tasks, policy, batch, pairs, cfg['horizon'], device, M)
    linked = np.array([raw_d[group[b]].get(int(batch['keys'][t, b]), math.inf) != math.inf for t, b in pairs])
    rows = []
    for est in estimators:
        A, _, cost, _ = compute_advantages(est, batch, group, tasks, policy, cfg, device, None, np.random.default_rng(seed))
        a_hat = np.array([A[t, b] for t, b in pairs])
        for name, mask in (('all', np.ones(len(pairs), bool)), ('linked', linked), ('frontier', ~linked)):
            cos = float(np.dot(a_hat[mask], A_true[mask]) / (np.linalg.norm(a_hat[mask]) * np.linalg.norm(A_true[mask]) + 1e-12)) if mask.sum() else np.nan
            m = mask & (np.abs(A_true) > 1e-9)
            if m.sum() < 5:
                rows.append(dict(estimator=est, subset=name, n=int(m.sum()), cosine=cos, spearman=np.nan,
                                 sign_agree=np.nan, extra_env_steps=cost))
                continue
            rho = spearmanr(a_hat[m], A_true[m]).correlation
            ms = m & (np.abs(A_true) > 0.05)
            sa = float(np.mean(np.sign(a_hat[ms]) == np.sign(A_true[ms]))) if ms.sum() else np.nan
            rows.append(dict(estimator=est, subset=name, n=int(m.sum()), cosine=cos, spearman=float(rho),
                             sign_agree=sa, extra_env_steps=cost))
    return rows, dict(batch_success=float(batch['success'].mean()), linked_frac=float(linked.mean()))
