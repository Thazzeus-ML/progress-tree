"""Merge-rate probe for ALFWorld, using GraphGPO's exact state representation.

Measures how often states merge within a rollout group (exactly, and with GraphGPO's
similarity canonicalizer), and whether a mid-episode state can be restored by
replaying the action prefix on a fresh copy of the same game (the premise of tree
branching in real environments).

The anchor string, state tracking (location / holding / item history) and the
similarity canonicalizer mirror verl-agent's recipe/GraphGPO (Apache-2.0):
  - AlfworldWorker in recipe/GraphGPO/alfworld_envs.py
  - AlfWorldEnvironmentManager.get_location_holding / step in graphgpo_env_manager.py
  - StateCanonicalizer in core_graph.py (here with an exact quick_ratio pre-filter)

Usage (Python 3.10/3.11; TextWorld's grammar eval breaks on 3.13):
  export ALFWORLD_DATA=/path/to/alfworld_data
  python alfworld_merge_probe.py --verl-agent /path/to/verl-agent --games 16 --group 8 \
         --policies uniform,expert0.5,expert0.8 --out alfworld_merge.json
Policies: 'uniform' (random admissible command), 'expertP' (hand-coded expert with
probability P, otherwise random). To probe your own LLM policy, pass a callable to
run_group(policy_fn=...).
"""
import argparse, json, math, os, random, re, sys, time
from collections import defaultdict, deque
from difflib import SequenceMatcher
try:
    from rapidfuzz.distance import Indel   # 2*LCS/(len1+len2) >= difflib's ratio, so it is an exact pre-filter
except ImportError:
    Indel = None

SUCCESS = 'success'


# ----------------------------------------------------------------------------------------
# State tracking and anchor string (mirrors GraphGPO's AlfworldWorker + env manager)
# ----------------------------------------------------------------------------------------
class Tracker:
    def __init__(self):
        self.history_items, self.location, self.holding = {}, "middle of a room", "nothing"
        self.item_location, self.done = {}, False

    def _update_item_location(self, old_holding):
        obj = old_holding if old_holding != 'nothing' else self.holding
        if obj == 'nothing':
            return
        if obj not in self.item_location:
            self.item_location[obj] = {'old_location': self.location, 'new_location': self.location}
        else:
            if old_holding == obj:
                self.item_location[obj]['new_location'] = self.location
            elif self.holding == obj:
                self.item_location[obj]['new_location'] = self.item_location[obj]['old_location']

    def _update_item_state(self, action, obs):
        action = action.lower().strip()
        m = re.match(r"^(heat|cool|clean|slice)\s+([\w\s\d]+?)(?:\s+with\s+[\w\s\d]+)?$", action)
        if not m:
            return
        verb, obj = m.groups(); obj = obj.strip()
        if verb not in obs:
            return
        self.history_items.setdefault(obj, {"heated": False, "cooled": False, "cleaned": False, "slice": False})
        key = {"heat": "heated", "cool": "cooled", "clean": "cleaned", "slice": "sliced"}[verb]
        self.history_items[obj][key] = True

    def _update_position(self, action, obs):
        m = re.match(r"^go to\s+(.+)$", action.lower().strip())
        if m and m.group(1).strip() in obs:
            self.location = m.group(1).strip()

    def _update_held_items(self, action, obs):
        action = action.lower().strip(); old = self.holding
        m = re.match(r"^take\s+(.+?)(?:\s+from\s+.+)?$", action)
        if m and m.group(1).strip() in obs and self.holding == 'nothing':
            self.holding = m.group(1).strip()
        m = re.match(r"^drop\s+(.+)$", action)
        if m and m.group(1).strip() in obs:
            self.holding = "nothing"
        m = re.match(r"^(put|place)\s+(.+?)\s+(?:in|on)\s+.+$", action)
        if m and m.group(2).strip() in obs:
            self.holding = "nothing"
        m = re.match(r"^move\s+(.+?)(?:\s+to\s+.+)?$", action)
        if m and m.group(1).strip() in obs:
            self.holding = "nothing"
        if old != self.holding:
            self._update_item_location(old)

    def step(self, action, obs):
        if "Nothing happens" not in obs and not self.done:
            self._update_item_state(action, obs)
            self._update_position(action, obs)
            self._update_held_items(action, obs)

    def location_holding(self):
        hs = ''
        for k, v in self.history_items.items():
            if k in self.holding:
                for s, ok in v.items():
                    if ok:
                        hs += f"{s} "
        if hs != '':
            hs = f"({hs.strip()})"
        elif self.holding != 'nothing':
            hs = "(unprocessed)"
        oil = ''
        for obj, loc in self.item_location.items():
            st = ''
            if obj in self.history_items:
                for s, ok in self.history_items[obj].items():
                    if ok:
                        st += f"{s} "
            st = f"({st.strip()})" if st != '' else "(unprocessed)"
            if loc['old_location'] != loc['new_location']:
                oil += f"{obj}{st} from {loc['old_location']} move to {loc['new_location']};"
            elif st != "(unprocessed)" and self.holding != obj:
                oil += f"{obj}{st} from {loc['old_location']} move to {loc['new_location']};"
        if oil != "":
            oil = f"History moving list by you: {oil}"
        return f"Location: {self.location}. Items in hand (status): {self.holding}{hs}.", oil


def anchor_string(text_obs, tracker, commands):
    lh, il = tracker.location_holding()
    return f"{text_obs} {lh} {il} {','.join(sorted(commands))}"


# ----------------------------------------------------------------------------------------
# GraphGPO's similarity canonicalizer (exact pre-filter: ratio <= quick_ratio <= real_quick_ratio)
# ----------------------------------------------------------------------------------------
class StateCanonicalizer:
    def __init__(self, enable=False, threshold=0.95):
        self.enable, self.threshold, self._clusters, self._cache = enable, threshold, [], {}

    def canonicalize(self, s):
        if not self.enable:
            return s
        if s in self._cache:
            return self._cache[s]
        for ch, cs in self._clusters:
            if Indel is not None and Indel.normalized_similarity(s, cs) < self.threshold:
                continue
            sm = SequenceMatcher(None, s, cs)
            if sm.real_quick_ratio() >= self.threshold and sm.quick_ratio() >= self.threshold and sm.ratio() >= self.threshold:
                self._cache[s] = ch
                return ch
        self._clusters.append((s, s)); self._cache[s] = s
        return s


# ----------------------------------------------------------------------------------------
# Environment helpers
# ----------------------------------------------------------------------------------------
def install_textworld_parse_cache():
    """TextWorld re-parses the same logic-query strings with a PEG parser (tatsu) on every
    text derivation, which dominates step time. Parsing a fixed string is a pure function,
    so memoising it cannot change the environment (checked by --verify-cache)."""
    from textworld.logic import Rule
    if getattr(Rule, '_h2lab_cached', False):
        return
    orig = Rule.parse_conjunctive_query.__func__
    cache = {}

    def cached(cls, expr):
        r = cache.get(expr)
        if r is None:
            r = cache[expr] = orig(cls, expr)
        return r
    Rule.parse_conjunctive_query = classmethod(cached)
    Rule._h2lab_cached = True


def make_base_env(verl_agent_root, use_expert=True, parse_cache=True):
    import yaml
    sys.path.insert(0, os.path.join(verl_agent_root, 'agent_system/environments/env_package/alfworld'))
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    if parse_cache:
        install_textworld_parse_cache()
    cfg = yaml.safe_load(open(os.path.join(verl_agent_root, 'agent_system/environments/env_package/alfworld/configs/config_tw.yaml')))
    base = AlfredTWEnv(cfg, train_eval='train')
    base.use_expert = use_expert
    return base


def single_game_env(base, gamefile):
    """A fresh env over exactly one game file (how a branch restores a state)."""
    files = base.game_files
    base.game_files = [gamefile]
    try:
        env = base.init_env(batch_size=1)
    finally:
        base.game_files = files
    return env


def make_policy(name, rng):
    if name == 'uniform':
        return lambda infos: rng.choice([c for c in infos['admissible_commands'][0] if c != 'help'])
    if name.startswith('expert'):
        p = float(name[len('expert'):])
        def pol(infos):
            plan = infos.get('extra.expert_plan')
            if plan and plan[0] and rng.random() < p:
                return plan[0][0]
            return rng.choice([c for c in infos['admissible_commands'][0] if c != 'help'])
        return pol
    raise ValueError(name)


def run_group(env, game_seed, group, policy_name, rng, max_steps=50, policy_fn=None):
    """G rollouts of one game. Returns per-trajectory step records."""
    trajs = []
    for g in range(group):
        pol = policy_fn or make_policy(policy_name, random.Random(rng.random()))
        env.seed(game_seed)
        obs, infos = env.reset()
        gamefile = infos['extra.gamefile'][0]
        tr = Tracker()
        anchor = anchor_string(obs[0], tr, infos['admissible_commands'][0])
        steps, won = [], False
        for t in range(max_steps):
            a = pol(infos)
            obs, _, dones, infos = env.step([a])
            tr.step(a, obs[0]); tr.done = bool(dones[0])
            won = bool(infos['won'][0])
            nxt = SUCCESS if won else anchor_string(obs[0], tr, infos['admissible_commands'][0])
            steps.append(dict(t=t, anchor=anchor, action=a, next=nxt))
            anchor = nxt
            if dones[0]:
                break
        trajs.append(dict(gamefile=gamefile, steps=steps, won=won))
    return trajs


# ----------------------------------------------------------------------------------------
# Graph statistics (GraphGPO's graph: anchor -> next, success node, per-group canonicalizer)
# ----------------------------------------------------------------------------------------
def group_stats(trajs, similarity=False, threshold=0.95):
    canon = StateCanonicalizer(enable=similarity, threshold=threshold)
    T = max(len(tr['steps']) for tr in trajs)
    keyed = [[None] * len(tr['steps']) for tr in trajs]
    for t in range(T):                       # step-major order, like the rollout loop
        for i, tr in enumerate(trajs):
            if t < len(tr['steps']):
                s = tr['steps'][t]
                keyed[i][t] = (canon.canonicalize(s['anchor']),
                               SUCCESS if s['next'] == SUCCESS else canon.canonicalize(s['next']))
    visitors = defaultdict(set); rev = defaultdict(set); nodes = set()
    for i, ks in enumerate(keyed):
        for a, n in ks:
            visitors[a].add(i); rev[n].add(a); nodes.add(a); nodes.add(n)
    reach = {SUCCESS}; q = deque([SUCCESS])
    while q:
        u = q.popleft()
        for p in rev.get(u, ()):
            if p not in reach:
                reach.add(p); q.append(p)
    steps_t1 = [(i, t) for i, ks in enumerate(keyed) for t in range(1, len(ks))]
    shared = sum(1 for i, t in steps_t1 if len(visitors[keyed[i][t][0]]) >= 2)
    all_steps = [(i, t) for i, ks in enumerate(keyed) for t in range(len(ks))]
    frontier = sum(1 for i, t in all_steps if keyed[i][t][0] not in reach)
    return dict(steps=len(all_steps), shared_step_frac=shared / max(len(steps_t1), 1),
                nodes_per_step=len({k[0] for ks in keyed for k in ks}) / max(len(all_steps), 1),
                frontier_step_frac=frontier / max(len(all_steps), 1),
                success=sum(tr['won'] for tr in trajs) / len(trajs))


def replay_test(base, trajs, n_checks, rng):
    """Restore random mid-episode states by replaying the action prefix on a fresh
    single-game env; check that the anchor string matches the logged one."""
    ok, tot, ms = 0, 0, []
    cands = [(tr, t) for tr in trajs for t in range(1, len(tr['steps']))]
    for tr, t in rng.sample(cands, min(n_checks, len(cands))):
        t0 = time.time()
        env = single_game_env(base, tr['gamefile'])
        obs, infos = env.reset(); trk = Tracker()
        for s in tr['steps'][:t]:
            obs, _, dones, infos = env.step([s['action']]); trk.step(s['action'], obs[0]); trk.done = bool(dones[0])
        anchor = anchor_string(obs[0], trk, infos['admissible_commands'][0])
        ok += int(anchor == tr['steps'][t]['anchor']); tot += 1
        ms.append((time.time() - t0) * 1000 / max(t, 1))
        env.close()
    return dict(replay_checks=tot, replay_match=ok / max(tot, 1), replay_ms_per_step=sum(ms) / max(len(ms), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verl-agent', required=True)
    ap.add_argument('--games', type=int, default=16)
    ap.add_argument('--group', type=int, default=8)
    ap.add_argument('--max-steps', type=int, default=50)
    ap.add_argument('--policies', default='uniform,expert0.5,expert0.8')
    ap.add_argument('--replay-checks', type=int, default=24)
    ap.add_argument('--game-offset', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='alfworld_merge.json')
    args = ap.parse_args()
    rng = random.Random(args.seed)
    base = make_base_env(args.verl_agent, use_expert=True)
    env = base.init_env(batch_size=1)
    results = {}
    for pol in args.policies.split(','):
        t0 = time.time(); rows = []; all_trajs = []
        for k in range(args.game_offset, args.game_offset + args.games):
            trajs = run_group(env, k, args.group, pol, rng, args.max_steps)
            all_trajs += trajs
            ex, sim = group_stats(trajs, False), group_stats(trajs, True)
            print(f"  [{pol}] game {k}: success {ex['success']:.2f} shared exact {ex['shared_step_frac']:.2f} sim {sim['shared_step_frac']:.2f}", flush=True)
            rows.append(dict(game=k, gamefile=trajs[0]['gamefile'].split('/')[-3], success=ex['success'],
                             exact_shared=ex['shared_step_frac'], sim_shared=sim['shared_step_frac'],
                             exact_nodes_per_step=ex['nodes_per_step'], sim_nodes_per_step=sim['nodes_per_step'],
                             exact_frontier=ex['frontier_step_frac'], sim_frontier=sim['frontier_step_frac'],
                             steps=ex['steps']))
        rp = replay_test(base, all_trajs, args.replay_checks, rng)
        mean = lambda k: sum(r[k] for r in rows) / len(rows)
        results[pol] = dict(summary={k: mean(k) for k in ['success', 'exact_shared', 'sim_shared', 'exact_nodes_per_step',
                                                          'sim_nodes_per_step', 'exact_frontier', 'sim_frontier', 'steps']},
                            replay=rp, per_game=rows, seconds=time.time() - t0)
        s = results[pol]['summary']
        print(f"[{pol}] success {s['success']:.2f} | shared steps exact {s['exact_shared']:.2f} sim {s['sim_shared']:.2f} | "
              f"nodes/step exact {s['exact_nodes_per_step']:.2f} sim {s['sim_nodes_per_step']:.2f} | "
              f"replay match {rp['replay_match']:.2f} ({rp['replay_ms_per_step']:.0f} ms/step) | {time.time()-t0:.0f}s", flush=True)
    json.dump(results, open(args.out, 'w'), indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
