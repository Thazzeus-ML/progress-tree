"""Restorable ALFWorld environments for tree rollouts (branching from visited states).

A branch restores an environment to the state a main rollout was in before its t-th
action, by replaying that rollout's first t environment actions on a private single-game
TextWorld env (ALFWorld is deterministic with domain_randomization=False; the collector
checks every restore by comparing the rebuilt GraphGPO anchor with the logged one). The
worker's main env is left untouched, so the group-seeded game sequence used by later
resets is unchanged. The environment manager also rebuilds the agent's prompt state
(history memory, task text, last observation) from a snapshot taken after the main
rollouts, so a branch's first prompt is exactly the prompt its parent saw at step t.
"""
import copy
from collections import OrderedDict

import ray

from recipe.GraphGPO.alfworld_envs import AlfworldEnvs, AlfworldWorker, load_config_file
from recipe.GraphGPO.graphgpo_env_manager import AlfWorldEnvironmentManager
from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment


class BranchableAlfworldWorker(AlfworldWorker):
    def __init__(self, config, seed, base_env, max_cached_envs=1):
        super().__init__(config, seed, base_env)
        self._base_env = base_env
        self._main_env = self.env
        self._branch_envs = OrderedDict()
        self._max_cached = max_cached_envs
        self._idle = False
        self._last = None

    # -- bookkeeping so an idle worker can answer step() without touching any env --------
    def reset(self):
        obs, infos = super().reset()
        self._last = (obs, infos)
        return obs, infos

    def step(self, action):
        if self._idle:
            obs, infos = copy.deepcopy(self._last)
            return obs, [0.0], [True], infos
        obs, scores, dones, infos = super().step(action)
        self._last = (obs, infos)
        return obs, scores, dones, infos

    # -- branching ---------------------------------------------------------------------
    def _env_for(self, gamefile):
        env = self._branch_envs.pop(gamefile, None)
        if env is None:
            files = self._base_env.game_files
            self._base_env.game_files = [gamefile]
            try:
                env = self._base_env.init_env(batch_size=1)
            finally:
                self._base_env.game_files = files
        self._branch_envs[gamefile] = env
        while len(self._branch_envs) > self._max_cached:
            _, old = self._branch_envs.popitem(last=False)
            try:
                old.close()
            except Exception:
                pass
        return env

    def restore(self, gamefile, actions):
        """Replay `actions` on a fresh episode of `gamefile`; later step() calls continue it."""
        self._idle = False
        self.env = self._env_for(gamefile)
        obs, infos = AlfworldWorker.reset(self)
        for a in actions:
            obs, _, _, infos = AlfworldWorker.step(self, a)
        self._last = (obs, infos)
        return obs, infos

    def set_idle(self):
        self._idle = True
        return copy.deepcopy(self._last)

    def end_branch(self):
        self._idle = False
        self.env = self._main_env
        for env in self._branch_envs.values():
            try:
                env.close()
            except Exception:
                pass
        self._branch_envs.clear()
        return True


class BranchableAlfworldEnvs(AlfworldEnvs):
    """AlfworldEnvs whose workers can be restored to logged states (see module docstring)."""

    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        if not ray.is_initialized():
            ray.init()
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n
        env_worker = ray.remote(**resources_per_worker)(BranchableAlfworldWorker)
        self.workers = [env_worker.remote(config, seed + (i // self.group_n), base_env) for i in range(self.num_processes)]
        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    supports_branching = True

    def restore(self, specs):
        """specs[i] is None (worker idles: its step() returns its last result, done=True) or
        (gamefile, [env actions]). Returns (text_obs list, info list) for every worker."""
        futures = [w.set_idle.remote() if sp is None else w.restore.remote(*sp) for w, sp in zip(self.workers, specs)]
        text_obs, infos = [], []
        for i, (obs, info) in enumerate(ray.get(futures)):
            info = {k: v[0] for k, v in info.items()}
            text_obs.append(obs[0])
            infos.append(info)
            self.prev_admissible_commands[i] = info['admissible_commands']
        return text_obs, infos

    def end_branching(self):
        ray.get([w.end_branch.remote() for w in self.workers])


def build_branchable_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    cls = BranchableAlfworldEnvs if is_train else AlfworldEnvs
    return cls(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)


class BranchableAlfWorldEnvironmentManager(AlfWorldEnvironmentManager):
    """Adds snapshot()/restore()/end_branching() to verl-agent's ALFWorld manager."""

    @property
    def supports_branching(self):
        return getattr(self.envs, 'supports_branching', False)

    def snapshot(self):
        return dict(memory=[[dict(r) for r in m] for m in self.memory._data],
                    tasks=list(self.tasks), gamefile=list(self.gamefile))

    def anchors_from(self, text_obs, infos):
        lh, il = self.get_location_holding(infos)
        new_text = [f"{o} {a} {b}" for o, a, b in zip(text_obs, lh, il)]
        anchors = [f"{o} {a} {b} {','.join(sorted(c))}"
                   for o, a, b, c in zip(text_obs, lh, il, self.envs.get_admissible_commands)]
        return new_text, anchors

    def restore(self, specs, snap):
        """specs[i] = None (idle) or (parent_env_index, t). Rebuilds env i in the state the
        parent's episode had before its t-th action, including the prompt history.
        Returns (observations like reset()/step(), infos, anchors)."""
        env_specs = []
        for sp in specs:
            if sp is None:
                env_specs.append(None)
            else:
                p, t = sp
                env_specs.append((snap['gamefile'][p], [rec['action'] for rec in snap['memory'][p][:t]]))
        text_obs, infos = self.envs.restore(env_specs)
        for i, sp in enumerate(specs):
            if sp is not None:
                p, t = sp
                self.memory._data[i] = [dict(r) for r in snap['memory'][p][:t]]
                self.tasks[i] = snap['tasks'][p]
                self.gamefile[i] = snap['gamefile'][p]
        new_text, anchors = self.anchors_from(text_obs, infos)
        self.pre_text_obs = new_text
        cmds = self.envs.get_admissible_commands
        with_hist = self.build_text_obs(new_text, cmds, init=False)
        no_hist = self.build_text_obs(new_text, cmds, init=True)
        full = [no_hist[i] if (sp is not None and sp[1] == 0) else with_hist[i] for i, sp in enumerate(specs)]
        return {'text': full, 'image': None, 'anchor': anchors}, infos, anchors

    def end_branching(self):
        self.envs.end_branching()
