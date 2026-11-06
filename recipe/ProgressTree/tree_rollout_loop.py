"""Tree rollouts for verl-agent: main group rollouts, then branches from selected visited states.

TreeTrajectoryCollector is a drop-in replacement for recipe.GraphGPO.rollout_loop.
TrajectoryCollector. With branching disabled (arm graphgpo / progress) it returns exactly
what the original collector returns, plus a few bookkeeping fields (step_idx, is_branch,
parent_traj_uid, branch_step) and a per-iteration CSV log used for equal-GPU-hours
comparisons. With branching enabled it:
  1. runs the usual G rollouts per task (vanilla_multi_turn_loop, unchanged);
  2. picks branch points per group (core_progress.select_branch_points: smart / spread /
     random), each realised as (parent rollout, step t);
  3. restores idle env workers to those states in waves (branch_envs), generating only
     for active rows, and continues each branch until done or the episode step limit;
  4. returns main rows + branch rows in one DataProto. Branch rows keep the parent's uid
     (same task group) and get their own traj_uid; their episode reward is the branch's
     own outcome.
"""
import csv
import os
import time
import uuid
from collections import defaultdict

import numpy as np

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from recipe.GraphGPO.rollout_loop import TrajectoryCollector
from recipe.GraphGPO.utils import to_list_of_dict, torch_to_numpy

from recipe.ProgressTree.core_progress import (select_branch_points, make_canonicalizer_factory, to_hashable, SUCCESS,
                                               group_shared_step_frac)

BRANCH_MODE = {'progress_tree': 'smart', 'progress_tree_spread': 'spread', 'tree_grpo': 'random',
               'tree_graphgpo': 'smart'}


class TreeTrajectoryCollector(TrajectoryCollector):
    def __init__(self, config, tokenizer, processor=None):
        super().__init__(config, tokenizer, processor)
        pt = config.algorithm.get('progress_tree', {}) or {}
        self.pt = pt
        self.arm = pt.get('arm', 'graphgpo')
        self.branch_mode = pt.get('branch_mode', None) or BRANCH_MODE.get(self.arm)
        self.rng = np.random.default_rng(int(pt.get('seed', 0)))
        gcfg = config.algorithm.get('graphgpo', {}) or {}
        self.canon_factory = make_canonicalizer_factory(gcfg.get('enable_similarity', False),
                                                        gcfg.get('similarity_thresh', 0.95))
        self.trainer = None
        self.t0 = None
        self.val_seconds = 0.0
        self.iter_times = []          # full training-step times (rollout + update + logging), validation excluded
        self._step_start = None
        self._val_at_step_start = 0.0
        self.iteration = 0
        self.last_adv_metrics = {}
        self.pending_val = {}
        log_dir = pt.get('log_dir', None) or config.trainer.get('default_local_dir', '.')
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, 'progress_tree_log.csv')
        self._log_fields = None

    # ---------------------------------------------------------------- trainer hooks -----
    def attach_trainer(self, trainer):
        self.trainer = trainer

    def on_validation(self, metrics, seconds, global_step):
        if self.t0 is not None:            # the clock starts with the first training rollout
            self.val_seconds += seconds
        keep = {k: float(v) for k, v in (metrics or {}).items() if 'success_rate' in k or k.startswith('val')}
        row = dict(kind='val', global_step=global_step, train_seconds=self._train_seconds(), val_seconds=seconds)
        row.update(keep)
        self._log(row)

    def _train_seconds(self):
        return 0.0 if self.t0 is None else time.time() - self.t0 - self.val_seconds

    def _wall_clock_guard(self):
        max_h = self.pt.get('max_hours', None)
        if not max_h or self.trainer is None or not self.iter_times:
            return
        avg = float(np.mean(self.iter_times[-5:]))
        # Called at the start of step k, after upstream has already decided whether step k is the
        # last one, so capping total_training_steps here can only stop after step k+1. Steps k and
        # k+1 must both fit: stop after k+1 once a step k+2 would not (3 x avg). Because the guard
        # did not fire at step k-1, steps k and k+1 then end within the budget.
        if self._train_seconds() + 3.0 * avg > float(max_h) * 3600.0:
            self.trainer.total_training_steps = min(self.trainer.total_training_steps, self.trainer.global_steps + 1)
            print(f"[progress_tree] wall-clock budget {max_h} h reached: global step "
                  f"{self.trainer.global_steps + 1} will be the last one", flush=True)

    def _log(self, row):
        try:
            if self._log_fields is None and os.path.exists(self.log_path):
                # a previous run used this directory: keep its log aside, start a fresh one
                os.replace(self.log_path, self.log_path.replace('.csv', f'.{int(time.time())}.csv'))
            if self._log_fields is None or set(row) - set(self._log_fields):
                # (re)write the header when new columns appear, keeping earlier rows
                old = []
                if os.path.exists(self.log_path):
                    with open(self.log_path) as f:
                        old = list(csv.DictReader(f))
                self._log_fields = sorted(set(row) | set(self._log_fields or []) | {k for r in old for k in r})
                with open(self.log_path, 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=self._log_fields)
                    w.writeheader()
                    for r in old:
                        w.writerow(r)
            with open(self.log_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=self._log_fields).writerow(row)
        except Exception as e:                                           # logging must never kill training
            print(f"[progress_tree] log failed: {e}", flush=True)

    # ---------------------------------------------------------------- main entry ------
    def multi_turn_loop(self, gen_batch, actor_rollout_wg, envs, is_train=True):
        if not is_train:
            return super().multi_turn_loop(gen_batch, actor_rollout_wg, envs, is_train=False)
        if self.t0 is None:
            self.t0 = time.time()
        t_start = time.time()
        if self._step_start is not None:      # the previous training step, start to start, minus validation
            self.iter_times.append(t_start - self._step_start - (self.val_seconds - self._val_at_step_start))
        self._step_start, self._val_at_step_start = t_start, self.val_seconds
        self._wall_clock_guard()
        if self.config.algorithm.filter_groups.enable:
            raise NotImplementedError("progress_tree: dynamic sampling (filter_groups) is not supported")

        gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        lists, ep_rewards, ep_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
            gen_batch=gen_batch, actor_rollout_wg=actor_rollout_wg, envs=envs)
        for rows in lists:
            for t, d in enumerate(rows):
                d['step_idx'] = t
                d['is_branch'] = False
                d['parent_traj_uid'] = ''
                d['branch_step'] = -1
        groups = self._groups(lists, ep_rewards)
        merge_stat = group_shared_step_frac(groups, self.canon_factory)
        out = self.gather_rollout_data(lists, ep_rewards, ep_lengths, success, traj_uid, tool_callings)
        main_steps = int(sum(ep_lengths))
        stats = dict(branch_points=0, branch_steps=0, branch_success=float('nan'), restore_mismatch=0,
                     branch_waves=0, branch_seconds=0.0)
        if self.branch_mode is not None:
            if not getattr(envs, 'supports_branching', False):
                raise RuntimeError("progress_tree: this environment manager cannot restore states "
                                   "(branching is implemented for ALFWorld only)")
            tb = time.time()
            br, stats = self.branch_rollouts(gen_batch, actor_rollout_wg, envs, lists, ep_rewards, traj_uid, success, groups)
            stats['branch_seconds'] = time.time() - tb
            if br is not None:
                out = DataProto.concat([out, br])
        dt = time.time() - t_start
        self.iteration += 1
        row = dict(kind='train', iteration=self.iteration,
                   global_step=getattr(self.trainer, 'global_steps', self.iteration),
                   train_seconds=self._train_seconds(), rollout_seconds=dt, main_steps=main_steps,
                   main_success=float(np.mean(ep_rewards > 0)), main_episodes=len(ep_rewards),
                   main_shared_step_frac=merge_stat,
                   rows=len(out), arm=self.arm)
        row.update(stats)
        row.update({k: float(v) for k, v in self.last_adv_metrics.items()})
        self._log(row)
        return out

    # ---------------------------------------------------------------- branching -------
    @staticmethod
    def _groups(lists, ep_rewards):
        """{uid: [(env_index, anchors, nexts)]} over the active steps of the main rollouts."""
        groups = defaultdict(list)
        for i, rows in enumerate(lists):
            act = [d for d in rows if d['active_masks']]
            if not act:
                continue
            anchors = [to_hashable(d['anchor_obs']) for d in act]
            nexts = [to_hashable(d['next_obs']) for d in act]
            if ep_rewards[i] > 0:
                nexts[-1] = SUCCESS
            groups[rows[0]['uid']].append((i, anchors, nexts))
        return groups

    def branch_rollouts(self, gen_batch, actor_rollout_wg, envs, lists, ep_rewards, traj_uid, success, groups=None):
        pt = self.pt
        n_env = len(lists)
        G = max(int(self.config.env.rollout.n), 1)
        max_steps = int(self.config.env.max_steps)
        uid_of = [rows[0]['uid'] for rows in lists]
        groups = groups if groups is not None else self._groups(lists, ep_rewards)
        points = select_branch_points(groups, mode=self.branch_mode, max_nodes=int(pt.get('branch_max_nodes', 12)),
                                      K=int(pt.get('branch_K', 1)), canon_factory=self.canon_factory, rng=self.rng,
                                      min_remaining=1, max_steps=max_steps)
        stats = dict(branch_points=len(points), branch_steps=0, branch_success=float('nan'), restore_mismatch=0,
                     branch_waves=0)
        if not points:
            return None, stats
        horizon_cap = pt.get('branch_horizon', None)
        snap = envs.snapshot()
        b_lists, b_rewards, b_lengths, b_uids, b_tools = [], [], [], [], []
        try:
            for w0 in range(0, len(points), n_env):
                wave = points[w0:w0 + n_env]
                stats['branch_waves'] += 1
                specs, meta, free = [None] * n_env, [None] * n_env, set(range(n_env))
                for (g, p, t, s) in wave:
                    grp = p // G
                    same = [j for j in range(grp * G, min(grp * G + G, n_env)) if j in free]
                    j = same[0] if same else min(free)
                    free.discard(j)
                    specs[j], meta[j] = (p, t), (g, p, t)
                obs, infos, anchors = envs.restore(specs, snap)
                active = np.zeros(n_env, dtype=bool)
                for j, m in enumerate(meta):
                    if m is None:
                        continue
                    g, p, t = m
                    if to_hashable(anchors[j]) == to_hashable(lists[p][t]['anchor_obs']):
                        active[j] = True
                    else:
                        stats['restore_mismatch'] += 1
                steps_left = np.array([0 if m is None else max_steps - m[2] for m in meta])
                if horizon_cap:
                    steps_left = np.minimum(steps_left, int(horizon_cap))
                res = self._run_wave(gen_batch, actor_rollout_wg, envs, obs, active, steps_left, meta, uid_of, traj_uid)
                for j, (rows, rew, length) in res.items():
                    if rows:
                        b_lists.append(rows)
                        b_rewards.append(rew)
                        b_lengths.append(length)
                        b_uids.append(rows[0]['traj_uid'])
                        b_tools.append(0.0)
        finally:
            envs.end_branching()
        if not b_lists:
            return None, stats
        stats['branch_steps'] = int(sum(b_lengths))
        stats['branch_success'] = float(np.mean(np.array(b_rewards) > 0))
        br = self.gather_rollout_data(b_lists, np.array(b_rewards, dtype=np.float32), np.array(b_lengths, dtype=np.float32),
                                      success, np.array(b_uids, dtype=object), np.array(b_tools, dtype=np.float32))
        return br, stats

    def _run_wave(self, gen_batch, actor_rollout_wg, envs, obs, active, steps_left, meta, uid_of, traj_uid):
        n_env = len(meta)
        is_done = ~active
        new_traj = np.array([str(uuid.uuid4()) for _ in range(n_env)], dtype=object)
        rows_out = {j: [] for j in range(n_env) if meta[j] is not None}
        rewards_out = defaultdict(float)
        n_taken = np.zeros(n_env, dtype=np.int64)
        for _ in range(int(steps_left.max(initial=0))):
            act_idx = np.where(~is_done)[0]
            if len(act_idx) == 0:
                break
            # tokenise and generate only for the active rows
            sub_gen = gen_batch.select_idxs(act_idx)
            sub_obs = {k: (None if v is None else [v[j] for j in act_idx]) for k, v in obs.items()}
            sub = self.preprocess_batch(gen_batch=sub_gen, obs=sub_obs)
            pop_nt = ["raw_prompt_ids"] + [k for k in ("multi_modal_data", "raw_prompt", "tools_kwargs")
                                           if k in sub.non_tensor_batch]
            sub_in = sub.pop(batch_keys=["input_ids", "attention_mask", "position_ids"], non_tensor_batch_keys=pop_nt)
            sub_in.meta_info = gen_batch.meta_info
            padded, pad = pad_dataproto_to_divisor(sub_in, actor_rollout_wg.world_size)
            sub_out = unpad_dataproto(actor_rollout_wg.generate_sequences(padded), pad_size=pad)
            sub = sub.union(sub_out)
            texts = self.tokenizer.batch_decode(sub.batch['responses'], skip_special_tokens=True)
            text_actions = [''] * n_env
            for k, j in enumerate(act_idx):
                text_actions[j] = texts[k]
            next_obs, rewards, dones, infos = envs.step(text_actions)
            rewards = np.asarray(torch_to_numpy(rewards)).reshape(-1)
            dones = np.asarray(torch_to_numpy(dones)).reshape(-1).astype(bool)
            valid = np.array([bool(infos[j].get('is_action_valid', True)) for j in act_idx], dtype=bool)
            anchor_next = next_obs.get('anchor', None)
            sub.non_tensor_batch['uid'] = np.array([uid_of[meta[j][1]] for j in act_idx], dtype=object)
            sub.non_tensor_batch['traj_uid'] = new_traj[act_idx]
            sub.non_tensor_batch['is_action_valid'] = valid
            sub.non_tensor_batch['rewards'] = rewards[act_idx].astype(object)
            sub.non_tensor_batch['active_masks'] = np.ones(len(act_idx), dtype=bool).astype(object)
            sub.non_tensor_batch['text_actions'] = np.array([text_actions[j] for j in act_idx], dtype=object)
            sub.non_tensor_batch['next_obs'] = np.array([anchor_next[j] for j in act_idx], dtype=object)
            sub.non_tensor_batch['step_idx'] = np.array([meta[j][2] + n_taken[j] for j in act_idx], dtype=object)
            sub.non_tensor_batch['is_branch'] = np.ones(len(act_idx), dtype=bool).astype(object)
            sub.non_tensor_batch['parent_traj_uid'] = np.array([traj_uid[meta[j][1]] for j in act_idx], dtype=object)
            sub.non_tensor_batch['branch_step'] = np.array([meta[j][2] for j in act_idx], dtype=object)
            for k, d in enumerate(to_list_of_dict(sub)):
                j = act_idx[k]
                rows_out[j].append(d)
                rewards_out[j] += float(rewards[j])
                n_taken[j] += 1
            is_done[act_idx] = np.logical_or(dones[act_idx], n_taken[act_idx] >= steps_left[act_idx])
            obs = next_obs
        return {j: (rows_out[j], rewards_out[j], int(n_taken[j])) for j in rows_out}
