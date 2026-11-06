"""Entry point: verl-agent GraphGPO trainer + Progress+Tree arms.

    python3 -m recipe.ProgressTree.main_progress_tree algorithm.progress_tree.arm=progress_tree ...

Arms (algorithm.progress_tree.arm):
  graphgpo              upstream GraphGPO (credit and rollouts unchanged; adds logging + budget guard)
  progress              progress credit with leave-one-out baselines, no extra rollouts
  progress_tree         progress credit + branches from the most-visited states (v2 concept)
  progress_tree_spread  progress credit + branches spread along each trajectory
  tree_grpo             Tree-GRPO-style proxy: random branch points + intra/inter-tree advantages
  tree_graphgpo         attribution arm: same branches as progress_tree, GraphGPO credit

Everything is applied by patching upstream modules inside the Ray TaskRunner process before
the upstream TaskRunner logic runs, so the upstream recipe files stay untouched.
"""
import time

import hydra
import ray


def _original_class(actor_cls):
    """The undecorated class behind a @ray.remote class (Ray 2.4x layouts)."""
    for attr in ('__ray_actor_class__',):
        c = getattr(actor_cls, attr, None)
        if isinstance(c, type):
            return c
    meta = getattr(actor_cls, '__ray_metadata__', None)
    mc = getattr(meta, 'modified_class', None) if meta is not None else None
    if mc is not None:
        return getattr(mc, '__ray_actor_class__', mc)
    raise RuntimeError("cannot find the class behind recipe.GraphGPO.main_graphgpo.TaskRunner; "
                       "please report your ray version")


def apply_patches(config):
    import recipe.GraphGPO.alfworld_envs as alf_envs
    import recipe.GraphGPO.core_graph as core_graph
    import recipe.GraphGPO.graphgpo_env_manager as env_mgr
    import recipe.GraphGPO.graphgpo_ray_trainer as trainer_mod
    import recipe.GraphGPO.main_graphgpo as main_mod
    import recipe.GraphGPO.rollout_loop as rollout_mod
    import torch

    from recipe.ProgressTree.branch_envs import BranchableAlfWorldEnvironmentManager, build_branchable_alfworld_envs
    from recipe.ProgressTree.core_progress import compute_progress_tree_advantage
    from recipe.ProgressTree.tree_rollout_loop import TreeTrajectoryCollector

    pt = config.algorithm.get('progress_tree', {}) or {}
    arm = pt.get('arm', 'graphgpo')
    known = ('graphgpo', 'progress', 'progress_tree', 'progress_tree_spread', 'tree_grpo', 'tree_graphgpo')
    if arm not in known:
        raise ValueError(f"unknown progress_tree.arm={arm}; choose one of {known}")
    assert str(config.algorithm.adv_estimator) in ('graphgpo', 'AdvantageEstimator.GraphGPO'), \
        "keep algorithm.adv_estimator=graphgpo (the Progress+Tree arms replace its computation)"

    rollout_mod.TrajectoryCollector = TreeTrajectoryCollector
    alf_envs.build_alfworld_envs = build_branchable_alfworld_envs
    env_mgr.AlfWorldEnvironmentManager = BranchableAlfWorldEnvironmentManager

    class ProgressTreeTrainer(main_mod.RayPPOTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if hasattr(self.traj_collector, 'attach_trainer'):
                self.traj_collector.attach_trainer(self)
            # let the estimator report its diagnostics to the collector's CSV log
            trainer_mod.compute_advantage.collector = self.traj_collector

        def _validate(self):
            t = time.time()
            metrics = super()._validate()
            if hasattr(self.traj_collector, 'on_validation'):
                self.traj_collector.on_validation(metrics, time.time() - t, self.global_steps)
            return metrics

    main_mod.RayPPOTrainer = ProgressTreeTrainer

    if arm in ('graphgpo', 'tree_graphgpo'):
        return                      # upstream GraphGPO credit (graph built from all rows, branches included)

    # The graph is not used by these arms: skip building it (it can be slow with similarity merging).
    class _NoGraph:
        @staticmethod
        def from_data(*a, **k):
            return {}

        @staticmethod
        def Calculate_all_shortest_path(Gs):
            return None

    core_graph.ComplexGraph = _NoGraph
    core_graph.compute_graph_path_returns = lambda batch, Gs, gamma, normalize_distance: torch.zeros(
        len(batch), dtype=torch.float32, device=batch.batch['input_ids'].device)

    gcfg = config.algorithm.get('graphgpo', {}) or {}
    actor = config.actor_rollout_ref.actor
    mode = 'tree_grpo' if arm == 'tree_grpo' else pt.get('credit', 'progress_loo')
    original = trainer_mod.compute_advantage

    def compute_advantage(data, adv_estimator, *args, **kwargs):
        if "response_mask" not in data.batch:
            data.batch["response_mask"] = trainer_mod.compute_response_mask(data)
        adv, ret, metrics = compute_progress_tree_advantage(
            data, mode=mode,
            enable_similarity=gcfg.get('enable_similarity', False),
            similarity_thresh=gcfg.get('similarity_thresh', 0.95),
            sigma_min=float(pt.get('sigma_min', 0.2)),
            reward_scale=pt.get('reward_scale', None),
            invalid_action_penalty_coef=float(actor.get('invalid_action_penalty_coef', 0.0)),
            use_invalid_action_penalty=bool(actor.get('use_invalid_action_penalty', True)),
            penalty_weight=float(pt.get('penalty_weight', 1.0)))
        data.batch['advantages'] = adv
        data.batch['returns'] = ret
        col = getattr(compute_advantage, 'collector', None)
        if col is not None:
            col.last_adv_metrics = metrics
        return data

    compute_advantage.original = original
    trainer_mod.compute_advantage = compute_advantage


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        apply_patches(config)
        import recipe.GraphGPO.main_graphgpo as main_mod
        return _original_class(main_mod.TaskRunner).run(self, config)


@hydra.main(config_path="config", config_name="progress_tree_trainer", version_base=None)
def main(config):
    import recipe.GraphGPO.main_graphgpo as main_mod
    from omegaconf import OmegaConf
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env = OmegaConf.merge(default_runtime_env, ray_init_kwargs.get("runtime_env", {}))
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


if __name__ == "__main__":
    main()
