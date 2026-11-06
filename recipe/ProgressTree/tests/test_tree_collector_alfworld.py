"""CPU integration test: tree rollouts on real ALFWorld through verl-agent's own code paths,
with a mock LLM (random admissible actions, some malformed) instead of vLLM.

Checks
  * main + branch rows come back as one DataProto (same keys, concatenation works);
  * every restore reproduces the logged state (anchor match) and every branch's first
    prompt is token-for-token the prompt its parent saw at the branch step;
  * reward manager -> invalid-action penalty -> Progress+Tree advantage runs on the batch;
  * after branching, the next iteration's groups still play one game per group;
  * the CSV log is written.

Run from the verl-agent repo root with recipe/ProgressTree installed and ALFWORLD_DATA set:
    python recipe/ProgressTree/tests/test_tree_collector_alfworld.py
"""
import os
import re
import sys
import tempfile
import time

import numpy as np
import torch

REPO = os.getcwd()
sys.path.insert(0, REPO)
os.environ['PYTHONPATH'] = REPO + os.pathsep + os.environ.get('PYTHONPATH', '')

import ray                                                        # noqa: E402
from omegaconf import OmegaConf                                   # noqa: E402
from verl import DataProto                                        # noqa: E402


def make_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = ["You arrive at countertop 1. On the countertop 1, you see a mug 1, and a apple 2.",
              "go to cabinet 3", "take mug 1 from countertop 1", "put mug 1 in/on coffeemachine 1",
              "Your admissible actions of the current situation are", "<think> </think> <action> </action>"] * 50
    tok.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=700, special_tokens=['<pad>', '<eos>', '<|user|>', '<|assistant|>'],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token='<pad>', eos_token='<eos>')
    hf.chat_template = ("{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}"
                        "{% if add_generation_prompt %}<|assistant|>{% endif %}")
    return hf


class MockActor:
    """Stands in for actor_rollout_wg: picks a random admissible action from the prompt."""
    world_size = 1

    def __init__(self, tok, response_length=48, invalid_p=0.1, seed=0):
        self.tok, self.L, self.p, self.rng = tok, response_length, invalid_p, np.random.default_rng(seed)
        self.calls = 0

    def generate_sequences(self, prompts):
        self.calls += 1
        ids, attn, pos = prompts.batch['input_ids'], prompts.batch['attention_mask'], prompts.batch['position_ids']
        texts = self.tok.batch_decode(ids, skip_special_tokens=True)
        resp = []
        for text in texts:
            tail = text[text.rfind('admissible actions'):]
            acts = re.findall(r"'([^']+)'", tail)
            a = acts[int(self.rng.integers(len(acts)))] if acts else 'look'
            resp.append(f"<think>ok</think><action>{a}</action>" if self.rng.random() > self.p else f"I will {a}")
        pad, eos = self.tok.pad_token_id, self.tok.eos_token_id
        out = torch.full((len(texts), self.L), pad, dtype=ids.dtype)
        for i, r in enumerate(resp):
            enc = (self.tok.encode(r, add_special_tokens=False) + [eos])[:self.L]
            out[i, :len(enc)] = torch.tensor(enc)
        rmask = (out != pad).to(attn.dtype)
        rpos = pos[:, -1:] + torch.arange(1, self.L + 1).unsqueeze(0)
        return DataProto.from_dict(tensors=dict(prompts=ids, responses=out, input_ids=torch.cat([ids, out], 1),
                                                attention_mask=torch.cat([attn, rmask], 1),
                                                position_ids=torch.cat([pos, rpos], 1)))


def make_config(arm, log_dir, n_tasks, G, max_steps):
    return OmegaConf.create(dict(
        data=dict(max_prompt_length=8192, truncation='error', return_raw_chat=True, train_batch_size=n_tasks),
        env=dict(env_name='alfworld/AlfredTWEnv', seed=0, max_steps=max_steps, history_length=2,
                 rollout=dict(n=G), resources_per_worker=dict(num_cpus=0.1)),
        algorithm=dict(filter_groups=dict(enable=False), adv_estimator='graphgpo',
                       graphgpo=dict(enable_similarity=False, similarity_thresh=0.95),
                       progress_tree=dict(arm=arm, branch_max_nodes=3, branch_K=1, log_dir=log_dir, seed=0,
                                          sigma_min=0.2, reward_scale=10.0)),
        trainer=dict(default_local_dir=log_dir)))


def gen_batch_for(n_tasks):
    return DataProto.from_dict(tensors=dict(input_ids=torch.zeros(n_tasks, 4, dtype=torch.long)),
                               non_tensors=dict(raw_prompt=np.array([[{'role': 'user', 'content': 'x'}]] * n_tasks, dtype=object),
                                                data_source=np.array(['alfworld'] * n_tasks, dtype=object)))


def main():
    from functools import partial
    import agent_system.environments.env_package.alfworld as alf_pkg
    from agent_system.environments.env_package.alfworld import alfworld_projection
    from agent_system.reward_manager import EpisodeRewardManager
    from recipe.GraphGPO.graphgpo_ray_trainer import apply_invalid_action_penalty, compute_response_mask
    from recipe.ProgressTree.branch_envs import BranchableAlfWorldEnvironmentManager, build_branchable_alfworld_envs
    from recipe.ProgressTree.core_progress import compute_progress_tree_advantage
    from recipe.ProgressTree.tree_rollout_loop import TreeTrajectoryCollector

    n_tasks, G, max_steps = 2, 2, 12
    ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True,
             runtime_env=dict(env_vars=dict(PYTHONPATH=os.environ['PYTHONPATH'],
                                            ALFWORLD_DATA=os.environ.get('ALFWORLD_DATA', ''))))
    tok = make_tokenizer()
    log_dir = tempfile.mkdtemp(prefix='pt_test_')
    alf_cfg = os.path.join(os.path.dirname(os.path.abspath(alf_pkg.__file__)), 'configs/config_tw.yaml')
    t0 = time.time()
    _envs = build_branchable_alfworld_envs(alf_cfg, 0, n_tasks, G, {'num_cpus': 0.1}, is_train=True,
                                           env_kwargs={'eval_dataset': 'eval_in_distribution'})
    print(f'envs up in {time.time() - t0:.0f}s')

    for arm in ('progress_tree', 'tree_grpo'):
        cfg = make_config(arm, log_dir, n_tasks, G, max_steps)
        envs = BranchableAlfWorldEnvironmentManager(_envs, partial(alfworld_projection), cfg)
        col = TreeTrajectoryCollector(cfg, tok)
        actor = MockActor(tok, seed=1)
        for it in range(2):
            t0 = time.time()
            out = col.multi_turn_loop(gen_batch_for(n_tasks), actor, envs, is_train=True)
            nt = out.non_tensor_batch
            is_b = nt['is_branch'].astype(bool)
            print(f'[{arm} it{it}] rows {len(out)} (branch rows {is_b.sum()}) in {time.time() - t0:.0f}s')
            import csv
            last = list(csv.DictReader(open(col.log_path)))[-1]
            assert int(float(last['restore_mismatch'])) == 0, last
            assert int(float(last['branch_points'])) > 0 and is_b.sum() > 0, last
            assert 0.0 <= float(last['main_shared_step_frac']) <= 1.0, last
            print(f"  main_shared_step_frac {float(last['main_shared_step_frac']):.2f}")
            # main rows of a group share the start state (one game per group), also after branching
            for u in set(nt['uid']):
                starts = {nt['anchor_obs'][i] for i in range(len(out)) if nt['uid'][i] == u and not is_b[i] and nt['step_idx'][i] == 0}
                assert len(starts) == 1, (u, len(starts))
            # each branch's first prompt == its parent's prompt at the branch step
            prompts = out.batch['prompts']
            idx = {(nt['traj_uid'][i], int(nt['step_idx'][i])): i for i in range(len(out))}
            n_checked = 0
            for tr in set(nt['traj_uid'][is_b]):
                rows = [i for i in range(len(out)) if nt['traj_uid'][i] == tr]
                first = min(rows, key=lambda i: nt['step_idx'][i])
                p, t = nt['parent_traj_uid'][first], int(nt['branch_step'][first])
                assert int(nt['step_idx'][first]) == t
                j = idx[(p, t)]
                assert nt['anchor_obs'][first] == nt['anchor_obs'][j]
                assert torch.equal(prompts[first], prompts[j]), 'branch prompt differs from parent prompt'
                n_checked += 1
            print(f'  {n_checked} branches: restored state and first prompt identical to the parent at the branch step')
            # credit
            out.batch['response_mask'] = compute_response_mask(out)
            out.batch['token_level_scores'] = EpisodeRewardManager(tok, 0)(out)
            out, _ = apply_invalid_action_penalty(out, invalid_action_penalty_coef=0.1)
            out.batch['token_level_rewards'] = out.batch['token_level_scores']
            mode = 'tree_grpo' if arm == 'tree_grpo' else 'progress_loo'
            adv, ret, m = compute_progress_tree_advantage(out, mode=mode, invalid_action_penalty_coef=0.1, reward_scale=10.0)
            assert adv.shape == out.batch['responses'].shape and torch.isfinite(adv).all()
            # the penalty enters exactly as in GraphGPO's episode-level term
            from gigpo.core_gigpo import episode_norm_reward
            from recipe.ProgressTree.core_progress import graphgpo_penalty_term
            mask = out.batch['response_mask']
            ep = episode_norm_reward(out.batch['token_level_rewards'], mask, nt['uid'], nt['traj_uid'], 1e-6, remove_std=False)
            ep_row = ((ep * mask).sum(-1) / mask.sum(-1).clamp(min=1)).numpy()
            pen = -0.1 * (1 - nt['is_action_valid'].astype(float))
            R = out.batch['token_level_rewards'].sum(-1).numpy() - pen
            mine = graphgpo_penalty_term(nt['uid'], R, pen)
            for u in set(nt['uid']):
                ix = np.array([i for i in range(len(out)) if nt['uid'][i] == u])
                sc = R[ix] + pen[ix]
                mine[ix] += (R[ix] - R[ix].mean()) / (sc.std(ddof=1) + 1e-6)
            assert np.allclose(mine, ep_row, atol=1e-4), np.abs(mine - ep_row).max()
            per_row = (adv * out.batch['response_mask']).sum(-1) / out.batch['response_mask'].sum(-1).clamp(min=1)
            inval = ~nt['is_action_valid'].astype(bool)
            print(f'  credit ok: {m}; invalid rows {inval.sum()} mean adv {per_row[torch.tensor(inval)].mean():.3f} '
                  f'vs valid {per_row[torch.tensor(~inval)].mean():.3f}')
    print('log rows:', sum(1 for _ in open(col.log_path)) - 1, '->', col.log_path)
    print('integration test passed')
    ray.shutdown()


if __name__ == '__main__':
    main()
