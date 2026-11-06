#!/bin/bash
# One arm, one seed, fixed wall-clock budget, on ONE GPU (Colab G4 RTX PRO 6000 96 GB, A100 80 GB or H100).
#
#   bash recipe/ProgressTree/run_alfworld_1gpu.sh ARM SEED HOURS [extra hydra overrides]
#
#   ARM    graphgpo | progress | progress_tree | progress_tree_spread | tree_grpo | tree_graphgpo
#   SEED   game-sequence seed (use the same seeds for every arm: common random numbers)
#   HOURS  training wall-clock budget; validation time is excluded (equal GPU-hours protocol)
#
# Environment variables (defaults in brackets):
#   MODEL [Qwen/Qwen2.5-1.5B-Instruct]  TRAIN_TASKS [8]  GROUP [8]  VAL_TASKS [64]  MAX_STEPS [50]
#   OUT [$HOME/progress_tree_runs]  ATTN [flash_attention_2 | sdpa: see README, Blackwell fallback]
#   MINI [256] rows per PPO minibatch (sized in samples, as in verl-agent)  MICRO [16]  GPU_MEM [0.4]
set -x
ARM=${1:-progress_tree}; SEED=${2:-0}; HOURS=${3:-6}
shift 3 2>/dev/null || shift $#
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
TRAIN_TASKS=${TRAIN_TASKS:-8}
GROUP=${GROUP:-8}
VAL_TASKS=${VAL_TASKS:-64}
MAX_STEPS=${MAX_STEPS:-50}
OUT=${OUT:-$HOME/progress_tree_runs}
MINI=${MINI:-256}
MICRO=${MICRO:-16}
GPU_MEM=${GPU_MEM:-0.4}
ATTN=${ATTN:-flash_attention_2}
RUN_NAME="${ARM}_s${SEED}"
mkdir -p "$OUT/$RUN_NAME"
export VERL_ATTN_IMPL=$ATTN
if [ "$ATTN" = "sdpa" ]; then REMOVE_PADDING=False; else REMOVE_PADDING=True; fi
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

python3 -m examples.data_preprocess.prepare --mode 'text' \
    --train_data_size $TRAIN_TASKS --val_data_size $VAL_TASKS

python3 -m recipe.ProgressTree.main_progress_tree \
    algorithm.progress_tree.arm=$ARM \
    algorithm.progress_tree.seed=$SEED \
    algorithm.progress_tree.max_hours=$HOURS \
    algorithm.progress_tree.reward_scale=10.0 \
    algorithm.progress_tree.log_dir="$OUT/$RUN_NAME" \
    algorithm.adv_estimator=graphgpo \
    algorithm.gamma=0.10 \
    algorithm.use_kl_in_reward=False \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$TRAIN_TASKS \
    data.val_batch_size=$VAL_TASKS \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=$REMOVE_PADDING \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=$SEED \
    env.max_steps=$MAX_STEPS \
    env.rollout.n=$GROUP \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='progress_tree_alfworld' \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=10000 \
    trainer.default_local_dir="$OUT/$RUN_NAME" \
    trainer.val_before_train=True "$@" 2>&1 | tee "$OUT/${RUN_NAME}.console.log"
