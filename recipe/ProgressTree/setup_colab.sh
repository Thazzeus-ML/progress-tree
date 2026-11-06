#!/bin/bash
# Colab / single-GPU setup for the Progress+Tree ALFWorld runs.
#   bash setup_colab.sh /path/to/ProgressTree   (the folder this script came in)
# Installs verl-agent (as in its README: vllm 0.11.0, flash-attn, editable install, ALFWorld),
# copies this recipe to verl-agent/recipe/ProgressTree, lets verl's actor read its attention
# implementation from $VERL_ATTN_IMPL (Blackwell fallback), and runs the GPU preflight.
set -e
SRC=${1:-$(cd "$(dirname "$0")" && pwd)}
cd ${WORKDIR:-$HOME}
[ -d verl-agent ] || git clone https://github.com/langfengQ/verl-agent.git
cd verl-agent
pip install -q vllm==0.11.0
# Prebuilt flash-attn wheels exist for some torch/CUDA/GPU combinations only; try newer first.
pip install -q flash-attn==2.8.3 --no-build-isolation --no-cache-dir \
  || pip install -q flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir \
  || echo "WARNING: flash-attn could not be installed; verl's actor imports it (see README)."
pip install -q -e .
pip install -q gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld rapidfuzz
[ -d "${ALFWORLD_DATA:-$HOME/.cache/alfworld}/json_2.1.1" ] || alfworld-download -f
rm -rf recipe/ProgressTree && cp -r "$SRC" recipe/ProgressTree
# attention implementation switch (default unchanged: flash_attention_2)
grep -q VERL_ATTN_IMPL verl/workers/fsdp_workers.py || sed -i \
  's/attn_implementation="flash_attention_2"/attn_implementation=__import__("os").environ.get("VERL_ATTN_IMPL", "flash_attention_2")/' \
  verl/workers/fsdp_workers.py
python recipe/ProgressTree/preflight_gpu.py
