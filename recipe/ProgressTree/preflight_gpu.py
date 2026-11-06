"""GPU preflight for the Progress+Tree ALFWorld runs (run from the verl-agent repo root).

Checks, in order, and prints what to set if something fails:
  1. CUDA device and bf16 matmul;
  2. flash-attn kernels (verl's actor uses attn_implementation=flash_attention_2);
  3. vLLM generation with a small Qwen model and the configured attention backend;
  4. ALFWorld game loading + one step, and a state restore by replay.

    python recipe/ProgressTree/preflight_gpu.py [--model Qwen/Qwen2.5-0.5B-Instruct] [--skip-vllm]
"""
import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.getcwd())


def check(name, fn):
    t = time.time()
    try:
        msg = fn()
        print(f"[ OK ] {name} ({time.time() - t:.1f}s) {msg or ''}", flush=True)
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc(limit=2)
        return False


def cuda():
    import torch
    assert torch.cuda.is_available(), "no CUDA device"
    p = torch.cuda.get_device_properties(0)
    a = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
    (a @ a).sum().item()
    return f"{p.name}, sm_{p.major}{p.minor}, {p.total_memory / 2**30:.0f} GiB, torch {torch.__version__}"


def flash():
    import torch
    from flash_attn import flash_attn_func
    q = torch.randn(1, 128, 8, 64, device='cuda', dtype=torch.bfloat16)
    flash_attn_func(q, q, q, causal=True).sum().item()
    import flash_attn
    return f"flash_attn {flash_attn.__version__}"


def vllm_gen(model):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, gpu_memory_utilization=0.3, max_model_len=2048, enforce_eager=True)
    out = llm.generate(["Say hello in one word."], SamplingParams(max_tokens=8, temperature=0))
    import vllm
    return f"vllm {vllm.__version__}, backend env={os.environ.get('VLLM_ATTENTION_BACKEND')}: {out[0].outputs[0].text!r}"


def alfworld():
    import random
    import yaml
    import agent_system.environments.env_package.alfworld as alf_pkg
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(alf_pkg.__file__)), 'configs/config_tw.yaml')
    cfg = yaml.safe_load(open(cfg_path))
    base = get_environment(cfg['env']['type'])(cfg, train_eval='train')
    env = base.init_env(batch_size=1)
    env.seed(0)
    obs, infos = env.reset()
    gamefile = infos['extra.gamefile'][0]
    rng = random.Random(0)
    acts = []
    for _ in range(5):
        a = rng.choice([c for c in infos['admissible_commands'][0] if c != 'help'])
        obs, _, _, infos = env.step([a])
        acts.append(a)
    files = base.game_files
    base.game_files = [gamefile]
    env2 = base.init_env(batch_size=1)
    base.game_files = files
    obs2, infos2 = env2.reset()
    for a in acts:
        obs2, _, _, infos2 = env2.step([a])
    assert obs2 == obs and infos2['admissible_commands'] == infos['admissible_commands'], "replay mismatch"
    return "game loaded, 5 steps, replay restore identical"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    ap.add_argument('--skip-vllm', action='store_true')
    args = ap.parse_args()
    ok = {'cuda': check('CUDA', cuda)}
    ok['flash'] = check('flash-attn kernels', flash)
    if not args.skip_vllm:
        ok['vllm'] = check('vLLM generation', lambda: vllm_gen(args.model))
    ok['alfworld'] = check('ALFWorld + replay restore', alfworld)
    print()
    if not ok['flash']:
        print("-> flash-attn kernels unavailable on this GPU (e.g. Blackwell sm_120 with an older wheel):\n"
              "   run with ATTN=sdpa (the setup script patched verl to read VERL_ATTN_IMPL; this also turns off\n"
              "   use_remove_padding), or use an A100/H100 runtime.")
    if not ok.get('vllm', True):
        print("-> vLLM failed: try `export VLLM_ATTENTION_BACKEND=FLASHINFER` or `TRITON_ATTN` (or unset it),\n"
              "   then re-run this check; if it still fails on this GPU, use an A100/H100 runtime.")
    if not ok['alfworld']:
        print("-> ALFWorld: run `alfworld-download -f` (data goes to ~/.cache/alfworld or $ALFWORLD_DATA).")
    print("preflight:", "all good" if all(ok.values()) else "see above")


if __name__ == '__main__':
    main()
