# ===== Optional LLM track: same environment rendered as text, LLM + LoRA policy =====
# The policy scores the four action words after a chat prompt (constrained
# action decoding). It exposes the same interface as MLPPolicy, so the rollout,
# graph, frontier-MC and PPO code above is reused unchanged.
import numpy as np
import torch
import torch.nn as nn

LEGEND = ("Legend: '#' wall, 'A' you, 'K' the correct key, 'X' a wrong key "
          "(picking it up makes the task impossible), '^' trap (freezes you), "
          "'D' locked door (walk into it while holding K), 'G' goal, '.' floor.")


def render_obs_text(o, S):
    g = o[:7 * S * S].reshape(7, S, S)
    inv = int(np.argmax(o[7 * S * S:7 * S * S + 3])); trapped = o[-1] > 0.5
    rows = []
    for r in range(S):
        line = ''
        for c in range(S):
            ch = '.'
            if g[0, r, c]: ch = '#'
            if g[1, r, c]: ch = '^'
            if g[3, r, c]: ch = 'D'
            if g[2, r, c]: ch = 'G'
            if g[4, r, c]: ch = 'K'
            if g[5, r, c]: ch = 'X'
            if g[6, r, c]: ch = 'A'
            line += ch
        rows.append(line)
    holding = ['nothing', 'the correct key K', 'the wrong key X'][inv]
    status = ' You are frozen in a trap.' if trapped else ''
    return '\n'.join(rows) + f"\nYou are holding {holding}.{status}"


class LLMPolicy(nn.Module):
    def __init__(self, model, tokenizer, S, action_words=('up', 'down', 'left', 'right'),
                 infer_batch=64):
        super().__init__()
        self.model, self.tok, self.S = model, tokenizer, S
        self.infer_batch = infer_batch
        self.tok.padding_side = 'left'
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        ids = [self._first_token(w) for w in action_words]
        if len(set(ids)) < len(ids):  # fall back to single letters if words collide
            action_words = ('U', 'D', 'L', 'R'); ids = [self._first_token(w) for w in action_words]
        assert len(set(ids)) == len(ids), "action tokens must be distinct"
        self.action_words = action_words
        self.register_buffer('action_ids', torch.tensor(ids), persistent=False)

    def _first_token(self, w):
        return self.tok.encode(w, add_special_tokens=False)[0]

    def prompt(self, o):
        user = (f"You control agent A in a grid world. Pick up the correct key K, open the door D, "
                f"then reach the goal G.\n{LEGEND}\n\n{render_obs_text(o, self.S)}\n\n"
                f"Reply with exactly one word: {', '.join(self.action_words)}.")
        if getattr(self.tok, 'chat_template', None):
            return self.tok.apply_chat_template([{'role': 'user', 'content': user}],
                                                tokenize=False, add_generation_prompt=True)
        return user + "\nAnswer: "

    def forward(self, obs):
        """Logits of the 4 action words at the last position only (no full-vocabulary projection)."""
        obs_np = obs.detach().float().cpu().numpy() if torch.is_tensor(obs) else obs
        enc = self.tok([self.prompt(o) for o in obs_np], return_tensors='pt', padding=True).to(self.action_ids.device)
        base = self.model.get_base_model() if hasattr(self.model, 'get_base_model') else self.model
        h = base.model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask']).last_hidden_state[:, -1, :]
        logits = h @ base.lm_head.weight[self.action_ids].T
        if getattr(base.lm_head, 'bias', None) is not None:
            logits = logits + base.lm_head.bias[self.action_ids]
        return logits.float()

    @torch.no_grad()
    def act(self, obs_np, device):
        logits = torch.cat([self.forward(obs_np[i:i + self.infer_batch])
                            for i in range(0, len(obs_np), self.infer_batch)])
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a.cpu().numpy(), dist.log_prob(a).cpu().numpy()


def _disable_incompatible_torchao():
    """Colab preinstalls an old torchao (e.g. 0.10). Recent PEFT raises ImportError
    from is_torchao_available() when it finds torchao < 0.16, even for plain,
    unquantized LoRA. We never use torchao, so if the installed version is too old,
    make PEFT treat it as absent. Works in a running session (no restart needed)."""
    import sys, importlib.metadata as md
    from packaging import version
    try:
        v = version.parse(md.version('torchao'))
    except md.PackageNotFoundError:
        return
    if v >= version.parse('0.16.0'):
        return
    import peft, peft.import_utils  # noqa: F401
    try:
        import peft.tuners.lora.torchao, peft.utils.quantization_utils  # noqa: F401
    except Exception:
        pass
    absent = lambda: False
    for name, mod in list(sys.modules.items()):
        if name.startswith('peft') and hasattr(mod, 'is_torchao_available'):
            setattr(mod, 'is_torchao_available', absent)
    print(f"[h2lab] torchao {v} is too old for this PEFT version; ignoring it (not needed for LoRA).")


def load_llm_policy(model_name, S, device, lora_r=16, infer_batch=256):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _disable_incompatible_torchao()
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(model_name)
    use_bf16 = device == 'cuda' and torch.cuda.is_bf16_supported()
    dt = torch.bfloat16 if use_bf16 else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dt)        # transformers >= 4.56
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dt)  # older versions
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0,
                                             target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']))
    return LLMPolicy(model, tok, S, infer_batch=infer_batch).to(device)