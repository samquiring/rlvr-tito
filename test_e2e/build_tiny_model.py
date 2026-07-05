"""Build a tiny Qwen3-architecture model + tokenizer locally (no downloads).

- Qwen3Config from transformers, ~1M params, random init
- word-level tokenizer over a small vocab, Qwen-style special tokens
- chat template that mimics Qwen3's history rewriting: <think> blocks are
  STRIPPED from previous assistant turns but kept in the current generation.
  This reproduces the exact property that breaks whole-conversation masking.
"""

import json
import os

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

OUT = os.path.abspath("./tiny-qwen3")

# ---- vocab: special tokens + digits + small word list --------------------
words = (
    [str(i) for i in range(10)]
    + "the a is are you i say now ready number word please yes no ok and then "
      "password secret it that this to of in READY seven hello".split()
    + list(".,:!?()<>/")
)
specials = ["<pad>", "<|im_start|>", "<|im_end|>", "<think>", "</think>",
            "system", "user", "assistant", "\n"]
vocab = {tok: i for i, tok in enumerate(specials + sorted(set(words)))}

tok_model = models.WordLevel(vocab=vocab, unk_token="<pad>")
tk = Tokenizer(tok_model)
tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

# Qwen3-style template, including the critical behavior: think content is
# removed from PRIOR assistant messages when re-rendering the conversation.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' }} {{ message['role'] }} {{ '\n' }}"
    "{% if message['role'] == 'assistant' and not loop.last %}"
    "{% set c = message['content'] %}"
    "{% if '</think>' in c %}{% set c = c.split('</think>')[-1] %}{% endif %}"
    "{{ c }}"
    "{% else %}{{ message['content'] }}{% endif %}"
    "{{ '\n' }} {{ '<|im_end|>' }} {{ '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>' }} assistant {{ '\n' }}"
    "{% endif %}"
)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tk,
    pad_token="<pad>", eos_token="<|im_end|>",
    additional_special_tokens=["<|im_start|>", "<think>", "</think>"],
)
tokenizer.chat_template = CHAT_TEMPLATE

cfg = Qwen3Config(
    vocab_size=len(vocab),
    hidden_size=128, intermediate_size=256,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
    head_dim=32, max_position_embeddings=512,
    pad_token_id=vocab["<pad>"], eos_token_id=vocab["<|im_end|>"],
    tie_word_embeddings=True,
)
model = Qwen3ForCausalLM(cfg)

os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)
print(json.dumps({
    "params": sum(p.numel() for p in model.parameters()),
    "vocab_size": len(vocab),
    "saved_to": OUT,
}))

# quick template sanity: think content must vanish from prior turns only
msgs = [
    {"role": "user", "content": "say READY"},
    {"role": "assistant", "content": "<think> secret plan </think> READY"},
    {"role": "user", "content": "now say the number 7"},
]
rendered = tokenizer.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
assert "secret plan" not in rendered and "READY" in rendered
print("think-stripping template verified")
