# from transformers import AutoModelForCausalLM, AutoTokenizer
# checkpoint = "HuggingFaceTB/SmolLM-1.7B-Instruct"

# device = "cuda" # for GPU usage or "cpu" for CPU usage
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# # for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
# model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# messages = [{"role": "user", "content": "What is the capital of France."}]
# input_text=tokenizer.apply_chat_template(messages, tokenize=False)
# print(input_text)
# inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
# outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
# print(tokenizer.decode(outputs[0]))



# print("==="*10)
# print("==="*10)
# print("==="*10)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

# Load the model and tokenizer
checkpoint = "HuggingFaceTB/SmolLM-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
device = "cuda"  # or "cpu"

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x / torch.sqrt(mean_sq + self.eps)
        return x_normed * self.weight

class SelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5  # Scaling factor
        self.layer_idx = layer_idx

        # Rotary Embedding
        self.rotary_emb = LlamaRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_type="default",
            scaling_factor=1.0,
            device=device,
            config=config
        )

        # Projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Dropout
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(self, x, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        batch_size, seq_length, _ = x.size()

        # Project inputs
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split into heads
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Rotary Embedding
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device).unsqueeze(0)
        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle past_key_value
        if past_key_value is not None:
            # Update the cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)
        else:
            past_key_value = None

        # Compute scaled dot-product attention
        attn_scores = torch.matmul(q * self.scale, k.transpose(-2, -1))

        if attention_mask is not None:
            # Adjust attention_mask shape to match attn_scores
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask[:, None, :, :]
            attn_scores = attn_scores + attention_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Multiply by values
        attn_output = torch.matmul(attn_probs, v)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)

        # Output projection
        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        x = self.gate_proj(x) * self.act_fn(self.up_proj(x))
        x = self.down_proj(x)
        return x

class CustomLlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = SelfAttention(config, layer_idx)
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(
        self,
        x,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs
    ):
        if position_ids is None:
            position_ids = torch.arange(x.size(1), dtype=torch.long, device=x.device).unsqueeze(0)

        residual = x
        x = self.input_norm(x)
        x, _, past_key_value = self.self_attn(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        x = residual + x

        residual = x
        x = self.post_attn_norm(x)
        x = self.mlp(x)
        x = residual + x

        outputs = (x,)

        if output_attentions:
            outputs += (None,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs

# Replace the model's decoder layers with the custom layers
model = model.to(device)
from tqdm import tqdm
for i in tqdm(range(len(model.model.layers)), total=len(model.model.layers)):
    original_layer = model.model.layers[i]
    custom_layer = CustomLlamaDecoderLayer(model.config, layer_idx=i).to(device).to(model.dtype)
    # print("=")
    print('orig layer state dict keys:', original_layer.state_dict().keys())
    print('custom layer state dict keys:', custom_layer.state_dict().keys())

    # Copy weights
    # print([x for x in dir(original_layer.self_attn.q_proj) if '_' not in x])
    # print([x for x in dir(original_layer.self_attn.q_proj.bias) if '_' not in x])
    # print([x for x in dir(original_layer.self_attn.k_proj) if '_' not in x])
    # print([x for x in dir(original_layer.self_attn.v_proj) if '_' not in x])
    # print([x for x in dir(original_layer.self_attn.o_proj) if '_' not in x])
    # print([x for x in dir(original_layer.mlp.gate_proj) if '_' not in x])
    # print([x for x in dir(original_layer.mlp.up_proj) if '_' not in x])
    # print([x for x in dir(original_layer.mlp.down_proj) if '_' not in x])
    print([x for x in dir(original_layer.input_layernorm) if '_' not in x])
    print([x for x in dir(original_layer.post_attention_layernorm) if '_' not in x])

    custom_layer.self_attn.q_proj.weight.data.copy_(original_layer.self_attn.q_proj.weight.data)
    custom_layer.self_attn.k_proj.weight.data.copy_(original_layer.self_attn.k_proj.weight.data)
    custom_layer.self_attn.v_proj.weight.data.copy_(original_layer.self_attn.v_proj.weight.data)
    custom_layer.self_attn.out_proj.weight.data.copy_(original_layer.self_attn.o_proj.weight.data)

    custom_layer.self_attn.q_proj.bias = original_layer.self_attn.q_proj.bias
    custom_layer.self_attn.k_proj.bias = original_layer.self_attn.k_proj.bias
    custom_layer.self_attn.v_proj.bias = original_layer.self_attn.v_proj.bias
    custom_layer.self_attn.out_proj.bias = original_layer.self_attn.o_proj.bias

    custom_layer.mlp.gate_proj.weight.data.copy_(original_layer.mlp.gate_proj.weight.data)
    custom_layer.mlp.up_proj.weight.data.copy_(original_layer.mlp.up_proj.weight.data)
    custom_layer.mlp.down_proj.weight.data.copy_(original_layer.mlp.down_proj.weight.data)

    custom_layer.mlp.gate_proj.bias = original_layer.mlp.gate_proj.bias
    custom_layer.mlp.up_proj.bias = original_layer.mlp.up_proj.bias
    custom_layer.mlp.down_proj.bias = original_layer.mlp.down_proj.bias

    custom_layer.input_norm.weight.data.copy_(original_layer.input_layernorm.weight.data)
    custom_layer.post_attn_norm.weight.data.copy_(original_layer.post_attention_layernorm.weight.data)

    # Replace the layer
    model.model.layers[i] = custom_layer


# Prepare input
messages = [{"role": "user", "content": "What is the capital of France."}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

# Create attention mask
attention_mask = torch.ones_like(inputs)

# Generate output
model.eval()
outputs = model.generate(
    inputs,
    attention_mask=attention_mask,
    max_new_tokens=50,
    temperature=0.2,
    top_p=0.9,
    do_sample=True,
    use_cache=True,
)
print(tokenizer.decode(outputs[0]))
