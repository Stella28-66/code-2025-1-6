#！qwen3_debug_20250528_161646.log
2025-05-28 16:16:46,778 - Loading tokenizer...
2025-05-28 16:16:50,269 - Tokenizer loaded successfully
2025-05-28 16:16:50,269 - Loading model with debug hooks...
2025-05-28 16:17:04,169 - Model loaded successfully
2025-05-28 16:17:04,171 - Model loaded with debug hooks
2025-05-28 16:17:04,171 - 
=== MODEL ARCHITECTURE OVERVIEW ===
2025-05-28 16:17:04,171 - Model class: Qwen3ForCausalLM
2025-05-28 16:17:04,171 - Model device: cuda:9
2025-05-28 16:17:04,171 - Model dtype: torch.bfloat16
2025-05-28 16:17:04,172 - Number of parameters: 596,049,920
2025-05-28 16:17:04,173 - Number of trainable parameters: 596,049,920

2025-05-28 16:17:04,207 - Starting text generation...
2025-05-28 16:17:04,691 - 
=== EMBEDDING LAYER: model.embed_tokens ===
Input: ['shape=(1, 18) dtype=torch.int64 device=cuda:9']
Output: shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9

2025-05-28 16:17:04,986 - 
=== ATTENTION LAYER: model.layers.0.self_attn.q_proj ===
Input: ['shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9']
Output: shape=(1, 18, 2048) dtype=torch.bfloat16 device=cuda:9
Attention Weights: shape=(1, 18, 2048) dtype=torch.bfloat16 device=cuda:9

2025-05-28 16:17:04,991 - 
=== ATTENTION LAYER: model.layers.0.self_attn.k_proj ===
Input: ['shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9']
Output: shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9
Attention Weights: shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9

2025-05-28 16:17:04,991 - 
=== ATTENTION LAYER: model.layers.0.self_attn.v_proj ===
Input: ['shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9']
Output: shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9
Attention Weights: shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9

2025-05-28 16:17:05,011 - 
=== ATTENTION LAYER: model.layers.0.self_attn.o_proj ===
Input: ['shape=(1, 18, 2048) dtype=torch.bfloat16 device=cuda:9']
Output: shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9
Attention Weights: shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9

2025-05-28 16:17:05,011 - 
=== FEED_FORWARD LAYER: model.layers.0.mlp.gate_proj ===
Input: ['shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9']
Output: shape=(1, 18, 3072) dtype=torch.bfloat16 device=cuda:9

2025-05-28 16:17:05,017 - 
=== FEED_FORWARD LAYER: model.layers.0.mlp.up_proj ===
Input: ['shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9']
Output: shape=(1, 18, 3072) dtype=torch.bfloat16 device=cuda:9

2025-05-28 16:17:05,018 - 
=== FEED_FORWARD LAYER: model.layers.0.mlp.down_proj ===
Input: ['shape=(1, 18, 3072) dtype=torch.bfloat16 device=cuda:9']
Output: shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9

2025-05-28 16:17:05,018 - 
=== FEED_FORWARD LAYER: model.layers.0.mlp ===
Input: ['shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9']
Output: shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9

2025-05-28 16:17:05,018 - 
=== ATTENTION LAYER: model.layers.1.self_attn.q_proj ===
Input: ['shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9']
Output: shape=(1, 18, 2048) dtype=torch.bfloat16 device=cuda:9
Attention Weights: shape=(1, 18, 2048) dtype=torch.bfloat16 device=cuda:9

2025-05-28 16:17:05,019 - 
=== ATTENTION LAYER: model.layers.1.self_attn.k_proj ===
Input: ['shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9']
Output: shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9
Attention Weights: shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9

2025-05-28 16:17:05,019 - 
=== ATTENTION LAYER: model.layers.1.self_attn.v_proj ===
Input: ['shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9']
Output: shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9
Attention Weights: shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9

2025-05-28 16:17:05,020 - 
=== ATTENTION LAYER: model.layers.1.self_attn.o_proj ===
Input: ['shape=(1, 18, 2048) dtype=torch.bfloat16 device=cuda:9']
Output: shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9
Attention Weights: shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9

2025-05-28 16:17:05,021 - 
=== FEED_FORWARD LAYER: model.layers.1.mlp.gate_proj ===
Input: ['shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9']
Output: shape=(1, 18, 3072) dtype=torch.bfloat16 device=cuda:9

2025-05-28 16:17:05,021 - 
=== FEED_FORWARD LAYER: model.layers.1.mlp.up_proj ===
Input: ['shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9']
Output: shape=(1, 18, 3072) dtype=torch.bfloat16 device=cuda:9

2025-05-28 16:17:05,021 - 
=== FEED_FORWARD LAYER: model.layers.1.mlp.down_proj ===
Input: ['shape=(1, 18, 3072) dtype=torch.bfloat16 device=cuda:9']
Output: shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9

2025-05-28 16:17:05,021 - 
=== FEED_FORWARD LAYER: model.layers.1.mlp ===
Input: ['shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9']
Output: shape=(1, 18, 1024) dtype=torch.bfloat16 device=cuda:9
