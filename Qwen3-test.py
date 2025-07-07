# ！test_Qwen3-0.6B.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from datetime import datetime

from transformers import Qwen3ForCausalLM

# 在文件顶部添加
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding
)

# Setup logging with both file and console output
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(f'/data2/syzeng/code/qwen3_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(console_handler)

# Enhanced debug hook function with detailed layer tracking
def debug_hook(module, input, output, name):
    # Skip if not a tensor operation
    if not any(isinstance(i, torch.Tensor) for i in input if isinstance(i, (torch.Tensor, list, tuple))):
        return
    
    # Process input shapes
    def process_tensor(t):
        return f"shape={tuple(t.shape)} dtype={t.dtype} device={t.device}"
        
    input_shapes = []
    for i in input:
        if isinstance(i, torch.Tensor):
            input_shapes.append(process_tensor(i))
        elif isinstance(i, (list, tuple)) and any(isinstance(x, torch.Tensor) for x in i):
            tensor_shapes = [process_tensor(x) for x in i if isinstance(x, torch.Tensor)]
            input_shapes.append(f"list[len={len(i)}] tensors: {tensor_shapes}")
    
    # Process output shape
    if isinstance(output, torch.Tensor):
        output_shape = process_tensor(output)
    elif isinstance(output, (list, tuple)) and any(isinstance(x, torch.Tensor) for x in output):
        tensor_shapes = [process_tensor(x) for x in output if isinstance(x, torch.Tensor)]
        output_shape = f"list[len={len(output)}] tensors: {tensor_shapes}"

# Enhanced logging for different layer types
    if "embed" in name.lower():
        layer_type = "EMBEDDING"
        logging.info(f"\n=== {layer_type} LAYER: {name} ===\nInput: {input_shapes}\nOutput: {output_shape}\n")
    elif "attn" in name.lower():
        layer_type = "ATTENTION"
        attn_weights = output
        if isinstance(attn_weights, torch.Tensor):
            # Get q,k,v projections if available
            qkv_info = ""
            if hasattr(module, 'in_proj_weight'):
                qkv_info = f"\nQKV Projections: in_proj_weight={process_tensor(module.in_proj_weight)}"
            elif hasattr(module, 'q_proj'):
                q_info = f"q_proj={process_tensor(module.q_proj.weight)}" if hasattr(module.q_proj, 'weight') else ""
                k_info = f"k_proj={process_tensor(module.k_proj.weight)}" if hasattr(module.k_proj, 'weight') else ""
                v_info = f"v_proj={process_tensor(module.v_proj.weight)}" if hasattr(module.v_proj, 'weight') else ""
                qkv_info = f"\nQKV Projections: {q_info}, {k_info}, {v_info}"
            
            logging.info(f"\n=== {layer_type} LAYER: {name} ===\n"
                        f"Input: {input_shapes}\n"
                        f"Output: {output_shape}\n"
                        f"Attention Weights: {process_tensor(attn_weights)}"
                        f"{qkv_info}\n")
    elif "mlp" in name.lower() or "ffn" in name.lower():
        layer_type = "FEED_FORWARD"
        logging.info(f"\n=== {layer_type} LAYER: {name} ===\nInput: {input_shapes}\nOutput: {output_shape}\n")
    elif "norm" in name.lower():
        layer_type = "NORMALIZATION"
        logging.info(f"\n=== {layer_type} LAYER: {name} ===\nInput: {input_shapes}\nOutput: {output_shape}\n")
    elif "output" in name.lower() or "project" in name.lower():
        layer_type = "OUTPUT_PROJECTION"
        logging.info(f"\n=== {layer_type} LAYER: {name} ===\nInput: {input_shapes}\nOutput: {output_shape}\n")
    else:
        layer_type = "GENERAL"
        logging.info(f"\n=== {layer_type} LAYER: {name} ===\nInput: {input_shapes}\nOutput: {output_shape}\n")

model_name = "/data2/syzeng/code/Qwen3-0.6B/output/v8-20250612-134428/checkpoint-21/"

# load the tokenizer and the model from local files with progress feedback
try:
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    logging.info("Tokenizer loaded successfully")
    
    logging.info("Loading model with debug hooks...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True
    )
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise

# Register debug hooks for all relevant layers with more comprehensive coverage
for name, module in model.named_modules():
    if isinstance(module, (Qwen3Attention, Qwen3MLP, torch.nn.Linear, torch.nn.LayerNorm, torch.nn.Embedding,
                         torch.nn.MultiheadAttention, torch.nn.Conv1d, torch.nn.Dropout,
                         torch.nn.ModuleList, torch.nn.ModuleDict)):
        module.register_forward_hook(
            lambda module, input, output, name=name: debug_hook(module, input, output, name)
        )
logging.info("Model loaded with debug hooks")

# Log model architecture overview
logging.info("\n=== MODEL ARCHITECTURE OVERVIEW ===")
logging.info(f"Model class: {model.__class__.__name__}")
logging.info(f"Model device: {model.device}")
logging.info(f"Model dtype: {model.dtype}")
logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
logging.info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

# prepare the model input
prompt = "How's everthing going?"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion with error handling
try:
    logging.info("Starting text generation...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128,  # Reduced for faster debugging
        early_stopping=True
    )


    logging.info("Text generation completed successfully")
except Exception as e:
    logging.error(f"Error during text generation: {str(e)}")
    raise
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
