from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load model and tokenizer
model_path = "/data2/ygong/code/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
model.eval()

class TextRequest(BaseModel):
    text: str
    max_length: int = 50

@app.post("/generate")
async def generate_text(request: TextRequest):
    input_ids = tokenizer.encode(request.text, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=request.max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
