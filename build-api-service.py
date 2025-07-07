from fastapi import FastAPI
from pydantic import BaseModel
import hashlib

app = FastAPI()

class RequestData(BaseModel):
    text: str
    repeat: int = 1

@app.post("/md5")
async def calculate_md5(data: RequestData):
    current_hash = data.text
    for _ in range(data.repeat):
        current_hash = hashlib.md5(current_hash.encode()).hexdigest()
        result=current_hash*data.repeat
    return {
        "original_text": data.text,
        "repeat_times": data.repeat,
        "final_hash": result
    }

@app.post("/md6")
async def calculate_md5(data: RequestData):
    current_hash = data.text
    for _ in range(data.repeat):
        current_hash = hashlib.md5(current_hash.encode()).hexdigest()
    return {
        "original_text": data.text,
        "repeat_times": data.repeat,
        "final_hash": current_hash
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
