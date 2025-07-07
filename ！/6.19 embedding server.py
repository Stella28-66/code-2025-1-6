import io
import logging
from PIL import Image
from typing import List, Dict, Any

from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, HTTPException
import psycopg2
from psycopg2.extras import DictCursor

import torch
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="图片相似度搜索API",
    description="接收图片并返回数据库中相似图片的API服务",
    version="1.0.0"
)

# 数据库配置
DB_URL = "postgres://postgres:crop1234@10.200.131.75:54333/postgres?sslmode=disable"

def get_db_connection():
    """获取数据库连接"""
    try:
        conn = psycopg2.connect(DB_URL)
        return conn
    except Exception as e:
        logger.error(f"数据库连接失败: {str(e)}")
        raise HTTPException(status_code=500, detail="数据库连接失败")

# 加载模型和处理器
model_name = "/nfs6/pub/services/LayoutEmbedding/docling-layout-heron"
try:
    image_processor = RTDetrImageProcessor.from_pretrained(model_name)
    model = RTDetrV2ForObjectDetection.from_pretrained(model_name).to("cuda:0")
    logger.info("模型加载成功")
except Exception as e:
    logger.error(f"模型加载失败: {str(e)}")
    raise RuntimeError("模型加载失败")

@app.post("/api/embed")
async def get_image_embedding(file: UploadFile = File(...)):
    """
    计算图片embedding
    - file: 上传的图片文件
    - 返回: JSON格式的embedding向量
    """
    try:
        logger.info(f"开始处理图片: {file.filename}")
        
        # 读取图片二进制数据
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 处理图片并获取模型输出
        inputs = image_processor(images=[image], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 获取encoder最后隐藏层并计算平均embedding
        last_hidden = outputs.encoder_last_hidden_state[-1]  # shape: [1, 256, 30, 30]
        embedding = torch.nn.functional.normalize(
            last_hidden.mean(dim=[2, 3]).squeeze(),
            p=2, dim=0
        ).tolist()  # shape: [256]
        
        logger.info(f"图片处理完成: {file.filename}")
        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "embedding": embedding
        })
        
    except torch.cuda.OutOfMemoryError:
        logger.error(f"CUDA内存不足: {file.filename}")
        raise HTTPException(
            status_code=500,
            detail="CUDA内存不足，请尝试使用更小的图片"
        )
    except Exception as e:
        logger.error(f"处理图片失败: {file.filename}, 错误: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"处理图片失败: {str(e)}"
        )
    @app.post("/api/search")
async def search_similar_images(file: UploadFile = File(...)):
    """
    搜索相似图片
    - file: 上传的图片文件
    - 返回: JSON格式的相似图片列表(最多5个)
    """
    try:
        logger.info(f"开始搜索相似图片: {file.filename}")
        
        # 读取图片并获取embedding
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        inputs = image_processor(images=[image], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        
        last_hidden = outputs.encoder_last_hidden_state[-1]
        query_embedding = torch.nn.functional.normalize(
            last_hidden.mean(dim=[2, 3]).squeeze(),
            p=2, dim=0
        ).tolist()

        # 查询数据库获取相似图片
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        cursor.execute("""
            SELECT image_path, embedding <=> %s AS distance
            FROM image_embedding
            ORDER BY distance ASC
            LIMIT 5
        """, (query_embedding,))
        
        results = cursor.fetchall()
        conn.close()

        # 格式化返回结果
        similar_images = [
            {
                "path": row["image_path"],
                "distance": float(row["distance"])
            } for row in results
        ]
        
        logger.info(f"找到 {len(similar_images)} 张相似图片")
        return JSONResponse(content={
            "status": "success",
            "query_image": file.filename,
            "similar_images": similar_images
        })
        
    except torch.cuda.OutOfMemoryError:
        logger.error(f"CUDA内存不足: {file.filename}")
        raise HTTPException(
            status_code=500,
            detail="CUDA内存不足，请尝试使用更小的图片"
        )
    except Exception as e:
        logger.error(f"搜索相似图片失败: {file.filename}, 错误: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"搜索相似图片失败: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("启动服务...")
    uvicorn.run(app, host="0.0.0.0", port=8003)  # 修改端口避免冲突