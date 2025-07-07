from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import requests
import psycopg2
from psycopg2.extras import DictCursor
import os
import tempfile
import logging

app = FastAPI()

# 配置参数
EMBEDDING_URL = 'http://10.200.131.45:8003/api/embed'  # 用户指定的embedding服务地址
DB_URL = "postgres://postgres:crop1234@10.200.131.75:54333/postgres?sslmode=disable"  # 更新为任务指定的数据库地址

async def get_image_embedding(image_path: str) -> list:
    """获取图片的embedding向量"""
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(EMBEDDING_URL, files=files)
            if response.status_code == 200:
                return response.json()['embedding']
    except Exception as e:
        print(f"Error processing image: {str(e)}")
    return None

async def find_similar_images(embedding: list, top_k: int = 5) -> list:
    """查询数据库中相似的图片"""
    print("Finding similar images...")
    conn = None
    try:
        conn = psycopg2.connect(DB_URL)
        print("Connected to the database")
        with conn.cursor(cursor_factory=DictCursor) as cursor:
            # 使用pgvector的余弦相似度查询
            query = f"""
            SELECT image_path
            FROM image_embeddings
            ORDER BY embedding <=> '{embedding}'::vector
            LIMIT {top_k}
            """
            cursor.execute(query)
            print("Query executed")
            # 获取结果集
            results = cursor.fetchall()
            print("Results fetched")
            return [row['image_path'] for row in results]
    except Exception as e:
        print(f"Database query error: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()
    @app.post("/find_similar")
async def handle_similarity_request(file: UploadFile = File(...)):
    """处理相似图片查询请求"""
    logging.info(f"开始处理图片相似度查询请求，文件名: {file.filename}")
    
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        logging.error(f"无效的文件类型: {file.filename}")
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name
            logging.info(f"已保存临时文件到: {temp_path}")
        
        # 获取embedding
        logging.info("开始获取图片embedding")
        embedding = await get_image_embedding(temp_path)
        if not embedding:
            logging.error("获取embedding失败")
            raise HTTPException(status_code=500, detail="Failed to get embedding")
        logging.info("成功获取图片embedding")

        # 查询相似图片
        logging.info("开始查询相似图片")
        similar_images = await find_similar_images(embedding)
        logging.info(f"找到 {len(similar_images)} 张相似图片")
        
        return JSONResponse({
            "query_image": file.filename,
            "similar_images": similar_images,
            "status": "success"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    uvicorn.run(app, host="0.0.0.0", port=8099)