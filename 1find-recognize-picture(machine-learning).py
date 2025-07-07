import os
import json
import requests
import base64
import io
from tqdm import tqdm
from PIL import Image

def process_images(directory):
    # 获取所有符合条件的文件
    files = [f for f in os.listdir(directory)
            if "India" in f and f.lower().endswith('.jpg')]
    
docker run --hostname=6204c62fb112 --env=PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin --env=GPG_KEY=7169605F62C751356D054A26A821E680E5FA6305 --env=PYTHON_VERSION=3.13.5 --env=PYTHON_SHA256=93e583f243454e6e9e4588ca2c2662206ad961659863277afcdb96801647d640 --volume=/Users/gy/Documents/volumes:/test --network=bridge -p 2000:1000 -it --restart=no --runtime=runc -d python:latest bash
    # 添加进度条
    for filename in tqdm(files, desc="Processing images"):
        if "India" in filename and filename.lower().endswith('.jpg'):
            json_path = os.path.join(directory, filename.replace('.jpg', '.json'))
            
            if not os.path.exists(json_path):
                continue
                
            # 读取json文件
            with open(json_path, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
                
            # 打开图片
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            
            # 处理每个文本框(带进度描述)
            for shape in tqdm(annotation['shapes'],
                            desc=f"Processing text boxes in {filename}",
                            leave=False):
                if shape['shape_type'] == 'polygon':
                    # 获取文本框坐标
                    points = shape['points']
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    
                    # 计算裁剪区域
                    left = min(x_coords)
                    top = min(y_coords)
                    right = max(x_coords)
                    bottom = max(y_coords)
                    
                    # 裁剪文本框区域
                    cropped = img.crop((left, top, right, bottom))
                    
                    # 转换为base64
                    buffered = io.BytesIO()
                    cropped.save(buffered, format="JPEG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    # 调用OCR API
                    payload = {
                        "Imgs": [img_base64],
                        "Param": {"UniModel": True}
                    }
                    response = requests.post(
                        "http://10.200.131.42:10030/TextRec",
                        json=payload
                    )
                    
                    # 处理OCR结果
                    if response.status_code == 200:
                        result = response.json()
                        if result['Result'] and len(result['Data']['TextLines']) > 0:
                            shape['group_id'] = result['Data']['TextLines'][0]['Text']
            
            # 保存修改后的json文件
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(annotation, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    target_dir = "/nfs6/pub/dataset/ocr/layout/"
    process_images(target_dir)
