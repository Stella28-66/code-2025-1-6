import argparse
import os
import base64
import requests
import json
from tqdm import tqdm
from PIL import Image
from glob import glob

curdir = os.path.dirname(__file__)

def load_base64(file_path):
    with open(file_path, "rb") as File:
        str_base64 = base64.b64encode(File.read()).decode("utf-8")#字符串格式
    return str_base64

def convert_to_labelme(original_result, image_path, image_data=None):
    """Convert API annotation result to labelme format"""
    try:
        # Get image dimensions from the image file
        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size
        
        labelme_data = {
            "version": "5.3.1",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(image_path),
            "imageData": image_data,
            "imageHeight": height,
            "imageWidth": width
        }
        
        # Process text lines from API response
        text_lines = original_result.get('Data', {}).get('TextLines', [])
        
        for line in text_lines:
            # Get polygon points from TextRegion
            region = line.get('TextRegion', {})
            points = [
                region.get('P0', [0, 0]),
                region.get('P1', [0, 0]),
                region.get('P2', [0, 0]),
                region.get('P3', [0, 0])
            ]
            
            # Create shape for labelme
            shape = {
                "label": "text",
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            labelme_data['shapes'].append(shape)
            
        return labelme_data
        
    except Exception as e:
        print(f"Error converting annotation: {str(e)}")
        return None
def process_image(image_path, output_dir, url):
    """Process single image and save labelme json with enhanced error handling"""
    try:
        # Verify image exists
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return False
            
        # Load and encode image
        img_base64 = load_base64(image_path)
        if not img_base64:
            print(f"Failed to load image: {image_path}")
            return False
            
        # Prepare request data
        post_data = {
            "Img": img_base64,
        }
        
        # Send request with timeout
        try:
            req = requests.post(url, json=post_data, timeout=30)
            req.raise_for_status()
            result = req.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed for {image_path}: {str(e)}")
            return False
            
        # Convert to labelme format with flexible parsing
        labelme_data = convert_to_labelme(result, image_path)
        
        # Ensure basic structure exists
        if not isinstance(labelme_data, dict):
            print(f"Invalid conversion result for {image_path}")
            return False
            
        # Fill required fields if missing
        labelme_data.setdefault("version", "5.3.1")
        labelme_data.setdefault("flags", {})
        labelme_data.setdefault("shapes", [])
        labelme_data.setdefault("imagePath", os.path.basename(image_path))
            
        # Save result
        output_path = os.path.join(output_dir,
                                 os.path.splitext(os.path.basename(image_path))[0] + '.json')
        output_dir 
        with open(output_path, 'w') as f:
            json.dump(labelme_data, f, indent=2)
            
        return True
        
    except Exception as e:
        print(f"Unexpected error processing {image_path}: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to labelme format")
    parser.add_argument("--root_port", type=int, default=21777, help="the port of root service.")
    parser.add_argument("--input_dir", type=str, required=True, help="input image directory")
    parser.add_argument("--output_dir", type=str, help="output directory (default: same as input)")
    args = parser.parse_args()

    # Set output dir same as input if not specified


    output_dir = args.output_dir if args.output_dir else args.input_dir
    os.makedirs(output_dir, exist_ok=True)  #如果目录已存在，不会报错；如果目录不存在，则创建它

    # Service URL - fix IP/port formatting
    ip = "10.200.131.44"
    port = 10010  # Fixed port number
    root_port = args.root_port
    url = f'http://{ip}:{port}/TextDet'  # 格式化重点为f字符串格式化，使用花括号{}
    
    # Get all image paths
    image_paths = glob(os.path.join(args.input_dir, "*.jpg")) + \
                 glob(os.path.join(args.input_dir, "*.png"))
    # global修改全局变量，os.listdir获取目录内容，不递归遍历子目录
    # \表示两行实际为一行，由于后面的写不下，所以用\表示换行

    # Batch processing to avoid memory issues
    batch_size = 100 # 100张图片为一批次，防止内存溢出，batch size批处理大小
    total_images = len(image_paths)

    success = 0
    for i in tqdm(range(0, total_images, batch_size), desc="Processing batches"):  # tqdm显示进度条
        batch = image_paths[i:i + batch_size]
        for img_path in batch:
            try:
                if process_image(img_path, output_dir, url):
                    success += 1
            except Exception as e:
                print(f"\nError processing {img_path}: {str(e)}")
                continue

    print(f"\nFinished processing: {success}/{total_images} images successfully")
