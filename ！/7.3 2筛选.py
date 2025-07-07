import os
import json
import shutil

def filter_and_move_files(src_dir, dest_dir, keywords):
    """
    筛选并移动符合条件的OCR文件
    
    参数:
        src_dir: 源目录路径
        dest_dir: 目标目录路径
        keywords: 需要匹配的关键词列表
    """
    # 确保目标目录存在
    os.makedirs(dest_dir, exist_ok=True)
    
    # 遍历源目录中所有以India开头的json文件
    for filename in os.listdir(src_dir):
        if filename.startswith('India') and filename.endswith('.json'):
            json_path = os.path.join(src_dir, filename)
            jpg_path = os.path.join(src_dir, filename.replace('.json', '.jpg'))
            
            try:
                # 读取json文件
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 检查是否有匹配的group_id
                has_match = any(
                    any(shape.get('group_id') and keyword in shape['group_id'].lower() for keyword in keywords)
                    for shape in data.get('shapes', [])
                )
                
                if has_match:
                    # 移动json文件
                    dest_json = os.path.join(dest_dir, filename)
                    shutil.move(json_path, dest_json)
                    
                    # 移动对应的jpg文件
                    if os.path.exists(jpg_path):
                        dest_jpg = os.path.join(dest_dir, filename.replace('.json', '.jpg'))
                        shutil.move(jpg_path, dest_jpg)
                    
                    print(f"Moved: {filename}")
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    # 设置目录路径
    src_directory = '/nfs6/pub/dataset/ocr/layout/'
    dest_directory = '/nfs6/pub/dataset/ocr/revised/'
    
    # 设置关键词
    target_keywords = ['significant', 'accompanying']
    
    # 执行筛选和移动
    filter_and_move_files(src_directory, dest_directory, target_keywords)
    print("Processing completed.")