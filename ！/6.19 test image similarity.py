import requests

# 测试图片路径
# 使用项目demo目录中的测试图片
image_path = "demo/images/EDB_12_01_0014_25.jpg"

# FastAPI服务地址
service_url = "http://127.0.0.1:8099/find_similar"

# 发送测试请求
try:
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(service_url, files=files)
        
    if response.status_code == 200:
        result = response.json()
        print("测试成功！返回结果：")
        print(f"查询图片: {result['query_image']}")
        print("最相似的5张图片：")
        for i, img_path in enumerate(result['similar_images'], 1):
            print(f"{i}. {img_path}")
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"测试过程中出错: {str(e)}")