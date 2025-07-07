
import os
import numpy as np
from sklearn.cluster import KMeans
from shutil import copy2

#Paths
images_dir='demo/images'
embeddings_path='demo/images_embedding.txt'
output_dir='demo/images-cluster'

#get the number of images
image_files=os.listdir(images_dir)
N=len(image_files)
n_clusters=N//2

#parse embeddings file
image_names=[]
embeddings=[]
with open(embeddings_path,'r')as f:
    for line in f:
        parts=line.strip().split('\t')
        if len(parts)==2:
            image_names.append(parts[0])
            embeddings.append([float(x) for x in parts[1].split(',')])

#convert to numpy array
X=np.array(embeddings)

#perform K-means clustering
kmeans=KMeans(n_clusters=n_clusters,random_state=42)#创建Kmeans聚类器
clusters=kmeans.fit_predict(X)#执行聚类并预测

#create output directory structure
os.makedirs(output_dir,exist_ok=True)
for cluster_num in range (n_clusters):
    cluster_dir=os.path.join(output_dir,str(cluster_num))
    os.makedirs(cluster_dir,exist_ok=True)

#copy images to cluster directories
for image_name,cluster_num in zip(image_names,clusters):
    src_path=os.path.join(images_dir,image_name)
    dst_path=os.path.join(output_dir,str(cluster_num),image_name)
    copy2(src_path,dst_path)

print(f'Clustering completed. Images organized into {n_clusters} clusters.')