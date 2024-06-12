from image_config import image_shape, image_train_dir, image_val_dir, image_test_dir, processed_dir, image_test_json2h5_dir, processed_dark_dir, processed_light_dir
import os
import json
import numpy as np
import h5py
from scipy.ndimage.filters import gaussian_filter


def json_to_density_map(json_file, image_shape):
    # 从JSON文件中读取点的位置数据
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 初始化密度图
    density_map = np.zeros(image_shape, dtype=np.float32)
    
    # 将点的位置转换为密度图上的坐标，并在密度图中增加点的密度
    for point in data["points"]:
        x, y = int(point[0]), int(point[1])
        if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]:
            density_map[y, x] += 1  # 增加点的密度
    
    # 使用高斯滤波平滑密度图
    density_map = gaussian_filter(density_map, sigma=15)
    
    return density_map

def json_to_h5(json_file, h5_file, image_shape):
    # 将JSON文件转换为密度图
    density_map = json_to_density_map(json_file, image_shape)
    
    # 创建HDF5文件并写入密度图数据
    with h5py.File(h5_file, 'w') as hf:
        hf['density'] = density_map

def convert_json_to_h5_in_directory(json_dir):
    # 获取目录下的所有JSON文件
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    # 遍历每个JSON文件并转换为HDF5文件
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        h5_file = os.path.join(json_dir, json_file.replace('.json', '.h5'))
        
        # 转换JSON文件为HDF5文件
        json_to_h5(json_path, h5_file, image_shape) 



if __name__ == "__main__":
    convert_json_to_h5_in_directory(image_test_dir)
    convert_json_to_h5_in_directory(image_train_dir)
    convert_json_to_h5_in_directory(image_val_dir)
