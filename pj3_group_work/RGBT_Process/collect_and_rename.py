from image_config import image_shape, image_train_dir, image_val_dir, image_test_dir, processed_dir, image_test_json2h5_dir, processed_dark_dir, processed_light_dir
import os
import shutil
from PIL import Image
import h5py
import numpy as np
import json

def collect_files(source_dir, target_dir):
    # 创建目标子目录
    rgb_dir = os.path.join(target_dir, 'rgb')
    tir_dir = os.path.join(target_dir, 'tir')
    hdf5s_dir = os.path.join(target_dir, 'hdf5s')
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(tir_dir, exist_ok=True)
    os.makedirs(hdf5s_dir, exist_ok=True)
    
    # 遍历源目录中的所有文件
    for file_name in os.listdir(source_dir):
        source_file = os.path.join(source_dir, file_name)
        
        # 确定文件类型并复制到相应的子目录
        if file_name.endswith('_RGB.jpg'):
            shutil.copy(source_file, rgb_dir)
        elif file_name.endswith('_T.jpg'):
            shutil.copy(source_file, tir_dir)
        elif file_name.endswith('.h5'):
            shutil.copy(source_file, hdf5s_dir)


def collect_dark_files(source_dir, target_dir):
    # 读取json文件中的数据
    json_file_path = './RGBT-CC-CVPR2021/dark_list.json'
    with open(json_file_path, 'r') as f:
        dark_list = json.load(f)
    # dark_list 每个元素去前4个字符
    dark_list = [x[:4] for x in dark_list]
    
    # 创建目标子目录
    rgb_dir = os.path.join(target_dir, 'rgb')
    tir_dir = os.path.join(target_dir, 'tir')
    hdf5s_dir = os.path.join(target_dir, 'hdf5s')
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(tir_dir, exist_ok=True)
    os.makedirs(hdf5s_dir, exist_ok=True)
    
    # 遍历源目录中的所有文件
    for file_name in os.listdir(source_dir):
        if file_name[:4] in dark_list:
            source_file = os.path.join(source_dir, file_name)
            
            # 确定文件类型并复制到相应的子目录
            if file_name.endswith('_RGB.jpg'):
                shutil.copy(source_file, rgb_dir)
            elif file_name.endswith('_T.jpg'):
                shutil.copy(source_file, tir_dir)
            elif file_name.endswith('.h5'):
                shutil.copy(source_file, hdf5s_dir)

def collect_light_files(source_dir, target_dir):
    # 读取json文件中的数据
    json_file_path = './RGBT-CC-CVPR2021/bright_list.json'
    with open(json_file_path, 'r') as f:
        light_list = json.load(f)
    # light_list 每个元素去前4个字符
    light_list = [x[:4] for x in light_list]
    
    # 创建目标子目录
    rgb_dir = os.path.join(target_dir, 'rgb')
    tir_dir = os.path.join(target_dir, 'tir')
    hdf5s_dir = os.path.join(target_dir, 'hdf5s')
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(tir_dir, exist_ok=True)
    os.makedirs(hdf5s_dir, exist_ok=True)
    
    # 遍历源目录中的所有文件
    for file_name in os.listdir(source_dir):
        if file_name[:4] in light_list:
            
            source_file = os.path.join(source_dir, file_name)
            
            # 确定文件类型并复制到相应的子目录
            if file_name.endswith('_RGB.jpg'):
                shutil.copy(source_file, rgb_dir)
            elif file_name.endswith('_T.jpg'):
                shutil.copy(source_file, tir_dir)
            elif file_name.endswith('.h5'):
                shutil.copy(source_file, hdf5s_dir)

def rename_and_resize(rgb_dir, tir_dir, hdf5s_dir, start_index=1):
    # 获取目录下的所有文件
    rgb_files = os.listdir(rgb_dir)
    tir_files = os.listdir(tir_dir)
    hdf5s_files = os.listdir(hdf5s_dir)
    
    # 遍历每个文件并重命名
    for i in range(start_index, start_index + len(rgb_files)):
        # 处理RGB文件
        rgb_file = os.path.join(rgb_dir, rgb_files[i-start_index])
        new_rgb_file = os.path.join(rgb_dir, f'{i}.jpg')
        image = Image.open(rgb_file)
        resized_image = image.resize((640, 512), resample=Image.BILINEAR)  # 使用双线性插值
        resized_image.save(new_rgb_file)
        os.remove(rgb_file)
        
        # 处理TIR文件
        tir_file = os.path.join(tir_dir, tir_files[i-start_index])
        new_tir_file = os.path.join(tir_dir, f'{i}R.jpg')
        image = Image.open(tir_file)
        resized_image = image.resize((640, 512), resample=Image.BILINEAR)  # 使用双线性插值
        resized_image.save(new_tir_file)
        os.remove(tir_file)
        
        # 处理HDF5s文件
        hdf5s_file = os.path.join(hdf5s_dir, hdf5s_files[i-start_index])
        new_hdf5s_file = os.path.join(hdf5s_dir, f'{i}.h5')
        # 调整HDF5文件中的density
        adjust_density(hdf5s_file)
        os.rename(hdf5s_file, new_hdf5s_file)

def adjust_density(h5_file):
    with h5py.File(h5_file, 'r+') as hf:
        density_map = hf['density'][:]
        resized_density_map = resize_density_map(density_map, (640, 512))  # 使用双线性插值
        del hf['density']
        hf.create_dataset('density', data=resized_density_map)

def resize_density_map(density_map, target_size):
    # 使用双线性插值调整密度图大小
    image = Image.fromarray(density_map)
    resized_image = image.resize(target_size, resample=Image.BILINEAR)
    return np.array(resized_image)

if __name__ == "__main__":
    # collect_files(image_test_dir, processed_dir)
    # collect_files(image_train_dir, processed_dir)
    # collect_files(image_val_dir, processed_dir)
    #rename_and_resize(os.path.join(processed_dir, 'rgb'), os.path.join(processed_dir, 'tir'), os.path.join(processed_dir, 'hdf5s'),2000)
    
    # collect_dark_files(image_test_dir, processed_dark_dir)
    # collect_dark_files(image_train_dir, processed_dark_dir)
    # collect_dark_files(image_val_dir, processed_dark_dir)
    # collect_light_files(image_test_dir, processed_light_dir)
    # collect_light_files(image_train_dir, processed_light_dir)
    # collect_light_files(image_val_dir, processed_light_dir)    
    
    rename_and_resize(os.path.join(processed_dark_dir, 'rgb'), os.path.join(processed_dark_dir, 'tir'), os.path.join(processed_dark_dir, 'hdf5s'),10000)
    rename_and_resize(os.path.join(processed_light_dir, 'rgb'), os.path.join(processed_light_dir, 'tir'), os.path.join(processed_light_dir, 'hdf5s'),20000)
