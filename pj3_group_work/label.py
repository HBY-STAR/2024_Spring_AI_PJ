import numpy as np
import os
import xml.etree.ElementTree as ET
import h5py
from scipy.ndimage.filters import gaussian_filter


def parse_xml(xml_file, image_shape):
    """
    解析XML文件，生成一个与图像形状相同的密度图
    """

    # 解析xml文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 初始化密度图
    density_map = np.zeros(image_shape, dtype=np.float32)

    # 遍历xml文件中的所有 <object>
    for obj in root.findall("object"):

        # 获取目标的坐标
        try:
            x = int(obj.find("point/x").text)
            y = int(obj.find("point/y").text)
        except:
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)
            x = (xmin + xmax) // 2
            y = (ymin + ymax) // 2

        # 将目标的坐标转换为密度图上的坐标
        if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]:
            density_map[y, x] = 1

    # 使用高斯滤波平滑密度图
    # https://baike.baidu.com/item/%E9%AB%98%E6%96%AF%E6%BB%A4%E6%B3%A2/9032353
    density_map = gaussian_filter(density_map, sigma=15)
    return density_map


def label(xml_path, mat_path):
    # 宽640，高512
    image_shape = (512, 640)  # Assuming the image shape is fixed

    for xml_file in os.listdir(xml_path):
        if xml_file.endswith(".xml"):
            # print(xml_file)
            density_map = parse_xml(os.path.join(xml_path, xml_file), image_shape)

            # hdf view
            # https://www.hdfgroup.org/downloads/hdfview/#download
            with h5py.File(
                os.path.join(mat_path, xml_file.replace("R.xml", ".h5")), "w"
            ) as hf:
                hf["density"] = density_map


if __name__ == "__main__":
    xml_path = "./dataset/train/labels/"
    mat_path = "./dataset/train/hdf5s/"
    label(xml_path, mat_path)