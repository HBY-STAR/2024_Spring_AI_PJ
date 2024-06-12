import xml.etree.ElementTree as ET
from PIL import Image
import os
import random
import shutil
from expansion.split_utils import (
    get_tir_path,
    get_xml_path,
    get_rgb_paths,
)

# 640*512


class split:
    def __init__(self, output_rgb_dir, output_tir_dir, output_labels_dir):
        self.output_rgb_dir = output_rgb_dir
        self.output_tir_dir = output_tir_dir
        self.output_labels_dir = output_labels_dir

    def split(self):
        img_paths = get_rgb_paths()

        for index in range(len(img_paths)):
            image_path = img_paths[index]
            xml_path = get_xml_path(image_path)
            tir_path = get_tir_path(image_path)
            # print(image_path, tir_path, xml_path)
            self.split_image(image_path, tir_path, xml_path)
            # break
        # random.shuffle(img_paths)

    def split_image(self, image_path, tir_path, xml_path):
        # 打开图片
        image = Image.open(image_path)
        width, height = image.size

        # 切分图片
        box1 = (0, 0, width // 2, height // 2)
        box2 = (width // 2, 0, width, height // 2)
        box3 = (0, height // 2, width // 2, height)
        box4 = (width // 2, height // 2, width, height)

        image1 = image.crop(box1)
        image2 = image.crop(box2)
        image3 = image.crop(box3)
        image4 = image.crop(box4)

        path_base = os.path.splitext(os.path.basename(image_path))[0]

        image1_path = os.path.join(self.output_rgb_dir, path_base + "i1.jpg")
        image2_path = os.path.join(self.output_rgb_dir, path_base + "i2.jpg")
        image3_path = os.path.join(self.output_rgb_dir, path_base + "i3.jpg")
        image4_path = os.path.join(self.output_rgb_dir, path_base + "i4.jpg")

        image1.save(image1_path)
        image2.save(image2_path)
        image3.save(image3_path)
        image4.save(image4_path)

        tir_image = Image.open(tir_path)
        tir_image1 = tir_image.crop(box1)
        tir_image2 = tir_image.crop(box2)
        tir_image3 = tir_image.crop(box3)
        tir_image4 = tir_image.crop(box4)

        tir_image1_path = os.path.join(self.output_tir_dir, path_base + "i1R.jpg")
        tir_image2_path = os.path.join(self.output_tir_dir, path_base + "i2R.jpg")
        tir_image3_path = os.path.join(self.output_tir_dir, path_base + "i3R.jpg")
        tir_image4_path = os.path.join(self.output_tir_dir, path_base + "i4R.jpg")

        tir_image1.save(tir_image1_path)
        tir_image2.save(tir_image2_path)
        tir_image3.save(tir_image3_path)
        tir_image4.save(tir_image4_path)

        # 解析XML并提取坐标
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 创建新的XML根节点
        root1 = ET.Element("annotation")
        root2 = ET.Element("annotation")
        root3 = ET.Element("annotation")
        root4 = ET.Element("annotation")

        # 复制size节点
        size = root.find("size")
        for r in [root1, root2, root3, root4]:
            r.append(size)
        segmented = root.find("segmented")
        for r in [root1, root2, root3, root4]:
            r.append(segmented)

        def create_object_element(name, x, y):
            obj = ET.Element("object")
            name_element = ET.Element("name")
            name_element.text = name
            obj.append(name_element)

            pose_element = ET.Element("pose")
            pose_element.text = "Unspecified"
            obj.append(pose_element)

            truncated_element = ET.Element("truncated")
            truncated_element.text = "0"
            obj.append(truncated_element)

            difficult_element = ET.Element("difficult")
            difficult_element.text = "0"
            obj.append(difficult_element)

            point = ET.Element("point")
            x_element = ET.Element("x")
            x_element.text = str(x)
            point.append(x_element)

            y_element = ET.Element("y")
            y_element.text = str(y)
            point.append(y_element)

            obj.append(point)
            return obj

        # 将点重新分配到相应的图像区域
        for obj in root.findall("object"):
            name = obj.find("name").text
            point = obj.find("point")
            if point is not None:
                x = int(point.find("x").text)
                y = int(point.find("y").text)

                if x < width // 2 and y < height // 2:
                    new_obj = create_object_element(name, x, y)
                    root1.append(new_obj)
                elif x >= width // 2 and y < height // 2:
                    new_obj = create_object_element(name, x - width // 2, y)
                    root2.append(new_obj)
                elif x < width // 2 and y >= height // 2:
                    new_obj = create_object_element(name, x, y - height // 2)
                    root3.append(new_obj)
                else:
                    new_obj = create_object_element(name, x - width // 2, y - height // 2)
                    root4.append(new_obj)
            else:
                # print("bndbox")
                bnd = obj.find("bndbox")
                minx = int(bnd.find("xmin").text)
                miny = int(bnd.find("ymin").text)
                maxx = int(bnd.find("xmax").text)
                maxy = int(bnd.find("ymax").text)

                if minx < width // 2 and miny < height // 2:
                    new_obj = obj
                    root1.append(new_obj)
                elif minx >= width // 2 and miny < height // 2:
                    new_obj = obj
                    new_obj.find("bndbox").find("xmin").text = str(minx - width // 2)
                    new_obj.find("bndbox").find("xmax").text = str(maxx - width // 2)
                    root2.append(new_obj)
                elif minx < width // 2 and miny >= height // 2:
                    new_obj = obj
                    new_obj.find("bndbox").find("ymin").text = str(miny - height // 2)
                    new_obj.find("bndbox").find("ymax").text = str(maxy - height // 2)
                    root3.append(new_obj)
                else:
                    new_obj = obj
                    new_obj.find("bndbox").find("xmin").text = str(minx - width // 2)
                    new_obj.find("bndbox").find("xmax").text = str(maxx - width // 2)
                    new_obj.find("bndbox").find("ymin").text = str(miny - height // 2)
                    new_obj.find("bndbox").find("ymax").text = str(maxy - height // 2)
                    root4.append(new_obj)

        # 保存新的XML文件
        tree1 = ET.ElementTree(root1)
        tree2 = ET.ElementTree(root2)
        tree3 = ET.ElementTree(root3)
        tree4 = ET.ElementTree(root4)

        # base_path2 = os.path.splitext(os.path.basename(xml_path))[0]

        tree1.write(
            os.path.join(self.output_labels_dir, path_base + "i1R.xml")
        )
        tree2.write(
            os.path.join(self.output_labels_dir, path_base + "i2R.xml")
        )
        tree3.write(
            os.path.join(self.output_labels_dir, path_base + "i3R.xml")
        )
        tree4.write(
            os.path.join(self.output_labels_dir, path_base + "i4R.xml")
        )

        # # 重新拼接图片
        # new_image = Image.new("RGB", (width, height))
        # new_image.paste(image1, (0, 0))
        # new_image.paste(image2, (width // 2, 0))
        # new_image.paste(image3, (0, height // 2))
        # new_image.paste(image4, (width // 2, height // 2))

        # # 保存拼接后的图片
        # new_image_path = os.path.join(output_dir, "image.jpg")
        # new_image.save(new_image_path)

        # # 合并XML文件数据
        # root_combined = ET.Element("annotation")
        # root_combined.append(size)
        # root_combined.append(segmented)

        # for r in [root1, root2, root3, root4]:
        #     for obj in r.findall("object"):
        #         root_combined.append(obj)

        # tree_combined = ET.ElementTree(root_combined)
        # combined_xml_path = os.path.join(output_dir, "combined_annotation.xml")
        # tree_combined.write(combined_xml_path)


# if __name__ == "__main__":
#     output_dir = get_output_dir()
#     rm_dir(output_dir)
#     img_paths = get_rgb_paths()

#     for index in range(len(img_paths)):
#         image_path = img_paths[index]
#         xml_path = get_xml_path(image_path)
#         tir_path = get_tir_path(image_path)
#         print(image_path, tir_path, xml_path)
#         split_image(image_path, tir_path, xml_path)
#         break
#     # random.shuffle(img_paths)
