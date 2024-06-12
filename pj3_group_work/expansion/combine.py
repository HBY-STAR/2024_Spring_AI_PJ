import xml.etree.ElementTree as ET
from PIL import Image
import expansion.split_utils as split_utils
import os


class combine:
    def __init__(
        self,
        temp_rgb_dir,
        temp_tir_dir,
        temp_labels_dir,
        output_image_path,
        output_tir_path,
        output_xml_path,
    ):
        self.temp_rgb_dir = temp_rgb_dir
        self.temp_tir_dir = temp_tir_dir
        self.temp_labels_dir = temp_labels_dir
        self.output_image_path = output_image_path
        self.output_tir_path = output_tir_path
        self.output_xml_path = output_xml_path

    def combine(self):
        rgb_paths = split_utils.get_rgb_paths(self.temp_rgb_dir + "/")

        length = len(rgb_paths) // 4

        for idx in range(length):
            # print(f"Combining {idx}...")

            output_image_path = f"{self.output_image_path}/combine{idx}.jpg"
            output_tir_image_path = f"{self.output_tir_path}/combine{idx}R.jpg"
            output_xml_path = f"{self.output_xml_path}/combine{idx}R.xml"

            image_paths = [
                rgb_paths[idx],
                rgb_paths[idx + length],
                rgb_paths[idx + 2 * length],
                rgb_paths[idx + 3 * length],
            ]

            tir_paths = split_utils.get_tir_paths(image_paths, self.temp_tir_dir + "/")

            xml_paths = split_utils.get_xml_paths(
                image_paths, self.temp_labels_dir + "/"
            )

            self.merge_images_and_annotations(
                image_paths,
                tir_paths,
                xml_paths,
                output_tir_image_path,
                output_image_path,
                output_xml_path,
            )

    def merge_images_and_annotations(
        self,
        image_paths,
        tir_paths,
        xml_paths,
        output_tir_image_path,
        output_image_path,
        output_xml_path,
    ):
        # print(image_paths)
        # 打开四个图像文件
        images = [Image.open(image_path) for image_path in image_paths]

        # 获取图像的宽度和高度
        width, height = images[0].size
        assert all(
            img.size == (width, height) for img in images
        ), "All images must have the same dimensions"
        merged_image = Image.new("RGB", (width * 2, height * 2))
        merged_image.paste(images[0], (0, 0))
        merged_image.paste(images[1], (width, 0))
        merged_image.paste(images[2], (0, height))
        merged_image.paste(images[3], (width, height))
        merged_image.save(output_image_path)

        merged_tir = Image.new("L", (width * 2, height * 2))
        tir_images = [Image.open(tir_path) for tir_path in tir_paths]
        merged_tir.paste(tir_images[0], (0, 0))
        merged_tir.paste(tir_images[1], (width, 0))
        merged_tir.paste(tir_images[2], (0, height))
        merged_tir.paste(tir_images[3], (width, height))
        merged_tir.save(output_tir_image_path)

        # 创建一个新的XML根节点
        root_combined = ET.Element("annotation")
        size = ET.Element("size")
        ET.SubElement(size, "width").text = str(width * 2)
        ET.SubElement(size, "height").text = str(height * 2)
        ET.SubElement(size, "depth").text = "3"
        root_combined.append(size)

        for i, xml_path in enumerate(xml_paths):
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall("object"):
                point = obj.find("point")
                if point is not None:
                    x = int(point.find("x").text)
                    y = int(point.find("y").text)

                    if i == 1:  # top-right
                        x += width
                    elif i == 2:  # bottom-left
                        y += height
                    elif i == 3:  # bottom-right
                        x += width
                        y += height

                    point.find("x").text = str(x)
                    point.find("y").text = str(y)
                    root_combined.append(obj)
                else:
                    bnd = obj.find("bndbox")
                    minx = int(bnd.find("xmin").text)
                    miny = int(bnd.find("ymin").text)
                    maxx = int(bnd.find("xmax").text)
                    maxy = int(bnd.find("ymax").text)

                    if i == 1:
                        minx += width
                        maxx += width
                    elif i == 2:
                        miny += height
                        maxy += height
                    elif i == 3:
                        minx += width
                        maxx += width
                        miny += height
                        maxy += height

                    bnd.find("xmin").text = str(minx)
                    bnd.find("ymin").text = str(miny)
                    bnd.find("xmax").text = str(maxx)
                    bnd.find("ymax").text = str(maxy)
                    root_combined.append(obj)

        # 保存合并后的XML文件
        tree_combined = ET.ElementTree(root_combined)
        tree_combined.write(output_xml_path)


if __name__ == "__main__":
    image_paths = [
        "split_test/1image1.jpg",
        "split_test/1image2.jpg",
        "split_test/1image3.jpg",
        "split_test/1image4.jpg",
    ]

    tir_paths = [
        "split_test/1tir1.jpg",
        "split_test/1tir2.jpg",
        "split_test/1tir3.jpg",
        "split_test/1tir4.jpg",
    ]

    xml_paths = [
        "split_test/1Rannotation1.xml",
        "split_test/1Rannotation2.xml",
        "split_test/1Rannotation3.xml",
        "split_test/1Rannotation4.xml",
    ]

    output_image_path = "split_test/reconstructed_image.jpg"
    output_tir_path = "split_test/reconstructed_tir.jpg"
    output_xml_path = "split_test/reconstructed_annotation.xml"

    # merge_images_and_annotations(
    #     image_paths,
    #     tir_paths,
    #     xml_paths,
    #     output_image_path,
    #     output_tir_path,
    #     output_xml_path,
    # )
