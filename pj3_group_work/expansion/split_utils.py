import os
import shutil
import random

default_xml_dir = "./dataset/train/labels/"
default_tir_dir = "./dataset/train/tir/"
default_rgb_dir = "./dataset/train/rgb/"


def get_output_dir():
    return "./split_test"


def get_rgb_paths(rgb_dir=default_rgb_dir) -> list[str]:
    img_paths = [
        os.path.join(rgb_dir, filename)
        for filename in os.listdir(rgb_dir)
        if filename.endswith(".jpg")
    ]
    # return random.shuffle(img_paths)
    return img_paths


def get_tir_paths(rgb_paths, tir_dir=default_tir_dir) -> list[str]:
    tir_paths = []
    for rgb_path in rgb_paths:
        base_path = os.path.splitext(os.path.basename(rgb_path))[0]
        tir_path = os.path.join(tir_dir, base_path + "R.jpg")
        tir_paths.append(tir_path)
    return tir_paths


def get_xml_paths(rgb_paths, xml_dir=default_xml_dir) -> list[str]:
    xml_paths = []
    for rgb_path in rgb_paths:
        base_path = os.path.splitext(os.path.basename(rgb_path))[0]
        xml_path = os.path.join(xml_dir, base_path + "R.xml")
        xml_paths.append(xml_path)
    return xml_paths


def get_tir_path(rgb_path, tir_dir=default_tir_dir) -> str:
    base_path = os.path.splitext(os.path.basename(rgb_path))[0]
    tir_path = os.path.join(tir_dir, base_path + "R.jpg")
    return tir_path


def get_xml_path(rgb_path, xml_dir=default_xml_dir) -> str:
    base_path = os.path.splitext(os.path.basename(rgb_path))[0]
    xml_path = os.path.join(xml_dir, base_path + "R.xml")
    return xml_path


def rm_dir(output_dir):
    for entry in os.listdir(output_dir):
        entry_path = os.path.join(output_dir, entry)

        # If the entry is a directory, recursively clear it
        if os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
        else:
            os.remove(entry_path)
