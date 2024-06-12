import PIL.Image as Image
import numpy as np
import os


def tir():
    # 设置热力图的路径
    test_tir_path = "./dataset/train/tir/"
    tir_img_paths = [f"{test_tir_path}{i}R.jpg" for i in range(1, len(os.listdir(test_tir_path))+1)]

    mean = 0.0
    std = 0.0

    for tir_img_path in tir_img_paths:
        # print(tir_img_path)

        img = Image.open(tir_img_path).convert("L")

        img_array = np.array(img) / 255.0
        # print("Image array:", img_array)
        # print("Image shape:", img_array.shape)
        mean += np.mean(img_array)
        std += np.std(img_array)
        # print("Mean:", mean)
        # print("Standard Deviation:", std)

    print("Mean:", mean / len(tir_img_paths))
    print("Standard Deviation:", std / len(tir_img_paths))


def rgb():
    test_tir_path = "./dataset/train/rgb/"
    tir_img_paths = [f"{test_tir_path}{i}.jpg" for i in range(1, len(os.listdir(test_tir_path))+1)]

    meanR = 0.0
    stdR = 0.0

    meanG = 0.0
    stdG = 0.0

    meanB = 0.0
    stdB = 0.0

    for tir_img_path in tir_img_paths:
        # print(tir_img_path)

        img = Image.open(tir_img_path).convert("RGB")

        img_array = np.array(img) / 255.0
        # print("Image array:", img_array)
        # print("Image shape:", img_array.shape)
        # print("Image shape:", img_array.shape)
        # print(img_array)
        meanR += np.mean(img_array[:, :, 0])
        stdR += np.std(img_array[:, :, 0])

        meanG += np.mean(img_array[:, :, 1])
        stdG += np.std(img_array[:, :, 1])

        meanB += np.mean(img_array[:, :, 2])
        stdB += np.std(img_array[:, :, 2])
        # print("Mean:", mean)
        # print("Standard Deviation:", std)

    print("Mean R:", meanR / len(tir_img_paths))
    print("Standard Deviation R:", stdR / len(tir_img_paths))
    print("Mean G:", meanG / len(tir_img_paths))
    print("Standard Deviation G:", stdG / len(tir_img_paths))
    print("Mean B:", meanB / len(tir_img_paths))
    print("Standard Deviation B:", stdB / len(tir_img_paths))


if __name__ == "__main__":
    rgb()
    tir()

# [0.452, 0.411, 0.362, 0.397]
# [0.188, 0.167, 0.162, 0.181]

[0.349, 0.335, 0.352, 0.495]
[ 0.151 , 0.145, 0.146, 0.159]

# Mean R: 0.34899641658986247
# Standard Deviation R: 0.15108294179196446
# Mean G: 0.33474687217888716
# Standard Deviation G: 0.1454851169492599
# Mean B: 0.3524778065145986
# Standard Deviation B: 0.14642199683164456
# Mean: 0.49493573325286977
# Standard Deviation: 0.15939872789155612

