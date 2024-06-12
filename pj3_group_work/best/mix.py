import numpy as np
from scipy.optimize import minimize
import h5py
import json
import os
import time
import random
import cv2


def mix():
    # 假设有2000个数据点
    np.random.seed(0)  # For reproducibility
    num_points = 20

    # 生成随机数据点作为示例
    x = np.random.rand(num_points)
    y1 = np.random.rand(num_points)
    y2 = np.random.rand(num_points)
    y3 = np.random.rand(num_points)
    y4 = np.random.rand(num_points)
    y5 = np.random.rand(num_points)

    print(x)
    print(y1)
    print(y2)
    print(y3)
    print(y4)
    print(y5)

    # 定义目标函数，计算误差的平方和
    def objective(params):
        k1, k2, k3, k4, k5 = params
        predicted_x = k1 * y1**2 + k2 * y2**2 + k3 * y3**2 + k4 * y4**2 + k5 * y5**2
        return np.sum((predicted_x - x) ** 2)

    # 初始参数猜测
    initial_guess = [1, 1, 1, 1, 1]

    # 使用最小化函数来拟合参数
    result = minimize(objective, initial_guess)

    # 提取拟合的参数
    k1, k2, k3, k4, k5 = result.x

    print(f"Fitted parameters: k1 = {k1}, k2 = {k2}, k3 = {k3}, k4 = {k4}, k5 = {k5}")
    print(f"Sum of squared errors: {result.fun}")


def get_targets():
    labels_dir = "../dataset/train/hdf5s/"
    # labels_paths = [
    #     os.path.join(labels_dir, filename)
    #     for filename in os.listdir(labels_dir)
    # ]
    numbers = [int(filename[:-3]) for filename in os.listdir(labels_dir)]
    targets = np.zeros(len(numbers) + 1)

    for num in numbers:
        gt_path = f"{labels_dir}{num}.h5"
        gt_file = h5py.File(gt_path, "r")
        target = np.asarray(gt_file["density"])

        # 对密度地图进行缩放操作，将其大小调整为原始大小的1/8，并使用双立方插值法进行插值。
        # https://baike.baidu.com/item/%E5%8F%8C%E4%B8%89%E6%AC%A1%E6%8F%92%E5%80%BC/11055947
        target = (
            cv2.resize(
                target,
                (target.shape[1] // 8, target.shape[0] // 8),
                interpolation=cv2.INTER_CUBIC,
            )
            * 64
        )
        targets[(num)] = round(target.sum())
        # print(gt_path, target.sum())

    # for i in range(1, len(targets)):
    #     print(f"{i},{targets[i]}")
    return targets


def process_file(input_files):
    train = {}
    for input_file in input_files:
        train[input_file] = np.zeros(2000)
        with open(input_file, "r") as file:
            lines = file.readlines()

        # processed_lines = []
        for line in lines:
            # 分割每一行的数据
            index, value = line.strip().split(",")
            # 将字符串转换为浮点数，检查是否小于5
            value = float(value)
            if value < 6:
                value = 6.00
            train[input_file][int(index)] = round(value, 2)
        #     # 重新格式化字符串，保留两位小数
        #     processed_line = f"{index},{value:.2f}"
        #     processed_lines.append(processed_line)

        # # 将处理后的内容写回文件
        # with open(output_file, 'w') as file:
        #     file.write('\n'.join(processed_lines))

    return train


input_files = [
    "train/55.txt",
    "train/64.txt",
    "train/67.txt",
    "train/642.txt",
    "train/1263.txt",
]
train = process_file(input_files)
target = get_targets()
get_param = True
test_mae = False
# for key, value in train.items():
#     print(key)
#     print(value.shape)
#     print(value)

# print(target.shape)
# print(target)

y1 = train["train/55.txt"][1 : len(target)]
y2 = train["train/64.txt"][1 : len(target)]
y3 = train["train/67.txt"][1 : len(target)]
y4 = train["train/642.txt"][1 : len(target)]
y5 = train["train/1263.txt"][1 : len(target)]
x = target[1:]

if get_param:

    def objective(params):
        k1, k2, k3, k4, k5, k6 = params
        predicted_x = k1 * y1 + k2 * y2 + k3 * y3 + k4 * y4 + k5 * y5 + k6
        return np.sum(abs(predicted_x - x)[ x > 100])

    # 初始参数猜测
    initial_guess = [0.2, 0.2, 0.2, 0.2, 0.2, 0]

    bounds = [(0.1, 0.3), (0.1, 0.3), (0.1, 0.3), (0.1, 0.3), (0.1, 0.3), (-5, 5)]

    # 使用最小化函数来拟合参数，带上 bounds 参数
    result = minimize(objective, initial_guess, bounds=bounds)

    # 使用最小化函数来拟合参数
    # result = minimize(objective, initial_guess)

    # 提取拟合的参数
    k1, k2, k3, k4, k5, k6 = result.x

    print(
        f"Fitted parameters: k1 = {k1}, k2 = {k2}, k3 = {k3}, k4 = {k4}, k5 = {k5}, k6 = {k6}"
    )
    print(f"Sum of squared errors: {result.fun}")


if test_mae:
    # Fitted parameters: k1 = 0.9745025291948695, k2 = 0.3624114610425393, k3 = 0.11456916368600852, k4 = -0.19082736947780196, k5 = -0.24744109093218028, k6 = -3.357016789422827
    # k1 = 0.3, k2 = 0.148977377065179, k3 = 0.166430474342344, k4 = 0.3, k5 = 0.1, k6 = -1.1425566431240712
    k1, k2, k3, k4, k5, k6 = 0.3, 0.148977377065179, 0.166430474342344, 0.3, 0.1, -1.1425566431240712
    Mae = 0
    for i in range(0, len(target) - 1):
        predicted_x = (
            k1 * y1[i] + k2 * y2[i] + k3 * y3[i] + k4 * y4[i] + k5 * y5[i] + k6
        )
        Mae += abs(predicted_x - target[i])
    print(f"Mean absolute error: {Mae/len(target)}")
