import numpy as np
from scipy.optimize import minimize
import h5py
import json
import os
import random


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

    return train


input_files = [
    "test/ans55.txt",
    "test/ans64.txt",
    "test/ans67.txt",
    "test/ans642.txt",
    "test/ans1263.txt",
]
train = process_file(input_files)
get_param = False
test_mae = True
# for key, value in train.items():
#     print(key)
#     print(value.shape)
#     print(value)

# print(target.shape)
# print(target)

y1 = train["test/ans55.txt"][1:1001]
y2 = train["test/ans64.txt"][1:1001]
y3 = train["test/ans67.txt"][1:1001]
y4 = train["test/ans642.txt"][1:1001]
y5 = train["test/ans1263.txt"][1:1001]


k1, k2, k3, k4, k5, k6 = (
    0.3,
    0.148977377065179,
    0.166430474342344,
    0.3,
    0.1,
    -1.1425566431240712,
)


k11, k12, k13, k14, k15, k16 = (
    0.3,
    0.1,
    0.1,
    0.2847135208414331,
    0.1,
    1.9177197361822278,
)
k21, k22, k23, k24, k25, k26 = (
    0.3,
    0.17963508650682025,
    0.1300163075797279,
    0.10606727934656125,
    0.19997052422250228,
    5.0,
)
k31, k32, k33, k34, k35, k36 = (
    0.2688548579181595,
    0.3,
    0.1,
    0.2257526709073257,
    0.1,
    1.3253165661119983,
)

y_result = []
for i in range(1000):
    if np.std([y1[i], y2[i], y3[i], y4[i], y5[i]]) < 8:
        mean_value = np.mean([y1[i], y2[i], y3[i], y4[i], y5[i]])
        if mean_value < 50:
            y_result.append(
                k11 * y1[i]
                + k12 * y2[i]
                + k13 * y3[i]
                + k14 * y4[i]
                + k15 * y5[i]
                + k16
            )
        elif mean_value < 100:
            y_result.append(
                k21 * y1[i]
                + k22 * y2[i]
                + k23 * y3[i]
                + k24 * y4[i]
                + k25 * y5[i]
                + k26
            )
        else:  # mean_value >= 100
            y_result.append(
                k31 * y1[i]
                + k32 * y2[i]
                + k33 * y3[i]
                + k34 * y4[i]
                + k35 * y5[i]
                + k36
            )
        # y_result.append(
        #     k1 * y1[i] + k2 * y2[i] + k3 * y3[i] + k4 * y4[i] + k5 * y5[i] + k6
        # )
    else:
        max_value = max(y1[i], y2[i], y3[i], y4[i], y5[i])
        min_value = min(y1[i], y2[i], y3[i], y4[i], y5[i])
        y_result.append((min_value))
# y_result = k1 * y1 + k2 * y2 + k3 * y3 + k4 * y4 + k5 * y5 + k6
# print(y_result)
# print(y1.shape)
# print(y_result.shape)
print(len(y_result))

processed_lines = []
for index, value in enumerate(y_result):
    processed_line = f"{index+1},{value:.2f}"
    processed_lines.append(processed_line)

output_file = "ansmix2.txt"
with open(output_file, "w") as file:
    file.write("\n".join(processed_lines))


# < 50
# Fitted parameters: k1 = 0.3, k2 = 0.1, k3 = 0.1, k4 = 0.2847135208414331, k5 = 0.1, k6 = 1.9177197361822278
# Sum of squared errors: 3714.676669388508
# 50-100
# Fitted parameters: k1 = 0.3, k2 = 0.17963508650682025, k3 = 0.1300163075797279, k4 = 0.10606727934656125, k5 = 0.19997052422250228, k6 = 5.0
# Sum of squared errors: 3774.0109512714307
# > 100
# Fitted parameters: k1 = 0.2688548579181595, k2 = 0.3, k3 = 0.1, k4 = 0.2257526709073257, k5 = 0.1, k6 = 1.3253165661119983
# Sum of squared errors: 4060.378717049801
