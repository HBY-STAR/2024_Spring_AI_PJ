import numpy as np

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

# 将所有数组堆叠成一个二维数组
arrays = np.vstack([y1, y2, y3, y4, y5])

# 计算每个位置上的标准差
std_devs = arrays.std(axis=0)

# 获取标准差的索引，并按标准差值降序排序
sorted_indices = np.argsort(-std_devs)

# 输出标准差最大的位置及其标准差值
print("Index\tStandard Deviation")
for idx in sorted_indices:
    print(f"{idx+1}\t{std_devs[idx]}")
