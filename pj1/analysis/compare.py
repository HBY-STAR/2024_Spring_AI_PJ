import matplotlib.pyplot as plt
import numpy as np

"""
比较 SVM, LinearSVM, LR 的 accuracy
"""
label = ['SVM-poly', 'SVM-rbf', 'SVM-sigmoid', 'LinearSVM', 'LR']
train_acc = [0.9027, 0.9022, 0.8650, 0.7927, 0.9021]
test_acc = [0.9774, 0.9782, 0.9267, 0.8506, 0.9732]

x = np.arange(len(label))
width = 0.35

fig, ax = plt.subplots()
bars1 = ax.bar(x - width / 2, train_acc, width, label='Train Accuracy', color='skyblue')
bars2 = ax.bar(x + width / 2, test_acc, width, label='Test Accuracy', color='salmon')

ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_title('Train and Test Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.15))

plt.show()

"""
比较 SVM, LinearSVM, LR 的 运行时间
"""

label = ['SVM-poly', 'SVM-rbf', 'SVM-sigmoid', 'LinearSVM', 'LR']
run_time = [15.65, 9.38, 2.72, 6.24, 1.11]

plt.figure(figsize=(6, 6))
plt.bar(label, run_time, color='skyblue')

plt.xlabel('Models')
plt.ylabel('Run Time (seconds)')
plt.title('Run Time')

plt.show()

"""
比较 SVM sigmoid 不同正则化参数的差异
"""

C = [0.01, 0.1, 0.5, 1, 5, 10, 15, 20, 25, 50, 100]
train_acc = [0.9020, 0.9020, 0.9020, 0.9022, 0.8817, 0.8760, 0.8689, 0.8649, 0.8619, 0.8511, 0.8434]
test_acc = [0.9784, 0.9784, 0.9778, 0.9766, 0.9470, 0.9387, 0.9308, 0.9267, 0.9239, 0.9110, 0.9023]
run_time = [3.59, 3.44, 3.25, 3.43, 3.076, 3.239, 2.94, 2.93, 2.98, 2.81, 2.78]

x = np.arange(len(C))
width = 0.35

fig, ax = plt.subplots()
bars1 = ax.bar(x - width / 2, train_acc, width, label='Train Accuracy', color='skyblue')
bars2 = ax.bar(x + width / 2, test_acc, width, label='Test Accuracy', color='salmon')

ax.set_xlabel('C')
ax.set_ylabel('Accuracy')
ax.set_title('Train and Test Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(C)
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.15))

plt.show()


plt.bar(range(len(C)), run_time, color='skyblue')

plt.xlabel('C')  # 设置x轴标签
plt.ylabel('Run Time')  # 设置y轴标签
plt.title('Run Time')  # 设置标题

plt.xticks(range(len(C)), C)  # 设置x轴刻度和标签，确保均匀分布

plt.show()  # 显示图形
