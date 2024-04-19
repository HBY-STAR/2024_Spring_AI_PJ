from matplotlib import pyplot as plt

# using data from result_model.log

model = ['ResNet-18', 'AlexNet', 'MobileNet-V3-Small', 'VGG']

total_params = [11689512, 61100840, 2542856, 138357544]
test_acc = [86.23, 80.75, 87.28, 81.93]
run_time = [8170.24, 3523.40, 4598.54, 31783.50]

fig, axs = plt.subplots(1, 3, figsize=(15, 9), dpi=300)

# model vs total_params
axs[0].bar(model, total_params, color='skyblue')
axs[0].set_xlabel('Model')
axs[0].set_ylabel('Total Parameters')
axs[0].set_title('Model vs Total Parameters')

# model vs test_acc
axs[1].bar(model, test_acc, color='red')
axs[1].set_xlabel('Model')
axs[1].set_ylabel('Test Accuracy')
axs[1].set_title('Model vs Test Accuracy')

# model vs run_time
axs[2].bar(model, run_time, color='green')
axs[2].set_xlabel('Model')
axs[2].set_ylabel('Run Time')
axs[2].set_title('Model vs Run Time')

plt.tight_layout()
plt.show()
plt.savefig('result_model.png')

