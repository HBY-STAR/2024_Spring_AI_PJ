import torch
import torch.nn.functional as F


# ref: https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html
# 由于LeNet原始对应1*32*32的图像，而这里我想要对应3*224*224的输入以与torchvision的模型对应，故修改了部分参数


class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 卷积层
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        # 全连接层
        self.fc1 = torch.nn.Linear(16 * 54 * 54, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # 使用ReLU和最大汇聚层

        # 2x2 池操作
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        x = x.view(-1, 16 * 54 * 54)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

