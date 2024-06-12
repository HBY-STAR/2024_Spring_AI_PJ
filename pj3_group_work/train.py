import json
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import time
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import h5py
import cv2
import shutil
from CSRNet_RGBT.csrnet_rgbt import CSRNet_RGBT
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Res50.model.Res50 import Res50
from ECAN.model import CANNet
import aiconfig
from CLIP_EBC import get_model

# from CLIP_EBC import get_model


# 用于保存最佳模型
def save_checkpoint(
    state,
    is_best,
    last_prec,
    best_prec1,
    task_id,
    filename="checkpoint.pth.tar",
    save_dir="./model/",
):
    checkpoint_path = os.path.join(save_dir, task_id + filename)
    torch.save(state, checkpoint_path)
    if is_best and best_prec1 < 7:
        best_model_path = os.path.join(
            save_dir, task_id + "model_best.pth.best" + str(best_prec1.item()) + ".tar"
        )
        shutil.copyfile(checkpoint_path, best_model_path)
    elif is_best:
        best_model_path = os.path.join(
            save_dir, task_id + "model_best.pth.batch" + str(batch_size) + ".tar"
        )
        shutil.copyfile(checkpoint_path, best_model_path)


# 加载数据
def load_data(img_path, tir_img_path, gt_path, train=True):
    """
    img_path: 图片文件路径
    gt_path: h5文件路径
    train: 是否为训练过程
    """

    # 打开图像文件并转换为RGB格式
    # img = Image.open(img_path).convert("RGB")
    rgb_img = Image.open(img_path).convert("RGB")

    # tir_img = Image.open(tir_img_path).convert('RGB')
    # print(tir_img.size)
    tir_img = Image.open(tir_img_path).convert("L")

    # img = cv2.merge((rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2],
    #                      tir_img[:, :, 0], tir_img[:, :, 1], tir_img[:, :, 2]))
    # # tir_img = Image.open(os.path.join(tir_img_dir, os.path.basename(img_path))).convert('RGB')
    # 将PIL图像转换为NumPy数组
    rgb_img = np.array(rgb_img)
    tir_img = np.array(tir_img)

    # 将两个图像合并为一个6通道图像
    # img = cv2.merge((rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2],
    #                      tir_img[:, :, 0], tir_img[:, :, 1]))
    # (512, 640, 6) will ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has 3 dimension(s) and the array at index 1 has 4 dimension(s)
    # if random.random() < 0.5:
    #     tir_img = 255 - tir_img
    img_np = np.concatenate((rgb_img, np.expand_dims(tir_img, axis=2)), axis=2)
    # print(img_np.shape)
    # print(img_path)
    img = Image.fromarray(img_np)
    # img.show()

    # 打开 HDF5 文件，该文件包含了密度地图（density map）的标注数据
    gt_file = h5py.File(gt_path)

    # 从 HDF5 文件中获取名为 'density' 的数据集，并将其转换为 NumPy 数组。
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

    return img, target


class ImgDataset(Dataset):
    def __init__(
        self,
        img_dir,
        tir_img_dir,
        gt_dir,
        shape=None,
        shuffle=True,
        transform=None,
        train=False,
        batch_size=1,
        num_workers=4,
    ):
        """
        img_dir: 图像文件路径
        gt_dir: h5文件路径
        shape: 图像大小
        shuffle: 是否打乱数据
        transform: 数据增强
        train: 是否为训练集
        batch_size: 批大小
        num_workers: 工作线程数
        """

        self.img_dir = img_dir
        self.tir_img_dir = tir_img_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.train = train
        self.shape = shape
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.img_paths = [
            os.path.join(img_dir, filename)
            for filename in os.listdir(img_dir)
            if filename.endswith(".jpg")
        ]
        # self.tir_img_paths = [os.path.join(tir_img_dir, filename) for filename in os.listdir(
        #     tir_img_dir) if filename.endswith('.jpg')]

        if shuffle:
            random.shuffle(self.img_paths)
            # random.shuffle(self.tir_img_paths)

        # 数据集大小，即图像数量
        self.nSamples = len(self.img_paths)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        """
        获取图像和标注数据（密度图）
        """

        assert index <= len(self), "index range error"
        img_path = self.img_paths[index]
        img_name = os.path.basename(img_path)
        gt_path = os.path.join(self.gt_dir, os.path.splitext(img_name)[0] + ".h5")
        # tir_img_dir = self.tir_img_paths[index]
        # tir_img_name = os.path.basename(tir_img_dir)
        tir_img_path = os.path.join(
            self.tir_img_dir, os.path.splitext(img_name)[0] + "R.jpg"
        )
        img, target = load_data(img_path, tir_img_path, gt_path, self.train)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


# 学习率
lr = aiconfig.lr
original_lr = lr

# 批大小
batch_size = aiconfig.batch_size

# 动量
momentum = 0.95

# 权重衰减
decay = 5 * 1e-4

# 训练轮数
epochs = 400

# decay_interval = 30
# steps = [i * decay_interval for i in range(1, epochs // decay_interval + 1)]
# # 每次衰减的系数为 0.1（即学习率降低 10 倍）
# scales = [decay_interval] * len(steps)


# 工作线程数
workers = 4

# 随机种子
seed = time.time()

# 打印频率
print_freq = 30

# 图像路径
# TODO check the data
img_dir = "./dataset/train/rgb/"
tir_img_dir = "./dataset/train/tir/"
gt_dir = "./dataset/train/hdf5s/"

# 预训练模型
# pre = "model/best/model_best.pth6.403.tar"
pre = None

# 任务名称
task = ""


def main():
    # 初始化起始轮次和最佳准确率
    start_epoch = 0
    best_prec1 = 1e6

    # 设置随机种子
    torch.cuda.manual_seed(seed)

    # 创建模型实例，并将其移动到GPU上
    # model = CSRNet_RGBT(True)
    model = CSRNet_RGBT()
    # model = CANNet()
    model = model.cuda()

    # 定义损失函数和优化器
    criterion = nn.MSELoss(size_average=False).cuda()
    # criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(), lr, momentum=momentum, weight_decay=decay
    )
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    # TODO1
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.2,
        patience=10,
        threshold=0.0001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-10,
        eps=1e-8,
        verbose=True,
    )

    # 数据预处理
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # TODO check the mean and std
            transforms.Normalize(
                mean=[0.349, 0.335, 0.352, 0.495], std=[0.151 , 0.145, 0.146, 0.159]
                # mean=[0.452, 0.411, 0.362, 0.397], std=[0.188, 0.167, 0.162, 0.181]
            ),
        ]
    )

    # TODO2
    # 创建数据集实例，并分割为训练集和验证集
    dataset = ImgDataset(
        img_dir, tir_img_dir, gt_dir, shuffle=False, transform=transform, train=True
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=workers
    )

    # 如果指定了预训练模型，则加载预训练参数
    if pre:
        if os.path.isfile(pre):
            print("=> loading checkpoint '{}'".format(pre))
            checkpoint = torch.load(pre)
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            best_prec1 = 10.0
            best_prec1 = checkpoint["best_prec1"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(pre, checkpoint["epoch"])
            )
        else:
            print("=> no checkpoint found at '{}'".format(pre))

    last_update = 0
    # 循环训练epochs轮
    for epoch in range(start_epoch, epochs):
        # 调整学习率
        # adjust_learning_rate(optimizer, epoch)

        # 训练模型
        train(
            model,
            criterion,
            optimizer,
            epoch,
            train_loader,
            optimizer.param_groups[0]["lr"],
        )
        # 在验证集上评估模型性能
        prec1 = validate(model, val_loader)

        scheduler.step(prec1)

        # 判断当前模型是否是最佳模型
        last_prec = best_prec1
        is_best = prec1 < best_prec1
        if is_best:
            last_update = epoch
            best_prec1 = min(prec1, best_prec1)
        print(
            " * best MAE {mae:.3f} in epoch {epoch}".format(
                mae=best_prec1, epoch=last_update
            )
        )

        # 保存模型参数
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": pre,
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best,
            last_prec,
            best_prec1,
            task,
        )

# TODO3 data
def train(model, criterion, optimizer, epoch, train_loader, curr_lr):
    """
    model: 训练的模型
    criterion: 损失函数
    optimizer: 优化器
    epoch: 当前的训练轮次
    train_loader: 用于训练的数据加载器
    """

    # 创建用于记录训练过程中损失、批处理时间和数据加载时间的 AverageMeter 对象
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # 打印当前训练轮次、已处理的样本数量以及学习率
    print(
        "epoch %d, processed %d samples, lr %.10f"
        % (epoch, epoch * len(train_loader.dataset), curr_lr)
    )

    # 将模型设置为训练模式
    model.train()
    end = time.time()

    # 迭代训练数据加载器中的每个批次
    for i, (img, target) in enumerate(train_loader):
        # 记录数据加载所需的时间
        data_time.update(time.time() - end)

        # 将输入图像移动到 GPU 上
        img = img.cuda()
        # 将输入图像包装成 Variable 对象
        img = Variable(img)
        # 将输入图像传递给模型，获取模型的输出
        output = model(img)
        # 将目标转换为 FloatTensor 类型并在第一维度上增加维度，然后移动到 GPU 上并包装成 Variable 对象
        target = target.type(torch.FloatTensor).unsqueeze(1).cuda()
        target = Variable(target)

        # 计算模型输出与目标之间的损失
        loss = criterion(output, target)
        
        # TODO res50 need
        # ----------------
        loss = loss.requires_grad_(True)
        # ----------------

        # 更新损失记录器
        losses.update(loss.item(), img.size(0))

        # 清除之前的梯度
        optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 使用优化器更新模型参数
        
        # TODO res50 need
        # ----------------
        optimizer_ft = torch.optim.SGD(
            model.parameters(), lr, momentum=momentum, weight_decay=decay
        )
        optimizer = optimizer_ft
        # ----------------
        
        optimizer.step()

        # 记录批处理所需的时间
        batch_time.update(time.time() - end)
        end = time.time()

        torch.cuda.empty_cache()

        # 如果满足打印频率条件，则打印当前训练轮次、当前处理的批次、总批次数以及损失、批处理时间和数据加载时间的平均值
        if i % print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                )
            )
        # break

# TODO
def validate(model, val_loader):
    """
    在验证集上评估模型的性能
    """

    print("begin test")

    # 将模型设置为评估模式
    model.eval()
    # 初始化 MAE（Mean Absolute Error）
    mae = 0

    # 迭代验证数据加载器中的每个批次
    for i, (img, target) in enumerate(val_loader):
        # 将输入图像移动到 GPU 上
        img = img.cuda()
        # 将输入图像包装成 Variable 对象
        img = Variable(img)
        # 将输入图像传递给模型，获取模型的输出
        output = model(img)

        # 计算预测值和目标值的绝对值误差，并累加到 MAE 中
        mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())

    # 计算平均 MAE
    mae = mae / (len(val_loader))  # * batch_size
    # 打印平均 MAE
    print(" * MAE {mae:.3f} ".format(mae=mae))

    return mae


# ReduceLROnPlateau
# CosineAnnealingWarmRestarts

# def adjust_learning_rate(optimizer, epoch):
#     """
#     根据当前轮次调整学习率，每隔30个轮次学习率衰减为初始学习率的1/10
#     optimizer: 优化器
#     epoch: 当前轮次
#     """

#     # 初始学习率
#     lr = original_lr

#     # 遍历学习率更新阶段
#     for i in range(len(steps)):
#         # 如果当前轮次大于等于设定的阶段轮次
#         if epoch >= steps[i]:
#             # 获取当前阶段的学习率缩放比例
#             scale = scales[i] if i < len(scales) else 1
#             # 根据缩放比例更新学习率
#             lr = lr * scale
#             # 如果当前轮次正好等于设定的阶段轮次，结束循环
#             if epoch == steps[i]:
#                 break
#         else:
#             break

#     # 更新优化器中每个参数组的学习率
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr


class AverageMeter(object):
    """
    用于计算并存储平均值和当前值的类
    """

    def __init__(self):
        # 初始化各项统计指标
        self.reset()

    def reset(self):
        # 重置统计指标
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.sum = 0  # 总和
        self.count = 0  # 计数

    def update(self, val, n=1):
        """
        更新统计指标
        val: 当前值
        n: 当前值的数量，默认为1
        """
        # 更新当前值、总和和计数
        self.val = val
        self.sum += val * n
        self.count += n
        # 计算平均值
        self.avg = self.sum / self.count


if __name__ == "__main__":
    main()
