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
from model import CSRNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim


# 用于保存最佳模型
def save_checkpoint(
    state, is_best, task_id, filename="checkpoint.pth.tar", save_dir="./model/"
):
    checkpoint_path = os.path.join(save_dir, task_id + filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_model_path = os.path.join(save_dir, task_id + "model_best.pth.tar")
        shutil.copyfile(checkpoint_path, best_model_path)


# 加载数据
def load_data(img_path, gt_path, train=True):
    """
    img_path: 图片文件路径
    gt_path: h5文件路径
    train: 是否为训练过程
    """

    # 打开图像文件并转换为RGB格式
    img = Image.open(img_path).convert("RGB")

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

        if shuffle:
            random.shuffle(self.img_paths)

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
        img, target = load_data(img_path, gt_path, self.train)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


# 学习率
lr = 1e-5
original_lr = lr

# 批大小
batch_size = 4
# batch_size = 1

# 动量
momentum = 0.95

# 权重衰减
decay = 5 * 1e-4

# 训练轮数
epochs = 400


# 工作线程数
workers = 4

# 随机种子
seed = time.time()

# 打印频率
print_freq = 30

# 图像路径
img_dir = "./dataset/train/rgb/"
gt_dir = "./dataset/train/hdf5s/"
# img_dir = "./expansion_dataset/rgb/"
# gt_dir = "./expansion_dataset/hdf5s/"

# 预训练模型
pre = None

# 任务名称
task = ""


#     lr = 1e-2
# batch_size = 8 # 51
# decay_interval = 30
# steps = [i * decay_interval for i in range(1, epochs // decay_interval + 1)]
# # 每次衰减的系数为 0.1（即学习率降低 10 倍）
# scales = [decay_interval] * len(steps)
#     criterion = nn.MSELoss().cuda()
def main():
    # 初始化起始轮次和最佳准确率
    start_epoch = 0
    best_prec1 = 1e6

    # 设置随机种子
    torch.cuda.manual_seed(seed)

    # 创建模型实例，并将其移动到GPU上
    model = CSRNet()
    model = model.cuda()

    # 定义损失函数和优化器
    #
    criterion = nn.MSELoss(reduction="sum").cuda()
    # criterion = nn.MSELoss(size_average=False).cuda()
    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr, momentum=momentum, weight_decay=decay
    # )
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.05,
        patience=2,
        threshold=0.0001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-08,
        verbose=True,
    )

    # 数据预处理
    # TODO 此处可设置数据增强
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 创建数据集实例，并分割为训练集和验证集
    dataset = ImgDataset(img_dir, gt_dir, transform=transform, train=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers
    )

    # 如果指定了预训练模型，则加载预训练参数
    if pre:
        if os.path.isfile(pre):
            print("=> loading checkpoint '{}'".format(pre))
            checkpoint = torch.load(pre)
            start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(pre, checkpoint["epoch"])
            )
        else:
            print("=> no checkpoint found at '{}'".format(pre))

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

        # scheduler.get_last_lr()[0]
        # optimizer.param_groups[0]["lr"]

        # 判断当前模型是否是最佳模型
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(" * best MAE {mae:.3f} ".format(mae=best_prec1))

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
            task,
        )


def data_augmentation(img, target):
    input_tensor, target_tensor = random_flip(img, target)
    if random.random() > 0.5:
        input_tensor2, target_tensor2 = split_and_merge(input_tensor, target_tensor)
    else:
        input_tensor2, target_tensor2 = split_and_merge(img, target)
    ran = random.random()
    if ran > 0.67:
        input_tensor1, target_tensor1 = downsample_and_combine(img, target)
    elif ran > 0.33:
        input_tensor1, target_tensor1 = downsample_and_combine(
            input_tensor, target_tensor
        )
    else:
        input_tensor1, target_tensor1 = downsample_and_combine(
            input_tensor2, target_tensor2
        )

    return torch.cat((input_tensor1, input_tensor2, input_tensor), dim=0), torch.cat(
        (target_tensor1, target_tensor2, target_tensor), dim=0
    )


def random_flip(input_tensor, target_tensor):
    for i in range(input_tensor.size(0)):
        if random.random() > 0.5:
            input_tensor[i] = torch.flip(input_tensor[i], dims=[2])
            target_tensor[i] = torch.flip(target_tensor[i], dims=[1])
        if random.random() > 0.5:
            input_tensor[i] = torch.flip(input_tensor[i], dims=[1])
            target_tensor[i] = torch.flip(target_tensor[i], dims=[0])
    return input_tensor, target_tensor


def merge(tl, tr, bl, br):
    return torch.cat((torch.cat((tl, tr), dim=-1), torch.cat((bl, br), dim=-1)), dim=-2)


def shuffle_two_arrays(arr1, arr2):
    """
    Shuffle two arrays such that the corresponding elements in both arrays maintain their relationship.

    Parameters:
    arr1 (list): The first array to shuffle.
    arr2 (list): The second array to shuffle, corresponding to arr1.

    Returns:
    tuple: A tuple containing the two shuffled arrays.
    """
    if len(arr1) != len(arr2):
        raise ValueError("Both arrays must have the same length")

    combined = list(zip(arr1, arr2))
    random.shuffle(combined)
    shuffled_arr1, shuffled_arr2 = zip(*combined)

    return list(shuffled_arr1), list(shuffled_arr2)


def split_and_merge(input_tensor, target_tensor):
    batch_size, channels, height, width = input_tensor.size()

    input_top_left = input_tensor[:, :, : height // 2, : width // 2]
    input_top_right = input_tensor[:, :, : height // 2, width // 2 :]
    input_bottom_left = input_tensor[:, :, height // 2 :, : width // 2]
    input_bottom_right = input_tensor[:, :, height // 2 :, width // 2 :]

    target_top_left = target_tensor[:, : height // 2, : width // 2]
    target_top_right = target_tensor[:, : height // 2, width // 2 :]
    target_bottom_left = target_tensor[:, height // 2 :, : width // 2]
    target_bottom_right = target_tensor[:, height // 2 :, width // 2 :]

    input_top_left, target_top_left = shuffle_two_arrays(
        input_top_left, target_top_left
    )
    input_top_right, target_top_right = shuffle_two_arrays(
        input_top_right, target_top_right
    )
    input_bottom_left, target_bottom_left = shuffle_two_arrays(
        input_bottom_left, target_bottom_left
    )
    input_bottom_right, target_bottom_right = shuffle_two_arrays(
        input_bottom_right, target_bottom_right
    )
    new_images = []
    new_targets = []
    for i in range(batch_size):
        idx1 = i
        idx2 = (i + 1) % batch_size
        idx3 = (i + 2) % batch_size
        idx4 = (i + 3) % batch_size
        new_images.append(
            merge(
                input_top_left[idx1],
                input_top_right[idx2],
                input_bottom_left[idx3],
                input_bottom_right[idx4],
            )
        )
        new_targets.append(
            merge(
                target_top_left[idx1],
                target_top_right[idx2],
                target_bottom_left[idx3],
                target_bottom_right[idx4],
            )
        )

    new_images_tensor = torch.stack(new_images)
    new_targets_tensor = torch.stack(new_targets)

    return new_images_tensor, new_targets_tensor


def down_sample(tensor):
    """
    Downsample the image tensor by a factor of 2.
    Only keep the top-left pixel of each 2x2 block.
    """
    if tensor.dim() == 2:
        return tensor[::2, ::2]
    if tensor.dim() == 3:
        return tensor[:, ::2, ::2]
    if tensor.dim() == 4:
        return tensor[:, :, ::2, ::2]


def downsample_and_combine(input_tensor, target_tensor):
    batch_size, channels, height, width = input_tensor.size()
    if batch_size % 4 != 0:
        return input_tensor, target_tensor

    # input_top_left, input_top_right, input_bottom_left, input_bottom_right, target_top_left, target_top_right, target_bottom_left, target_bottom_right = split(
    #     input_tensor, target_tensor
    # )
    data = []
    target = []
    for i in range(batch_size):
        data.append(down_sample(input_tensor[i]))
        target.append(down_sample(target_tensor[i]))
    # wash card
    data, target = shuffle_two_arrays(data, target)

    new_images = []
    new_targets = []

    for i in range(batch_size // 4):
        tl, tr, bl, br = data[i], data[i + 1], data[i + 2], data[i + 3]
        new_images.append(merge(tl, tr, bl, br))
        tl, tr, bl, br = target[i], target[i + 1], target[i + 2], target[i + 3]
        new_targets.append(merge(tl, tr, bl, br))

    return torch.stack(new_images), torch.stack(new_targets)


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
        if batch_size == 4:
            img, target = data_augmentation(img, target)
        else:
            img, target = random_flip(img, target)
        # print(img.shape)
        # print(target.shape)

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

        # 更新损失记录器
        losses.update(loss.item(), img.size(0))

        # 清除之前的梯度
        optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 使用优化器更新模型参数
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
        mae += abs(
            output.data.sum() - target.sum().type(torch.FloatTensor).cuda()
        ) / img.size(0)

    # 计算平均 MAE
    mae = mae / len(val_loader)
    # 打印平均 MAE
    print(" * MAE {mae:.3f} ".format(mae=mae))

    return mae


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
