# Importing Libraries
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm


def design_model(cnn_model):
    # https://pytorch.org/vision/main/models.html#classification
    if cnn_model == 'ResNet-18':
        return torchvision.models.resnet18()
        # return ResNet(BasicBlock, [2, 2, 2, 2])
    elif cnn_model == 'AlexNet':
        return torchvision.models.alexnet()
    elif cnn_model == 'VGG':
        return torchvision.models.vgg16()
    elif cnn_model == 'MobileNet-V3-Small':
        return torchvision.models.mobilenet_v3_small()
    else:
        exit('Model not supported')


# 训练代码

def model_training(model, device, train_dataloader, optimizer, train_acc, train_losses):
    model.train()
    pbar = tqdm(train_dataloader)
    correct = 0
    processed = 0
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        # TODO
        # 补全内容:optimizer的操作，获取模型输出，loss设计与计算，反向传播
        optimizer.zero_grad()
        y_pred = model(data)
        loss = F.cross_entropy(y_pred, target)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        # print statistics
        running_loss += loss.item()
        pbar.set_description(desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
        train_acc.append(100 * correct / processed)


# 验证代码

def model_testing(model, device, test_dataloader, test_acc, test_losses, misclassified=[]):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for index, (data, target) in enumerate(test_dataloader):
            data, target = data.to(device), target.to(device)

            # TODO
            # 补全内容:获取模型输出，loss计算
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_dataloader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))

    test_acc.append(100. * correct / len(test_dataloader.dataset))


def main(cnn_model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # prepare datasets and transforms
    train_transforms = transforms.Compose([
        # https://pytorch.org/vision/0.9/transforms.html
        # 50%的概率对图像进行水平翻转
        # torchvision.transforms.RandomHorizontalFlip(),
        # 随机旋转角度范围为-10到10度
        # transforms.RandomRotation(10),
        # 随机裁剪图像，裁剪后的图像大小为原图像的0.9到1之间
        # transforms.RandomResizedCrop(32, scale=(0.9, 1.0), ratio=(0.9, 1.1)),

        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # comvert the image to tensor so that it can work with torch
        transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))  # Normalize all the images
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
    ])

    data_dir = './data'
    trainset = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transforms)
    testset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=4)

    # Importing Model and printing Summary,默认是ResNet-18
    # TODO,分析讨论其他的CNN网络设计

    model = design_model(cnn_model).to(device)
    summary(model, input_size=(3, 224, 224))

    # Training the model

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=2, threshold=0.0001,
                                  threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)

    train_acc = []
    train_losses = []
    test_acc = []
    test_losses = []
    model_path = './checkpoints'
    os.makedirs(model_path, exist_ok=True)

    print(f'Running with Model: {cnn_model}')

    EPOCHS = 40

    for i in range(EPOCHS):
        print(f'EPOCHS : {i}')
        model_training(model, device, trainloader, optimizer, train_acc, train_losses)
        scheduler.step(train_losses[-1])
        model_testing(model, device, testloader, test_acc, test_losses)

        # 保存模型权重
        torch.save(model.state_dict(), os.path.join(model_path, cnn_model + '_' + 'model.pth'))

    return max(test_acc)


if __name__ == '__main__':

    models = ['ResNet-18', 'AlexNet', 'MobileNet-V3-Small', 'VGG']
    log_file = open("result_models.log", "w")
    original_stdout = sys.stdout
    sys.stdout = log_file

    test_acc = []
    run_time = []

    for model in models:
        start = time.time()
        temp = main(model)
        end = time.time()
        print(f'Model: {model}, Test Accuracy: {temp:.2f}, Run Time: {end - start:.2f}')
        test_acc.append(temp)
        run_time.append(end - start)

    # 使用柱状图分别展示不同模型的准确率和运行时间
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    axs[0].bar(models, test_acc)
    axs[0].set_title('Test Accuracy')
    axs[0].set_xlabel('Model')
    axs[0].set_ylabel('Accuracy (%)')

    axs[1].bar(models, run_time)
    axs[1].set_title('Run Time')
    axs[1].set_xlabel('Model')
    axs[1].set_ylabel('Time (s)')

    plt.show()
    plt.savefig('result_models.png')

    sys.stdout = original_stdout
    log_file.close()
