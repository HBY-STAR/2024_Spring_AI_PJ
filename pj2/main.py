# Importing Libraries
import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
import os
from tqdm import tqdm


#Define the model, here we take resnet-18 as an example

class BasicBlock(nn.Module):
    expansion = 1
    

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        DROPOUT = 0.1

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(DROPOUT)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(DROPOUT)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                nn.Dropout(DROPOUT)
            )

    def forward(self, x):
        out = F.relu(self.dropout(self.bn1(self.conv1(x))))
        out = self.dropout(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)


def design_model():
    return ResNet(BasicBlock, [2, 2, 2, 2])




#训练代码

def model_training(model, device, train_dataloader, optimizer, train_acc, train_losses):
            
    model.train()
    pbar = tqdm(train_dataloader)
    correct = 0
    processed = 0
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        #TODO,补全代码,填在下方

        #补全内容:optimizer的操作，获取模型输出，loss设计与计算，反向传播

        #TODO,补全代码,填在上方

        train_losses.append(loss.item())
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        # print statistics
        running_loss += loss.item()
        pbar.set_description(desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)


#验证代码

def model_testing(model, device, test_dataloader, test_acc, test_losses, misclassified = []):
    
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():

        for index, (data, target) in enumerate(test_dataloader):
            data, target = data.to(device), target.to(device)
            
            #TODO,补全代码,填在下方

            #补全内容:获取模型输出，loss计算

            #TODO,补全代码,填在上方

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_dataloader.dataset)
    test_losses.append(test_loss)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))
    
    test_acc.append(100. * correct / len(test_dataloader.dataset))


def main():
    device = "cuda" if torch.cuda.is_available else "cpu"
    print(device)


    #prepare datasets and transforms
    train_transforms = transforms.Compose([
            #TODO,设计针对训练数据集的图像增强
            transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)) #Normalize all the images
            ])
    test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
        ])

    data_dir = './data'
    trainset = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transforms)
    testset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                            shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                            shuffle=False, num_workers=4)
    

    # Importing Model and printing Summary,默认是ResNet-18
    #TODO,分析讨论其他的CNN网络设计

    model = design_model().to(device)
    summary(model, input_size=(3,32,32))

    # Training the model

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=2, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)


    train_acc = []
    train_losses = []
    test_acc = []
    test_losses = []
    model_path = './checkpoints'
    os.makedirs(model_path,exist_ok=True)

    EPOCHS = 40

    for i in range(EPOCHS):
        print(f'EPOCHS : {i}')
        #TODO,补全model_training里的代码
        model_training(model, device, trainloader, optimizer, train_acc, train_losses)
        scheduler.step(train_losses[-1])
        #TODO,补全model_testing里的代码
        model_testing(model, device, testloader, test_acc, test_losses)

        # 保存模型权重
        torch.save(model.state_dict(), os.path.join(model_path,'model.pth'))


    fig, axs = plt.subplots(2,2, figsize=(25,20))

    axs[0,0].set_title('Train Losses')
    axs[0,1].set_title(f'Training Accuracy (Max: {max(train_acc):.2f})')
    axs[1,0].set_title('Test Losses')
    axs[1, 1].set_title(f'Test Accuracy (Max: {max(test_acc):.2f})')

    axs[0,0].plot(train_losses)
    axs[0,1].plot(train_acc)
    axs[1,0].plot(test_losses)
    axs[1,1].plot(test_acc)

    # 保存图像
    plt.savefig('curves.png')  # 保存为名为 'plot.png' 的图片文件



if __name__ == '__main__':
    main()