import torch.nn as nn
import torch
from torchvision import models

# CSRNet 论文
# https://arxiv.org/abs/1802.10062


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0

        # 前端配置，for 2D feature extraction
        # 10个卷积层（VGG16的前10层），3个最大池化层。
        self.frontend_feat = [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
        ]

        # 后端配置，to deliver larger reception fields and  replace pooling operations
        # 使用 Dilated convolution （扩展的卷积层）
        self.backend_feat = [512, 512, 512, 256, 128, 64]

        # 前端和后端
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)

        # 输出层
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            # frontend_state_dict = self.frontend.state_dict()
            # mod_state_dict = mod.state_dict()
            # for i, (key, value) in enumerate(list(frontend_state_dict.items())[1:]):  # 跳过第一层
            #     frontend_state_dict[key].data[:] = list(mod_state_dict.items())[i][1].data[:]
            for i, (key, value) in enumerate(list(self.frontend.state_dict().items())):
                self.frontend.state_dict()[key].data[:] = list(
                    mod.state_dict().items()
                )[i][1].data[:]
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        # x = self.relu(x)
        return x

    # 初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    """
    cfg: list, 卷积层配置，这里分别接收前后端的配置
    in_channels: 输入通道数
    batch_norm: 是否使用BN
    dilation: 是否使用扩展卷积
    """
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(
                in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate
            )
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
