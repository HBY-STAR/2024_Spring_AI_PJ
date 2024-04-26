import torch
from vit_pytorch import ViT
from einops.layers.torch import Rearrange
import torch.nn as nn
from torchvision import models


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128,
                              'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(
            self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            for i, (key, value) in enumerate(list(self.frontend.state_dict().items())):
                self.frontend.state_dict()[key].data[:] = list(
                    mod.state_dict().items())[i][1].data[:]

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

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
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                               padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class Vit4R(nn.Module):
    def __init__(self, load_weights=False):
        super(Vit4R, self).__init__()
        self.seen = 0
        self.vit = ViT(
            image_size=64,
            patch_size=16,
            num_classes=1,  # Output regression score
            dim=768,  # You can adjust this based on your needs
            depth=12,  # Number of transformer blocks
            heads=12,  # Number of attention heads
            mlp_dim=1024,  # Hidden layer size in MLP
            dropout=0.1,  # Dropout rate
            emb_dropout=0.1  # Dropout rate for token embeddings
        )

        self.output_layer = nn.Conv2d(768, 1, kernel_size=1)

        if not load_weights:
            # Initialize weights of ViT
            self._initialize_weights()

    def forward(self, x):
        # Rearrange input for ViT
        x = Rearrange('b c h w -> b (h w) c')(x)
        x = self.vit(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        # Initialize weights of ViT
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'vit' in name:
                    # Initialize weights of ViT
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # Initialize biases
                nn.init.constant_(param, 0.0)
