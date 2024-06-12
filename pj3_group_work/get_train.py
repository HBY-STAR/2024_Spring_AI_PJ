import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image


from CSRNet_RGBT.csrnet_rgbt import CSRNet_RGBT
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Res50.model.Res50 import Res50
from ECAN.model import CANNet
from train import ImgDataset
import aiconfig


test_path = "./dataset/train/rgb/"
test_tir_path = "./dataset/train/tir/"
img_paths = [f"{test_path}{i}.jpg" for i in range(1, 1808)]
tir_img_paths = [f"{test_tir_path}{i}R.jpg" for i in range(1, 1808)]

model = CSRNet_RGBT()
model.eval()
model = model.cuda()
checkpoint = torch.load(aiconfig.modelPath)
model.load_state_dict(checkpoint["state_dict"])


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.452, 0.411, 0.362, 0.397], std=[0.188, 0.167, 0.162, 0.181]
        ),
    ]
)

for i in range(len(img_paths)):
    rgb_img = Image.open(img_paths[i]).convert("RGB")
    tir_img = Image.open(tir_img_paths[i]).convert("L")
    rgb_img = np.array(rgb_img)
    tir_img = np.array(tir_img)
    img_np = np.concatenate((rgb_img, np.expand_dims(tir_img, axis=2)), axis=2)
    img = Image.fromarray(img_np)

    img = transform(img)
    img = img.cuda()
    img = Variable(img)

    output = model(img.unsqueeze(0))
    ans = output.detach().cpu().sum()
    ans = "{:.2f}".format(ans.item())
    print(f"{i+1},{ans}")
