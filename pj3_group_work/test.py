import PIL.Image as Image
import numpy as np
import torchvision.transforms.functional as F
import torch
from model import CSRNet
from torchvision import transforms
from torch.autograd import Variable
from CSRNet_RGBT.csrnet_rgbt import CSRNet_RGBT
from ECAN.model import CANNet
import aiconfig

test_path = "./dataset/test/rgb/"
test_tir_path = "./dataset/test/tir/"
img_paths = [f"{test_path}{i}.jpg" for i in range(1, 1001)]
tir_img_paths = [f"{test_tir_path}{i}R.jpg" for i in range(1, 1001)]

model = CSRNet_RGBT()
# model = CANNet()
model = model.cuda()
# ./best/model_best.pth7.5.tar
checkpoint = torch.load(aiconfig.modelPath)
model.load_state_dict(checkpoint["state_dict"])

# for i in range(len(img_paths)):
#     img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))

#     img[0, :, :] = img[0, :, :]-92.8207477031
#     img[1, :, :] = img[1, :, :]-95.2757037428
#     img[2, :, :] = img[2, :, :]-104.877445883
#     img = img.cuda()
#     output = model(img.unsqueeze(0))
#     ans = output.detach().cpu().sum()
#     ans = "{:.2f}".format(ans.item())
#     print(f"{i+1},{ans}")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.452, 0.411, 0.362, 0.397], std=[0.188, 0.167, 0.162, 0.181]),
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
