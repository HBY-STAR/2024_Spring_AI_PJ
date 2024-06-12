import PIL.Image as Image
import torchvision.transforms.functional as F
import torch
from model import CSRNet
from torchvision import transforms
from torch.autograd import Variable
import os
import numpy as np

test_path = "./dataset/test/rgb/"
img_paths = [f"{test_path}{i}.jpg" for i in range(1, 1001)]

model = CSRNet()
model = model.cuda()

checkpoint = torch.load("./model/model_best.pth.tar")  # ./best/model_best.pth7.5.tar
model.load_state_dict(checkpoint["state_dict"])

for i in range(len(img_paths)):
    img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))

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
        transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.225]),
    ]
)

for i in range(len(img_paths)):
    rgb_path = img_paths[i]
    tir_path = "./dataset/test/tir/" + os.path.basename(img_paths[i])[:-4] + "R.jpg"
    rgb_image = Image.open(rgb_path).convert("RGB")
    tir_image = Image.open(tir_path).convert("L")
    rgb_np = np.array(rgb_image)
    tir_np = np.array(tir_image)
    img = np.concatenate((rgb_np, np.expand_dims(tir_np, axis=2)), axis=2)
    Image.fromarray(img).show()
    img = transform(img)
    img = img.cuda()
    img = Variable(img)
    output = model(img.unsqueeze(0))
    ans = output.detach().cpu().sum()
    ans = "{:.2f}".format(ans.item())
    print(f"{i+1},{ans}")
