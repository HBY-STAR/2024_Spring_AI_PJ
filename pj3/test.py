import PIL.Image as Image
import torchvision.transforms.functional as F
import torch
from model import CSRNet


test_path = "./dataset/test/rgb/"
img_paths = [f"{test_path}{i}.jpg" for i in range(1, 1001)]

model = CSRNet()
model = model.cuda()
checkpoint = torch.load('./model/model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

for i in range(len(img_paths)):
    img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))

    img[0, :, :] = img[0, :, :]-92.8207477031
    img[1, :, :] = img[1, :, :]-95.2757037428
    img[2, :, :] = img[2, :, :]-104.877445883
    img = img.cuda()
    output = model(img.unsqueeze(0))
    ans = output.detach().cpu().sum()
    ans = "{:.2f}".format(ans.item())
    print(f"{i+1},{ans}")
