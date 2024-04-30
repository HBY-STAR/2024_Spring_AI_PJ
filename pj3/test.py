import PIL.Image as Image
import torchvision.transforms.functional as F
import torch
from model import CSRNet
from torchvision import transforms
from torch.autograd import Variable

test_path = "./dataset/test/rgb/"
img_paths = [f"{test_path}{i}.jpg" for i in range(1, 1001)]

model = CSRNet()
model = model.cuda()
checkpoint = torch.load('./model/model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225]),
])

for i in range(len(img_paths)):
    img = transform((Image.open(img_paths[i]).convert('RGB')))
    img = img.cuda()
    img = Variable(img)
    output = model(img.unsqueeze(0))
    ans = output.detach().cpu().sum()
    ans = "{:.2f}".format(ans.item())
    print(f"{i+1},{ans}")
