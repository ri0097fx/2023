from PIL import Image
from torchvision import transforms
import os
import glob
import torch

def make_tensor_img(path, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
        ])
    if os.path.isdir(path):
        paths = glob.glob(os.path.join(path, '*.*'))
        imgs = []
        for img_path in paths:
            img = Image.open(img_path,)
            tensor_img = transform(img)
            imgs.append(tensor_img)
        return torch.stack(imgs)
    elif os.path.isfile(path):
        img = Image.open(path)
        tensor_img = transform(img)
        return tensor_img.unsqueeze(0)

torch.no_grad()
