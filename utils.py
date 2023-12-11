from PIL import Image
from torchvision import transforms
import os
import glob
import torch
import torch.nn as nn

def make_tensor_img(path, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
        ])
    if os.path.isdir(path):
        paths = sorted(glob.glob(os.path.join(path, '*.*')))
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

def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)
