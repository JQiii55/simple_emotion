import torch
from torchvision import transforms, datasets
import os

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(48, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(48),
        transforms.CenterCrop(48),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ]),
}

data_dir = '/root/cqy/simpleconv3_emotion/data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

# 打印一些样本的形状信息
for inputs, labels in dataloaders['train']:
    print(inputs.shape, labels.shape)
    break
