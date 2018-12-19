import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
FOLDER_DATASET = "./Track_1_Wheel_Test/"
plt.ion()

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self,root_dir,transform=None):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.name_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.name_frame.iloc[idx, 0])
        image = Image.open(img_name)
        image = self.transform(image)

        #labels = labels.reshape(-1, 2)
        sample = image

        return sample

def load_dataset():
    data_path = 'images/'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader
transform = transforms.Compose(
[transforms.ToTensor(),
 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

ds = CustomDataset(root_dir='images')

loader = torch.utils.data.DataLoader(ds,batch_size=32,shuffle=True, num_workers=1)

ds = load_dataset()
for i_batch,sample_batched in enumerate(ds):
    print(i_batch)