import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, target_transform=None):
        self.img_paths = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths.iloc[idx, 0])
        normal_path = os.path.join(self.img_dir, self.img_paths.iloc[idx, 0])[:-4] + "_normal.npy"
        image = read_image(img_path)
        normal = np.load(normal_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            normal = self.target_transform(normal)

        sample = {'image': image, 'normal': normal}
        return sample


def load_data(data_folder, batch_size):
    test_indoor = CustomImageDataset("Data/val_indoors.csv", data_folder, target_transform=ToTensor())
    test_outdoor = CustomImageDataset("Data/val_outdoor.csv", data_folder, target_transform=ToTensor())
    test_data = test_indoor + test_outdoor

    train_indoor = CustomImageDataset("Data/train_indoors.csv", data_folder, target_transform=ToTensor())
    train_outdoor = CustomImageDataset("Data/train_outdoor.csv", data_folder, target_transform=ToTensor())
    train_data = train_indoor + train_outdoor

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data)

    return train_dataloader, test_dataloader

