import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def load_data(data_folder, batch_size):
    test_indoor = CustomImageDataset("Data/val_indoors.csv", data_folder)
    test_outdoor = CustomImageDataset("Data/val_outdoor.csv", data_folder)
    test_data = test_indoor + test_outdoor

    train_indoor = CustomImageDataset("Data/train_indoors.csv", data_folder)
    train_outdoor = CustomImageDataset("Data/train_outdoor.csv", data_folder)
    train_data = train_indoor + train_outdoor

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader
