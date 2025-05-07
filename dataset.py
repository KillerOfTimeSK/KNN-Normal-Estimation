import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, ToPILImage
import matplotlib.pyplot as plt
from PIL import Image
from torch import from_numpy

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
        image = Image.open(img_path)
        try:
            normal = np.load(normal_path)
            normal = from_numpy(normal).permute(2, 0, 1)

        except EOFError:
            print(normal_path)
            exit(1)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            normal = self.target_transform(normal)

        sample = {'image': image, 'normal': normal}
        return sample


def load_data(data_folder, batch_size, size=(224, 224)):
    in_transform_train = Compose([
        Resize(size),
        ToTensor(),
    ])

    out_transform_train = Compose([
        Resize(size)
    ])

    in_transform_test = Compose([
        Resize(size),
        ToTensor(),
    ])

    out_transform_test = Compose([
        Resize(size)
    ])

    test_indoor = CustomImageDataset(os.path.join(data_folder, "val_indoors.csv"), data_folder, transform=in_transform_test, target_transform=out_transform_test)
    test_outdoor = CustomImageDataset(os.path.join(data_folder, "val_outdoor.csv"), data_folder, transform=in_transform_test, target_transform=out_transform_test)
    test_data = test_indoor + test_outdoor

    train_indoor = CustomImageDataset(os.path.join(data_folder, "train_indoors.csv"), data_folder, transform=in_transform_train, target_transform=out_transform_train)
    train_outdoor = CustomImageDataset(os.path.join(data_folder, "train_outdoor.csv"), data_folder, transform=in_transform_train, target_transform=out_transform_train)
    train_data = train_indoor + train_outdoor

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data)

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    train, test = load_data("Data", 64, size=(224, 224))
    print(len(test))
    for data in test:
        print(data["image"].shape)
        print(data["normal"].shape)
        img = data['image'].squeeze(0).permute(1, 2, 0).numpy()
        normal = data['normal'].squeeze(0).permute(1, 2, 0).numpy()
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(normal)
        plt.title('Normal Image')
        plt.show()

        exit(0)