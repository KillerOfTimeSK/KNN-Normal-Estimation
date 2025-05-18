import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize
from PIL import Image
from torch import from_numpy

class ImageNormalDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, target_transform=None, include_depth=False):
        self.img_paths = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.include_depth = include_depth
        self.depth_transform = ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths.iloc[idx, 0])
        normal_path = os.path.join(self.img_dir, self.img_paths.iloc[idx, 0])[:-4] + "_normal.npy"
        image = Image.open(img_path)

        if self.include_depth:
            depth_path = os.path.join(self.img_dir, self.img_paths.iloc[idx, 1])
            depth = self.depth_transform(np.load(depth_path)).repeat(3, 1, 1)

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
            if self.include_depth:
                depth = self.target_transform(depth)

        if self.include_depth:
            sample = {'image': image, 'depth': depth, 'normal': normal}
        else:
            sample = {'image': image, 'normal': normal}
        return sample


def load_data(data_folder, batch_size, size=(224, 224), indoor=True, outdoor=True, depth=False):
    if not (indoor or outdoor):
        raise RuntimeError('When loading dataset you need so select at least one subdataset. But neither indoor nor outdoor was selected.')

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

    if indoor:
        test_indoor = ImageNormalDataset(os.path.join(data_folder, "val_indoors.csv"), data_folder,
                                         transform=in_transform_test, target_transform=out_transform_test,
                                         include_depth=depth)
        train_indoor = ImageNormalDataset(os.path.join(data_folder, "train_indoors.csv"), data_folder,
                                          transform=in_transform_train, target_transform=out_transform_train,
                                          include_depth=depth)
    if outdoor:
        test_outdoor = ImageNormalDataset(os.path.join(data_folder, "val_outdoor.csv"), data_folder,
                                          transform=in_transform_test, target_transform=out_transform_test,
                                          include_depth=depth)
        train_outdoor = ImageNormalDataset(os.path.join(data_folder, "train_outdoor.csv"), data_folder,
                                           transform=in_transform_train, target_transform=out_transform_train,
                                           include_depth=depth)

    if indoor and outdoor:
        test_data = test_indoor + test_outdoor
        train_data = train_indoor + train_outdoor
    elif indoor:
        test_data = test_indoor
        train_data = train_indoor
    elif outdoor:
        test_data = test_outdoor
        train_data = test_outdoor

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data)

    return train_dataloader, test_dataloader