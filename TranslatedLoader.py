from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np
import pandas as pd
from PIL import Image
import torch

import os
def FindMissingPath(path):
    if os.path.exists(path): return []
    # Iterate directory by directory, going from root to file and check if the file exists
    # If it doesn't exist, throw more informing error
    ret = [f'ile {path} does not exist. Trying to find it...']
    dirs = path.split('/')
    tested_path = dirs[0]
    last_path = '/'
    for dir in dirs:
        if dir == '' or dir == '~': continue
        tested_path += '/' + dir
        ret.append('Testing: ' + tested_path)
        if not os.path.exists(tested_path):
            other_childs = os.listdir(last_path)
            if len(other_childs) == 0:
                ret.append('No other child found in ' + last_path)
            else:
                ret.append('Other children found in ' + last_path + ':')
                for child in other_childs:
                    ret.append(child)
            raise FileNotFoundError(f"File {path} does not exist :(. \n{'\n'.join(ret)}")
        last_path = tested_path
    return ret

class NormalMapDataset(Dataset):
    def __init__(self, csv_dir, translation_table, transform_rgb=None, transform_normal=None, valid_rgb=None, more=None):
        self.data = pd.read_csv(csv_dir, header=None)
        self.data = self.data[self.data[3] != 'Unavailable']
        # self.data = self.data.iloc[::2, :].reset_index(drop=True)
        # self.data = self.data.iloc[::2, :].reset_index(drop=True)
        # self.data = self.data.iloc[::3, :].reset_index(drop=True)
        # self.data = self.data.iloc[::3, :].reset_index(drop=True)

        self.rgb_paths = self.data[0].tolist()
        self.normal_paths = self.data[3].tolist()
        self.N = len(self.rgb_paths)
        # Translation table is dictionary<string, string> to create absolute path from relative path
        for key, value in translation_table.items():
            if self.rgb_paths[0].startswith(key): self.rgb_paths = [path.replace(key, value) for path in self.rgb_paths]
            if self.normal_paths[0].startswith(key): self.normal_paths = [path.replace(key, value) for path in self.normal_paths]
        
        if more is not None: # Not enough time to do this the right way :()
            moreData = pd.read_csv(more, header=None)
            moreData = moreData[moreData[3] != 'Unavailable']
            moreRGBPaths = moreData[0].tolist()
            moreNormalPaths = moreData[3].tolist()
            for key, value in translation_table.items():
                if moreRGBPaths[0].startswith(key): moreRGBPaths = [path.replace(key, value) for path in moreRGBPaths]
                if moreNormalPaths[0].startswith(key): moreNormalPaths = [path.replace(key, value) for path in moreNormalPaths]
            self.rgb_paths += moreRGBPaths
            self.normal_paths += moreNormalPaths
            self.N = len(self.rgb_paths)
            
        self.transform_rgb = transform_rgb
        self.transform_normal = transform_normal
        self.valid_rgb = valid_rgb
    
    def __len__(self): return self.N

    def __getitem__(self, idx):
        rgb_path = self.rgb_paths[idx]
        normal_path = self.normal_paths[idx]
        if not os.path.exists(rgb_path):
            info = FindMissingPath(rgb_path)
            raise FileNotFoundError(f"RGB file not found: {rgb_path}\nInfo: {'\n'.join(info)}")
        if not os.path.exists(normal_path):
            info = FindMissingPath(normal_path)
            raise FileNotFoundError(f"Normal file not found: {normal_path}\nInfo: {'\n'.join(info)}")
        
        rgb = Image.open(rgb_path)
        normal = np.load(normal_path)
        normal = torch.from_numpy(normal).permute(2, 0, 1).float()
        
        if self.transform_rgb: rgb = self.transform_rgb(rgb)
        if self.transform_normal: normal = self.transform_normal(normal)
        
        return rgb, normal
    
    def GetImage(self, idx):
        rgb_path = self.rgb_paths[idx]
        normal_path = self.normal_paths[idx]

        rgb = Image.open(rgb_path)
        normal = np.load(normal_path)
        normal = torch.from_numpy(normal).permute(2, 0, 1).float()
        rgbT = rgb.copy()
        tensor = rgb.copy()
        if self.valid_rgb: rgbT = self.valid_rgb(rgbT)
        if self.transform_rgb: tensor = self.transform_rgb(rgb)
        if self.transform_normal: normal = self.transform_normal(normal)
        
        return rgb, rgbT, tensor, normal

def GetDataset(csv_dir, translation_table, desiredOutSize=None, more=None):
    transform_normal = None
    if desiredOutSize is not None:
        transform_normal = transforms.Compose([
            transforms.Resize((desiredOutSize, desiredOutSize)),
        ])

    return NormalMapDataset(csv_dir, translation_table,
                            transform_rgb=transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ]),
                            transform_normal=transform_normal,
                            valid_rgb=transforms.Compose([
                                transforms.Resize((224, 224)),
                            ]), more=more)