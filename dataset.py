import os
import torch
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt

class LabeledDataset(Dataset):
    def __init__(self, path2data='./data/train/labeled/', transform=None):
        self.transform = transform
        self.path = path2data
        walker = os.walk(path2data)
        _, classes, _ = next(walker)
        classes = sorted(classes)
        self.classes = classes
        self.class2ind = {classes[i]:i for i in range(len(classes))}
        self.files = []
        
        for path, folders, files in walker:
            for file in files:
                self.files.append(f'{path}/{file}')
        
        self.length_dataset = len(self.files)
        
    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        filename = self.files[idx]
        classname = filename.split('/')[-2]
        class_idx = self.class2ind[classname]
        
        img = torch.tensor(plt.imread(filename), dtype=torch.float32)
        img = img.transpose_(0,2)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, class_idx


class UnlabeledDataset:
    def __init__(self, path2data='./data/train/unlabeled/', transform=None):
        self.transform = transform
        self.path = path2data
        walker = os.walk(path2data)
        self.files = []
        
        for path, folders, files in walker:
            for file in files:
                self.files.append(f'{path}/{file}')
        self.length_dataset = len(self.files)
        
    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        filename = self.files[idx]
        img = torch.tensor(plt.imread(filename), dtype=torch.float32)
        img = img.transpose_(0,2)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img