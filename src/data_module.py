import numpy as np

import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from src.utils import read_image, read_class_description
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader 

from lightning import LightningDataModule

import os
from glob import glob

from cfg import CFG



class DatasetForCLIP(Dataset):
    def __init__(self, dict_cls_imgpaths: dict, transform=None, target_transform=None):
        self.dict_cls_imgpaths = dict_cls_imgpaths
        self.transform = transform
        self.target_transform = target_transform
        
        self.cls2decs = read_class_description()
        self.classes = list(self.cls2decs.keys())
    
        # calculate num of total datas
        self._len = sum(len(img_paths) for img_paths in self.dict_cls_imgpaths.values())
    
    def __len__(self):
        # A train batch consists of image of every each class 
        return self._len // CFG.num_classes
    
    def __getitem__(self, idx):
        # target must be matched with order of classes
        images = []
        
        for cls in self.classes: 
            img_path = np.random.choice(self.dict_cls_imgpaths[cls])
            image = read_image(img_path)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        
        images = torch.stack(images)
        return images
    

class DataModuleForCLIP(LightningDataModule):
    def __init__(
        self, 
        train_dset_path: os.PathLike, 
        test_size: float=0.2,
        batch_size: str=32,
        width: int=224, height: int=224,
        train_transforms: transforms=None,
        val_transforms: transforms=None,
    ) -> None:
        super().__init__()
        self.train_dset_path = train_dset_path
        self.test_size=test_size
        self.batch_size=batch_size
        self.width = width
        self.height = height
        self.train_transforms=train_transforms
        self.val_transforms=val_transforms
    
    def setup(self, stage: str) -> None:
        class_desc_dict = read_class_description()
        train_class_img_paths_dict = {}
        val_class_img_paths_dict = {}
        for cls in class_desc_dict.keys():
            img_paths = glob(os.path.join(CFG.train_dataset_dir, cls, "*"))
            train_img_paths, val_img_paths = train_test_split(img_paths, test_size=self.test_size)
            
            train_class_img_paths_dict[cls] = train_img_paths
            val_class_img_paths_dict[cls] = val_img_paths
            
        self.train_dataset = DatasetForCLIP(train_class_img_paths_dict, transform=self.train_transforms)
        self.val_dataset = DatasetForCLIP(val_class_img_paths_dict, transform=self.val_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader( # type: ignore
            self.train_dataset, shuffle=True, batch_size=self.batch_size,
            num_workers=8, pin_memory=True, drop_last=True)
    def val_dataloader(self) -> DataLoader:
        return DataLoader( # type: ignore
            self.val_dataset, batch_size=self.batch_size, 
            num_workers=8, pin_memory=True)
    def test_dataloader(self) -> DataLoader:
        return DataLoader( # type: ignore
            self.val_dataset, batch_size=self.batch_size, 
            num_workers=8, pin_memory=True)
    def predict_dataloader(self) -> DataLoader:
        return DataLoader( # type: ignore
            self.val_dataset, batch_size=self.batch_size, 
            num_workers=8, pin_memory=True)