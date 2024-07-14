import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, RandomCrop, ToTensor, CenterCrop, Normalize

from einops import rearrange
from functools import partial

from cv_common_utils import read_file_lst_txt

def flower_train_t(img_size, mean, std):
    t = Compose([
        Resize(img_size),
        RandomCrop((img_size, img_size)),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return t

def flower_test_t(img_size, mean, std):
    t = Compose([
        Resize(img_size),
        CenterCrop((img_size, img_size)),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return t


class ImageDataset(Dataset):
    def __init__(self, root_dir, file_lst_txt=None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        files = None
        if file_lst_txt is not None:
            files = read_file_lst_txt(file_lst_txt)
        
        if files is not None:
            self.image_files = [f for f in files if os.path.isfile(os.path.join(root_dir, f))]
        else:
            self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image
    


def get_flower_train_data(data_config):
    dataset = ImageDataset(**data_config.dataset_config, transform=flower_train_t(**data_config.transform_config))
    data_loader = DataLoader(dataset=dataset, 
                            **data_config.data_loader_config, 
                            shuffle=True)
    return dataset, data_loader

def get_flower_test_data(data_config):
    dataset = ImageDataset(**data_config.dataset_config, transform=flower_test_t(**data_config.transform_config))
    data_loader = DataLoader(dataset=dataset, 
                            **data_config.data_loader_config, 
                            shuffle=False)
    return dataset, data_loader