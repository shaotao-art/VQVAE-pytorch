import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from copy import deepcopy
import random
from typing import List
from functools import partial
import torch
from torchvision.utils import make_grid
from typing import Dict, List, Tuple


def split_dataset(input_folder, train_ratio=0.9, seed=42, suffix='jpg'):
    """
    将 input_folder 文件夹内的所有文件按比例分为训练集和验证集，并保存到 train.txt 和 val.txt 文件中。
    
    :param input_folder: 存放文件的文件夹路径
    :param train_ratio: 训练集所占的比例，默认值为 0.8
    :param seed: 随机种子，默认值为 42
    """
    # 设置随机种子
    random.seed(seed)

    # 获取文件夹中的所有文件
    all_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and f.endswith(suffix)]
    
    # 打乱文件列表
    random.shuffle(all_files)

    # 计算训练集和验证集的大小
    train_size = int(train_ratio * len(all_files))
    
    # 分割文件列表
    train_files = all_files[:train_size]
    val_files = all_files[train_size:]
    
    # 将文件列表保存到 txt 文件中
    with open('train.txt', 'w') as train_file:
        for file in train_files:
            train_file.write(f"{file}\n")
    
    with open('val.txt', 'w') as val_file:
        for file in val_files:
            val_file.write(f"{file}\n")
    
    print(f"训练集文件数: {len(train_files)}")
    print(f"验证集文件数: {len(val_files)}")
    print("文件列表已保存到 train.txt 和 val.txt 中")
    
    
def read_file_lst_txt(text_p, suffix='jpg'):
    with open(text_p, 'r') as f:
        data_lst = f.read().split()
    data_lst = [f for f in data_lst if f.endswith(suffix)]
    return data_lst



def array2tensor(inp):
    assert isinstance(inp, np.ndarray)
    return torch.tensor(inp)

def print_lst_tensor_shape(lst, name=None):
    if name is not None:
        print(f'{name} is a list of tensor with shape')
    for x in lst:
        print('\t', x.shape)


def append_dims(inp: torch.Tensor, target_len: int):
    """unsqueeze tensor at dim=-1, until ndim=target_len"""
    assert len(inp.shape) < target_len
    while len(inp.shape) < target_len:
        inp = inp.unsqueeze(-1)
    return inp

def denorm_img(inp: torch.Tensor,
               mean=torch.tensor([0.485, 0.456, 0.406]), 
               std=torch.tensor([0.229, 0.224, 0.225])):
    """denorm tensor of shape (b[optional], c, h, w) 
    """
    mean, std = list(map(partial(append_dims, target_len=3), [mean, std]))
    return inp * std + mean




def batch_img_tensor_to_img_lst(inp: torch.Tensor) -> List[np.ndarray]:
    """convert tensor of shape (b, 3, h, w)
    to a list of np array of shape (h, w, 3)

    Args:
        inp (torch.Tensor): _description_

    Returns:
        List[np.ndarray]: _description_
    """
    assert len(inp.shape) == 4 and isinstance(inp, torch.Tensor)
    denormed_inp = denorm_img(inp)
    out = []
    for i in range(denormed_inp.shape[0]):
        out.append((denormed_inp[i].permute(1, 2, 0).contiguous().numpy() * 255).astype(np.uint8))
        
    return out



def show_img(img: np.ndarray):
    print(f'showing img with shape: {img.shape}')
    plt.tight_layout()
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def dprint(debug, *args):
    if debug == True:
        print(args)
    else:
        pass

def show_or_save_batch_img_tensor(img_tensor: torch.Tensor, 
                                  num_sample_per_row: int, 
                                  denorm: bool = True, 
                                  mode: str = 'show', 
                                  save_p: str = None):
    assert mode in ['show', 'save', 'all', 'return']
    if img_tensor.device != torch.device('cpu'):
        img_tensor = img_tensor.cpu()
    if denorm:
        img_tensor = torch.clip((img_tensor + 1.0) / 2.0, 0.0, 1.0)
    img = make_grid(img_tensor, nrow=num_sample_per_row)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    if mode == 'show':
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    if mode == 'save':
        assert save_p is not None
        img = Image.fromarray(img)
        img.save(save_p)
        print(f'saving sample img to {save_p}')
    if mode == 'all':
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        assert save_p is not None
        img = Image.fromarray(img)
        img.save(save_p)
        print(f'saving sample img to {save_p}')
    if mode == 'return':
        return img



def print_model_num_params_and_size(model):
    MB = 1024 * 1024
    cal_num_parameters = lambda module: sum([p.numel() for p in module.parameters() if p.requires_grad == True])
    num_param_to_MB = lambda num_parameters: num_parameters * 4  / MB
    total_num_params = cal_num_parameters(model) 
    print(f'model #params: {total_num_params / (10 ** 6)}M, fp32 model size: {num_param_to_MB(total_num_params)} MB') 
    