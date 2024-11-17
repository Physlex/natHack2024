import os, sys
import numpy as np
import torch
from einops import rearrange
from PIL import Image
import torchvision.transforms as transforms
from config import *
import wandb
import datetime
import argparse


from config import Config_Generative_Model
from dataset import create_EEG_dataset
from dc_ldm.ldm_for_eeg import eLDM

def to_image(img):
    if img.shape[-1] != 3:
        img = rearrange(img, 'c h w -> h w c')
    img = 255. * img
    return Image.fromarray(img.astype(np.uint8))

def channel_last(img):
    if img.shape[-1] == 3:
        return img
    return rearrange(img, 'c h w -> h w c')

def normalize(img):
    # Convert PIL Image to numpy array if necessary
    if isinstance(img, Image.Image):
        img = np.array(img)  # Convert to numpy.ndarray

    # Check if image has channels last
    if img.ndim == 3 and img.shape[-1] == 3:  # HWC format
        img = rearrange(img, 'h w c -> c h w')  # Convert to CHW

    # Convert to PyTorch tensor and normalize
    img = torch.tensor(img, dtype=torch.float32)
    img = img / 255.0  # Scale to [0, 1]
    img = img * 2.0 - 1.0  # Normalize to [-1, 1]
    return img

def wandb_init(config):
    wandb.init( project="dreamdiffusion",
                group='eval',
                anonymous="allow",
                config=config,
                reinit=True)

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img


if __name__ == '__main__':
    target='EEG'
    sd = torch.load('../data/checkpoint.pth', map_location='cpu')
    config = sd['config']
    # update paths
    config.root_path = '../data'
    config.pretrain_mbm_path = os.path.join(config.root_path, 'checkpoint-eeg-500.pth')
    config.pretrain_gm_path = os.path.join(config.root_path, 'gm_pretrain')
    print(config.__dict__)

    output_path = os.path.join(config.root_path, '../results', 'eval',  
                    '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    crop_pix = int(config.crop_ratio*config.img_size)
    img_transform_train = transforms.Compose([
        normalize,
        transforms.Resize((512, 512)),
        # random_crop(config.img_size-crop_pix, p=0.5),
        # transforms.Resize((256, 256)), 
        channel_last
    ])
    img_transform_test = transforms.Compose([
        normalize, transforms.Resize((512, 512)), 
        channel_last
    ])

    
    splits_path = "../data/block_splits_by_image_single_fixed.pth"
    # NOTE: change depending on the received data from API
    dataset_train, dataset_test = create_EEG_dataset(eeg_signals_path = '../data/processed_eeg_data_updated.pth', splits_path = splits_path, 
                image_transform=[img_transform_train, img_transform_test], subject = 4)
    num_voxels = dataset_test.dataset.data_len

    ### DEBUG CODE: START
    splits = torch.load("../data/block_splits_by_image_single_fixed.pth")
    print(splits)

    # Check the train/test splits
    train_indices = splits["splits"][0]["train"]
    test_indices = splits["splits"][0]["test"]

    print(f"Train indices: {train_indices}")
    print(f"Test indices: {test_indices}")
    ### DEBUG CODE: END

    # num_voxels = dataset_test.num_voxels
    print(len(dataset_test))
    # prepare pretrained mae 
    pretrain_mbm_metafile = torch.load(config.pretrain_mbm_path, map_location='cpu')
    # create generateive model
    generative_model = eLDM(pretrain_mbm_metafile, num_voxels,
                device=device, pretrain_root=config.pretrain_gm_path, logger=config.logger,
                ddim_steps=config.ddim_steps, global_pool=config.global_pool, use_time_cond=config.use_time_cond)
    # m, u = model.load_state_dict(pl_sd, strict=False)
    generative_model.model.load_state_dict(sd['model_state_dict'], strict=False)
    print('load ldm successfully')
    state = sd['state']
    os.makedirs(output_path, exist_ok=True)
    grid, _ = generative_model.generate(dataset_train, config.num_samples, 
                config.ddim_steps, config.HW, 2) # generate 2 instances
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    
    grid_imgs.save(os.path.join(output_path, f'./samples_train.png'))

    """
    grid, samples = generative_model.generate(dataset_test, config.num_samples, 
                config.ddim_steps, config.HW, limit=None, state=state, output_path = output_path) # generate 10 instances
    grid_imgs = Image.fromarray(grid.astype(np.uint8))


    grid_imgs.save(os.path.join(output_path, f'./samples_test.png'))
    """
