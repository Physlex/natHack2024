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

# update paths

config.root_path = "../data"
config.pretrain_mbm_path = os.path.join(config.root_path, "checkpoint-eeg-500.pth")
config.pretrain_gm_path = os.path.join(config.root_path, "gm_pretrain")
print(config.__dict__)

output_path = os.path.join(
    config.root_path,
    "results",
    "eval",
    "%s" % (datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

crop_pix = int(config.crop_ratio * config.img_size)
img_transform_train = transforms.Compose(
    [
        normalize,
        transforms.Resize((512, 512)),
        # random_crop(config.img_size-crop_pix, p=0.5),
        # transforms.Resize((256, 256)),
        channel_last,
    ]
)
img_transform_test = transforms.Compose(
    [normalize, transforms.Resize((512, 512)), channel_last]
)

splits_path = "../data/block_splits_by_image_single_fixed.pth"
# NOTE: change depending on the received data from API
dataset_train, dataset_test = create_EEG_dataset(
    eeg_signals_path="../data/processed_eeg_data_updated.pth",
    splits_path=splits_path,
    image_transform=[img_transform_train, img_transform_test],
    subject=4,
)
num_voxels = dataset_test.dataset.data_len

### DEBUG CODE: START
splits = torch.load("../data/block_splits_by_image_single_fixed.pth")
print(splits)

# Check the train/test splits
train_indices = splits["splits"][0]["train"]
test_indices = splits["splits"][0]["test"]
