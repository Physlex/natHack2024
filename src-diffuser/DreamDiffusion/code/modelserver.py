import os
import numpy as np
import torch
from einops import rearrange
from PIL import Image
import torchvision.transforms as transforms
import datetime
from config import Config_Generative_Model
from dataset import create_EEG_dataset
from dc_ldm.ldm_for_eeg import eLDM

class ModelServer:
    def __init__(self, device=torch.device('cpu'), pretrain_root='../data'):
        self.device = device
        self.pretrain_root = pretrain_root
        self.eLDM_model = None
        self.config = None

    def load_model(self, config_path='checkpoint.pth'):
        sd = torch.load(config_path, map_location='cpu')
        self.config = sd['config']
        self.config.root_path = self.pretrain_root
        self.config.pretrain_mbm_path = os.path.join(self.config.root_path, 'checkpoint-eeg-500.pth')
        self.config.pretrain_gm_path = os.path.join(self.config.root_path, 'gm_pretrain')
        pretrain_mbm_metafile = torch.load(self.config.pretrain_mbm_path, map_location='cpu')

        self.eLDM_model = eLDM(pretrain_mbm_metafile, num_voxels=440,
                               device=self.device, pretrain_root=self.config.pretrain_gm_path, logger=self.config.logger,
                               ddim_steps=self.config.ddim_steps, global_pool=self.config.global_pool, use_time_cond=self.config.use_time_cond)
        self.eLDM_model.model.load_state_dict(sd['model_state_dict'], strict=False)
        print('Model loaded successfully')

    def infer(self, pth_file, num_samples, ddim_steps, HW=None, limit=None, state=None, output_path=None):
        eeg_signals = torch.load(pth_file)
        
        splits_path = os.path.join(self.config.root_path, "block_splits_by_image_single_fixed.pth")
        dataset_train, dataset_test = create_EEG_dataset(
            eeg_signals_path=pth_file,
            splits_path=splits_path,
            image_transform=[self.get_img_transform_train(), self.get_img_transform_test()],
            subject=4
        )
        num_voxels = dataset_test.dataset.data_len

        if self.eLDM_model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        results, _ = self.eLDM_model.generate(eeg_signals, num_samples, ddim_steps, HW=HW, limit=limit, state=state, output_path=output_path)
        grid_imgs = Image.fromarray(results.astype(np.uint8))
        return grid_imgs

    def get_img_transform_train(self):
        crop_pix = int(self.config.crop_ratio * self.config.img_size)
        return transforms.Compose([
            self.normalize,
            transforms.Resize((512, 512)),
            self.channel_last
        ])

    def get_img_transform_test(self):
        return transforms.Compose([
            self.normalize,
            transforms.Resize((512, 512)),
            self.channel_last
        ])

    @staticmethod
    def normalize(img):
        if img.shape[-1] == 3:
            img = rearrange(img, 'h w c -> c h w')
        img = torch.tensor(img)
        img = img * 2.0 - 1.0  # to -1 ~ 1
        return img

    @staticmethod
    def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')

# Example usage:
# server = ModelServer(device=torch.device('cuda'))
# server.load_model(config_path='path_to_config_pth_file')
# results = server.infer('path_to_inference_pth_file', num_samples=5, ddim_steps=250)