
import os
import numpy as np
import torch
from DreamDiffusion.code.dc_ldm.ldm_for_eeg import eLDM
from PIL import Image


class ModelServer:
    def __init__(self, device=torch.device('cpu'), pretrain_root='../data'):
        self.metafile = os.path.join(pretrain_root, "checkpoint-eeg-500.pth")
        self.device = device
        self.pretrain_root = pretrain_root
        self.eLDM_model = None

    def load_model(self, num_voxels):
        self.eLDM_model = eLDM(self.metafile, num_voxels, device=self.device, pretrain_root=self.pretrain_root)

    def infer(self, pth_file, num_samples, ddim_steps, HW=None, limit=None, state=None, output_path=None):
        eeg_signals = torch.load(pth_file)
        num_voxels = eeg_signals['num_voxels']  # Assuming the pth file contains num_voxels
        if self.eLDM_model is None:
            self.load_model(num_voxels)
        results, _ = self.eLDM_model.generate(eeg_signals, num_samples, ddim_steps, HW=HW, limit=limit, state=state, output_path=output_path)
        grid_imgs = Image.fromarray(results.astype(np.uint8))
        return grid_imgs
    
    
# Example usage:
# server = ModelServer(metafile='path_to_metafile')
# results = server.infer('path_to_inference_pth_file', num_samples=5, ddim_steps=250)