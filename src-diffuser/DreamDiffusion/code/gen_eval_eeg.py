import os
import datetime
import torch
from torchvision import transforms
from some_module import create_EEG_dataset  # Replace 'some_module' with the actual module name

# update paths
config.root_path = '../data'
config.pretrain_mbm_path = os.path.join(config.root_path, 'checkpoint-eeg-500.pth')
config.pretrain_gm_path = os.path.join(config.root_path, 'gm_pretrain')
print(config.__dict__)

output_path = os.path.join(config.root_path, 'results', 'eval',  
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

class ModelServer:
    def __init__(self, metafile, device=torch.device('cpu'), pretrain_root='../data'):
        self.metafile = metafile
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
        results = self.eLDM_model.generate(eeg_signals, num_samples, ddim_steps, HW=HW, limit=limit, state=state, output_path=output_path)
        return results

# Example usage:
# server = ModelServer(metafile='path_to_metafile')
# results = server.infer('path_to_inference_pth_file', num_samples=5, ddim_steps=250)
