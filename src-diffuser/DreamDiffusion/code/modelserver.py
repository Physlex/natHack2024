import os
import torch
from einops import rearrange
from PIL import Image
import torchvision.transforms as transforms
from dataset import create_EEG_dataset
from dc_ldm.ldm_for_eeg import eLDM
from config import Config_MBM_EEG  # Ensure this import is present

class ModelServer:
    def __init__(self, checkpoint_path, pretrain_root="../data", device=torch.device("cpu")):
        """
        Initialize ModelServer and load static models and configurations.
        """
        self.device = device
        self.pretrain_root = pretrain_root
        self.checkpoint_path = checkpoint_path
        self.eLDM_model = None
        self.num_voxels = None
        self.config = None

        self._load_static_models_and_config()

    def _load_static_models_and_config(self):
        """
        Load static models, configurations, and other resources required for inference.
        """
        print("Loading static models and configurations...")

        # Custom `torch.load` function to make Config_MBM_EEG available during deserialization
        def load_checkpoint_with_globals(filepath):
            # Add Config_MBM_EEG to the global namespace for deserialization
            from config import Config_MBM_EEG  # Ensure this matches the definition in your checkpoint
            return torch.load(filepath, map_location="cpu")

        # Load checkpoint
        sd = load_checkpoint_with_globals(self.checkpoint_path)
        self.config = sd["config"]

        # Set paths
        self.config.root_path = self.pretrain_root
        self.config.pretrain_mbm_path = os.path.join(self.config.root_path, "checkpoint-eeg-500.pth")
        self.config.pretrain_gm_path = os.path.join(self.config.root_path, "gm_pretrain")

        # Load pre-trained model
        pretrain_mbm_metafile = torch.load(self.config.pretrain_mbm_path, map_location="cpu")

        # Initialize eLDM model
        self.eLDM_model = eLDM(
            metafile=pretrain_mbm_metafile,
            num_voxels=None,  # Will be set dynamically during inference
            device=self.device,
            pretrain_root=self.config.pretrain_gm_path,
            logger=self.config.logger,
            ddim_steps=self.config.ddim_steps,
            global_pool=self.config.global_pool,
            use_time_cond=self.config.use_time_cond
        )
        self.eLDM_model.model.load_state_dict(sd["model_state_dict"], strict=False)
        print("Static models and configurations loaded successfully.")

    def infer(self, eeg_file, num_samples, ddim_steps, HW=None, limit=None, state=None, output_path=None):
        """
        Perform inference using the loaded model and a new EEG file.
        """
        if self.eLDM_model is None:
            raise RuntimeError("Static models are not loaded. Call _load_static_models_and_config() first.")

        # Prepare dataset dynamically for the current EEG file
        dataset_test = self._prepare_dataset_for_inference(eeg_file)

        # Run the inference
        print("Generating images from EEG signals...")
        results, _ = self.eLDM_model.generate(
            fmri_embedding=dataset_test,
            num_samples=num_samples,
            ddim_steps=ddim_steps,
            HW=HW,
            limit=limit,
            state=state,
            output_path=output_path
        )

        # Convert results to a grid image
        grid_imgs = Image.fromarray(results.astype(np.uint8))
        print("Inference complete.")
        return grid_imgs

    def _prepare_dataset_for_inference(self, eeg_file):
        """
        Prepare the EEG dataset dynamically for the given EEG file.
        """
        print(f"Preparing dataset for inference using EEG file: {eeg_file}")
        splits_path = os.path.join(self.config.root_path, "block_splits_by_image_single_fixed.pth")

        # Dynamically create a dataset using the EEG file
        _, dataset_test = create_EEG_dataset(
            eeg_signals_path=eeg_file,
            splits_path=splits_path,
            image_transform=[self.get_img_transform_train(), self.get_img_transform_test()],
            subject=4
        )

        # Update num_voxels based on the new dataset
        self.num_voxels = dataset_test.dataset.data_len
        self.eLDM_model.num_voxels = self.num_voxels  # Update model with the dynamic num_voxels

        return dataset_test

    def get_img_transform_train(self):
        """
        Returns the image transformation pipeline for training data.
        """
        return transforms.Compose([
            self.normalize,
            transforms.Resize((512, 512)),
            self.channel_last
        ])

    def get_img_transform_test(self):
        """
        Returns the image transformation pipeline for test data.
        """
        return transforms.Compose([
            self.normalize,
            transforms.Resize((512, 512)),
            self.channel_last
        ])

    @staticmethod
    def normalize(img):
        """
        Normalizes an image to the range [-1, 1].
        """
        if img.shape[-1] == 3:
            img = rearrange(img, "h w c -> c h w")
        img = torch.tensor(img)
        img = img * 2.0 - 1.0  # Normalize to [-1, 1]
        return img

    @staticmethod
    def channel_last(img):
        """
        Converts channel-first image format to channel-last format.
        """
        if img.shape[-1] == 3:
            return img
        return rearrange(img, "c h w -> h w c")


