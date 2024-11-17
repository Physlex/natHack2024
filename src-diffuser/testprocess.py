import torch
import pandas as pd
import numpy as np
from scipy.signal import butter, sosfilt, resample

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Design a bandpass filter."""
    sos = butter(order, [lowcut, highcut], analog=False, btype='band', fs=fs, output='sos')
    return sos

def apply_bandpass_filter(data, sos):
    """Apply bandpass filter to EEG data."""
    return sosfilt(sos, data, axis=-1)

def pad_to_length(data, target_length):
    """Pad or truncate EEG data to a fixed length."""
    if data.shape[-1] > target_length:
        return data[..., :target_length]  # Truncate
    elif data.shape[-1] < target_length:
        padding = np.zeros((data.shape[0], target_length - data.shape[-1]))
        return np.hstack((data, padding))  # Pad
    return data

def process_eeg_data(csv_file_path, output_path="processed_eeg_data_updated.pth", sampling_rate=128, target_length=512):
    # Read the CSV file
    eeg_data = pd.read_csv(csv_file_path).values

    # Define preprocessing parameters
    lowcut = 5  # 5 Hz
    highcut = 95  # 95 Hz
    original_fs = 125  # Original sampling rate (adjust based on your data)
    sos = butter_bandpass(lowcut, highcut, original_fs)

    # Process EEG data: filter, resample, and pad
    eeg_data_filtered = apply_bandpass_filter(eeg_data, sos)
    eeg_data_resampled = resample(eeg_data_filtered, int(sampling_rate * eeg_data_filtered.shape[-1] / original_fs), axis=-1)
    eeg_data_processed = pad_to_length(eeg_data_resampled, target_length)

    # Assuming fixed time window per sample (e.g., 512 time steps per sample)
    num_samples = eeg_data_processed.shape[0] // target_length
    eeg_data_reshaped = eeg_data_processed[:num_samples * target_length].reshape(num_samples, -1, target_length)

    # Function to pad EEG data to 128 channels
    def pad_to_128_channels(eeg_tensor, target_channels=128):
        current_channels, time_steps = eeg_tensor.shape
        if current_channels >= target_channels:
            return eeg_tensor[:target_channels, :]  # Truncate if too many channels
        padding = torch.zeros(target_channels - current_channels, time_steps)
        return torch.cat((eeg_tensor, padding), dim=0)

    # Convert to a list of padded tensors
    eeg_tensors = [pad_to_128_channels(torch.tensor(sample, dtype=torch.float32)) for sample in eeg_data_reshaped]

    # Generate metadata for each EEG sample
    data_dicts = []
    for i, eeg_tensor in enumerate(eeg_tensors):
        # Tokenize the EEG data (group every 4 time steps into a single token)
        tokens = eeg_tensor.unfold(dimension=1, size=4, step=4).mean(dim=-1)
        data_dicts.append({
            'eeg': tokens,  # The tokenized EEG data as a tensor
            'image': i,     # Assign a unique integer index as image ID
            'label': 0,     # Dummy label (set all to 0 for now)
            'subject': 4    # Set the subject ID to 4
        })

    # Dummy labels and images (replace with actual data if available)
    labels = [f'label_{i}' for i in range(len(data_dicts))]
    images = [f'image_{i}' for i in range(len(data_dicts))]

    # Construct the full dataset dictionary
    processed_data = {
        'dataset': data_dicts,
        'labels': labels,
        'images': images
    }

    # Save the processed data to the specified output path
    torch.save(processed_data, output_path)
    print(f"File saved to: {output_path}")

# Example usage
# process_eeg_data('path_to_your_csv_file.csv')
