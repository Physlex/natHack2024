import mne
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy.fft import fft, ifft
from scipy.interpolate import CubicSpline
import glob
from moviepy.editor import VideoFileClip, concatenate_videoclips
import time
import jwt

def concatenate_videos(video_files_path):

    video_file_list = glob.glob(f"{video_files_path}/*.mp4")

    loaded_video_list = []

    for video in video_file_list:
        loaded_video_list.append(VideoFileClip(video))

    final_clip = concatenate_videoclips(loaded_video_list)

    final_clip.write_videofile(f"test.webm")

def interpolateData(data_path, interpolation_type='fourier', num_points=500):
    
    data_path = "ds004745/sub-001/eeg/sub-001_task-unnamed_eeg.set"
    raw = mne.io.read_raw_eeglab(data_path)
    data = raw.get_data()
    
    # Cutting data just to see how it works for 100 points
    # data = [i[:100] for i in data]
    channels = len(data)
    N = len(data[0]) # number of samples
    
    
    print(f'channels: {channels} N: {N} ')
    padded_data = []
    if interpolation_type == 'fourier':
        for channel in data:
            # # Apply Hanning window function to rid of artifacts
            # (this doesn't work, makes signal taper to 0 at edges)
            # window = np.hanning(len(channel))
            # # turn channel into windowed channel
            # channel = channel * window
            
            # Perform FFT
            freq_data = fft(channel)
            new_length = N + num_points

            # Zero-padding in the frequency domain to double the number of points
            padded_freq_data = np.concatenate([freq_data[:N//2], 
                                               np.zeros(num_points), freq_data[N//2:]])
            # Perform inverse FFT to get the upsampled time-domain signal
            upsampled_signal = ifft(padded_freq_data).real
            # Scale the upsampled signal (optional)
            scaling = new_length / N
            upsampled_signal *= scaling
            padded_data.append(upsampled_signal)
    
    elif interpolation_type == 'cubic':
        
        for channel in data:
            # Create parametric curve
            t_path = np.linspace(0, 1, channel.size)

            # Use `channel` as the position vector (no need for a 2D array here)
            # Create the cubic spline interpolator
            cs = CubicSpline(t_path, channel)

            # Generate new x-values for interpolation
            x_new = np.linspace(0, 1, num_points)

            # Interpolate y-values using the spline
            y_new = cs(x_new)
            
            padded_data.append(y_new)

        
    padded_data = np.array(padded_data)
    print(f'Original points #: {N} New point #: {len(padded_data[0])}')
    # TESTING
    # Re-scale graphs to show proper overlapping
    time_original = np.linspace(0, 1, len(data[0]))
    time_upsampled = np.linspace(0, 1, len(padded_data[0]))     
    
    # Mean Squared Error 
    # MSE = np.square(np.subtract(Y_true,Y_pred)).mean() 
    
    plt.plot(time_upsampled, padded_data[0], 'o', label='padded data', marker='.')
    plt.plot(time_original, data[0], 'o', label='data', marker='.')
    plt.show()
    
def generate_cling_video(image_path):
    # does nothing rn, useless, might delete
    authorization = encode_jwt_token(ak, sk)
    print(authorization) # Printing the generated API_TOKEN

def encode_jwt_token(ak, sk):
    headers = {
        "alg": "HS256",
        "typ": "JWT"
    }
    payload = {
        "iss": ak,
        "exp": int(time.time()) + 1800, # The valid time, in this example, represents the current time+1800s(30min)
        "nbf": int(time.time()) - 5 # The time when it starts to take effect, in this example, represents the current time minus 5s
    }
    token = jwt.encode(payload, sk, headers=headers)
    return token


    

def main():
    # asyncio.run(start_srv())
    interpolation_type = 'cubic'
    # interpolateData(interpolation_type, 500)
    concatenate_videos('./videos')

if __name__ == "__main__":
    
    main()