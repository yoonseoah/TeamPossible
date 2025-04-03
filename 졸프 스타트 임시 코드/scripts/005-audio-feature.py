import os
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from tqdm import tqdm

# ==========================
# AENet Audio Feature Extractor
# ==========================
class AENetFeatureExtractor:
    def __init__(self, model, sample_rate=16000, target_length=256):
        self.model = model
        self.sample_rate = sample_rate
        self.target_length = target_length

    def extract_features(self, waveform):
        with torch.no_grad():
            features = self.model(waveform.unsqueeze(0))  # Input [1, C, T]
            features = features.squeeze(0).numpy()  # Output [N, 1024]

            # Adjust features to target length
            if features.shape[0] < self.target_length:
                padding = np.zeros((self.target_length - features.shape[0], features.shape[1]))
                features = np.vstack((features, padding))
            elif features.shape[0] > self.target_length:
                features = features[:self.target_length, :]

            return features

# ==========================
# Utility Functions
# ==========================
def is_silent_audio(waveform, threshold=1e-4):
    """Check if the audio is silent based on RMS energy."""
    rms_energy = torch.sqrt(torch.mean(waveform ** 2))
    return rms_energy.item() < threshold

def extract_audio_features(audio_path, output_path, extractor, no_sound_list, class_name):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample to 16kHz if necessary
        if sample_rate != extractor.sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=extractor.sample_rate)
            waveform = resampler(waveform)

        # Check for silent audio
        if is_silent_audio(waveform):
            print(f"Silent audio detected: {audio_path}")
            no_sound_list.append({"class": class_name, "file": os.path.basename(audio_path)})
            return

        # Extract features using AENet
        features = extractor.extract_features(waveform)
        np.save(output_path, features)
        print(f"Extracted features for: {audio_path}")

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

# ==========================
# Main Processing Function
# ==========================
def process_audio_files(audio_root, output_root, no_sound_csv, extractor):
    no_sound_files = []  # Track silent audio files

    for class_name in os.listdir(audio_root):
        class_audio_path = os.path.join(audio_root, class_name)
        class_output_path = os.path.join(output_root, class_name)

        if not os.path.isdir(class_audio_path):
            print(f"Skipping non-directory: {class_audio_path}")
            continue

        os.makedirs(class_output_path, exist_ok=True)

        audio_files = [f for f in os.listdir(class_audio_path) if f.endswith(".wav")]

        with ThreadPoolExecutor() as executor:
            futures = []
            for audio_file in tqdm(audio_files, desc=f"Processing {class_name}"):
                audio_path = os.path.join(class_audio_path, audio_file)
                output_file = os.path.join(class_output_path, audio_file.replace(".wav", ".npy"))
                futures.append(executor.submit(extract_audio_features, audio_path, output_file, extractor, no_sound_files, class_name))

            # Wait for all threads to complete
            for future in futures:
                future.result()

    # Save no-sound audio file list to CSV
    if no_sound_files:
        df = pd.DataFrame(no_sound_files)
        df.to_csv(no_sound_csv, index=False)
        print(f"Saved no-sound audio list to: {no_sound_csv}")
    else:
        print("All audio files have sound. No CSV generated.")

# ==========================
# Script Execution
# ==========================
if __name__ == "__main__":
    # Input and Output Paths
    audio_root = r"D:/aud-processed"
    output_root = r"D:/aud-features"
    no_sound_csv = r"Audio-Feature-Generation/results/no_sound_audio.csv"

    # AENet Model Initialization (Placeholder for actual model loading)
    # Replace `AENet()` with your actual AENet model instance
    class AENet(torch.nn.Module):
        def forward(self, x):
            return torch.rand((x.shape[-1] // 16000, 1024))  # Simulating output of [N, 1024]

    aenet_model = AENet().eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    extractor = AENetFeatureExtractor(aenet_model, target_length=256)

    # Process Audio Files
    print("Starting audio feature extraction...")
    process_audio_files(audio_root, output_root, no_sound_csv, extractor)
    print("Audio feature extraction completed!")
