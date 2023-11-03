import os
import librosa
import numpy as np
import pandas as pd


# Function to extract MFCC features from an audio file
def extract_mfcc_features(file_path, n_mfcc=40):
    audio_data, sampling_rate = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=n_mfcc, n_mels=n_mfcc)
    return mfccs.T  # Transpose to have features in columns


def process_audio_files(directory, output_csv, n_mfcc=40):
    data = {'species': []}
    for i in range(n_mfcc):
        data[f'mfcc_{i}'] = []
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                count = count + 1
                print(f"{count}/{len(files)} | Extracting MFCC Data from: {file_path}")
                subfolder_name = os.path.basename(root)
                mfccs = extract_mfcc_features(file_path, n_mfcc)
                mfccs_scaled_features = np.mean(mfccs.T, axis=0)
                # print(mfccs_scaled_features)
                data['species'].append(subfolder_name)
                for i in range(n_mfcc):
                    if i < len(mfccs_scaled_features):
                        data[f'mfcc_{i}'].append(mfccs_scaled_features[i])
                    else:
                        # Pad with zeros for missing features
                        data[f'mfcc_{i}'].append(0)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

def main():
    data_processed_dir = "Data_Processed"
    output_csv = "mfcc_features.csv"

    process_audio_files(data_processed_dir, output_csv)
    print(f"MFCC features saved to {output_csv}")


if __name__ == "__main__":
    main()
