import os
import librosa
import numpy as np
import pandas as pd


# Function to extract MFCC features from an audio file
def extract_mfcc_features(file_path, n_mfcc=40):
    audio_data, sampling_rate = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=n_mfcc, n_mels=n_mfcc)
    return mfccs.T  # Transpose to have features in columns


# Function to extract additional features (Chroma, Spectral Contrast, Spectral Bandwidth, etc.)
def extract_additional_features(file_path, feature_types=[]):
    audio_data, sampling_rate = librosa.load(file_path)

    features = {}

    # Chroma features represent the distribution of energy across musical pitch classes.
    # Captures the harmonic content of the audio and tonal characteristics
    if 'chroma' in feature_types:
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sampling_rate)
        features['chroma'] = np.mean(chroma.T, axis=0)

    # Spectral contrast measures the difference in amplitude between peaks and valleys in the spectrum.
    # Highlights the distinction between harmonic and non-harmonic components.
    if 'spectral_contrast' in feature_types:
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sampling_rate)
        features['spectral_contrast'] = np.mean(spectral_contrast.T, axis=0)

    # Spectral bandwidth represents the width of the spectral band. Higher values may indicate a broader spread of
    # frequencies in the signal.
    if 'spectral_bandwidth' in feature_types:
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sampling_rate)
        features['spectral_bandwidth'] = np.mean(spectral_bandwidth.T, axis=0)

    # Spectral centroid indicates the "center of mass" of the spectrum, providing a measure of where the "center" of
    # the frequencies is. It can offer insights into the perceived brightness or tonal quality of the sound.
    if 'spectral_centroid' in feature_types:
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sampling_rate)
        features['spectral_centroid'] = np.mean(spectral_centroid)

    # Spectral rolloff is the frequency below which a certain percentage of the total spectral energy lies.
    # Gives an indication of the spread of higher frequencies in the signal.
    if 'spectral_rolloff' in feature_types:
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sampling_rate)
        features['spectral_rolloff'] = np.mean(spectral_rolloff)

    # Zero-crossing rate measures the rate at which the signal changes its sign. It is useful for capturing
    # characteristics related to the noisiness or percussiveness of the audio.
    if 'zero_crossing_rate' in feature_types:
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_data)
        features['zero_crossing_rate'] = np.mean(zero_crossing_rate)

    # RMS (Root Mean Square) Energy represents the square root of the mean squared values of the signal.
    # It is a measure of the signal's energy and can be indicative of the overall amplitude or loudness.
    if 'rmse' in feature_types:
        rmse = librosa.feature.rms(y=audio_data)
        features['rmse'] = np.mean(rmse)

    return features


def process_audio_files(directory, output_csv, n_mfcc=40, additional_features=[]):
    data = {'species': []}
    for i in range(n_mfcc):
        data[f'mfcc_{i}'] = []

    for feature_type in additional_features:
        data[feature_type] = []

    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                count = count + 1
                print(f"{count}/{len(files)} | Extracting Features from: {file_path}")
                subfolder_name = os.path.basename(root)

                # Extract MFCC features
                mfccs = extract_mfcc_features(file_path, n_mfcc)
                mfccs_scaled_features = np.mean(mfccs.T, axis=0)

                # Extract additional features
                additional_features_dict = extract_additional_features(file_path, additional_features)

                # Append data to dictionary
                data['species'].append(subfolder_name)
                for i in range(n_mfcc):
                    if i < len(mfccs_scaled_features):
                        data[f'mfcc_{i}'].append(mfccs_scaled_features[i])
                    else:
                        data[f'mfcc_{i}'].append(0)

                # Append additional features
                for feature_type in additional_features:
                    data[feature_type].append(np.mean(additional_features_dict[feature_type]))

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

def main():
    data_processed_dir = "Data_Processed"
    output_csv = "features.csv"
    n_mfcc = 40
    additional_features = ['chroma',
                           'spectral_contrast',
                           'spectral_bandwidth',
                           'spectral_centroid',
                           'spectral_rolloff',
                           'zero_crossing_rate',
                           'rmse'
                           ]

    process_audio_files(data_processed_dir, output_csv, n_mfcc, additional_features)
    print(f"Features saved to {output_csv}")


if __name__ == "__main__":
    main()
