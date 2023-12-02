import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# import noisereduce as nr
# from scipy.io import wavfile


def visualize_audio_features(file_path):
    # load data
    # rate, data = wavfile.read(file_path)
    # perform noise reduction
    # reduced_noise = nr.reduce_noise(y=data, sr=rate)
    # reduced_noise = nr.reduce_noise(y=data, sr=rate, time_mask_smooth_ms=427, n_std_thresh_stationary=0.75,
    #                                 stationary=True)
    # wavfile.write("mywav_reduced_noise.wav", rate, reduced_noise)

    # Load audio file
    # audio_data, sampling_rate = librosa.load("mywav_reduced_noise.wav")
    audio_data, sampling_rate = librosa.load(file_path)

    # Calculate features
    spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sampling_rate)
    chroma = librosa.feature.chroma_vqt(intervals='equal', y=audio_data, sr=sampling_rate)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sampling_rate)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sampling_rate)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sampling_rate)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sampling_rate)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_data)
    rmse = librosa.feature.rms(y=audio_data)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sampling_rate)

    # Static tempo estimation
    onset_env = librosa.onset.onset_strength(y=audio_data, sr=sampling_rate)
    tempo, utempo = librosa.beat.beat_track(onset_envelope=onset_env, sr=sampling_rate)

    mean_spectrogram = np.mean(spectrogram, axis=1)
    mean_chroma = np.mean(chroma, axis=1)
    mean_spectral_contrast = np.mean(spectral_contrast, axis=1)
    mean_spectral_bandwidth = np.mean(spectral_bandwidth, axis=1)
    mean_spectral_centroid = np.mean(spectral_centroid)
    mean_spectral_rolloff = np.mean(spectral_rolloff)
    mean_zero_crossing_rate = np.mean(zero_crossing_rate)
    mean_rmse = np.mean(rmse)
    mean_mfccs = np.mean(mfccs, axis=1)

    # Print mean values across time for the first frame
    print("Spectrogram (mean values):", mean_spectrogram)
    print("Chroma (mean values):", mean_chroma)
    print("Spectral Contrast (mean values):", mean_spectral_contrast)
    print("Spectral Bandwidth (mean values):", mean_spectral_bandwidth)
    print("Spectral Centroid (mean values):", mean_spectral_centroid)
    print("Spectral Rolloff (mean values):", mean_spectral_rolloff)
    print("Zero-Crossing Rate (mean values):", mean_zero_crossing_rate)
    print("RMSE (mean values):", mean_rmse)
    print("MFCCs (mean values):", mean_mfccs)

    # Plotting
    plt.figure(figsize=(12, 14))

    # Spectrogram
    plt.subplot(5, 2, 1)
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), x_axis='time', y_axis='mel', fmax=8000)
    plt.title('Spectrogram')

    # Chroma
    plt.subplot(5, 2, 2)
    librosa.display.specshow(chroma, y_axis='chroma')
    plt.title('Chroma')

    # Spectral Contrast
    plt.subplot(5, 2, 3)
    librosa.display.specshow(spectral_contrast, x_axis='time')
    plt.title('Spectral Contrast')

    # MFCC
    plt.subplot(5, 2, 4)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCCs')

    # Spectral Centroid
    plt.subplot(5, 2, 5)
    plt.plot(spectral_centroid.T)
    plt.title('Spectral Centroid')
    plt.xlabel('Time (Frames)')
    plt.ylim(0, 8192)
    plt.axhline(mean_spectral_centroid, color='r', linestyle='dashed', linewidth=2, label='Mean')
    plt.legend()

    # Spectral Rolloff
    plt.subplot(5, 2, 6)
    plt.plot(spectral_rolloff.T)
    plt.title('Spectral Rolloff')
    plt.xlabel('Time (Frames)')
    plt.ylim(0, 8192)
    plt.axhline(mean_spectral_rolloff, color='r', linestyle='dashed', linewidth=2, label='Mean')
    plt.legend()

    # Zero-Crossing Rate
    plt.subplot(5, 2, 7)
    plt.plot(zero_crossing_rate.T)
    plt.title('Zero-Crossing Rate')
    plt.xlabel('Time (Frames)')
    plt.ylim(0, 1)
    plt.axhline(mean_zero_crossing_rate, color='r', linestyle='dashed', linewidth=2, label='Mean')
    plt.legend()

    # RMSE
    plt.subplot(5, 2, 8)
    plt.plot(rmse.T)
    plt.title('Root Mean Square Energy (RMSE)')
    plt.xlabel('Time (Frames)')
    plt.axhline(mean_rmse, color='r', linestyle='dashed', linewidth=2, label='Mean')
    plt.legend()

    # Static Tempo Estimation
    hop_length = 512
    ac = librosa.autocorrelate(onset_env, max_size=2 * sampling_rate // hop_length)
    freqs = librosa.tempo_frequencies(len(ac), sr=sampling_rate, hop_length=hop_length)

    plt.subplot(5, 2, 9)
    plt.semilogx(freqs[1:], librosa.util.normalize(ac)[1:], label='Onset autocorrelation', base=2)
    plt.axvline(tempo, 0, 1, alpha=0.75, linestyle='--', color='r', label=f'Tempo: {tempo:.2f} BPM')
    plt.xlabel('Tempo (BPM)')
    plt.title('Static tempo estimation')
    plt.legend()


    plt.subplot(5, 2, 10)
    plt.imshow(np.mean(mfccs, axis=1).reshape(1, -1), cmap='coolwarm', aspect='auto',
               extent=[0, len(np.mean(mfccs, axis=1)), 0, 1])
    plt.title('Mean Values of MFCCs')
    plt.xlabel('MFCC Coefficient Index')
    plt.yticks([])  # Hide y-axis ticks
    plt.colorbar(orientation='vertical', pad=0.2)

    plt.tight_layout()
    plt.show()


file_pathA = "Data_Processed/sperm_whale/8301900C.wav"
file_pathB = "Data_Processed/shortfinned_pacific_pilot_whale/5702100K.wav"
# file_path = "PinkPanther.wav"
visualize_audio_features(file_pathA)
visualize_audio_features(file_pathB)
