import os
import shutil
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import noisereduce as nr
from scipy.io import wavfile

# Function to remove folders containing "seal" in the Data_Processed directory
def remove_seal_folders(data_processed_dir, kill_seals):
    if kill_seals.lower() == 'n':
        return
    print(f"\n\nRemoving all seals from Folders!")
    for dir_name in os.listdir(data_processed_dir):
        if 'seal' in dir_name.lower():
            folder_path = os.path.join(data_processed_dir, dir_name)
            if os.path.isdir(folder_path):
                print(f"Removing folder: {folder_path}")
                shutil.rmtree(folder_path)


# Function to trim audio files in a directory to a maximum length
def trim_audio_files(directory, max_length):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                try:
                    audio = AudioSegment.from_wav(file_path)
                    if len(audio) > max_length * 1000:  # Convert to milliseconds
                        print(f"Trimming {file_path} to {max_length} seconds")
                        trimmed_audio = audio[:max_length * 1000]  # Convert to milliseconds
                        trimmed_audio.export(file_path, format="wav")
                except CouldntDecodeError:
                    print(f"Could not decode {file_path}. Skipping.")


# Function to limit the total duration of audio files in a directory
def limit_subfolder_length(directory, max_length):
    for root, dirs, files in os.walk(directory):
        total_length = 0
        files_to_remove = []
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                try:
                    audio = AudioSegment.from_wav(file_path)
                    total_length += len(audio)
                    if total_length > max_length * 1000:  # Convert to milliseconds
                        files_to_remove.append(file_path)
                except CouldntDecodeError:
                    print(f"Could not decode {file_path}. Skipping.")
        for file_path in files_to_remove:
            print(f"Removing {file_path} to limit subfolder length to {max_length} seconds")
            os.remove(file_path)


# Function to remove subfolders and their contents if total length doesn't reach a threshold
def remove_subfolders_below_threshold(directory, min_length):
    for root, dirs, files in os.walk(directory, topdown=False):
        total_length = 0
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                try:
                    audio = AudioSegment.from_wav(file_path)
                    total_length += len(audio)
                except CouldntDecodeError:
                    print(f"Could not decode {file_path}. Skipping.")
        if (total_length < min_length * 1000) & (total_length != 0):  # Convert to milliseconds
            print(
                f"Folder {root} has length of {total_length / 1000} seconds and does not meet threshold of {min_length}")
            shutil.rmtree(root)

def apply_noisegate(directory, gate_noise):
    if gate_noise.lower() == 'n':
        return
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                try:
                    print("Working on " + file_path)
                    rate, data = wavfile.read(file_path)
                    reduced_noise = nr.reduce_noise(y=data, sr=rate, time_mask_smooth_ms=427, n_std_thresh_stationary=0.75, stationary=True)
                    wavfile.write(file_path, rate, reduced_noise)
                except ValueError:
                    print(f"Could not decode {file_path}. Skipping.")


def main():
    data_dir = "Data"
    data_processed_dir = "Data_Processed"

    # Step 1: Copy the contents of the Data folder into the Data_Processed folder
    if os.path.exists(data_processed_dir):
        print("Removing contents of Data_Processed folder.")
        shutil.rmtree(data_processed_dir)

    print("Copying contents of Data folder to Data_Processed folder.")
    shutil.copytree(data_dir, data_processed_dir)

    # Operation 1: Remove folders with "seal" in the name
    kill_seals = str(input("Kill the seals (Y/N)? (default Y): ") or 'y')
    remove_seal_folders(data_processed_dir, kill_seals)

    # Operation 2: Trim audio files to a maximum length
    max_audio_length = float(input("Enter the maximum audio file length in seconds (default 10): ") or 10)
    trim_audio_files(data_processed_dir, max_audio_length)

    # Operation 3: Limit total subfolder length
    max_subfolder_length = float(input("Enter the maximum total sub-folder length in seconds (default 400): ") or 400)
    limit_subfolder_length(data_processed_dir, max_subfolder_length)

    # Operation 4: Remove subfolders if total length is below a threshold
    min_folder_length = float(input("Enter the minimum sub-folder length in seconds (default 60): ") or 60)
    remove_subfolders_below_threshold(data_processed_dir, min_folder_length)

    # Operation 5: Apply noisegate
    gate_noise = str(input("Apply noisegating (Y/N)? (default Y): ") or 'y')
    apply_noisegate(data_processed_dir, gate_noise)


if __name__ == "__main__":
    main()
