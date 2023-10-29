import os
from pydub import AudioSegment
import pandas as pd
import matplotlib.pyplot as plt


# Function to calculate the duration of an audio file
def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    return audio.duration_seconds


# Function to iterate through folders and subfolders and collect data
def collect_audio_duration_data(root_folder):
    data = []
    filecount = 0
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(('.mp3', '.wav', '.ogg', '.flac', '.aac')):
                filecount = filecount + 1
                file_path = os.path.join(foldername, filename)
                duration = get_audio_duration(file_path)
                foldernameWrite = foldername.replace(root_folder, "")
                data.append((foldernameWrite, filename, duration))
    print(filecount)
    return data


# Main script
if __name__ == "__main__":
    root_folder = "Data_Processed"  # Replace with the path to your audio folder

    audio_data = collect_audio_duration_data(root_folder)

    # Create a DataFrame
    df = pd.DataFrame(audio_data, columns=["foldername", "filename", "file_duration"])

    print(df.head())

    # Calculate average duration for each folder
    avg_durations = df.groupby(["foldername", "filename"])["file_duration"].mean().unstack()

    # Plot the grouped bar chart
    plt.figure(figsize=(16, 16))
    avg_durations.plot(kind="bar", stacked=True, legend=False)
    plt.xlabel("Species of Whale")
    plt.ylabel("File Duration (seconds)")
    plt.title("Audio File Durations by Whale Type")
    # plt.xticks(rotation=90)
    plt.xticks([])
    plt.tight_layout()
    plt.savefig("audio_plot.png", dpi=300)
    plt.show()