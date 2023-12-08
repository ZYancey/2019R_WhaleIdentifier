import os 
from pydub import AudioSegment
# need to pip install ffmpeg-python

# convert the data_validation files to wav files 

def convert_wav_to_mp3(src_name, dst_name):
    sound = AudioSegment.from_mp3(src_name)
    sound.export(dst_name, format="wav")

