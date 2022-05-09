#Libraries
import librosa as lb
from pydub import AudioSegment
import pandas as pd
import numpy as np
import os

#Function for Loading and getting audio splits(<=n_max) of audio file of 's' seconds

def load_and_split(audio_path, n_max=5, s=3):
    '''
    default n_max is set at n_max = 5, which means that if possible  the audio track
    will be splitted into 5 parts each of 's' seconds and by default s = 3 secs

    Both n_max and s are development variable only, i.e they would be changed during the
    model development phase for finding an optimum value. For production/Deployment both
    will remain constant. (Not controllable through application front end)
    '''

    #Reading Audio File
    audio_format = audio_path[-3:] #Identifying the format
    audio_fn = os.path.split(audio_path)[-1] #Audio Filename
    try:
        audio_pd = AudioSegment.from_file(audio_path, audio_format) #Loading Audio
    except:
        print(f'{audio_format} is not recognizable or is broken, try using any file that ffmpeg supports.')
        return None

    #Calculating possible splits
    dur_sec = audio_pd.duration_seconds
    poss_splits = int(dur_sec//s)

    #Splitting Audio
    split_audios = []
    start = 0
    if poss_splits < n_max: n_max = poss_splits
    for i in range(n_max):
        end = start + s*1000
        split_audio = audio_pd[start:end]
        split_audios.append(split_audio)
        start = end

    return split_audios

if __name__ == '__main__':

    split_audios = load_and_split(os.path.join('Training_Audio_Data', 'GTZAN_Dataset', 'blues', 'blues.00035.wav'))
    for i, audio in enumerate(split_audios):
        print(audio)
        print('\n')

        audio.export(os.path.join('Test_Audio', str(i) + '.wav'), format='wav')
    end_pause = input('\n')
