#Libraries
import librosa
import numpy as np
import os
import pandas as pd
import numpy as np

def extract_features(audio_path):

    #load Audio
    audio_lb, sr = librosa.load(audio_path)

    #Feature Names
    features = ['Chroma_STFT', 'RMS', 'Spectral_Centroid', 'Spectral_Bandwidth', 'Spectral_Rolloff',
            'Zero_Crossing_Rate', 'Harmony', 'Perceptr'] + ['MFCC' + str(i) for i in range(1,21)]

    #Creating an empty structure inside dictionary
    feature_dic = {'Tempo' : []}
    for feature in features:
      feature_dic[feature + '_Mean'] = []
      feature_dic[feature + '_Var'] = []

    #Extraction
    #Chorma_STFT
    chroma_stft = librosa.feature.chroma_stft(audio_lb, sr=sr)
    feature_dic['Chroma_STFT_Mean'].append(np.mean(chroma_stft))
    feature_dic['Chroma_STFT_Var'].append(np.var(chroma_stft))

    #Root Mean Square
    rms = librosa.feature.rms(audio_lb)
    feature_dic['RMS_Mean'].append(np.mean(rms))
    feature_dic['RMS_Var'].append(np.var(rms))

    #Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(audio_lb, sr=sr)[0]
    feature_dic['Spectral_Centroid_Mean'].append(np.mean(spec_cent))
    feature_dic['Spectral_Centroid_Var'].append(np.var(spec_cent))

    #Spectral Bandwidth
    spec_band = librosa.feature.spectral_bandwidth(audio_lb, sr=sr)[0]
    feature_dic['Spectral_Bandwidth_Mean'].append(np.mean(spec_band))
    feature_dic['Spectral_Bandwidth_Var'].append(np.var(spec_band))

    #Spectral RollOff
    spec_roll = librosa.feature.spectral_rolloff(audio_lb, sr=sr)[0]
    feature_dic['Spectral_Rolloff_Mean'].append(np.mean(spec_roll))
    feature_dic['Spectral_Rolloff_Var'].append(np.var(spec_roll))

    #Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio_lb)
    feature_dic['Zero_Crossing_Rate_Mean'].append(np.mean(zcr))
    feature_dic['Zero_Crossing_Rate_Var'].append(np.var(zcr))

    #Harmony and Perceptr
    harmony, perceptr = librosa.effects.hpss(audio_lb)
    feature_dic['Harmony_Mean'].append(np.mean(harmony))
    feature_dic['Harmony_Var'].append(np.var(harmony))
    feature_dic['Perceptr_Mean'].append(np.mean(perceptr))
    feature_dic['Perceptr_Var'].append(np.var(perceptr))

    #Tempo
    tempo, _ = librosa.beat.beat_track(audio_lb, sr=sr)
    feature_dic['Tempo'].append(tempo)

    #Mel-Frequency Cepstrum Coefficients(20)
    mfccs = librosa.feature.mfcc(audio_lb, sr=sr)
    for k, mfcc in enumerate(mfccs):
      feature_name = 'MFCC' + str(k+1)
      feature_dic[feature_name + '_Mean'].append(np.mean(mfcc))
      feature_dic[feature_name + '_Var'].append(np.var(mfcc))

    #Creating datafram
    df = pd.DataFrame(feature_dic)

    return df.head(5)

if __name__ == '__main__':

    extract_features(os.path.join('static','blues.00007.wav'))
