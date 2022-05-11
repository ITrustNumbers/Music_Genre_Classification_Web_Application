#Libraries
import librosa
import numpy as np
import os
import pandas as pd
import numpy as np
import joblib

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
    cols = ['Tempo', 'Chroma_STFT_Mean', 'Chroma_STFT_Var', 'RMS_Mean', 'RMS_Var',
       'Spectral_Centroid_Mean', 'Spectral_Centroid_Var',
       'Spectral_Bandwidth_Mean', 'Spectral_Bandwidth_Var',
       'Spectral_Rolloff_Mean', 'Spectral_Rolloff_Var',
       'Zero_Crossing_Rate_Mean', 'Zero_Crossing_Rate_Var', 'Harmony_Mean',
       'Harmony_Var', 'Perceptr_Mean', 'Perceptr_Var', 'MFCC1_Mean',
       'MFCC1_Var', 'MFCC2_Mean', 'MFCC2_Var', 'MFCC3_Mean', 'MFCC3_Var',
       'MFCC4_Mean', 'MFCC4_Var', 'MFCC5_Mean', 'MFCC5_Var', 'MFCC6_Mean',
       'MFCC6_Var', 'MFCC7_Mean', 'MFCC7_Var', 'MFCC8_Mean', 'MFCC8_Var',
       'MFCC9_Mean', 'MFCC9_Var', 'MFCC10_Mean', 'MFCC10_Var', 'MFCC11_Mean',
       'MFCC11_Var', 'MFCC12_Mean', 'MFCC12_Var', 'MFCC13_Mean', 'MFCC13_Var',
       'MFCC14_Mean', 'MFCC14_Var', 'MFCC15_Mean', 'MFCC15_Var', 'MFCC16_Mean',
       'MFCC16_Var', 'MFCC17_Mean', 'MFCC17_Var', 'MFCC18_Mean', 'MFCC18_Var',
       'MFCC19_Mean', 'MFCC19_Var', 'MFCC20_Mean', 'MFCC20_Var']

    df = df[cols]

    #Scaling
    with open('scaler.save', 'rb') as f:
        scaler = joblib.load(f)

    df = pd.DataFrame(scaler.transform(df), columns = cols)
    return df.values[0]

if __name__ == '__main__':

    path = os.path.join('static','blues.00007.wav')
    print(extract_features(path))
    ch = input('')
