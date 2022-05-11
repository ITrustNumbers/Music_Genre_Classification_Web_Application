#Importing dependencies
from flask import Flask
from flask import render_template, request
import numpy as np
import joblib
import xgboost as xgb
import catboost as cb
import os
import Preprocessor as prep
import matplotlib.pyplot as plt
import librosa
import librosa.display

#Labels
label = {0: 'Blues', 1: 'Classical', 2: 'Country', 3: 'Disco',
        4: 'Hiphop', 5: 'Jazz', 6: 'Metal', 7: 'Pop', 8: 'Reggae', 9: 'Rock'}

#Loading Models
#XGB
import saved_models.XGBCNative as xgbc
#CBC
cbc = cb.CatBoostClassifier()
path = os.path.join('saved_models', 'CBClassifier.json')
cbc.load_model(path)
#Random Forest
import saved_models.RFCNative as rfc
#Ensembled Class
from MyEnsembledClassifier import EnsembledModel
models = [cbc, xgbc, rfc]
weights = [0.35, 0.35, 0.30]
emc = EnsembledModel(models, weights)

#Function for prediction
def predict(audio_path,model):

    #Extracting Features
    features = prep.extract_features(audio_path)

    #Predicting
    if model == 'X':
        model_name = 'XGBoost'
        probs = xgbc.predict_proba(features)

    elif model == 'C':
        model_name = 'CatBoost'
        probs = cbc.predict_proba(features)

    elif model == 'F':
        model_name = 'Random Forest'
        probs = rfc.predict_proba(features)

    else:
        model_name = 'Ensembled Model'
        probs = emc.predict_proba(features)

    #Decoding Label
    pred_prob = '{:.2f}'.format(probs[np.argmax(probs)])
    pred = label[np.argmax(probs)]

    return pred, pred_prob, model_name

#Function to visualize MFCC
def get_mfcc(audio_path):

    audio_lb, sr = librosa.load(audio_path)
    chromagram = librosa.feature.chroma_stft(audio_lb, sr=sr)
    print('Chromogram shape:', chromagram.shape, '\n')

    plt.figure(figsize=(16, 6))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm');
    #saving
    img_path = os.path.join('static', 'Chroma_Frequencies.png')
    plt.savefig(img_path, bbox_inches='tight', dpi=200)

    return img_path

#Application Framework
app = Flask(__name__)

@app.route("/", methods=['GET'])
def index_page():
    return  render_template("index.html")

@app.route('/predict', methods=['POST'])
def prediction():

    #Getting user Audio
    audio = request.files['audiofile']
    audio_path = 'static/' + audio.filename
    audio.save(audio_path)

    #getting mfcc visualization
    img_path = get_mfcc(audio_path)

    #Checking User Model Selection
    model = request.form['model'].split()[-1][0]

    #Getting Prediction
    pred, pred_prob, model_name = predict(audio_path, model)

    return render_template("prediction.html", pred=pred, pred_prob=pred_prob,
                            model_name=model_name, img_path=img_path)


if __name__ == "__main__":
    app.run()
