#Importing dependencies
from flask import Flask
from flask import render_template, request
import numpy as np
import joblib
import xgboost as xgb
import catboost as cb
import os

#Loading Models
#XGB
xgbc = xgb.XGBClassifier()
path = os.path.join('saved_models', 'XGBClassifier.json')
xgbc.load_model(path)

#CBC
cbc = cb.CatBoostClassifier()
path = os.path.join('saved_models', 'CBClassifier.json')
cbc.load_model(path)

#Random Forest
path = os.path.join('saved_models', 'RFClassifier.sav')
with open(path, 'rb') as f:
  rfc = joblib.load(f)

#Ensembled Class
from MyEnsembledClassifier import EnsembledModel
models = [cbc, xgbc, rfc]
weights = [0.35, 0.35, 0.30]
emc = EnsembledModel(models, weights)

#Labels
label = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco',
        4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}

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

    #Extracting Features


    model = request.form['model'].split()[-1][0]

    return render_template("Test.html", model=model)


if __name__ == "__main__":
    app.run(debug=True)
