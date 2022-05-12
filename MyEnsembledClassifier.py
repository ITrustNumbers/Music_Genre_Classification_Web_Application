import numpy as np

class EnsembledModel:

  #Initiation
  def __init__(self, models, weights):
    self.__models = models
    self.__weights = weights

  #Function to Calculate sepreat prediction probabilities
  def __weighted_probs(self, X):
    weighted_probs = []
    for model,weight in zip(self.__models, self.__weights):
      if weight == 0.3:
        weighted_probs.append(model.predict_proba([X])[0] * weight)
      else:
        weighted_probs.append(model.predict_proba(X) * weight)
    return weighted_probs
  
  #Funstion to calculate ensembler prediction probability
  def predict_proba(self, X):
    predict_probs = sum(self.__weighted_probs(X))
    return np.array(predict_probs)

  #Function to calculate prediction of the ensembler
  def predict(self, X):
    preds = []
    predict_probs = sum(self.__weighted_probs(X))
    for probs in predict_probs:
      preds.append(np.argmax(probs))
    return np.array(preds)