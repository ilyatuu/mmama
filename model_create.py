import joblib
import numpy as np
import xgboost
from sklearn.preprocessing import StandardScaler

class BasePredictor(object):
    def predict(self, X):
        raise NotImplementedError
    
class Predictor(BasePredictor):
    def __init__(self):
        self.model = joblib.load("model/mmama_model.joblib")
        self.scaler = joblib.load("model/mmama_scaler.joblib")

    def predict(self, X):
        # prediction
        data_to_pass = np.asarray(X)

        pred_data = self.scaler.transform(X = data_to_pass)

        result = self.model.predict(pred_data)

        # unpack results
        prediction = []

        for i in result:
            if i == 0:
                prediction.append("GH_Negative")
            else:
                prediction.append("GH_Positive")

        GH_prediction = np.asarray(prediction)

        return GH_prediction

joblib.dump(Predictor, "model/mmama_predictor.sav")