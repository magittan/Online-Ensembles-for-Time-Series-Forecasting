import numpy as np

class Expert(object):
    def __init__(self):
        pass
    def _predict(self,X):
        pass
    def _fit(self):
        pass
    
class CannedExpert(Expert):
    def __init__(self,outputs):
        self.time = 0
        self.outputs = outputs
        
    def _predict(self,X):
        output = np.nan
        try:
            output = self.outputs[self.time]
            self.time+=1
        except:
            print("Out of Values")
        return output
    
    def _reset(self):
        self.t=0
        
class EnsembleForecaster(object):
    
    def __init__(self, forecasters, ensemble_algorithm):
        self.ensemble_algorithm = ensemble_algorithm
        self.forecasters = forecasters
        
    def _predict(self,X):
        predictions = np.array([forecaster._predict(X) for forecaster in self.forecasters])
        output = self.ensemble_algorithm._predict(predictions)
        
        return output
    
    def _fit_experts(self,X,y):
        [forecaster._fit(X,y) for forecaster in self.forecasters]
        
    def _fit_ensemble(self,X,y):
        predictions = np.array([forecaster._predict(X) for forecaster in self.forecasters])
        actual_values = y
        
        self.ensemble_algorithm._update(predictions,actual_values)