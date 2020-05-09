import numpy as np

class Expert(object):
    
    def __init__(self):
        pass
    def _predict(self,X):
        pass
    def _fit(self,X,y):
        pass
    
class CannedExpert(Expert):
    
    def __init__(self,outputs):
        self.time = 0
        self.outputs = outputs
        
    def _predict(self,X):
        length = X.shape[0]
        
        output = np.array([])
        try:
            output = self.outputs[0:length]
        except:
            print("Out of Values")
            
        return output
    
    def _fit(self,X,y):
        self.outputs = y
        
class LSTMExpert(Expert):
    
    def __init__(self,outputs):
        self.time = 0
        self.outputs = outputs
        
    def _predict(self,X):
        length = X.shape[0]
        
        output = np.array([])
        try:
            output = self.outputs[0:length]
        except:
            print("Out of Values")
            
        return output
    
    def _fit(self,X,y):
        self.time = 0
    
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
        
    def _online_predict(self,X,y):
        online_predictions = []
        expert_predictions = np.array([forecaster._predict(X) for forecaster in self.forecasters])
        
        length = X.shape[0]
        for i in range(length):
            online_predictions.append(self.ensemble_algorithm._predict(expert_predictions[:,i]))
            self.ensemble_algorithm._update(expert_predictions[:,i][:,np.newaxis],y[i,np.newaxis])
                                      
        return np.array(online_predictions)
    
def calculate_losses(predicted,actual,loss_func = None):
    assert len(predicted)==len(actual)
    total_length = len(predicted)
    
    output = [loss_func(predicted[i],actual[i]) for i in range(total_length)]
    return output

def regret(predictions,expert_predictions,actual_values,loss_func=None):
    min_regret = loss_func(expert_predictions,actual_values).sum(axis=1)
    regret = loss_func(predictions,actual_values).sum()-np.min(min_regret)
    return regret