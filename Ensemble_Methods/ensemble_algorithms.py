from Expert_Prediction.online_experts import *

from scipy.optimize import nnls
import numpy as np

HEDGE_EXPERTS=(OnlineHedge,OnlineHedgeDoubling,OnlineHedgeIncremental)

class EnsembleAlgorithms(object):

    def __init__(self,n,loss_func=None,**kwargs):
        self.n = n
        self.weights = np.ones(n)/n
        self.loss_func = loss_func
        
    def _predict(self,expert_predictions):
        prediction = np.dot(self.weights,expert_predictions)
        return prediction
                      
    def _update(self, expert_predictions, actual_values):
        pass
    
class HedgeExpertEnsemble(EnsembleAlgorithms):
    
    def __init__(self,n,T,online_expert,loss_func=None):
        super().__init__(n=n,loss_func=loss_func)
        
        # Check if Online Expert is a Hedge Algorithm
        if type(online_expert) not in HEDGE_EXPERTS:
            raise ValueError("Online Expert is not a Online Hedge Algorithm")
        
        # Resident Online Expert
        self.online_expert = online_expert
    
    def _update(self,expert_predictions,actual_values):
        self.online_expert._update(expert_predictions,actual_values)
        self.weights = self.online_expert._predict(expert_predictions)
    
class NNLSEnsemble(EnsembleAlgorithms):
    
    def __init__(self,n,loss_func=None):
        self.n = n
        self.weights = np.ones(n)/n
        self.total_expert_predictions = np.empty((10,0))
        self.total_actual_values = np.empty((0))
        
    def _update(self, expert_predictions, actual_values):
        self.total_expert_predictions = np.concatenate((self.total_expert_predictions,expert_predictions),axis=1)
        self.total_actual_values = np.concatenate((self.total_actual_values,actual_values))
        
        weights, loss = nnls(self.total_expert_predictions.T, self.total_actual_values)
        self.weights = weights
            
        
class DiscrepancyEnsemble(EnsembleAlgorithms):
    
    def __init__(self, n, online_expert,loss_func=None):
        super().__init__(n=n,loss_func=loss_func)
        self.hypotheses = []  
        self.time = 0
    
    def _update(self, expert_predictions,actual_values):
        assert expert_predictions.shape[1]==len(actual_values), "Time Dimension Matches"
        time_length = expert_predictions.shape[1]
        
        #Could run into an error here
        updated_expert_predictions = expert_predictions[:,-1]
        trunc_expert_predictions = expert_predictions[:,:-1]
        trunc_actual_values = actual_values[:-1]
        
        for i in range(time_length):
            expert_array = online_expert._predict(expert_predictions[:,i])
            self.hypotheses.append(expert_array)
            
            prediction = np.dot(expert_array,expert_predictions[:,i])
            self.losses.append(self.loss_func(prediction,actual_values[i]))
            
            online_expert._update(expert_predictions[:,i][:,np.newaxis],actual_values[i,np.newaxis])
        
        weighted_loss = lambda x: np.dot(x, np.array(self.losses))