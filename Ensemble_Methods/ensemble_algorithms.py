from scipy.optimize import nnls
import numpy as np

# ONLINE_EXPERTS=[]

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
    
class OnlineExpertEnsemble(EnsembleAlgorithms):
    
    def __init__(self,n,T,online_expert,loss_func=None):
        super().__init__(n=n,loss_func=loss_func)
        
        #Resident Online Expert
        self.online_expert = online_expert
    
    def _update(self,expert_predictions,actual_values):
        self.online_expert._update(expert_predictions,actual_values)
        self.weights = self.online_expert._predict(expert_predictions)
    
class NNLSEnsemble(EnsembleAlgorithms):
    
    def __init__(self,n,loss_func=None):
        self.n = n
        self.weights = np.ones(n)/n
        
    def _update(self, expert_predictions, actual_values):
        weights, loss = nnls(expert_predictions.T, actual_values)
        self.weights = weights
        
class DiscrepancyEnsemble(EnsembleAlgorithms):
    
    def __init__(self, n, online_expert,loss_func=None):
        super().__init__(n=n,loss_func=loss_func)
        self.hypotheses = []  
        self.time = 0
        
        self.losses = []
    
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
        
        
    