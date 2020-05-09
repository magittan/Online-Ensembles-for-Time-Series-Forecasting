from expert_prediction.online_experts import *
from projective_gradient_descent import gd
import discrepancy as disc
from scipy.optimize import nnls
import numpy as np

HEDGE_EXPERTS=(OnlineHedgeDoubling,OnlineHedgeIncremental,NormalHedge)

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
        online_expert.n=n
        online_expert.T=T
        online_expert.loss_func=loss_func
        
        self.online_expert = online_expert
    
    def _update(self,expert_predictions,actual_values):
        self.online_expert._update(expert_predictions,actual_values)
        self.weights = self.online_expert._predict(expert_predictions)
    
class NNLSEnsemble(EnsembleAlgorithms):
    
    def __init__(self,n,loss_func=None):
        super().__init__(n=n,loss_func=loss_func)
        
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
        
        self.total_expert_predictions = np.empty((10,0))
        self.total_actual_values = np.empty((0))
        
        self.online_expert = online_expert
        self.total_online_predictions = []
    
    def _update(self, expert_predictions, actual_values):
        assert expert_predictions.shape[1]==len(actual_values), "Time Dimension Matches"
        time_length = expert_predictions.shape[1]
        
        # Add New Data
        self.total_expert_predictions = np.concatenate((self.total_expert_predictions,expert_predictions),axis=1)
        self.total_actual_values = np.concatenate((self.total_actual_values,actual_values))

        # Run the Online Learner
        for i in range(time_length):
            expert_array = self.online_expert._predict(expert_predictions[:,i])
            self.total_online_predictions.append(expert_array)
            self.online_expert._update(expert_predictions[:,i][:,np.newaxis],actual_values[i,np.newaxis])
            
        # Adapting the predictions
        online_predictions = np.array(self.total_online_predictions).T
        
        # Create objective function
        obj_func = lambda q: disc.objective_function(q, online_predictions, self.total_expert_predictions,self.total_actual_values,loss_func=se)
        
        # total examples
        total_time = len(self.total_actual_values)
        
        # Run and randomize
        results = np.array([gd.projective_simplex_gradient_descent_2(np.ones(total_time-1)/(total_time-1),obj_func,iterations=200,eta=5,epsilon=0.01,noise=0.001) for i in range(15)])
        
        self.weights = np.dot(online_predictions[:,:-1],results.T).mean(axis=1)
        
        