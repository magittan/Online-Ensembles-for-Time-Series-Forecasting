import numpy as np
import bisect

class OnlineExperts(object):

    def __init__(self,**kwargs):
        pass
    def _predict(self, expert_predictions):
        pass
    def _update(self, expert_predictions, actual_values):
        pass
    
class OnlineHedge(OnlineExperts):

    def __init__(self,n=10,T=10,a=1,loss_func=None):
        self.n = n
        self.T = T
        self.a = a
        self.weights = _set_uniform(n)
        self.time = 0
        self.loss_func = loss_func
        
    def _predict(self,expert_predictions):
        """
        Weights the expert predictions into a single prediction based on the weights that have been calculated by the
        hedge algorithm

        Args:
            expert predictions (np.array) (pred.float): np.array with the expert predictions

        Returns:
            a value for prediction based on the inputs of the experts and their respective weights.
        """
        
        choosen_expert = _choose_from_distribution(self.weights,sample_size=1)
        
        expert_array = np.zeros(self.n)
        expert_array[choosen_expert]=1
        
        return expert_array
    
    
    def _update(self, expert_predictions, actual_values):
        pass
        
    def _modify_weights(self,new_array):
        self.weights = self.weights * new_array
        self.weights /= np.sum(self.weights)
        
class OnlineHedgeDoubling(OnlineHedge):

    def __init__(self,n=10,T=10,a=1,loss_func=None):
        super().__init__(n=n,T=T,a=a,loss_func=loss_func)
        self.epsilon = _define_epsilon(n,T,a=a)
    
    def _update(self, expert_predictions, actual_values):
        """
        """
        assert expert_predictions.shape[1]==len(actual_values), "Time Dimension Matches"
        time_length = expert_predictions.shape[1]
        
        total_time = time_length+self.time
        
        a = int(np.floor(np.log2(total_time/self.T)))
        splits = [self.T*2**(i)-self.time for i in range(a)]
        # negative indices are ignored
        splits = list(filter(lambda x: x>=0,splits))
        partitions = np.split(np.arange(total_time-self.time),splits)
        
        for i in range(len(partitions)):
            self.time+=len(partitions[i])
            
#             print(partitions[i])
#             print(self.time)
            
            if self.time>self.T:
                self.T = 2*self.T
                self.epsilon = _define_epsilon(self.n,self.T,self.a)
                
            losses = np.array([self.loss_func(expert_predictions[:,part], actual_values[part]) for part in partitions[i]])
            f = lambda x: np.exp(-self.epsilon*x)
            self._modify_weights(np.prod(f(losses),0))
            
class OnlineHedgeIncrementalTime(OnlineHedge):

    def __init__(self,n=10,a=1,loss_func=None):
        super().__init__(n=n,T=None,a=a,loss_func=loss_func)
        
    def _update(self, expert_predictions, actual_values, loss_func = None):
        """
        """
        assert expert_predictions.shape[1]==len(actual_values), "Time Dimension Matches"
        time_length = expert_predictions.shape[1]
        
        for i in range(time_length):
            self.time+=1
            epsilon = _define_epsilon(self.n,self.time,self.a)
            losses = self.loss_func(expert_predictions[:,i], actual_values[i])
            f = lambda x: np.exp(-epsilon*x)
            self._modify_weights(f(losses))
            
class NormalHedge(OnlineHedge):
    
    # Will only work for losses between in interval of size 1

    def __init__(self,n=10,a=1,loss_func=None):
        super().__init__(n=n,T=None,a=a,loss_func=loss_func)
        self.R = np.zeros(n)
    
    def _update(self, expert_predictions, actual_values):
        
        assert expert_predictions.shape[1]==len(actual_values), "Time Dimension Matches"
        
        time_length = expert_predictions.shape[1]
        
        for i in range(time_length):
            loss_vector = np.array([self.loss_func(prediction,actual_values[i]) for prediction in expert_predictions[:,i]])
            average_loss = np.dot(self.weights,loss_vector)
           
            instant_regret = (average_loss - loss_vector)
            
            self.R += instant_regret
            
            self._update_weights()
    
    def _update_weights(self):
        # Calculating Normalizing Constant
        R_plus = np.array(list(map(lambda x: 0 if 0 > x else x , self.R)))
        
        low_c = (min(R_plus)**2)/2
        high_c = (max(R_plus)**2)/2
        pot = lambda c: np.mean(np.exp((R_plus**2)/(2*c)))-np.e
        
        c_t = bisection(low_c,high_c,pot)
        
        print(c_t)
        
        # Calculating Probabilities
        prob = lambda r, c_t: (r/c_t)*np.exp((r**2)/(2*c_t))
        
        self.weights = np.array([prob(r,c_t) for r in R_plus])
        self.weights /= np.sum(self.weights)
        
        print(self.weights)
        
class FollowTheLeader(object):
    
    def __init__(self,n=10, loss_func=None):
        self.losses = np.zeros(n)
        self.loss_func = loss_func
        
    def _predict(self,expert_predictions):
        
        choosen_expert = np.argmax(self.losses)
        expert_array = np.zeros(self.n)
        expert_array[choosen_expert]=1
        
        return expert_array
        
    def _update(self,expert_predictions, actual_values):
        assert expert_predictions.shape[1]==len(actual_values), "Time Dimension Matches"
        time_length = expert_predictions.shape[1]
        
        for i in range(time_length):
            self.losses+=np.array([self.loss_func(prediction, actual_values[i]) for prediction in expert_predictions[:,i]])
        
class AverageHedge(OnlineExperts):
    def __init__(self,n):
        self.n = n
    
    def _predict(self, expert_predictions):
        return np.ones(self.n)/self.n
        
def bisection(low,high,function,threshold=1e-8):
    left = low
    right = high
    
    if function(low)>0:
        left = high
        right = low
    
    while abs(left-right)>1e-8:
        mid = (left+right)/2
        if function(mid)>0:
            right=mid
        else:
            left=mid
            
    return (left+right)/2
            
def se(actual,expected):
    """
    Will return the squared error between the two arguments
    """
    return np.power(np.subtract(actual,expected),2)

def mse(actual,expected):
    """
    Will return the mean squared error between the two arguments
    """
    return np.mean(se(actual,expected))
            
def _choose_from_distribution(axis_weights,sample_size=1):
        """
        Parameters
        ----------
        axis_weights: np.array() 1-D Array

        Returns
        -------
        np.array() with indices chosen according to the probability distribution defined by the axis weights

        Functional Code
        ---------------
            weights = abs(np.random.randn(10))
            weights/=np.sum(weights)
            bins = np.cumsum(weights)
            selections = np.random.uniform(size=10)

        Test Code
        ---------
            a = choosing_with_respect_to_prob_dist([1,2,3],sample_size=10000)
            print("Should be around .16: {}".format(np.sum(a==0)/10000))
            print("Should be around .33: {}".format(np.sum(a==1)/10000))
            print("Should be around .50: {}".format(np.sum(a==2)/10000))
        """
        weights = axis_weights/np.sum(axis_weights)
        bins = np.cumsum(weights)

        selections = np.random.uniform(size=sample_size)
        indices = [bisect.bisect_left(bins,s) for s in selections]

        return np.array(indices)
    
def _set_uniform(n):
        return np.ones(n)/n
    
def _define_epsilon(n,T,a=1):
    """
    Calculates a factor that is used in determining loss in the hedge algorithm

    Args:
        n (int): number of experts present
        T (int): number of time steps taken
        a (float): value that we can use to scale our epsilon
    Return:
        epsilon (float): the theoretical episilon, but which can be customized by a
    """

    return np.sqrt(np.log(n)/T)*a