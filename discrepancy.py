from projective_gradient_descent import gd
from expert_prediction.online_experts import *
import numpy as np

def time_dependent_loss(hypotheses_seq, expert_predictions, actual_values,loss_func=None):
    predictions = (hypotheses_seq * expert_predictions).sum(axis=0)
    losses_t = loss_func(predictions,actual_values)
    
    return losses_t

# If we are considering choosing hypotheses individually
def discrepancy_simple(q, hypotheses_seq, expert_predictions, actual_values, updated_expert_predictions, loss_func=None):
    tdl = time_dependent_loss(hypotheses_seq, expert_predictions, actual_values,loss_func=se)
    
    set_of_hypothesis_losses = np.array([loss_func(np.dot(updated_expert_predictions,hypotheses_seq),h) for h in updated_expert_predictions])
    abs_total_losses = abs(np.dot(set_of_hypothesis_losses,q)-np.dot(tdl,q))
    sup_abs_total_losses = np.max(abs_total_losses)
    
    return sup_abs_total_losses


def discrepancy_optimize(q, hypotheses_seq, expert_predictions, actual_values, updated_expert_predictions, loss_func=None):
    tdl = time_dependent_loss(hypotheses_seq, expert_predictions, actual_values,loss_func=se)
    
    n = hypotheses_seq.shape[0]
    q_prime = np.ones(n)/n
    
    def inner_obj_func(x):
        set_of_hypothesis_losses = loss_func(np.dot(updated_expert_predictions,hypotheses_seq),np.dot(updated_expert_predictions,x))
        abs_total_losses = abs(np.dot(set_of_hypothesis_losses,q)-np.dot(tdl,q))
        return -abs_total_losses
        
    q_final = gd.projective_simplex_gradient_descent_2(q_prime,inner_obj_func,iterations=20,eta=0.2,epsilon=0.1)
    output = -inner_obj_func(q_final)
    
    return output

def discrepancy_two_stage_q(q, expert_predictions, actual_values, loss_func=None):
    n = len(q)
    uniform = np.ones(n)/n
    
    n_prime = expert_predictions.shape[0]
    w = np.ones(n_prime)/n_prime
    
    def inner_obj_func(w):
        predictions = np.dot(w,expert_predictions)
        losses_t = loss_func(predictions,actual_values)
        abs_total_losses = np.dot(uniform-q,losses_t)
        return -abs_total_losses
        
    w_final = gd.projective_simplex_gradient_descent_2(w,inner_obj_func,iterations=100,eta=0.1)
    output = -inner_obj_func(w_final)
    
    return output

def objective_function(q, hypotheses_seq, expert_predictions,actual_values,loss_func=se):
    obj_hypotheses_seq = hypotheses_seq[:,:-1]
    obj_expert_predictions = expert_predictions[:,:-1]
    obj_actual_values = actual_values[:-1]
    
    update_expert_predictions = expert_predictions[:,-1]
    
    tdl = time_dependent_loss(obj_hypotheses_seq, obj_expert_predictions, obj_actual_values,loss_func=se)
    weighted_q_losses=np.dot(tdl,q)
    
    discrepancy_term = discrepancy_simple(q, obj_hypotheses_seq, obj_expert_predictions, obj_actual_values, update_expert_predictions, loss_func = loss_func)
    
    if np.isnan(discrepancy_term):
        raise ValueError("discrepancy went to nan")
    
    return discrepancy_term + weighted_q_losses