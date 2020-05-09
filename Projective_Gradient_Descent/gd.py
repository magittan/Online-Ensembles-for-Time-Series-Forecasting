import numpy as np
from scipy.optimize import minimize

_plus = lambda x: np.maximum(x,np.zeros(len(x)))
_minus = lambda x: np.minimum(x,np.zeros(len(x)))
_ones = lambda x: np.ones(len(x))

_g = lambda v,x: (1/2)*np.linalg.norm(_minus(x-v*_ones(x)))**2+v*(np.dot(_ones(x),x)-1)-len(x)*v**2


def calculate_gradient(objective_function, x_0, epsilon = 0.1, noise_factor = 0.001):
    #Creating Perturbation Matrix
    dimension = len(x_0)
    
    a = epsilon*np.diag(np.ones(dimension))+noise_factor*np.random.randn(dimension,dimension)
    perturbations = np.concatenate((a,-a),axis=0)
    iterations = np.outer(x_0,np.ones(2*dimension))
    
    #Perturbing the values
    dx = iterations.T+perturbations
    
    #Applying along an axis
    grad = (np.apply_along_axis(objective_function, 1, dx[:dimension])-np.apply_along_axis(objective_function, 1, dx[dimension:]))/(2*epsilon)
    
    return grad

def project_to_simplex(x):
    # Using a basic minization function instead of another approach
    res = minimize(lambda v: -_g(v,x), 0.1, method='BFGS', options={'disp': True})
    max_v = res["x"][0]
    
    output=_plus(x-(max_v)*np.ones(len(x)))
    output/=np.sum(output)
    return output

def projective_simplex_gradient_descent(q, loss_func, iterations=1000,eta=0.001,epsilon=0.01,noise=0.001):
    q_0 = q.copy()
    for i in range(iterations):
        grad = calculate_gradient(loss_func,q_0,epsilon=epsilon,noise_factor=noise)
        if np.isnan(grad.any()):
            raise ValueError('gradient went to nan')
        q_0 = q_0-eta*grad
        q_0 = project_to_simplex(q_0)
        if np.isnan(q_0.any()):
            raise ValueError('q_0 went to nan')
        
    return q_0

def decider(t,sorted_x):
    t_hat = None
    for i in range(len(t)-1):
        if t[i]>=sorted_x[::-1][i+1]:
            return t[i]
    t_hat = t[-1]
    return t_hat

def project_to_simplex_2(x):
    sorted_x = sorted(x)
    t = [np.sum((np.array(sorted_x[-i:])-1)/i) for i in range(1,len(sorted_x)+1)]
    t_hat = decider(t,sorted_x)
    output = _plus(x-decider(t,sorted_x))
    output/=np.sum(output)
    
    return output

def projective_simplex_gradient_descent_2(q, loss_func, iterations=1000,eta=0.001,epsilon=0.01,noise=0.001):
    q_0 = q.copy()
    for i in range(iterations):
        grad = calculate_gradient(loss_func,q_0,epsilon=epsilon,noise_factor=noise)
        q_0 = q_0-eta*grad
        q_0 = project_to_simplex_2(q_0)
        
    return q_0

def projective_simplex_gradient_ascent_2(q, loss_func, iterations=1000,eta=0.001,epsilon=0.01,noise=0.001):
    q_0 = q.copy()
    for i in range(iterations):
        grad = calculate_gradient(loss_func,q_0,epsilon=epsilon,noise_factor=noise)
        q_0 = q_0+eta*grad
        q_0 = project_to_simplex_2(q_0)
        
    return q_0