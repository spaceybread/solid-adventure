import pandas as pd
import numpy as np
from tqdm import tqdm
import math

def h(x: np.array, theta: np.array): 
    return x @ theta

def j(xs: np.array, y: np.array, theta: np.array):
    s = 0
    for i in range(len(xs)): s += (h(xs[i], theta) - y[i])**2
    return s / (2 * len(xs))

def j_faster(xs: np.ndarray, y: np.ndarray, theta: np.ndarray):
    residuals = xs @ theta - y
    return (residuals @ residuals) / (2 * len(y))

def grad_descent_step(alpha: float, xs: np.ndarray, y: np.ndarray, theta: np.ndarray):
    grad = np.zeros(len(theta))
    for i in range(len(theta)):
        for j in range(len(xs)):
            grad[i] += (h(xs[j], theta) - y[j]) * xs[j][i]
    theta -= alpha * grad
        
def grad_descent_step_faster(alpha: float, xs: np.ndarray, y: np.ndarray, theta: np.ndarray):
    theta -= alpha * (xs.T @ (xs @ theta - y))

def grad_descent(alpha: float, xs: np.ndarray, y: np.ndarray, theta: np.ndarray, max_iter=10000):
    last_score = float('inf')
    for _ in range(max_iter):
        grad_descent_step_faster(alpha, xs, y, theta)
        curr_score = j_faster(xs, y, theta)
        if np.isnan(curr_score) or np.isinf(curr_score): break
        if abs(curr_score - last_score) < 0.001: break
        last_score = curr_score
    
    # print(theta)
    return theta


def cross_val(X, y, alpha, k=5):
    fold_size = len(X) // k
    costs = []
    
    for i in range(k):
        # do this so that you don't get an unlucky split
        X_test  = X[i*fold_size : (i+1)*fold_size]
        y_test  = y[i*fold_size : (i+1)*fold_size]
        X_train = np.vstack([X[:i*fold_size], X[(i+1)*fold_size:]])
        y_train = np.concatenate([y[:i*fold_size], y[(i+1)*fold_size:]])
        
        # this is required because without it, features in different units and stuff become
        # hard to use together
        mean, std = X_train.mean(axis=0), X_train.std(axis=0)
        X_train = (X_train - mean) / std
        X_test  = (X_test  - mean) / std
        
        # adding the first feature that is not impacted by the input
        X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        X_test  = np.hstack([np.ones((X_test.shape[0],  1)), X_test])
        
        theta = np.zeros(X_train.shape[1])
        theta = grad_descent(alpha, X_train, y_train, theta)
        costs.append(j_faster(X_test, y_test, theta))
    
    return np.mean(costs), np.std(costs)
    
    
