import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize 
from operator import itemgetter
import scipy.stats as scistat
import os

### generate data

def generate_sem_data(length, n, A, P):
    # order
    order = [np.argmax(P[i]) for i in range(n)]
    
    # Initialize series
    series = np.array(np.zeros((length, n)))
    
    # Precalculate inverse
    inv_P = np.linalg.inv(P)
    
	# precalculate B
    B = np.matmul(inv_P, np.matmul(A, P))
    
    # Generate series
    for t in range(1, length):
        # Generate SEM model in correct order
		
		# First the randomness
        series[t] = np.random.multivariate_normal(np.zeros(n), np.identity(n))
		
		# Then all other variables
        for i in range(n):
            series[t][order[i]] += np.matmul(B[order[i]], series[t])
    
	# Return series
    return series

def generate_var_2(length, n, A, P):
    
    # Initialize series
    series = np.array(np.zeros((length, n)))
    
    # Precalculate inverse
    inv_P = np.linalg.inv(P)
    
	# precalculate B
    B = np.matmul(inv_P, np.matmul(A, P))
    
    # Generate series
    for t in range(1, length):
		# First the randomness
        series[t] = np.matmul(series[t - 1], B) + np.random.multivariate_normal(np.zeros(n), np.identity(n))
    
	# Return series
    return series
	
def generate_var_res(T, W, residuals):
    
    # Initialize series
    p = np.shape(W)[0]
    series = np.array(np.zeros((T, p)))
    R = np.shape(residuals)[0]
	
    # Generate series
    for t in range(1, T):
		# First the randomness
        residual = np.array([residuals[np.random.choice(R), i] for i in range(p)])
        series[t] = series[t - 1] @ W + residual
    
	# Return series
    return series
	
def generate_var_data(length, n, A, P):
    
    # Initialize series
    series = np.array(np.zeros((length, n)))
    
    # Precalculate inverse
    inv_P = np.linalg.inv(P)
    
	# precalculate B
    B = np.matmul(inv_P, np.matmul(A, P))
    
    # Generate series
    for t in range(1, length):
		# First the randomness
        series[t] = np.matmul(B, series[t - 1]) + np.random.multivariate_normal(np.zeros(n), np.identity(n))
    
	# Return series
    return series

### conversions
def a_list_to_a_matrix(alist):
	# initialize with zeros
    A_matrix = np.zeros((n, n))
	
	# get indices of non-strict LT
    indices = np.tril_indices(n)
	
	# set these to the values in the list
    A_matrix[indices] = alist

	# return
    return A_matrix

def a_list_to_a_matrix_LT(alist):
	# initialize with zero
    A_matrix = np.zeros((n, n))
	
	# get indices of strict LT
    indices = np.tril_indices(n, -1)
	
	# set these to the values in the list
    A_matrix[indices] = alist
	
	# return
    return A_matrix

def p_list_to_p_matrix(ps):
    P_matrix = np.zeros((n, n))
    P_matrix[:n-1,:n-1] = ps.reshape((n - 1, n - 1))
    P_matrix[:n-1, n - 1] = 1 - P_matrix[:n-1,:].sum(axis = 1)
    P_matrix[n - 1, :] = 1 - P_matrix.sum(axis = 0)
    return P_matrix

def closest_perm(P_DS):
    # convert to get the maximum linear assignment problem
    P_MOD = np.ones((n, n)) * np.max(P_DS) - P_DS
	
	# apply the maximum linear assignment algorithm
    row_ind, col_ind = optimize.linear_sum_assignment(P_MOD)

    # initialize Permutation matrix to return
    P_perm = np.zeros((n, n))

	# fill in permutation matrix
    for row, col in zip(row_ind, col_ind):
        P_perm[row][col] = 1 
    
	# re
    return P_perm

### Cost functions

# C1: B = P^{-1}AP
def C_1(variables):
    P = p_list_to_p_matrix(variables[:(n - 1) ** 2])
    A = a_list_to_a_matrix_LT(variables[(n - 1) ** 2:])  
    P_inv = np.linalg.inv(P)
    B = np.matmul(P_inv, np.matmul(A, P))
    
    cost = 0

    for t in range(length):
        val = series[t]
        est = np.matmul(B, series[t])
        cost += np.linalg.norm(val - est, ord = 2) ** 2

    return cost / (length - 1) # regularization

def C_var_1(variables):
    P = p_list_to_p_matrix(variables[:(n - 1) ** 2])
    A = a_list_to_a_matrix(variables[(n - 1) ** 2:])  
    P_inv = np.linalg.inv(P)
    B = np.matmul(P_inv, np.matmul(A, P))
    
    cost = 0

    for t in range(1, length):
        val = series[t]
        est = np.matmul(B, series[t - 1])
        cost += np.linalg.norm(val - est, ord = 2) ** 2

    return cost / (length - 1) # regularization
	
# C2: B = P^TAP
def C_2(variables):
    P = p_list_to_p_matrix(variables[:(n - 1) ** 2])
    A = a_list_to_a_matrix_LT(variables[(n - 1) ** 2:])  
    B = np.matmul(P.transpose(), np.matmul(A, P))
    
    cost = 0

    for t in range(length):
        val = series[t]
        est = np.matmul(B, series[t])
        cost += np.linalg.norm(val - est, ord = 2) ** 2

    return cost / (length - 1) # + 0.01 * np.sum(np.abs(B)) # regularization
	
# C3: B = P^TAP / det(P)
def C_3(variables):
    P = p_list_to_p_matrix(variables[:(n - 1) ** 2])
    A = a_list_to_a_matrix_LT(variables[(n - 1) ** 2:])  
    B = np.matmul(P.transpose(), np.matmul(A, P)) / np.linalg.det(P)
    
    cost = 0

    for t in range(length):
        val = series[t]
        est = np.matmul(B, series[t])
        cost += np.linalg.norm(val - est, ord = 2) ** 2

    return cost / (length - 1) # + 0.01 * np.sum(np.abs(B)) # regularization

def gen_P(n):
    P = np.random.random((n, n))
    
    rsum = 0
    csum = 0

    while (np.any(np.round(rsum, 3) != 1)) | (np.any(np.round(csum, 3) != 1)):
        P /= P.sum(0)
        P = P / P.sum(1)[:, np.newaxis]
        rsum = P.sum(1)
        csum = P.sum(0)
        
    return P
	
def generate_A(n, num_edges, low = 0.5, high = 2.0, tril = False):
    edges = np.array([0.0] * (int(n * (n + 1) / 2) - num_edges -n * tril) + [1.0] * num_edges)
    
    edges[edges > 0] = (2 * np.random.randint(0, 2, size=(num_edges)) - 1) * np.random.uniform(low, high, num_edges)
    np.random.shuffle(edges)
    
    A = np.zeros((n, n))
    
    A[np.tril_indices(n, - tril)] = edges
	
    for i in range(n):
        if abs(A[i][i]) > 0.95:
            A[i][i] = np.random.uniform(0.75, 0.95)
    
    return A
	
def is_dag(W_input):
    n = np.shape(W_input)[0]
    
    W = W_input.copy()
    # remove diagonal entries
    np.fill_diagonal(W, 0)
    
    order, old_order = [], list(range(n))
    
    # for the number of elements
    for i in range(n):
        
        # find a row that contains only zeros
        for j in range(n - i):
            # if we find a zero row (excl. diags)
            if not W[j].any() != 0:
                
                # remove this row and column
                W = np.delete(W, j, 0)
                W = np.delete(W, j, 1)
            
                order.append(old_order[j])
                old_order.remove(old_order[j])
                
                # go to next variable
                break
        
            # if no zero row exist stop
            elif i == n - 1:
                return False
            
    return True, order

### Score a WAM given data X
def score(X, W, W_true, printing = True, rounding = 3, is_sem = False):
    """
    Score the weighted adjacency matrix "W" against the true W, "W_true".
    Two different score types:
        - Structural: TPR, FPR, etc.
        - Predictive: Loss.
    """
	
    # length of diagonal
    n = np.shape(W)[0]
	
    ### Structural
    ## TPR
	# Ratio of detected ground truth positives over ground truth positives
	
    truth_bin = W_true.copy()
    truth_bin[truth_bin != 0] = 1

    W_bin = W.copy()
    W_bin[W_bin != 0] = 1

    true_edges = np.flatnonzero(truth_bin)
    pred_edges = np.flatnonzero(W_bin)
    tpr = len(np.intersect1d(pred_edges, true_edges, assume_unique=True)) / max(len(true_edges), 1)

    ## TNR
	# Ratio of detected ground truth negatives over ground truth negatives
    true_non_edges = np.flatnonzero(truth_bin - 1)
    pred_non_edges = np.flatnonzero(W_bin - 1)
    tnr = len(np.intersect1d(pred_non_edges, true_non_edges, assume_unique=True)) / max(len(true_non_edges), 1)

    ## FPR
    pred_false_edges = np.setdiff1d(pred_edges, true_edges)
    fpr = len(pred_false_edges) / max(len(pred_edges), 1)

    # Accuracy
	## Ratio of detected ground truths over ground truths
    acc = (len(truth_bin[truth_bin == W_bin]) - is_sem * n) / max(len(truth_bin.flatten()) - is_sem * n, 1)
    
	## Structural Hamming Distance
	# Distance between true and incorrect
    shd = len(truth_bin[truth_bin != W_bin])
	
    ## Predictive
    mse = MSE(W, X, is_sem)
	
    rsquared = RSquared(W, X, is_sem)
	
    if printing:
        print(f"True Positive Rate: {round(tpr, rounding)}.\nTrue Negative Rate: {round(tnr, rounding)}.\nFalse Prediction Rate: {round(fpr, rounding)}\nAccuracy: {round(acc, rounding)}.")
        print("R-Squared:", round(rsquared, rounding))
        print("Mean Squared Error:", round(mse, rounding))
	
    return tpr, tnr, fpr, acc, shd, mse, rsquared

def RSquared(W, X, is_sem = False):
    """Compute R-Squared of matrix W on data X"""
	
    T, n = np.shape(X)
		
    if is_sem:
        X_regress = np.kron(np.eye(n, dtype=float), X) @ W.T.reshape(n ** 2)
        y_regress = X.T.reshape(T * n, 1) 
        _, _, r_value, _, _ = scistat.linregress(X_regress, y_regress[:, 0])
		
        return r_value ** 2
    else:
        X_regress = np.kron(np.eye(n, dtype=float), X[:-1]) @ W.T.reshape(n ** 2)
        y_regress = X[1:].T.reshape((T - 1) * n, 1) 
        _, _, r_value, _, _ = scistat.linregress(X_regress, y_regress[:, 0])
        
        return r_value ** 2
	
    
	
def MSE(W, X, is_sem = False):
    """Compute Mean Squared error of matrix W on data X."""
    
    if is_sem:
        X_val = X
        X_pred = X @ W
    
        return 1 / len(X_val) * np.linalg.norm(X_val - X_pred, 'f') ** 2
    else:
        X_val = X[1:]
        X_pred = X[:-1] @ W
    
        return 1 / len(X_val) * np.linalg.norm(X_val - X_pred, 'f') ** 2
	
def save_data(X, W, misc, expl, directory = ""):
    """Saves dataset X, matrix W"""
    
    ## The name of the dataset X is as follows:
    # n: Number of variables
    # T: number of samples (timesteps) per variable
    n, T = np.shape(X)
    
    ## The ground truth W (need not be the best!)
    # s: Number of edges
    s = len(W[W != 0])
        
    identifier = 0
    name = f"X_s{s}_n{n}_T{T}_{misc}_{identifier}"
    path = f"C:/Users/s165048/OneDrive - TU Eindhoven/QuinceyFinalProject/final-project/data/generated_data/{directory}{name}"
            
    while os.path.exists(f"{path}.txt"): 
        identifier += 1
        name = f"X_s{s}_n{n}_T{T}_{misc}_{identifier}"
        path = f"C:/Users/s165048/OneDrive - TU Eindhoven/QuinceyFinalProject/final-project/data/generated_data/{directory}{name}"
    
    np.savetxt(f"{path}.txt", (W, X, expl), fmt='%s')
    np.savez(path, W=W, X=X, expl=expl)
    
    print(f"Dataset saved with name {name}.")

def load_data(name, directory = ""):
    if name[-4:] == '.npz':
        name = name[:-4]
    path = f"C:/Users/s165048/OneDrive - TU Eindhoven/QuinceyFinalProject/final-project/data/generated_data/{directory}/{name}.npz"
    data = np.load(path)
    return data['W'], data['X'], data['expl']