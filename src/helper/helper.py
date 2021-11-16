import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize 
from operator import itemgetter

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