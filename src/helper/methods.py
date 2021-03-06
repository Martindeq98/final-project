### Methods.py

import numpy as np
import itertools

from tqdm import tqdm
from timeit import default_timer as timer

# OMP method
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import Lasso

# Exhaustive method
from tqdm import tqdm
import statsmodels.api as sm

# Notears method, old in week 12
import sys
path = "C:/Users/s165048/OneDrive - TU Eindhoven/QuinceyFinalProject/final-project/src/Week 12/notears/notears"
sys.path.append(path)
import linear

# notears method, new in main directory ("src/notears/...")
sys.path.append("..")
import notears.notears.notears.linear as linear2

# Lasso LARS
from sklearn.linear_model import LassoLars

# Helper functions
import helper.helper as h

T, p = -1, -1

#### General Functions
### Sample next P, method one for random walk
def sample_next_P(P):  
    # get copy
    P_return = P.copy()

    # get the two rows to swap
    i, j = np.random.choice(p, 2, replace = False)

    # swap row i and row j
    P_return[[i, j]] = P_return[[j, i]]

    # return new sample
    return P_return

### Sample next P, method 2 for random walk
def sample_next_P_2(P):  
    # get copy
    return np.random.permutation(np.identity(p))

## Loss function
def loss(W, X, Y):
    return 1 / len(X) * np.linalg.norm(Y - X @ W, 'f') ** 2

### Do OLS according to permutation
def k_ols_W(Psi, K, P, is_sem = True):
    # translate X
    psi = P @ Psi @ P.T
    k = P @ K @ P.T

    W_hat = np.zeros((p, p))

    # get parameters
    if is_sem:
        for i in range(1, p):
            psi_F = psi[np.array(range(i))[:, None], np.array(range(i))[None, :]]   
            W_hat[np.array(range(i))[:, None], i] = np.linalg.inv(psi_F) @ k[np.array(range(i))[:, None], i]
    else: 
        for i in range(p):
            psi_F = psi[np.array(range(i+1))[:, None], np.array(range(i+1))[None, :]]   
            W_hat[np.array(range(i+1))[:, None], i] = np.linalg.inv(psi_F) @ k[np.array(range(i+1))[:, None], i]

    return P.T @ W_hat @ P

#### Method 1: Exhaustive
def exh(X, Y, is_sem = False):
    perms = itertools.permutations(np.identity(p))
    total = np.math.factorial(p)

    Psi = X.T.dot(X)
    K   = X.T.dot(Y)

    P_best = np.identity(p)
    W_best = k_ols_W(Psi, K, P_best, is_sem = is_sem)
    L_best = loss(W_best, X, Y)

    for perm in tqdm(perms, total = total):
        perm = np.array(perm)
        W = k_ols_W(Psi, K, perm, is_sem = is_sem)
        L = loss(W, X, Y)

        if L < L_best:
            P_best, W_best, L_best = perm, W, L
            
    return P_best, W_best, L_best

#### Method 2.1: Random Walk v1
def rw_1(X, Y, P, max_it, verbose = False, total_time = 1e10, is_sem = False):
    
    Psi = X.T.dot(X)
    K = X.T.dot(Y)
    
    n, p = np.shape(X)

    start = timer()
    
    # smallest likelihood
    min_likelihood = 0 # loss(np.linalg.inv(X[:-1].T @ X[:-1]) @ X[:-1].T @ X[1:], X)
    
    # initial value
    likelihood_P = loss(k_ols_W(Psi, K, P, is_sem = is_sem), X, Y) - min_likelihood

    # current bests
    P_best, L_best, W_best = P.copy(), likelihood_P.copy(), k_ols_W(Psi, K, P, is_sem = is_sem)
    
    for i in tqdm(range(max_it)):   
        # check if time has passed
        if timer() - start > total_time: 
            break
            
        # sample next P
        P_prime = sample_next_P_2(P)
        P = P_prime
        W = k_ols_W(Psi, K, P_prime, is_sem = is_sem)
        L = loss(W, X, Y)

        if L < L_best:
            P_best, W_best, L_best = P_prime, W, L

    return P_best, W_best, L_best

#### Method 2.2: Random Walk v2
def rw_2(X, Y, P, max_it, verbose = False, total_time = 1e10, is_sem = False):
    
    Psi = X.T.dot(X)
    K = X.T.dot(Y)
    
    n, p = np.shape(X)

    start = timer()
    
    # smallest likelihood
    min_likelihood = 0 # loss(np.linalg.inv(X[:-1].T @ X[:-1]) @ X[:-1].T @ X[1:], X)
    
    # initial value
    likelihood_P = loss(k_ols_W(Psi, K, P, is_sem = is_sem), X, Y) - min_likelihood

    # current bests
    P_best, L_best, W_best = P.copy(), likelihood_P.copy(), k_ols_W(Psi, K, P, is_sem = is_sem)
    
    for i in tqdm(range(max_it)):   
        # check if time has passed
        if timer() - start > total_time: 
            break
            
        # sample next P
        P_prime = sample_next_P_2(P)

        W = k_ols_W(Psi, K, P_prime, is_sem = is_sem)
        L = loss(W, X, Y)

        if L < L_best:
            P_best, W_best, L_best = P_prime, W, L

    return P_best, W_best, L_best

#### Method 4: MCMC
def acc_prob_reg(L, L_new):
    return min(L / L_new, 1)

def acc_prob_trans(L, L_new, L_ols):
    return min((L - L_ols) / (L_new - L_ols), 1)

def acc_prob_pow(L, L_new, k):
    return min((L / L_new) ** k, 1)

def acc_prob_trans_pow(L, L_new, k, L_ols):
    return min(((L - L_ols) / (L_new - L_ols)) ** k, 1)

def acc_prob_greed(L, L_new):
    return L_new < L

def mcmc_1(X, Y, max_it, P, verbose = False, factor = 1.0, total_time = 1e10, is_sem = False):
    
    Psi = X.T.dot(X)
    K = X.T.dot(Y)
    
    n, p = np.shape(X)

    start = timer()
    
    # smallest likelihood
    min_likelihood = 0 # loss(np.linalg.inv(X[:-1].T @ X[:-1]) @ X[:-1].T @ X[1:], X)
    
    # initial value
    likelihood_P = loss(k_ols_W(Psi, K, P, is_sem = is_sem), X, Y) - min_likelihood

    # current bests
    P_best, L_best = P.copy(), likelihood_P.copy()
    
    # results
    results_likelihood = [L_best]
    results_l_iter = []
    results_l_transition = []
    results_l_next = []
    num_missed_edges = []

    # transitions
    transitions = 0
    
    for i in tqdm(range(max_it)):   
        # check if time has passed
        if timer() - start > total_time: 
            break
            
        # sample next P
        P_prime = sample_next_P(P)
    
        # compute acceptance probability
        likelihood_P_prime = loss(k_ols_W(Psi, K, P_prime, is_sem = is_sem), X, Y)
        alpha = acc_prob_reg(likelihood_P, likelihood_P_prime)
        
        # print results
        if verbose:
            print(f"Iteration {i+1}.\n")
            print(f"Old P:\n{P}\nLikelihood: {round(likelihood_P, 2)}\n\nNew P:\n{P_prime}\nLikelihood: {round(likelihood_P_prime, 2)}.")
            print(f"\nAcceptance probability: {round(alpha, 3)}.")
    
        # prepare for next iteration
        if np.random.rand() <= alpha:
            transitions += 1
            P, likelihood_P = P_prime, likelihood_P_prime
            results_l_transition.append(likelihood_P_prime)
            # save best
            if likelihood_P < L_best:
                L_best = likelihood_P
                P_best = P
        
        # results
        results_l_next.append(likelihood_P_prime)
        results_likelihood.append(L_best)
        results_l_iter.append(likelihood_P)
        # check_triu = np.triu(P_best @ P_true.T @ B_true @ P_true @ P_best.T)
        # num_missed_edges.append(len(check_triu[check_triu != 0]))
    
    if verbose:
        print(transitions)
        print(f"Best permutation:\n{P_best}\n\nLikelihood: {round(L_best, 2)}.")

    W_best = k_ols_W(Psi, K, P_best, is_sem)
    return P_best, W_best, [results_l_iter, results_likelihood, results_l_transition, results_l_next, num_missed_edges]

def mcmc_2(X, Y, max_it, P, verbose = False, factor = 1.0, total_time = 1e10, is_sem = False):

    Psi = X.T.dot(X)
    K = X.T.dot(Y)
    
    n, p = np.shape(X)

    start = timer()
    
    L_ols = loss(_OLS(np.vstack((X, Y[-1]))), X, Y)
                 
    # smallest likelihood
    min_likelihood = 0 # loss(np.linalg.inv(X[:-1].T @ X[:-1]) @ X[:-1].T @ X[1:], X)
    
    # initial value
    likelihood_P = loss(k_ols_W(Psi, K, P, is_sem = is_sem), X, Y) - min_likelihood

    # current bests
    P_best, L_best = P.copy(), likelihood_P.copy()
    
    # results
    results_likelihood = [L_best]
    results_l_iter = []
    results_l_transition = []
    results_l_next = []
    num_missed_edges = []

    # transitions
    transitions = 0
    
    for i in tqdm(range(max_it)):   
        # check if time has passed
        if timer() - start > total_time: 
            break
            
        # sample next P
        P_prime = sample_next_P(P)
    
        # compute acceptance probability
        likelihood_P_prime = loss(k_ols_W(Psi, K, P_prime, is_sem = is_sem), X, Y)
        alpha = acc_prob_trans(likelihood_P, likelihood_P_prime, L_ols)
        
        # print results
        if verbose:
            print(f"Iteration {i+1}.\n")
            print(f"Old P:\n{P}\nLikelihood: {round(likelihood_P, 2)}\n\nNew P:\n{P_prime}\nLikelihood: {round(likelihood_P_prime, 2)}.")
            print(f"\nAcceptance probability: {round(alpha, 3)}.")
    
        # prepare for next iteration
        if np.random.rand() <= alpha:
            transitions += 1
            P, likelihood_P = P_prime, likelihood_P_prime
            results_l_transition.append(likelihood_P_prime)
            # save best
            if likelihood_P < L_best:
                L_best = likelihood_P
                P_best = P
        
        # results
        results_l_next.append(likelihood_P_prime)
        results_likelihood.append(L_best)
        results_l_iter.append(likelihood_P)
        # check_triu = np.triu(P_best @ P_true.T @ B_true @ P_true @ P_best.T)
        # num_missed_edges.append(len(check_triu[check_triu != 0]))
    
    if verbose:
        print(transitions)
        print(f"Best permutation:\n{P_best}\n\nLikelihood: {round(L_best, 2)}.")

    W_best = k_ols_W(Psi, K, P_best, is_sem)
    return P_best, W_best, [results_l_iter, results_likelihood, results_l_transition, results_l_next, num_missed_edges]

def mcmc_3(X, Y, max_it, P, verbose = False, factor = 1.0, total_time = 1e10, is_sem = False, k = 2):
    
    Psi = X.T.dot(X)
    K = X.T.dot(Y)
    
    n, p = np.shape(X)

    start = timer()
    
    # smallest likelihood
    min_likelihood = 0 # loss(np.linalg.inv(X[:-1].T @ X[:-1]) @ X[:-1].T @ X[1:], X)
    
    # initial value
    likelihood_P = loss(k_ols_W(Psi, K, P, is_sem = is_sem), X, Y) - min_likelihood

    # current bests
    P_best, L_best = P.copy(), likelihood_P.copy()
    
    # results
    results_likelihood = [L_best]
    results_l_iter = []
    results_l_transition = []
    results_l_next = []
    num_missed_edges = []

    # transitions
    transitions = 0
    
    for i in tqdm(range(max_it)):   
        # check if time has passed
        if timer() - start > total_time: 
            break
            
        # sample next P
        P_prime = sample_next_P(P)
    
        # compute acceptance probability
        likelihood_P_prime = loss(k_ols_W(Psi, K, P_prime, is_sem = is_sem), X, Y)
        alpha = acc_prob_pow(likelihood_P, likelihood_P_prime, k)
        
        # print results
        if verbose:
            print(f"Iteration {i+1}.\n")
            print(f"Old P:\n{P}\nLikelihood: {round(likelihood_P, 2)}\n\nNew P:\n{P_prime}\nLikelihood: {round(likelihood_P_prime, 2)}.")
            print(f"\nAcceptance probability: {round(alpha, 3)}.")
    
        # prepare for next iteration
        if np.random.rand() <= alpha:
            transitions += 1
            P, likelihood_P = P_prime, likelihood_P_prime
            results_l_transition.append(likelihood_P_prime)
            # save best
            if likelihood_P < L_best:
                L_best = likelihood_P
                P_best = P
        
        # results
        results_l_next.append(likelihood_P_prime)
        results_likelihood.append(L_best)
        results_l_iter.append(likelihood_P)
        # check_triu = np.triu(P_best @ P_true.T @ B_true @ P_true @ P_best.T)
        # num_missed_edges.append(len(check_triu[check_triu != 0]))
    
    if verbose:
        print(transitions)
        print(f"Best permutation:\n{P_best}\n\nLikelihood: {round(L_best, 2)}.")

    W_best = k_ols_W(Psi, K, P_best, is_sem)
    return P_best, W_best, [results_l_iter, results_likelihood, results_l_transition, results_l_next, num_missed_edges]

def mcmc_4(X, Y, max_it, P, verbose = False, factor = 1.0, total_time = 1e10, is_sem = False):
    
    Psi = X.T.dot(X)
    K = X.T.dot(Y)
    
    n, p = np.shape(X)

    start = timer()
    
    # smallest likelihood
    min_likelihood = 0 # loss(np.linalg.inv(X[:-1].T @ X[:-1]) @ X[:-1].T @ X[1:], X)
    
    # initial value
    likelihood_P = loss(k_ols_W(Psi, K, P, is_sem = is_sem), X, Y) - min_likelihood

    # current bests
    P_best, L_best = P.copy(), likelihood_P.copy()
    
    # results
    results_likelihood = [L_best]
    results_l_iter = []
    results_l_transition = []
    results_l_next = []
    num_missed_edges = []

    # transitions
    transitions = 0
    
    for i in tqdm(range(max_it)):   
        # check if time has passed
        if timer() - start > total_time: 
            break
            
        # sample next P
        P_prime = sample_next_P(P)
    
        # compute acceptance probability
        likelihood_P_prime = loss(k_ols_W(Psi, K, P_prime, is_sem = is_sem), X, Y)
        alpha = acc_prob_greed(likelihood_P, likelihood_P_prime)
        
        # print results
        if verbose:
            print(f"Iteration {i+1}.\n")
            print(f"Old P:\n{P}\nLikelihood: {round(likelihood_P, 2)}\n\nNew P:\n{P_prime}\nLikelihood: {round(likelihood_P_prime, 2)}.")
            print(f"\nAcceptance probability: {round(alpha, 3)}.")
    
        # prepare for next iteration
        if np.random.rand() <= alpha:
            transitions += 1
            P, likelihood_P = P_prime, likelihood_P_prime
            results_l_transition.append(likelihood_P_prime)
            # save best
            if likelihood_P < L_best:
                L_best = likelihood_P
                P_best = P
        
        # results
        results_l_next.append(likelihood_P_prime)
        results_likelihood.append(L_best)
        results_l_iter.append(likelihood_P)
        # check_triu = np.triu(P_best @ P_true.T @ B_true @ P_true @ P_best.T)
        # num_missed_edges.append(len(check_triu[check_triu != 0]))
    
    if verbose:
        print(transitions)
        print(f"Best permutation:\n{P_best}\n\nLikelihood: {round(L_best, 2)}.")

    W_best = k_ols_W(Psi, K, P_best, is_sem)
    return P_best, W_best, [results_l_iter, results_likelihood, results_l_transition, results_l_next, num_missed_edges]

#### Method 5: NOTEARS
def notears_2(X, lambda1=0.0, loss_type="l2-var", h_tol=1e-18, rho_max=1e25, w_threshold=0.01, verbose = False):
	return linear2.notears_linear(X, lambda1=lambda1, loss_type=loss_type, h_tol=h_tol, rho_max=rho_max, w_threshold=w_threshold, verbose=verbose)

#### Method 6: DAG-OLS
def _OLS_LINGNAM(X):
	def OLS(X):
		# get regressor and variables
		y = X[1:]
		x = X[:-1]

		# initialize W_hat
		W_hat = np.array([])

		# get parameters
		for i in range(n):
			est = sm.OLS(y[:, i], x).fit()
			W_hat = np.append(W_hat, est.params)
	
		# return W
		return W_hat.reshape((n, n)).T

	T, n = np.shape(X)

	W_hat = OLS(X)
	
	while not h.is_dag(W_hat):
        # Get the indices of minimum element in numpy array
        # little trickery to skip zero values
		W_hat[W_hat == 0] = np.inf
		min_loc = np.where(np.abs(W_hat) == np.amin(np.abs(W_hat)))
		W_hat[W_hat == np.inf] = 0

		# set minimum value to zero
		W_hat[min_loc] = 0

	return W_hat, OLS(X)

## Orthogonal Matching Pursuit, add edges until it is a DAG.
def _OMP(X, get_order = False, verbose = False):
	"""
	Computes the Orthogonal Matching Pursuit approach for data matrix X.
	Greedily add edges. For every added edge, we check if adding this edge violates the DAG.
	If so, then we remove this edge and hard-fix this to be no edge.
	We continue until we have visited all edges, return will be a complete DAG.
	@params: X, shape p (#nodes), T (#observations).
	@returns: W, weighted adjacency matrix.
	"""
	T, n = np.shape(X)
	
	## Transform data to go from matrix to vector notation
	# Transforms X (the regressors)
	X_large = np.kron(np.eye(n, dtype=float), X[:-1])

	# Stack all X on top of each other for y
	y_large = X[1:].T.reshape((T - 1) * n, 1)
	
	# edges, scores to remember
	edges, scores = [], []
	
	# old coefficients to determine newest edge
	omp_coefs_old = np.zeros(n ** 2)

	# number of violations we have encountered so far
	viols = 0
	
	# for each possible edge in our graph
	for i in range(1, n ** 2 + 1):
		
		# fit OMP for i coefficients
		omp = OrthogonalMatchingPursuit(n_nonzero_coefs = i - viols)
		omp_fit = omp.fit(X_large, y_large)
		
		# get index of newly added edge
		omp_coefs_new = omp_fit.coef_.copy()
		
		# find the newest edge
		omp_coefs_diff = omp_coefs_new.copy()
		omp_coefs_diff[omp_coefs_old != 0] = 0
		
		if len(np.array(np.where(omp_coefs_diff != 0))[0]) == 0: break
		
		# get location of newest edge		
		edge_loc = np.array(np.where(omp_coefs_diff != 0))[0][0]
		
		
		# we check if we have violated the DAG constraint
		if not h.is_dag(omp_fit.coef_.reshape(n, n).T):
			## option 1: We terminate the procedure
			if verbose: print(f"Edge ({edge_loc % n + 1}, {edge_loc // n + 1}) added. violation! Removed.")
			# break
			
			## option 2: We force this edge out, and continue
			# we force the edge out by setting the corresponding data to zero
			# this makes it by default the least desirable option
			# the data that corresponds to this coefficient is 
			to_remove = (edge_loc // n)  * n + edge_loc % n
			X_large[:, to_remove] = np.zeros((T - 1) * n)
		
			# increase number of violations, i - viols = number of coefs
			viols += 1
			
            # edge case: if violation at last iteration, we need to refit
			if i == n ** 2:
                # fit OMP for i coefficients
				omp = OrthogonalMatchingPursuit(n_nonzero_coefs = i - viols)
				omp_fit = omp.fit(X_large, y_large)

			continue
		else:
			edges.append((edge_loc % n + 1, edge_loc // n + 1))
			if i == 1 and verbose:
				print(f"Edge ({edge_loc % n + 1}, {edge_loc // n + 1}) added w/ coef {round(omp_coefs_new[edge_loc], 2)}. Increased score by {round(omp.score(X_large, y_large), 2)}.")
			elif verbose:
				print(f"Edge ({edge_loc % n + 1}, {edge_loc // n + 1}) added w/ coef {round(omp_coefs_new[edge_loc], 2)}. Increased score by {round(omp.score(X_large, y_large) - scores[-1], 2)}.")
			
			
		# get results
		if verbose:
			print(np.round(omp_fit.coef_.reshape(n, n).T, 2))
			print(round(omp.score(X_large, y_large), 2))
			print(end="\n")
		
		# append results
		scores.append(omp.score(X_large, y_large))
		
		# update coefficients
		omp_coefs_old = omp_coefs_new

	W = omp_fit.coef_.reshape(n, n).T
	
	if get_order:
		return W, edges, scores
	
	return W

def _Exh(X):
	def ols_W(X, P):
		# translate X
		x = X @ P.T
		
		# get regressor and varibles
		y = x[1:]
		x = x[:-1]

		W_hat = np.array(np.zeros(n)).T

		# get parameters
		for i in range(n):
			est = sm.OLS(y[:, i], x[:, i:]).fit()   
			W_hat = np.vstack((W_hat, np.append(np.zeros(i), est.params).T))

		return P.T @ W_hat[1:].T @ P

	def loss(W):
		M = X @ W
		
		# Remove X[0] and XW[last]
		R = X[1:] - M[:-1]
		
		# Frobenius norm squared loss
		loss = 1 / (T - 1) * (R ** 2).sum()
		
		return loss
	
	def get_likelihood(P):    
		# Get using least squares
		W = ols_W(X, P)
    
		# get loss    
		return loss(W), W
	
	T, n = np.shape(X)
	
	perms = itertools.permutations(np.identity(n))
	total = np.math.factorial(n)

	P_best = np.identity(n)
	L_best, W_best = get_likelihood(np.identity(n))

	for perm in tqdm(perms, total = total):
		perm = np.array(perm)
		L, W = get_likelihood(perm)

		if L < L_best:
			P_best, W_best, L_best = perm, W, L
			
	return W_best

def _Exh_LT(X, verbose = False):
	def ols_W(X, P):
		# translate X
		x = X @ P.T
		
		# get regressor and varibles
		y = x[1:]
		x = x[:-1]

		W_hat = np.array(np.zeros(n)).T

		# get parameters
		for i in range(n):
			est = sm.OLS(y[:, i], x[:, :(i + 1)]).fit()   
			W_hat = np.vstack((W_hat, np.append(est.params, np.zeros(n - (i + 1))).T))
            
		return P.T @ W_hat[1:].T @ P

	def loss(W):
		M = X @ W
		
		# Remove X[0] and XW[last]
		R = X[1:] - M[:-1]
		
		# Frobenius norm squared loss
		loss = 1 / (T - 1) * (R ** 2).sum()
		
		return loss
	
	def get_likelihood(P):    
		# Get using least squares
		W = ols_W(X, P)
    
		# get loss    
		return loss(W), W
	
	T, n = np.shape(X)
	
	perms = itertools.permutations(np.identity(n))
	total = np.math.factorial(n)

	P_best = np.identity(n)
	L_best, W_best = get_likelihood(np.identity(n))

	for perm in tqdm(perms, total = total):
		perm = np.array(perm)
		L, W = get_likelihood(perm)
		
		if verbose: print(f"Permutation:\n{perm}\n\nW:\n{np.round(W, 2)}\n\nLikelihood: {round(L, 3)}.\n\n")

		if L < L_best:
			P_best, W_best, L_best = perm, W, L
			
	return W_best

def _notears(X, lambda1=0.0, loss_type="l2-var", h_tol=1e-18, rho_max=1e25, w_threshold=0.25, verbose=False):
	return linear.notears_linear(X, lambda1=lambda1, loss_type=loss_type, h_tol=h_tol, rho_max=rho_max, w_threshold=w_threshold, verbose=verbose)

def _notears_2(X, lambda1=0.0, loss_type="l2-var", h_tol=1e-18, rho_max=1e25, w_threshold=0.01, verbose = False):
	return linear2.notears_linear(X, lambda1=lambda1, loss_type=loss_type, h_tol=h_tol, rho_max=rho_max, w_threshold=w_threshold, verbose=verbose)
	
def _OLS_LINGNAM(X):
	def OLS(X):
		# get regressor and variables
		y = X[1:]
		x = X[:-1]

		# initialize W_hat
		W_hat = np.array([])

		# get parameters
		for i in range(n):
			est = sm.OLS(y[:, i], x).fit()
			W_hat = np.append(W_hat, est.params)
	
		# return W
		return W_hat.reshape((n, n)).T

	T, n = np.shape(X)

	W_hat = OLS(X)
	
	while not h.is_dag(W_hat):
        # Get the indices of minimum element in numpy array
        # little trickery to skip zero values
		W_hat[W_hat == 0] = np.inf
		min_loc = np.where(np.abs(W_hat) == np.amin(np.abs(W_hat)))
		W_hat[W_hat == np.inf] = 0

		# set minimum value to zero
		W_hat[min_loc] = 0

	return W_hat, OLS(X)

def _LASSO_LINGNAM(X, step_size = 0.01, normalize = False):
	"""Incrementally increase penalty by step_size until we have a DAG"""
	
	def lasso_lars_W(X, alpha):
		"""Performs LASSO with Lars on X with regularization value alpha"""

		# get regressor and variables
		y = X[1:]
		x = X[:-1]

		# initialize W_hat
		W_hat = np.array([])

		# Get our regularization method
		reg = LassoLars(alpha=alpha, normalize=normalize)

		# get parameters
		for i in range(n):
			est = reg.fit(x, y[:, i])
			W_hat = np.append(W_hat, est.coef_)

        # return W_hat after reshape
		return W_hat.reshape((n, n)).T

	T, n = np.shape(X)

	# initial L1 penalty
	alpha = 0.0

	# get W_hat with 0 penalty -> OLS
	W_ols = lasso_lars_W(X, alpha)
	W_hat = W_ols.copy()

	# while we do not have a dag
	while not h.is_dag(W_hat):
		# increase alpha and do lasso with increased alpha
		alpha += step_size
		W_hat = lasso_lars_W(X, alpha)

	# return W_hat and print smallest alpha
	return W_hat, alpha

def _OMP_2(X, Y, max_coefs = int(1e10), tol = 0.0, verbose = False, output = False):
    
    def normalize(X): return X / np.linalg.norm(X, 2)
	
	# number of variables
    T, n = np.shape(X)
	
    # intialize F
    F = np.full((n, n), False)
    
    # forbidden indices
    N = set()
    
    # output
    Ws, mses, max_gains = [], [], []
    
    mses.append(1 / T * np.linalg.norm(Y - X @ np.zeros((n, n)), 'f') ** 2)
	
    # initialize beta
    W = np.zeros((n, n))
    
    # greedy for loop
    for coef in range(min(n ** 2, max_coefs)):
        
        # get the gains
        gains = np.array([np.abs(normalize(X[:, i]) @ ((X @ W)[:, j] - Y[:, j])) for j in range(n) for i in range(n)])
        
        # set the forbidden coefficients to negative value, so that they will never be chosen
        gains[list(N)] = -np.ones(len(list(N)))
        
        # find the index that maximizes the gains
        i_max = np.argmax(gains)

        # check if the gain is large enough
        if max(gains) <= tol: break
        
        # add index to F
        F[i_max % n][i_max // n] = True
        
        # get 2D indices based on flattened index i_max
        row, col = i_max % n, i_max // n
        
        # check if we still have a DAG
        if h.is_dag(F):   
            # if so, calculate new betas using only atoms in F
            X_F = X[:, F[:, col]]
            W[F[:, col], col] = np.linalg.inv(X_F.T @ X_F) @ X_F.T @ Y[:, col]
            
            # append to iterative list of W
            Ws.append(W.copy().reshape(n, n))
            
            # append max_gains
            max_gains.append(max(gains))
            
            # check the current mean squared error    
            mses.append(1 / T * np.linalg.norm(Y - X @ Ws[-1], 'f') ** 2)
        else:
            # if we do not have a DAG, we remove it, and add it to the forbidden list
            F[row][col] = False
            N.add(i_max)
        
        # print for feedback
        if verbose:
            print(f"MaxGain: {np.round(max(gains), 2)}.")
            print(f"Gain: {np.round(gains.reshape(n, n).T, 2)}.")
            print(f"W:\n{np.round(W, 2)}.\n")
            print(f"F: {F}.")
        
    # return W and extra infor if we care about it
    if output:
        return W.reshape(n, n), Ws, mses, max_gains
    
    # if we do not care about output, we only return betas
    return W
	
def _OLS(X):
    """
       Performs OLS on the data X
       The output will NOT be a DAG necessarily.
    """
	
	# get dimensions
    _,  n = np.shape(X)
	
    # initialize OLS matrix
    W_OLS = np.zeros((n, n))
    
    # get per column
    for i in range(n):
        # closed form solution per OLS
        W_OLS[:, i] = np.linalg.inv(X[:-1].T @ X[:-1]) @ X[:-1].T @ X[1:, i]
        
    # return OLS solution
    return W_OLS

def _OMP_SEM(X, max_coefs = int(1e10), tol = 0.0, verbose = False, output = False):
    
    def normalize(X): return X / np.linalg.norm(X, 2)
	
	# number of variables
    T, n = np.shape(X)
	
    # intialize F
    F = np.full((n, n), False)
    
    # forbidden indices
    N = set([i * (n + 1) for i in range(n)])
    
    # output
    Ws, mses, max_gains = [], [], []
    
    # initialize beta
    W = np.zeros((n, n))
    
    # greedy for loop
    coef = 0
    while coef <= min(int(n * (n - 1) / 2), max_coefs - 1):
    #for coef in range(min(n ** 2, max_coefs)):
        coef += 1
        # get the gains
        gains = np.array([np.abs(normalize(X[:, i]) @ ((X @ W)[:, j] - X[:, j])) for j in range(n) for i in range(n)])
   
        # set the forbidden coefficients to negative value, so that they will never be chosen
        gains[list(N)] = -np.ones(len(list(N)))
        
        # find the index that maximizes the gains
        i_max = np.argmax(gains)

        # check if the gain is large enough
        if max(gains) <= tol: break
        
        # add index to F
        F[i_max % n][i_max // n] = True
        
        # get 2D indices based on flattened index i_max
        row, col = i_max % n, i_max // n
        
        # check if we still have a DAG
        if h.is_dag(F):   
            # if so, calculate new betas using only atoms in F
            X_F = X[:, F[:, col]]
            W[F[:, col], col] = np.linalg.inv(X_F.T @ X_F) @ X_F.T @ X[:, col]
            
            # append to iterative list of W
            Ws.append(W.copy().reshape(n, n))
            
            # append max_gains
            max_gains.append(max(gains))
            
            # check the current mean squared error    
            mses.append(1 / T * np.linalg.norm(X - X @ Ws[-1], 'f') ** 2)
        else:
            # if we do not have a DAG, we remove it, and add it to the forbidden list
            F[row][col] = False
            coef -= 1
            N.add(i_max)
        
        # print for feedback
        if verbose:
            print(f"MaxGain: {np.round(max(gains), 2)}.")
            print(f"Gain: {np.round(gains.reshape(p, p).T, 2)}.")
            print(f"W:\n{np.round(W, 2)}.\n")
            print(f"F: {F}.")
        
    # return W and extra infor if we care about it
    if output:
        return W.reshape(n, n), Ws, mses, max_gains
    
    # if we do not care about output, we only return betas
    return W
	
def _K_OMP(X, Y, max_coefs = 1e10, tol = 0.0, tol_res = 0.0, verbose = False, output = False, normalize = False, F = [], is_sem = False, normalize_2 = False):
    """Do Kernel OMP on X, Y."""
	
    def Lambda_to_adj(Lambda):
        """Convert Lambda list to adjacency matrix"""
        n = len(Lambda)
    
        adj_mat = np.zeros((n, n))
    
        for i, col in enumerate(Lambda):
            adj_mat[i, col] = 1 
    
        return adj_mat
    
	# get dimensions
    T, n = np.shape(X)
	
    if is_sem: F = [i * (n + 1) for i in range(n)]
    
	# compute kernel spaces
    Psi = X.T.dot(X)  					# p times p
    K = X.T.dot(Y)	  					# p  times p
    Theta = [y.T.dot(y) for y in Y.T] 	# 1 times 1
	
    # initialize Lambda, idx, betas
    Lambda, idx, betas = [[] for _ in range(n)], [], np.zeros((n, n))
    
	# compute norms if we want to normalize
    norms = [1] * n # initialize as harmless 1 array	
    if normalize: norms = [np.linalg.norm(x) for x in X.T]
	
    # for each possible coefficient
    for i in range(n ** 2):    
    
        # compute gains
        gains = np.abs([(k - betas.T @ Psi[i, :]) / norms[i] for i, k in enumerate(K)])
        
        # set forbidden set to -1, impossible to pick then
        gains = gains.flatten()
		
        gains[F] = - np.ones(len(F))
        gains = gains.reshape(n, n)
		
        #print(gains.max())
		# stopping criterion
        # print(np.round(gains.max(), 1), end = "\t")
        if np.round(gains, 8).max() <= tol: break
		
        # append best atom to Lambda
		# if tie, pick the one that minimizes residual
        row, col = np.argmax(gains) // n, np.argmax(gains) % n
		
        if row not in Lambda[col]: Lambda[col].append(row)
    
        # check if we have a DAG, not super efficient
        if h.is_dag(Lambda_to_adj(Lambda)): 
            # update only column col, use indices of 
            idx = Lambda[col]
            Psi_F = Psi[np.array(idx)[:, None], np.array(idx)[None, :]]
        
            # speedup: add transpose to forbidden set
            F.append(col * n + row)

            # update betas        	
            betas[np.array(idx)[:, None], col] = np.linalg.inv(Psi_F) @ K[np.array(idx)[:, None], col]
			
            # print(np.round(betas[row][col], 1), end = "\t")
            # if np.abs(betas[row][col]) <= tol: break
            if len(np.nonzero(betas)[0]) >= max_coefs: break
            
        else:
            # append forbidden entry to forbidden list
            F.append(int(np.argmax(gains)))
            # remove coefficient from Lambda
            Lambda[col].remove(row)
    
		# check residual squared
        # print(sum([Theta[i] - K[:, i] @ betas[:, i] for i in range(n)]))
		
        if sum([Theta[i] - K[:, i] @ betas[:, i] for i in range(n)]) < tol_res:
            print("Residual Limit, terminate")
            break
		
        # print info if verbose
        if verbose:
            print(f"Iteration {i + 1}.\n")
            print(f"Gains:\n{np.round(gains, 3)}.\n")
            print(f"Beta_{i + 1}:\n{np.round(betas, 3)}.\n")
            print(f"Residual Squared: {np.round([Theta[i] - K[:, i] @ betas[:, i] for i in range(n)], 32)}.\n\n")

    return betas, sum([Theta[i] - K[:, i] @ betas[:, i] for i in range(n)])
	
def _K_OMP_output(X, Y, max_coefs = 1e10, tol = 0.0, tol_res = 0.0, verbose = False, output = False, normalize = False, F = [], is_sem = False):
    """Do Kernel OMP on X, Y."""
	
    def Lambda_to_adj(Lambda):
        """Convert Lambda list to adjacency matrix"""
        n = len(Lambda)
    
        adj_mat = np.zeros((n, n))
    
        for i, col in enumerate(Lambda):
            adj_mat[i, col] = 1 
    
        return adj_mat
    
	# get dimensions
    T, n = np.shape(X)
	
    if is_sem: F = [i * (n + 1) for i in range(n)]
    
	# compute kernel spaces
    Psi = X.T.dot(X)  					# p times p
    K = X.T.dot(Y)	  					# p  times p
    Theta = [y.T.dot(y) for y in Y.T] 	# 1 times 1

	
    # initialize Lambda, idx, betas
    Lambda, idx, betas = [[] for _ in range(n)], [], np.zeros((n, n))
    
	# compute norms if we want to normalize
    norms = [1] * n # initialize as harmless 1 array	
    if normalize: norms = [np.linalg.norm(x) for x in X.T]
	
    Ws = []
	
    # for each possible coefficient
    for i in range(n ** 2):    
    
        # compute gains
        gains = np.abs([(k - betas.T @ Psi[i, :]) / norms[i] for i, k in enumerate(K)])
        
        # set forbidden set to -1, impossible to pick then
        gains = gains.flatten()
		
        gains[F] = - np.ones(len(F))
        gains = gains.reshape(n, n)
		
        #print(gains.max())
		# stopping criterion
        # print(np.round(gains.max(), 1), end = "\t")
        if np.round(gains, 8).max() <= tol: break
		
        # append best atom to Lambda
		# if tie, pick the one that minimizes residual
        row, col = np.argmax(gains) // n, np.argmax(gains) % n
		
        if row not in Lambda[col]: Lambda[col].append(row)
    
        # check if we have a DAG, not super efficient
        if h.is_dag(Lambda_to_adj(Lambda)): 
            # update only column col, use indices of 
            idx = Lambda[col]
            Psi_F = Psi[np.array(idx)[:, None], np.array(idx)[None, :]]
        
            # speedup: add transpose to forbidden set
            F.append(col * n + row)

            # update betas        	
            betas[np.array(idx)[:, None], col] = np.linalg.inv(Psi_F) @ K[np.array(idx)[:, None], col]
			
            Ws.append(betas.copy())
			
            # print(np.round(betas[row][col], 1), end = "\t")
            # if np.abs(betas[row][col]) <= tol: break
            if len(np.nonzero(betas)[0]) >= max_coefs: break
            
        else:
            # append forbidden entry to forbidden list
            F.append(int(np.argmax(gains)))
            # remove coefficient from Lambda
            Lambda[col].remove(row)
    
		# check residual squared
        #print(sum([Theta[i] - K[:, i] @ betas[:, i] for i in range(n)]))
		
        if sum([Theta[i] - K[:, i] @ betas[:, i] for i in range(n)]) < tol_res:
            print("Residual Limit, terminate")
            break
		
        # print info if verbose
        if verbose:
            print(f"Iteration {i + 1}.\n")
            print(f"Gains:\n{np.round(gains, 3)}.\n")
            print(f"Beta_{i + 1}:\n{np.round(betas, 3)}.\n")
            print(f"Residual Squared: {np.round([Theta[i] - K[:, i] @ betas[:, i] for i in range(n)], 32)}.\n\n")

    return betas, Ws, sum([Theta[i] - K[:, i] @ betas[:, i] for i in range(n)])
	
def _constrained_OLS(X, supp):
    """
       Performs OLS on the data X
       Where supp is the only allowed support for W
    """

    # get dimensions
    _,  n = np.shape(X)

    # initialize OLS matrix
    W_OLS = np.zeros((n, n))
    
    # get per column
    for i in range(n):
        indx = supp[:, i]
        
        X_F = X[:-1, indx]
        
        # closed form solution per OLS
        W_OLS[indx, i] = (np.linalg.inv(X_F.T @ X_F) @ X_F.T) @ X[1:, i]
        
    # return OLS solution
    return W_OLS
	
def _constrained_lasso(X, B, alpha = 0.01):
    
    T, p = np.shape(X)
    
    X_large = np.kron(np.eye(p, dtype=float), X[:-1])
    y_large = X[1:].T.reshape((T - 1) * p, 1)
    
    for i in range(p):
        for j in range(p):
            if B[i][j] == 0:
                to_remove = j * p + i
                X_large[:, to_remove] = np.zeros((T - 1) * p)
                
    clf = Lasso(alpha = alpha, fit_intercept = False)
    clf.fit(X_large, y_large)
    
    return clf.coef_.reshape(p, p).T