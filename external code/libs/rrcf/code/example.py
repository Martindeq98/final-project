import numpy as np
from scipy.optimize import linear_sum_assignment
from rrcfdag import Relaxation
from utility import RandomDAG
from utility import SimResult
#from utility import count_accuracy, is_dag

print("A")

p = 10
n = 50000
prob = 0.3
dat = RandomDAG(p, n, prob, type = "random", seed = 12335)
X = dat.X
B = dat.B
ord = dat.ord


B_adj = np.abs(B)
B_adj[B_adj > 0] = 1

B_pi = B_adj[np.ix_(ord,ord)]

P_true = np.zeros((p,p))
P_true[np.arange(p),np.array(ord)] = 1


rrcf = Relaxation(mu = 10, niter = 1, eps = 1e-9, s = 0.1, lbd = 1, 
                 permest= "permest", penalty = "MCP")

rrcf.fit(X, initP = P_true)

B_hat = rrcf.DAG
est_ord = rrcf.ord
P_hat = rrcf.P 

res = SimResult(rrcf, dat)

print("Real Non zero values", np.count_nonzero(B))
print("Estimated Non zero values", np.count_nonzero(B_hat))

print("Simulation result is \n", res.result)

print("true ord is", ord)
print("Estimated order is", est_ord)

print("Estimatet B \n", B_hat)
print("True B \n", B_pi)