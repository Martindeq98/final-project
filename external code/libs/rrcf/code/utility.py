import scipy.stats as stats
import numpy as np
#import random


### This class generated data

class RandomDAG():
    """
    This class generates data from the DAG.
    """
    
    def __init__(self, p, n, prob, type = "random", seed = 10034):
        self. p = p
        self.prob = prob
        self.n = n
        self.seed = seed
        self.type = type
        if self.type == "random":
            self.genDAG()
        else:
            self.genChain()
        self.genData()

        
    def set_seed(self, seed):
 #       random.seed(seed)
        np.random.seed(seed) 

    def genDAG(self):
        """
        Generates stritly lower trinagular weighted
        adjacency matrix B
        Args:  
        """
        DAG = np.zeros((self.p, self.p))
        self.set_seed(self.seed)
        causalOrder = np.random.choice(np.arange(self.p), size = self.p, replace = False)
        for i in range(self.p - 2):
            node = causalOrder[i]
            potParent = causalOrder[(i+1) : self.p]
            self.set_seed(self.seed * (i + 1))
            numParent = np.sum(np.random.binomial(n = 1, p = self.prob, size = (self.p - (i + 1))))
            self.set_seed(self.seed * (i + 1))
            Parent =  np.random.choice(potParent, size = numParent, replace = False)
            Parent =  np.append(causalOrder[i + 1], Parent)
            DAG[node, Parent] = 1
        node = causalOrder[self.p - 2]
        self.set_seed(self.seed)
        Parent = np.random.binomial(n = 1, size = 1, p = self.prob)
        DAG[node, causalOrder[self.p - 1]] = 1
        self.DAG = DAG
        self.ord = np.flip(causalOrder) 

    def genChain(self):
        DAG = np.zeros((self.p, self.p))
        self.set_seed(self.seed)
        causalOrder = np.random.choice(np.arange(self.p), size = self.p, 
                    replace = False)
        DAG[causalOrder[0], causalOrder[1]] = 1
        for i in range(2 , self.p):
            node = causalOrder[i]
            potParent = causalOrder[0 : (i)]
            index = np.sum(DAG[potParent, :], axis = 1) < 4
            potParent = potParent[index]
            if (len(potParent) > 0):
                self.set_seed(self.seed * (i + 1))
                Parents = np.random.choice(potParent, size = min(len(potParent), 2), 
                            replace = False)
                DAG[Parents, node] = np.ones(len(Parents))
            DAG[causalOrder[i - 1], node] = 1
        
        self.DAG = DAG
        self.ord = np.flip(causalOrder) 
        
    def genData(self):
        Bmin = 0.3
        #self.genDAG()
        B = np.copy(self.DAG)
        self.set_seed(self.seed)
        errs = np.random.normal(size = self.p * self.n).reshape(self.p, self.n)
        size = np.sum(B).astype(int)
        self.set_seed(self.seed)
        B[B == 1] = np.random.uniform(low = Bmin, high = 1, size = size ) * (2 * np.random.binomial(n = 1, p = 0.5, size = size) - 1)
        X = np.linalg.solve(np.eye(self.p) - B, errs)
        self.X = np.transpose(X)
        self.B = B
    
def gensample(L, n, seed = 10345):
    p = L.shape[0]
    np.random.seed(seed)
    Z = np.random.normal(size = n * p).reshape([n, p])
    X = np.linalg.solve(L, np.transpose(Z))
    return(np.transpose(X))


class SimResult():
    """
    This class measures performance
    of the proposed method
    """
    
    def __init__(self, est, true):

        self.true_ord = true.ord
        self.true_adj = true.DAG[np.ix_(self.true_ord, self.true_ord)]
        self.est_ord  = est.ord
        self.est_adj  = est.DAG
        self.result = {}
        self.result["SHD"] = self.hammingDistance()
        self.result["Kendall tau"] = self.kendall()
        self.result["recall"] = self.recall()
        self.result["precision"] = self.precision()
        self.result["fdr"] = self.fdr()
        
    def hammingDistance(self):
        hammingDis = np.sum(abs(self.est_adj - self.true_adj))
        hammingDis -= 0.5 * np.sum(self.est_adj * np.transpose(self.est_adj) * (1 - self.true_adj) * np.transpose(1 - self.true_adj) +
                                  self.true_adj * np.transpose(self.true_adj) * (1 - self.est_adj) * np.transpose(1 - self.est_adj))
        return hammingDis
    
    def kendall(self):
        tau, _ = stats.kendalltau(self.true_ord, self.est_ord)
        return tau
    
    def recall(self):
        if np.sum(self.true_adj) > 0:
            return (np.sum(self.true_adj * self.est_adj) / np.sum(self.true_adj))
        else:
            return 0
        
#    def flipped(self):
#        if np.sum(self.true_adj) > 0:
#            return (np.sum(self.true_adj * np.transpose(self.est_adj)) / np.sum(self.true_adj))
#        else:
#            return 0
        
    def precision(self):
        if np.sum(self.est_adj) > 0:
            return (np.sum(self.true_adj * self.est_adj) / np.sum(self.est_adj) )
        
    def fdr(self):
        if np.sum(self.est_adj) > 0:
            return (1 - np.sum(self.true_adj * self.est_adj) / np.sum(self.est_adj))
        else:
            return 0
        
        
# def count_accuracy(B_true, B_est):
#     """Compute various accuracy metrics for B_est.
#     true positive = predicted association exists in condition in correct direction
#     reverse = predicted association exists in condition in opposite direction
#     false positive = predicted association does not exist in condition
#     Args:
#         B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
#         B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG
#     Returns:
#         fdr: (reverse + false positive) / prediction positive
#         tpr: (true positive) / condition positive
#         fpr: (reverse + false positive) / condition negative
#         shd: undirected extra + undirected missing + reverse
#         nnz: prediction positive
#     """
#     if (B_est == -1).any():  # cpdag
#         if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
#             raise ValueError('B_est should take value in {0,1,-1}')
#         if ((B_est == -1) & (B_est.T == -1)).any():
#             raise ValueError('undirected edge should only appear once')
#     else:  # dag
#         if not ((B_est == 0) | (B_est == 1)).all():
#             raise ValueError('B_est should take value in {0,1}')
#         if not is_dag(B_est):
#             raise ValueError('B_est should be a DAG')
#     d = B_true.shape[0]
#     # linear index of nonzeros
#     pred_und = np.flatnonzero(B_est == -1)
#     pred = np.flatnonzero(B_est == 1)
#     cond = np.flatnonzero(B_true)
#     cond_reversed = np.flatnonzero(B_true.T)
#     cond_skeleton = np.concatenate([cond, cond_reversed])
#     # true pos
#     true_pos = np.intersect1d(pred, cond, assume_unique=True)
#     # treat undirected edge favorably
#     true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
#     true_pos = np.concatenate([true_pos, true_pos_und])
#     # false pos
#     false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
#     false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
#     false_pos = np.concatenate([false_pos, false_pos_und])
#     # reverse
#     extra = np.setdiff1d(pred, cond, assume_unique=True)
#     reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
#     # compute ratio
#     pred_size = len(pred) + len(pred_und)
#     cond_neg_size = 0.5 * d * (d - 1) - len(cond)
#     fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
#     tpr = float(len(true_pos)) / max(len(cond), 1)
#     fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
#     # structural hamming distance
#     pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
#     cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
#     extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
#     missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
#     shd = len(extra_lower) + len(missing_lower) + len(reverse)
#     return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}



# def is_dag(W):
#     G = ig.Graph.Weighted_Adjacency(W.tolist())
#     return G.is_dag()