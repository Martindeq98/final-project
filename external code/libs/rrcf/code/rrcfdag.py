import numpy as np
import pdb
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

class Relaxation:
    """
    This class estimates Bayesian Networks 
    by two step procedure described in Dallakyan et.al (2021+)
    The main function is rrcfdag().
    """
    def __init__(self,  mu = 0.03, niter = 100, proj_niter = 100,  eps = 1e-7, 
                s = 0.3, penalty = "MCP", gamma = 2,
                lbd = 0.1, alpha = 1.5, permest = "LSAP"):
        self.mu = mu
        self.niter = niter
        self.proj_niter = proj_niter
        self.eps = eps
        self.s = s
        self.lmbd = lbd
        self.alpha = alpha
        self.gamma = gamma
        self.penalty = penalty
        if( mu < 0) or niter < 0 or gamma < 0 or alpha < 0 or lbd < 0:
            raise TypeError("Negative value specified when positive needed")
        if (penalty is not "lasso") and (penalty is not "MCP") :
            raise TypeError("Specify currect penalty.")
        if (permest is not "LSAP") and (permest is not "permest") :
            raise TypeError("Specify currect permutation matrix estimation method.")
        self.permest = permest
        self.error = {}
        
    def fit(self, X, initP = None, initL = None ):
        self.X = X
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.S = np.cov(np.transpose(self.X))
        if initP is None:
            self.P = np.eye(self.p)
        else:
            self.P = initP
        if initL is None:
            self.L = np.diag(1 / np.sqrt(np.diag(self.S)))
        else:
            self.L = initL
        self.onevec = np.ones((self.p,1))
        self.Pii = np.eye(self.p) - 1.0 / self.p * np.outer(self.onevec, self.onevec)
        self.B = self.rrcfdag()
        self.DAG = abs(self.B) 
        self.DAG[self.DAG > 0] = 1
        
    ## Projects into doubly stochastic space

    
    def primalObj(self, P , P_0):
        """
        Estimates primal objective function for the Projection step

        Args:
        P - DS matrix
        P_0 - projected matrix

        Result:
        scalar
        """
        return 1/2.0 * (np.sum((P - P_0)**2)).item()

                 
    def gradient(self, P):
        """This function estimates gradient for the 
        projection gradient algorithm
        
        Args:
        P - pxp Doubly stochastic matrix
        self.L - Cholesky factor
        self.S - covariance matrix
        self.Pii - projection matrix
        self.mu  - regularization parameter that pushed DS matrix toward projection.
        Result:
        p x p gradient matrix
        """
 #       pdb.set_trace()
        grad_f = (np.transpose(self.L) @ self.L @ P @ self.S - 
        self.mu * np.transpose(self.Pii) @ self.Pii @  P)
        return grad_f
     
       
    def soft(self, a):
        """
        Estimated soft thresholding function
        for the lasso penalty
        Args:
        a - scalar
        self.lmbd - thresholding parameter

        Result:
        scalar
        """
#        pdb.set_trace()
        return np.sign(a) * np.maximum(0.0, np.abs(a) - self.lmbd)
    
    def rho(self, a):
        """
        Estimates MCP penalty function.

        Args:
        a - scalar
        self.lmbd, self.gamma - regularization parameters
        """
        result = [(self.lmbd *  np.abs(i) - (i)**2 / (2 * self.gamma)) if 
        np.abs(i) < self.lmbd * self.gamma else self.gamma * (self.lmbd**2) / 2 for i in a ]
        return sum(result)

    def subObj(self, P):
        """
        Estimated subobjective function for the 
        projection gradient algorithm.

        Args: 
        P - pxp DS matrix
        self.S - covariance matrix
        self.L - Cholesky factor
        self.Pii - projection matrix
        """
        return (1.0/2 * np.trace(P @ self.S @ np.transpose(P) @ np.transpose(self.L) @ self.L) 
        - 1.0 / 2 * self.mu * np.sum((self.Pii * P)**2)).item()
    
    def objFunc(self, P, L):
        """
        Estimated full objective function.
        Args:
        P - pxp permutation matrix
        L - Cholesky factor
        Results:
        scalar

        """
        if self.penalty == "lasso":
            obj = self.subObj(P) + self.lmbd * np.sum(np.abs(L - np.diag(np.diag(L))))
        elif self.penalty == "MCP":
            rowsum = sum(map(self.rho, L))
            obj = self.subObj(P) + rowsum
        return obj
    
    def dualObj(self, x, y, Z, P_0):
        """
        Estimated dual objective function 
        for the projection step.

        Args:
        x,y  -  px1 vector of Langrange multipliers
        Z    - p xp matrix of Langrange multipliers
        P_0 - DS matrix

        Results:
        scalar
        """
        return  (-1.0/ 2 * np.sum((np.repeat(x,self.p, axis = 1) + np.repeat(np.transpose(y),self.p, axis = 0) - Z)**2) - 
                np.trace(np.transpose(Z) @ P_0) + np.transpose(x) @ (P_0 @ self.onevec - self.onevec) + 
                np.transpose(y) @ (np.transpose(P_0) @ self.onevec - self.onevec)).item()

    # def dualObj_constr(self, x, y, Z, z, D, P_0):
    #     """
    #     Estimated dual objective function with constraint
    #     for the projection step.

    #     Args:
    #     x,y  -  px1 vector of Langrange multipliers
    #     Z    - p xp matrix of Langrange multipliers
    #     P_0 - DS matrix

    #     Results:
    #     scalar
    #     """
    #     f = -1.0/ 2 * np.sum((np.repeat(x,self.p, axis = 1) + np.repeat(np.transpose(y),self.p, axis = 0) - 
    #                         Z + D @ z @ np.transpose(self.g))**2)
    #     s = np.trace(np.transpose(Z) @ P_0)
    #     t = np.transpose(x) @ (np.sum(P_0,axis = 1).reshape((-1,1)) - 1) 
    #     fo = np.transpose(y) @ (np.sum(P_0, axis = 0).reshape((-1,1)) - 1)
    #     fi = np.transpose(z) @ (np.transpose(D) @ P_0 @ self.g + self.delta)

    #     return (f + s + t + fo + fi).item()
    #     # return  (-1.0/ 2 * np.sum((np.repeat(x,self.p, axis = 1) + np.repeat(np.transpose(y),self.p, axis = 0) - Z + D @ z @ np.transpose(self.g))**2) - 
    #     #         np.trace(np.transpose(Z) @ P_0) + np.transpose(x) @ (np.sum(P_0,axis = 1).reshape((-1,1)) - self.onevec) + 
    #     #         np.transpose(y) @ (np.sum(P_0, axis = 0).reshape((-1,1)) - self.onevec) + np.transpose(z) @ (np.transpose(D) @ P_0 @ self.g + self.delta)).item()
               
   
    def projDS(self, P, x, y, Z):
        """
        Dual Ascent algorithm for projecting onto the space of
        DS matrices.

        Args:
        P - pxp matrix
        x,y  -  px1 vector of Langrange multipliers
        Z    - p xp matrix of Langrange multipliers

        Results:
        P_ds - pxp DS matrix
        x,y    - px1 updated Langrage multipliers
        Z     - pxp Langrage multipliers
        err.values - vector of gap errors.
        """
        k = 0
        converged = False
        err = {}
        while converged == False and k <= self.proj_niter:
            Z = np.maximum(np.zeros((self.p, self.p)),np.outer(x, self.onevec) + np.outer(self.onevec, y) - P)
            x = (1.0 / self.p) * (P @ self.onevec - (np.sum(y) + 1.0) * self.onevec + Z @ self.onevec)
            y = (1.0 / self.p) * (np.transpose(P) @ self.onevec - (np.sum(x) + 1.0) * self.onevec +
                           np.transpose(Z) @ self.onevec)
            P_ds = P - np.outer(x, self.onevec) - np.outer(self.onevec, y) + Z
            err[k] = (abs(self.primalObj(P_ds, P) - self.dualObj(x, y, Z, P)))
 #     print("error value is {0}".format(err[k]))
            if err[k] <= self.eps:
                converged = True
            else:
                k = k + 1
        return P_ds, x, y, Z, np.array(list(err.values()))


#     def projDS_constr(self, P, x, y, Z, z):
#         k = 0
#         converged = False
#         err = {}
#         if (self.constr is None):
#             n_c = 1
#             D = np.zeros((self.p,1))
#             D[0] = 1
#             D[self.p - 1] = -1
#             self.delta = 1
#         else:
#             n_c = len(self.constr)
#             D = np.zeros((self.p, n_c - 1))
#             #D[0, 0] = 1
#             #D[self.p - 1, 0] = -1
#             self.delta = np.zeros((n_c - 1,1))
#        # self.delta[0,0] = 0
#             for i in range(1, n_c):
#                 D[self.constr[i - 1], i - 1] = 1
#                 D[self.constr[i], i - 1] = -1
#         while converged == False and k <= self.proj_niter:

#             Z = np.maximum(np.zeros((self.p, self.p)),np.outer(x, self.onevec) + 
#                             np.outer(self.onevec, y) - P - D @ z @ np.transpose(D) @ P @ S)

#             x = (1.0 / self.p) * (np.sum(P,axis = 1) .reshape((-1,1)) - (np.sum(y) + 1.0) * self.onevec + 
#                             np.sum(Z, axis = 1).reshape((-1,1)) + D @ z @ np.transpose(D) @ P @ np.sum(self.S, axis = 0).reshape((-1,1)))

#             y = (1.0 / self.p) * (np.sum(P,axis = 0) .reshape((-1,1)) - (np.sum(x) + 1.0) * self.onevec +
#                            np.sum(Z, axis = 0).reshape((-1,1)) + S @ np.transpose(P) @ D @ np.transpose(z) @ 
#                            np.sum(D, axis = 0).reshape((-1,1)))

# #            z = 1 / np.sum(self.g**2) * np.maximum(np.zeros((n_c, 1)),  
#  #                           np.linalg.solve(np.transpose(D) @ D, np.transpose(D) @ (Z + P) @ self.g +
#  #                           self.delta - np.sum(D, axis = 0).reshape((-1,1)) @ np.transpose(self.g) @ y  -
#  #                           np.transpose(D) @ x * np.sum(self.g)))
#             Dt_P_S_D = np.transpose(D) @ D
#             Dt_Z_P_g = np.transpose(D) @ (Z + P) @ self.g
#             z = np.maximum(np.zeros((n_c - 1, n_c - 1)),  
#                             np.linalg.solve(Dt_D, Dt_Z_P_g +
#                             self.delta - np.transpose(D) @ x * np.sum(self.g) - np.sum(D, axis = 0).reshape((-1,1)) * 
#                             np.sum(y * self.g)))                

#             P_ds = P - np.outer(x, self.onevec) - np.outer(self.onevec, y) + Z - D @ z @ np.transpose(self.g)
#             err[k] = (abs(self.primalObj(P_ds, P) - self.dualObj_constr(x, y, Z,z,D, P)))
#             print("error value is {0}".format(err[k]))
#             if err[k] <= self.eps:
#                 converged = True
#             else:
#                 k = k + 1
#         return P_ds, x, y, Z, z, np.array(list(err.values()))


            
    
    ## Selects the permutation matrix
    def permProj(self, D, oldP = None):
        """
        Implements permutation esimation 
        based on Bavonik suggestions

        Args:
        D - pxp DS matrix
        oldP - pxp permutaiton matrix

        Results:
        newP - pxp estimated permutation matrix
        """
 #       pdb.set_trace()
        storeObj = []
        min = 1e+8
        newP = oldP
        i = 0
        min = 1e+8
        converged = False
        if(self.permest == "LSAP"):
            newP = np.zeros((self.p, self.p))
            ind = linear_sum_assignment(D)[1]
            self.ord = ind
            newP[np.arange(self.p), np.array(ind)] = 1
        else:
            while (i <= self.proj_niter) and (converged is False) :
                oldObj = self.subObj(oldP)
                P = np.zeros((self.p, self.p))
                x = np.random.normal(size = self.p)
                temp = x.argsort()
                rnk_x = np.empty_like(temp)
                rnk_x[temp] = np.arange(len(x)) 
                D_x = D @ x
                temp_D = D_x.argsort()
                rnk_Dx = np.empty_like(temp_D)
                rnk_Dx[temp_D] = np.arange(len(D_x))
                ind = [rnk_x.tolist().index(x) if x in rnk_x else None for x in rnk_Dx ]
                self.ord = ind
                P[np.arange(self.p), np.array(ind)] = 1
                storeObj.append(self.subObj(P))
                if storeObj[i] < min:
                    min = storeObj[i]
                    self.ord = ind
                    newP = P
                if (np.abs(storeObj[i] - oldObj) < self.eps):
                    converged = True
                else:
                    i = i + 1
        return newP

    def RC_i(self, i, A, initx = None):
        """
        Estimates the ith row of the 
        Cholesky factor L through regularization

        Args:
        i - scalar, the row index
        A - ixi submatrix of S
        initx - initial value for the row, default is None

        Results:
        ix1 array
        """
#        pdb.set_trace()
        if i == 0:
            x = 1 / np.sqrt(A)
        else:
            iter = 0
            converged = False
            if initx is None:
                x = np.zeros((i+1,1))
                x[i] = 1 / np.sqrt(A[i,i])
            else:
                x = np.copy(initx)
            while ((converged == False) and (iter <= self.niter)):
                x_old = np.copy(x)
                ## Update nondiaognal
                for j in range(i):
                    A_jj = A[j, j]
                    #y_j = np.sum(A[-j, j] * x[-j])
                    
                    y_j = np.sum(A[np.arange(i + 1) != j, j] * x[np.arange(i + 1) != j])
                    if(self.penalty == "MCP"):
                        if (np.abs(y_j) / A_jj) <= (self.gamma * self.lmbd):
                            x[j] = self.soft(-2.0 * y_j) / (2.0 *  A_jj - 1.0/ self.gamma)
                        else:
                            x[j] = - y_j / A_jj
                    else:
                        x[j] = self.soft(-2.0 * y_j) / (2.0 * A_jj)
                ## Update diagonal
                A_ii = A[i, i]
                y_i = np.sum(A[np.arange(i + 1) != i, i] * x[np.arange(i + 1) != i])
                x[i] = (-y_i + np.sqrt(y_i**2 + 4.0 * A_ii)) / (2.0 * A_ii)
                if np.max(np.abs(x - x_old)) < self.eps:
                    converged = True
                else:
                    iter = iter + 1
        return x
    
    ### Function to update permutation P
    def updateP(self, x, y, Z):
        """
        Updates permutation matrix in two steps:
        1) Projects into Bikhoff polytope
        2) Finds the closes permutation matrix

        Args:
        x,y - px1 Langrange mutipliers
        Z   = pxp Langrange multipier
        self.L - Cholesky factor
        self.P - pxp initial value for permutation

        Results:
        pxp permutation matrix
        """
 #       pdb.set_trace()
        iter = 0
        converged = False
        err = {}
        P_proj = self.P
        while((converged == False) and (iter <= self.niter)):
            P_old = P_proj
            oldObj = self.objFunc(P_old, self.L)
            P_old = P_old - self.s * self.gradient(P_old)
            # if self.constr is None:
            D_proj, x, y, Z, _ = self.projDS(P_old, x, y, Z)
            # else:
            #D_proj, x, y, Z, z, _ =  self.projDS_constr(P_old, x, y, Z, z)
            P_proj = P_proj + self.alpha * (D_proj - P_proj)
            newObj = self.objFunc(P_proj, self.L)
            err[iter] = abs(oldObj - newObj)
#            print("inner error is",err[iter], "and eps is", self.eps)
            if err[iter] < self.eps:
                converged = True
            else:
                iter = iter + 1
        ## Find the closest permutation matrix    
        self.DS = P_proj    
        if self.permest == "LSAP":
            return(self.permProj(P_proj))
        else:
            return(self.permProj(P_proj, self.P))
    ## Funciton to update lower traingular L
    def updateL(self):
        """
        Updates Cholesky factor L 
        using RC_i function rowwise
        Args:
        self.P - pxp permutation matrix
        self.X - nxp data

        Results:
        pxp lower triangular matrix
        """
#        pdb.set_trace()
        X_p = self.X @ np.transpose(self.P)
        S_p = np.cov(np.transpose(X_p))
        L = np.diag(1 / np.sqrt(np.diag(self.S)))
        for i in range(self.p):
            index = i + 1
            A = S_p[0: index, 0 :index]
            initx = self.L[i, 0 : index]
            L[i, 0 : index] = self.RC_i(i, A, initx)
        return L

    
    ## This is the main function 
                    
    def rrcfdag(self):
        """
        This is the main function that combines two steps:
        1) permutation matrix update
        2) lower triangular matrix update
        
        Args:
        Results:
        pxp permutation matrix P
        pxp lower trinagular matrix L
        """
  #      pdb.set_trace()
        ## Initialize
        #initL = np.diag(1 / np.sqrt(np.diag(self.S)))
        # if self.constr is None:
        #     z = 1
        # else:
        #     z = np.ones((len(self.constr) - 1, 1))
        x = np.ones((self.p,1))
        y = np.ones((self.p,1))
        Z = np.ones((self.p,self.p))
  #      P = self.P
        iter = 0
        converged = False
        while((converged == False) and (iter <= self.niter)):
            print("iter is", iter)
            P_old = self.P
            L_old = self.L
#            print("Update P")
            self.P       = self.updateP(x, y, Z)
#            print("P is updated in iteration ", iter )
#            print("Update L")
            self.L = self.updateL()
#            print("L is updated in iteration", iter)
            oldObj = self.objFunc(P_old, L_old)
            newObj = self.objFunc(self.P, self.L)
            self.error[iter] = abs(oldObj - newObj)
#            print("outer error is", self.error[iter])
            if self.error[iter] < self.eps:
                converged = True
            else:
                iter = iter + 1
#                print("iter is", iter)
        B_p =  1 * self.L / np.diag(self.L)
        B_p = B_p - np.eye(self.p)
        return(B_p)
