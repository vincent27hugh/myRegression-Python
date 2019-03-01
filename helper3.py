#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:45:12 2018
O
@author: huwei

For Part 3-2 maximal marginal likelihood 
"""


import numpy as np
from cvxopt import matrix, solvers
# Only for quadratic programming & linear programming


class FeatureTransform(object):
    """
    Feature transformation of input X:
    * X is input data matrix w/ dim of N by K (np.ndarray);
    * degree is the order of polynomial (int);
    * interaction is bool: 
    True if we consider interaction terms 
    (We only consider the case of degree = 2);
    """
    def __init__(self, X, degree = 2, interaction = False):
        self.X = X
        self.degree = degree
        self.interaction  = interaction
        
    def ftr_poly_mat(self):
        """
        Feature transformation function of polynomial w/o interaction term:
        output is matrix of feature transformation of 
            X w/ dim of K+1 by N (np.ndarray);
        """
        if type(self.X[0]) != np.ndarray:
            K = 1
        else:
            K = len(self.X[0])
        N = len(self.X)
        D = K * self.degree + 1
        Phi = np.zeros((D, N))   
        for j in range(0,N):
            temp = [1.0]
            for i in range(1,self.degree+1):
                temp = np.append(temp, np.power(self.X[j],i))
            Phi[:,j] = temp    
        return Phi
    
    def ftr_inter_mat(self):
        """
        Feature transformation function of polynomial w/ interaction only term:
        output is matrix of feature transformation of 
            X w/ dim of K+1 by N (np.ndarray);
        """
        if type(self.X[0]) != np.ndarray:
            K = 1
        else:
            K = len(self.X[0])
        N = len(self.X)
        D = int(1 + K/2.0 + K**2/2.0)
        Phi = np.zeros((D, N))
        for j in range(0,N):
            temp = [1.0]
            temp = np.append(temp, self.X[j])
            for k in range(0,K):
                for l in range(k+1,K):
                    temp = np.append(temp, self.X[j][k]*self.X[j][l])
            Phi[:,j] = temp    
        return Phi
    
    def Run(self):
        if self.interaction == True and self.degree == 2:
            Phi = self.ftr_inter_mat()
        else:
            Phi = self.ftr_poly_mat()
        return Phi
        

class LeastSquare(object):
    """
    Least square method:
    Phi is input feature matrix w/ dim of D * N (np.ndarray);
    Y is the output vector w/ dim of N (np.ndarray);
    """
    def __init__(self, Phi, Y):
        self.Y = Y
        self.Phi = Phi
        
    def para_est(self):
        """
        Parameter estimate:
        theta is estimated parameters w/ dim D (np.ndarray);
        """
        temp = self.Phi @ self.Phi.T
        theta = np.linalg.inv(temp) @ self.Phi @ self.Y
        self.theta = theta
        return theta
    
    def predict(self, new_Phi):
        """
        Prediction f_* for input new_Phi:
        new_X is input data vector w/ dim of N' (np.ndarray);
        f is learned fcn output vector w/ dim of N' (np.ndarray);
        """
        f = new_Phi.T @ self.theta
        return f
    
    def score(self, new_Phi, true_Y):
        """
        Mean Square Error btw learned fcn output and the true fcn outputs:
        new_Phi is input feature matrix w/ dim of D * N' (np.ndarray);
        true_Y is the true fcn output w/ dim N' (np.ndarray);
        MSE is the mean square error, scalar (np.float64);
        """
        f = self.predict(new_Phi)
        temp = f - true_Y
        MSE = sum(temp**2)/len(true_Y)
        return MSE
    
    def MAE(self, new_Phi, true_Y):
        """
        Mean Absolute Error btw learned fcn output and the true fcn outputs:
        new_Phi is input feature matrix w/ dim of D * N' (np.ndarray);
        true_Y is the true fcn output w/ dim N' (np.ndarray)
        MAE is the mean square error, scalar (np.float64)
        """
        f = self.predict(new_Phi)
        temp = f - true_Y
        MAE = sum(abs(temp))/len(true_Y)
        return MAE
        
        
class RegularizedLS(object):
    """
    Regularized least square method:
    Phi is input feature matrix w/ dim of D * N (np.ndarray);
    Y is the output vector w/ dim of N (np.ndarray);
    lamb is scalar parameter (np.float64);
    """
    def __init__(self, Phi, Y, lamb):
        self.Y = Y
        self.Phi = Phi
        self.lamb = lamb
        
    def para_est(self):
        """
        Parameter estimate:
        theta is estimated parameters w/ dim D (np.ndarray);
        """
        D = len(self.Phi)
        temp = self.Phi @ self.Phi.T + self.lamb * np.identity(D)
        theta = np.linalg.inv(temp) @ self.Phi @ self.Y
        self.theta = theta
        return theta
    
    def predict(self, new_Phi):
        """
        Prediction f_* for input new_Phi:
        new_Phi is input feature matrix w/ dim of D * N' (np.ndarray);
        f is output data vector w/ dim of N' (np.ndarray)
        """
        f = new_Phi.T @ self.theta
        return f
    
    def score(self, new_Phi, true_Y):
        """
        Mean Square Error btw learned fcn output and the true fcn outputs
        new_Phi is input feature matrix w/ dim of D * N' (np.ndarray);
        true_Y is the true fcn output w/ dim N' (np.ndarray)
        MSE is the mean square error, scalar (np.float64)
        """
        f = self.predict(new_Phi)
        temp = f - true_Y
        MSE = sum(temp**2)/len(true_Y)
        return MSE
    
    def MAE(self, new_Phi, true_Y):
        """
        Mean Absolute Error btw learned fcn output and the true fcn outputs:
        new_Phi is input feature matrix w/ dim of D * N' (np.ndarray);
        true_Y is the true fcn output w/ dim N' (np.ndarray)
        MAE is the mean square error, scalar (np.float64)
        """
        f = self.predict(new_Phi)
        temp = f - true_Y
        MAE = sum(abs(temp))/len(true_Y)
        return MAE

class LASSO(object):
    """
    LASSO (least absolute shrinkage and selection operator):
    Phi is input feature matrix w/ dim of D * N (np.ndarray);
    Y is the output vector w/ dim of N (np.ndarray);
    lamb is scalar parameter (np.float64);
    """
    def __init__(self, Phi, Y, lamb):
        self.Y = Y
        self.Phi = Phi
        self.lamb = lamb
        
    def para_est(self):
        """
        Parameter estimate:
        theta is estimated parameters w/ dim D (np.ndarray);
        """
        #N = len(X)
        D = len(self.Phi)
        temp1 = np.concatenate((self.Phi @ self.Phi.T, -self.Phi @ self.Phi.T ),axis = 1)
        temp2 = np.concatenate((-self.Phi @ self.Phi.T, self.Phi @ self.Phi.T ),axis = 1)
        P = np.concatenate((temp1, temp2), axis = 0)
        q = self.lamb * np.ones(2*D) - np.concatenate((self.Phi@self.Y, -self.Phi@self.Y), axis=0)
        G = - np.identity(2*D)
        h = np.zeros(2*D)
        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
        theta = np.array(sol['x'][0:D] - sol['x'][D:2*D])
        theta = np.concatenate(theta, axis=0)
        self.theta = theta
        return theta
    
    def predict(self, new_Phi):
        """
        Prediction f_* for input new_Phi:
        new_Phi is input feature matrix w/ dim of D * N' (np.ndarray);
        f is output data vector w/ dim of N' (np.ndarray)
        """
        f = new_Phi.T @ self.theta
        return f
    
    def score(self, new_Phi, true_Y):
        """
        Mean Square Error btw learned fcn output and the true fcn outputs
        new_X is input data vector w/ dim of N' (np.ndarray)
        true_Y is the true fcn output w/ dim N' (np.ndarray)
        MSE is the mean square error, scalar (np.float64)
        """
        f = self.predict(new_Phi)
        temp = f - true_Y
        MSE = sum(temp**2)/len(true_Y)
        return MSE
    
    def MAE(self, new_Phi, true_Y):
        """
        Mean Absolute Error btw learned fcn output and the true fcn outputs:
        new_Phi is input feature matrix w/ dim of D * N' (np.ndarray);
        true_Y is the true fcn output w/ dim N' (np.ndarray)
        MAE is the mean square error, scalar (np.float64)
        """
        f = self.predict(new_Phi)
        temp = f - true_Y
        MAE = sum(abs(temp))/len(true_Y)
        return MAE

class RobustReg(object):
    """
    Robust Regression:
    Phi is input feature matrix w/ dim of D * N (np.ndarray)
    Y is the output vector w/ dim of N (np.ndarray)
    """
    def __init__(self, Phi, Y):
        self.Y = Y
        self.Phi = Phi
        
    def para_est(self):
        """
        Parameter estimate:
        theta is estimated parameters w/ dim D (np.ndarray);
        """
        N = len(self.Y)
        D = len(self.Phi)
        c = np.concatenate((np.zeros(D),np.ones(N)), axis = 0)
        temp1 = np.concatenate((-self.Phi.T, -np.identity(N)), axis = 1)
        temp2 = np.concatenate((self.Phi.T, -np.identity(N)), axis = 1)
        G = np.concatenate((temp1, temp2),axis = 0)
        h = np.concatenate((-self.Y, self.Y), axis = 0)
        sol = solvers.lp(matrix(c, (D+N,1)), matrix(G, (2*N, N+D)), matrix(h,(2*N, 1)))
        theta = np.array(sol['x'][0:D])
        theta = np.concatenate(theta,axis=0)
        self.theta = theta
        return theta
    
    def predict(self, new_Phi):
        """
        Prediction f_* for input new_Phi
        new_Phi is input feature matrix w/ dim of D * N' (np.ndarray);
        f is output data vector w/ dim of N' (np.ndarray)
        """
        f = new_Phi.T @ self.theta
        return f
    
    def score(self, new_Phi, true_Y):
        """
        Mean Square Error btw learned fcn output and the true fcn outputs:
        new_Phi is input feature matrix w/ dim of D * N' (np.ndarray);
        true_Y is the true fcn output w/ dim N' (np.ndarray)
        MSE is the mean square error, scalar (np.float64)
        """
        f = self.predict(new_Phi)
        temp = f - true_Y
        MSE = sum(temp**2)/len(true_Y)
        return MSE
    
    def MAE(self, new_Phi, true_Y):
        """
        Mean Absolute Error btw learned fcn output and the true fcn outputs:
        new_Phi is input feature matrix w/ dim of D * N' (np.ndarray);
        true_Y is the true fcn output w/ dim N' (np.ndarray)
        MAE is the mean square error, scalar (np.float64)
        """
        f = self.predict(new_Phi)
        temp = f - true_Y
        MAE = sum(abs(temp))/len(true_Y)
        return MAE
    
class BayesianReg(object):
    """
    Bayesian Regression:
    Phi is input feature matrix w/ dim of D * N (np.ndarray);
    Y is the output vector w/ dim of N (np.ndarray);
    parameter theta has normal prior with zero mean,
        alpha is the std of normal dist. of theta (np.float64);
    Noise of regression has normal dist. w/ zero mean,
        sigma is the std of normal dist. of noise (np.float64).
    """
    def __init__(self, Phi, Y, alpha, sigma):
        self.Y = Y
        self.Phi = Phi
        self.alpha = alpha
        self.sigma = sigma
        
    def para_est(self):
        """
        Parameter estimate:
        theta is posterior estimated distribution, a vector w/ dim 1,000 (np.ndarray)
        Sigma_th is the covirance matrix of distribution of theta w/ dim D by D
        mu_th is the mean vector of distribution of theta w/ dim of D
        """
        D = len(self.Phi)
        I = np.identity(D)
        temp = 1.0/self.alpha*I + 1.0 / self.sigma**2 * self.Phi @ self.Phi.T
        Sigma_th = np.linalg.inv(temp)
        mu_th = 1.0/self.sigma**2 * Sigma_th @ self.Phi @ self.Y
        self.Sigma_th = Sigma_th
        self.mu_th = mu_th
        return mu_th, Sigma_th
    
    def predict(self, new_Phi):
        """
        Prediction f_* for input new_Phi
        new_Phi is input feature matrix w/ dim of D by N' (np.ndarray)
        new_Sigma is the covirance matrix of distribution of 
         theta w/ dim N' by N' (np.ndarray);
        new_mu is the mean vector of distribution of 
         theta w/ dim of N' (np.ndarray);
        """
        new_mu = new_Phi.T @ self.mu_th
        new_Sigma = new_Phi.T @ self.Sigma_th @ new_Phi
        return new_mu, new_Sigma
    
    def score(self, new_Phi, true_Y):
        """
        Mean Square Error btw learned fcn output and the true fcn outputs:
        new_Phi is input feature matrix w/ dim of D by N' (np.ndarray)
        true_Y is the true fcn output w/ dim N' (np.ndarray)
        MSE is the mean square error, scalar (np.float64)
        """
        [new_mu,_] = self.predict(new_Phi)
        temp = new_mu - true_Y
        MSE = sum(temp**2)/len(true_Y)
        return MSE
    
    def MAE(self, new_Phi, true_Y):
        """
        Mean Absolute Error btw learned fcn output and the true fcn outputs:
        new_Phi is input feature matrix w/ dim of D by N' (np.ndarray)
        true_Y is the true fcn output w/ dim N' (np.ndarray)
        MSE is the mean square error, scalar (np.float64)
        """
        [new_mu,_] = self.predict(new_Phi)
        temp = new_mu - true_Y
        MAE = sum(abs(temp))/len(true_Y)
        return MAE
    
    def marginal_likelihood(self):
        D = len(self.Phi)
        N = len(self.Y)
        temp1 = -D*np.log(self.alpha)/2.0-N*np.log(self.sigma**2)/2.0
        temp2 = self.Y - self.Phi.T @ self.mu_th
        temp3 = -temp2.T @ temp2/(2.0 * self.sigma**2)
        temp4 = -self.mu_th.T @ self.mu_th/(2.0*self.alpha)
        temp5 = -np.log(np.linalg.det(np.linalg.inv(self.Sigma_th)))/2.0
        temp6 = -N*np.log(2*np.pi)/2.0
        MLL = temp1+temp3+temp4+temp5+temp6
        return MLL
        
def SelectSubsamp(Sample, percentage):
    """
    Selecting subsample from Sample with percentage
    Sample is vector w/ dim N (np.ndarray)
    
    """
    np.random.seed(127)
    p = int(np.rint(len(Sample)*percentage))
    subsamp = np.random.choice(Sample, size = p, replace = False)
    return subsamp

class Regression(object):
    """
    Five type of Regression Methods:
        X is sample input w/ dim of N by K (np.ndarray):
            K = 1, if X is a vector;
            K >1, if X is a matrix;
        Y is the sample output vector w/ dim of N (np.ndarray);
        treuX is new sample input w/ dim of N' by K (np.ndarray):
            K = 1, if X is a vector;
            K >1, if X is a matrix;
        Y is the new sample output vector w/ dim of N' (np.ndarray);
        lamb_1 is scalar parameter for RLS (np.float64);
        lamb_2 is scalar parameter for LASSO (np.float64);
        alpha & sigma are scalar parameter for BR (np.float64);
        degree is the order of polynomial (int);
        interaction is bool: 
            True if we consider interaction terms 
            (We only consider the case of degree = 2);
    ---
    """
    def __init__(self, X, Y,
                 trueX, trueY, 
                 lamb_1, lamb_2,
                 alpha, sigma,
                 degree = 2, interaction = False):
        self.X = X
        self.Y = Y
        self.lamb_1 = lamb_1
        self.lamb_2 = lamb_2
        self.trueX =trueX
        self.trueY = trueY
        self.sigma = sigma
        self.alpha = alpha
        self.degree = degree
        self.interaction = interaction
        
    def Run(self, method):
        """
        """
        Phi_X = FeatureTransform(self.X, self.degree, self.interaction).Run()
        Phi_trueX = FeatureTransform(self.trueX, self.degree, self.interaction).Run()
        if method == 'LS':
            LS = LeastSquare(Phi_X, self.Y)
            theta = LS.para_est()
            f = LS.predict(Phi_trueX)
            MSE = LS.score(Phi_trueX, self.trueY)
            MAE = LS.MAE(Phi_trueX, self.trueY)
            return theta,f,MSE,MAE
        elif method == 'RLS':
            RLS = RegularizedLS(Phi_X, self.Y, self.lamb_1)
            theta = RLS.para_est()
            f = RLS.predict(Phi_trueX)
            MSE = RLS.score(Phi_trueX, self.trueY)
            MAE = RLS.MAE(Phi_trueX, self.trueY)
            return theta,f,MSE,MAE
        elif method == 'LASSO':
            lass = LASSO(Phi_X, self.Y, self.lamb_2)
            theta = lass.para_est()
            f = lass.predict(Phi_trueX)
            MSE = lass.score(Phi_trueX, self.trueY)
            MAE = lass.MAE(Phi_trueX, self.trueY)
            return theta,f,MSE,MAE
        elif method == 'RR':
            RR = RobustReg(Phi_X, self.Y)
            theta = RR.para_est()
            f = RR.predict(Phi_trueX)
            MSE = RR.score(Phi_trueX, self.trueY)
            MAE = RR.MAE(Phi_trueX, self.trueY)
            return theta,f,MSE,MAE
        elif method == 'BR':
            BR = BayesianReg(Phi_X, self.Y, self.alpha, self.sigma)
            [theta_mu, theta_sigma] = BR.para_est()
            [f_mu,f_sigma] = BR.predict(Phi_trueX)
            MSE = BR.score(Phi_trueX, self.trueY)
            MAE = BR.MAE(Phi_trueX, self.trueY)
            MLL = BR.marginal_likelihood()
            return theta_mu, theta_sigma,f_mu,f_sigma,MSE,MAE, MLL
        else:
            print('No such method!!!!')
            
  

        
    
    
