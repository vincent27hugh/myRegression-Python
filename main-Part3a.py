#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 13:34:15 2018

@author: huwei
"""

from helper2 import FeatureTransform,RegularizedLS, LASSO, BayesianReg

import numpy as np
# Only for quadratic programming & linear programming
import matplotlib.pyplot as plt
import pandas as pd
import os


##############################################################################
# Part 2-b
##############################################################################
# Load data 
# Training input
trainx = np.loadtxt('PA-1-data-text/count_data_trainx.txt')
trainx = trainx.T
# Training output
trainy = np.loadtxt('PA-1-data-text/count_data_trainy.txt')
trainy = trainy.T
# Testing input
testx = np.loadtxt('PA-1-data-text/count_data_testx.txt')
testx = testx.T
# Output values for the true functions
testy = np.loadtxt('PA-1-data-text/count_data_testy.txt')
testy = testy.T
print(trainx.shape)
print(trainy.shape)
print(testx.shape)
print(testy.shape)

######################################################
# Set parameters
#
save_path_data = os.path.join(os.getcwd(), 'Data')
save_path_fig = os.path.join(os.getcwd(), 'Plots')
######################################################
save_path_fig_a = os.path.join(save_path_fig,'3-a-2')
save_path_data_a = os.path.join(save_path_data, '3-a-2')
######################################################
# Number of folds of Cross-Validation
N = 3
# seperate training set into N equal partitions
index = np.array(range(0,len(trainx)))
np.random.shuffle(index)
# list of partitions splited
partitions = np.array_split(index,N)
######################################################
# Tuning lambda_1 and lambda_2
######################################################
# Estimating hyperparameters
# Dictionary with parameter names (sting) and lists of parameter settings;
paras  = {'lamb_1': np.linspace(1e-3,1e1,200), \
          'lamb_2':np.linspace(1e-3,1e1,200), \

          'degree': [2], \
          'interaction': [True]}
###################################################### 
MSEs_RLS = []
MAEs_RLS = []
######################################################
for i in range(0,len(paras['lamb_1'])):
    print(str(int(1e4*(i+1)/len(paras['lamb_1']))/100)+'% completed...')
    my_para = paras['lamb_1'][i]
    MSE_RLS = []
    MAE_RLS = []
    for j in range(0,N):
        # New training set index
        tr_index = np.array([x for x in index if x not in partitions[j]])
        # New test set index
        te_index = partitions[j]
        
        ######################################################
        # RLS
        Phi_X = FeatureTransform(trainx[tr_index], \
                                 paras['degree'][0], \
                                 paras['interaction'][0]).Run()
        Phi_trueX = FeatureTransform(trainx[te_index], \
                                     paras['degree'][0], \
                                     paras['interaction'][0]).Run()
        RLS = RegularizedLS(Phi_X, trainy[tr_index], my_para)
        RLS.para_est()
        MSE = RLS.score(Phi_trueX, trainy[te_index])
        MAE = RLS.MAE(Phi_trueX, trainy[te_index])
        ######################################################
        # Save dat
        
        MSE_RLS.append(MSE)
        MAE_RLS.append(MAE)
        
        
    MSEs_RLS.append(np.sum(MSE_RLS))
    MAEs_RLS.append(np.sum(MAE_RLS))

# Error dataframe
df_MSEs_RLS = pd.DataFrame(MSEs_RLS)
df_MSEs_RLS.columns = ['MSE_RLS']
df_MAEs_RLS = pd.DataFrame(MAEs_RLS)
df_MAEs_RLS.columns = ['MAE_RLS']
errors = pd.DataFrame()
errors = pd.concat([df_MSEs_RLS,df_MAEs_RLS],axis=1)
errors.index = paras['lamb_1']
errors.to_csv(os.path.join(save_path_data_a,'errors_RLS_cv.csv'))

minidx1 = np.argmin(MSEs_RLS)
minidx2 = np.argmin(MAEs_RLS)

minerror = pd.DataFrame()
minerror = pd.concat([minerror,\
                      pd.DataFrame([paras['lamb_1'][minidx1],\
                                    paras['lamb_1'][minidx2]], \
                                    columns = ['RLS'])],axis=1)
#####
plt.figure
plt.plot(paras['lamb_1'],MSEs_RLS, label = 'result')
plt.axvline(x=paras['lamb_1'][minidx1], \
            color = 'r', \
            linestyle = '--',\
            label = 'minimum MSE')
plt.xlabel(r'$\lambda$',  fontsize = 18)
plt.ylabel('MSE', fontsize = 18)
plt.title('RLS', fontsize = 18)
plt.legend(loc = 'best', fontsize=18)
fig = plt.gcf()
fig.set_size_inches(15, 15)
plt.savefig(os.path.join(save_path_fig_a,'plot_RLS_cvMSE'+".png"), dpi = 100)
plt.show()

plt.figure
plt.plot(paras['lamb_1'],MAEs_RLS, label = 'result')
plt.axvline(x=paras['lamb_1'][minidx2], \
            color = 'r', \
            linestyle = '--',\
            label = 'minimum MAE')
plt.xlabel(r'$\lambda$',  fontsize = 18)
plt.ylabel('MAE', fontsize = 18)
plt.title('RLS', fontsize = 18)
plt.legend(loc = 'best', fontsize=18)
fig = plt.gcf()
fig.set_size_inches(15, 15)
plt.savefig(os.path.join(save_path_fig_a,'plot_RLS_cvMAE'+".png"), dpi = 100)

plt.show()

###################################################### 
MSEs_LASSO = []
MAEs_LASSO = []
######################################################
for i in range(0,len(paras['lamb_2'])):
    print(str(int(1e4*(i+1)/len(paras['lamb_2']))/100)+'% completed...')
    my_para = paras['lamb_2'][i]
    MSE_LASSO = []
    MAE_LASSO = []
    for j in range(0,N):
        # New training set index
        tr_index = np.array([x for x in index if x not in partitions[j]])
        # New test set index
        te_index = partitions[j]
        
        ######################################################
        # RLS
        Phi_X = FeatureTransform(trainx[tr_index], \
                                 paras['degree'][0], \
                                 paras['interaction'][0]).Run()
        Phi_trueX = FeatureTransform(trainx[te_index], \
                                     paras['degree'][0], \
                                     paras['interaction'][0]).Run()
        lass = LASSO(Phi_X, trainy[tr_index], my_para)
        lass.para_est()
        MSE = lass.score(Phi_trueX, trainy[te_index])
        MAE = lass.MAE(Phi_trueX, trainy[te_index])
        ######################################################
        # Save dat
        
        MSE_LASSO.append(MSE)
        MAE_LASSO.append(MAE)
        
        
    MSEs_LASSO.append(np.sum(MSE_LASSO))
    MAEs_LASSO.append(np.sum(MAE_LASSO))

# Error dataframe
df_MSEs_LASSO = pd.DataFrame(MSEs_LASSO)
df_MSEs_LASSO.columns = ['MSE_LASSO']
df_MAEs_LASSO = pd.DataFrame(MAEs_LASSO)
df_MAEs_LASSO.columns = ['MAE_LASSO']
errors = pd.DataFrame()
errors = pd.concat([df_MSEs_LASSO,df_MAEs_LASSO],axis=1)
errors.index = paras['lamb_2']

minidx1 = np.argmin(MSEs_LASSO)
minidx2 = np.argmin(MAEs_LASSO)

minerror = pd.concat([minerror,\
                      pd.DataFrame([paras['lamb_2'][minidx1],\
                                    paras['lamb_2'][minidx2]], \
                                    columns = ['LASSO'])],axis=1)
#####
plt.figure
plt.plot(paras['lamb_2'],MSEs_LASSO, label = 'result')
plt.axvline(x=paras['lamb_2'][minidx1], \
            color = 'r', \
            linestyle = '--',\
            label = 'minimum MSE')
plt.xlabel(r'$\lambda$',  fontsize = 18)
plt.ylabel('MSE', fontsize = 18)
plt.title('LASSO', fontsize = 18)
plt.legend(loc = 'best', fontsize=18)
fig = plt.gcf()
fig.set_size_inches(15, 15)
plt.savefig(os.path.join(save_path_fig_a,'plot_LASSO_cvMSE'+".png"), dpi = 100)
plt.show()

plt.figure
plt.plot(paras['lamb_2'],MAEs_LASSO, label = 'result')
plt.axvline(x=paras['lamb_2'][minidx2], \
            color = 'r', \
            linestyle = '--',\
            label = 'minimum MAE')
plt.xlabel(r'$\lambda$',  fontsize = 18)
plt.ylabel('MAE', fontsize = 18)
plt.title('LASSO', fontsize = 18)
plt.legend(loc = 'best', fontsize=18)
fig = plt.gcf()
fig.set_size_inches(15, 15)
plt.savefig(os.path.join(save_path_fig_a,'plot_LASSO_cvMAE'+".png"), dpi = 100)
plt.show()

###################################################### 
minerror.index = ['para with min MSE', 'para withh min MAE']
minerror.to_csv(os.path.join(save_path_data_a,'bestpara_cv.csv'))


