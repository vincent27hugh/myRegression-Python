#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 13:34:15 2018

@author: huwei
"""

from helper3 import FeatureTransform,BayesianReg

import numpy as np
# Only for quadratic programming & linear programming
import matplotlib.pyplot as plt
import pandas as pd
import os

import itertools


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
save_path_fig_a = os.path.join(save_path_fig,'3-b-3')
save_path_data_a = os.path.join(save_path_data, '3-b-3')
######################################################
# Estimating hyperparameters
# Dictionary with parameter names (sting) and lists of parameter settings;
paras  = {'lamb_1':[0.1], \
          'lamb_2':[0.1], \
          'alpha': np.linspace(0.1,0.9,9), \
          'sigma': np.linspace(5,15,11), \
          'degree': [1], \
          'interaction': [False]}
######################################################
# https://codereview.stackexchange.com/questions/171173/list-all-possible-permutations-from-a-python-dictionary-of-lists
keys, values = zip(*paras.items())
my_paras = [dict(zip(keys, v)) for v in itertools.product(*values)]
print(len(my_paras))
MLLs = []
######################################################
for i in range(0,len(my_paras)):
    print(str(int(1e4*(i+1)/len(my_paras))/100)+'% completed...')
    my_para = my_paras[i]
    MSEs = pd.DataFrame()
    MAEs = pd.DataFrame()
    # BR
    Phi_X = FeatureTransform(trainx, my_para['degree'], my_para['interaction']).Run()
    BR = BayesianReg(Phi_X, trainy, my_para['alpha'], my_para['sigma'])
    BR.para_est()
    MLL = BR.marginal_likelihood()
    ######################################################
    # Save dat
    MLLs.append(MLL)


# The index of my_paras for the best paras
maxidx = np.argmax(np.array(MLLs), axis = 0)

best_para_MLL = [my_paras[maxidx]]
best_MLL = MLLs[maxidx]

# Save list
with open(os.path.join(save_path_data_a,'Best_parameters_MLL.txt'), 'w') as filehandle:  
    for listitem in best_para_MLL:
        filehandle.write('%s\n' % listitem)
        
with open(os.path.join(save_path_data_a,'MLLs.txt'), 'w') as filehandle:  
    for listitem in MLLs:
        filehandle.write('%s\n' % listitem)
        