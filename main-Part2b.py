#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 13:34:15 2018

@author: huwei
"""

from helper2 import Regression

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
# std of noise
sigma = np.std(trainy)
# parameter for RLS
lamb_1 = 5
# parameter for LASSO
lamb_2 = 5
# parameters for BR 
# alpha = 0, strong belief 
# alpha = 1, uncertain about prior prob
alpha = 0.5
# Order of polynomial
degree = 2
interaction = False

save_path_data = os.path.join(os.getcwd(), 'Data')
save_path_fig = os.path.join(os.getcwd(), 'Plots')
######################################################
reg = Regression(trainx, trainy, testx, testy, \
                 lamb_1, lamb_2, alpha, sigma, degree, interaction)
######################################################
save_path_fig_b = os.path.join(save_path_fig,'2-b-1')
save_path_data_b = os.path.join(save_path_data, '2-b-1')
# LS
method = 'LS'
[theta_LS, f_LS, MSE_LS, MAE_LS] = reg.Run(method)
print(theta_LS)
print(MSE_LS)
print(MAE_LS)
# Plot
plt.scatter(f_LS,testy,c='g', s = 10, label = 'Result')
temp = np.linspace(min(testy),max(testy),100)
plt.plot(temp, temp, c = 'b', label = r'$y=x$')
plt.xlabel('Prediction Counts', fontsize=18)
plt.ylabel('True Counts', fontsize=18)
plt.legend(loc = 'best', fontsize=18)
plt.title(method, fontsize=18)
plt.grid(which = 'both')
fig = plt.gcf()
fig.set_size_inches(15, 15)
plt.savefig(os.path.join(save_path_fig_b,'plot_'+str(method)+".png"), dpi = 100)
plt.show()
######################################################
# RLS
method = 'RLS'
[theta_RLS, f_RLS, MSE_RLS, MAE_RLS] = reg.Run(method)
print(theta_RLS)
print(MSE_RLS)
print(MAE_RLS)
# Plot
plt.scatter(f_RLS,testy,c='g', s = 10, label = 'Result')
temp = np.linspace(min(testy),max(testy),100)
plt.plot(temp, temp, c = 'b', label = r'$y=x$')
plt.xlabel('Prediction Counts', fontsize=18)
plt.ylabel('True Counts', fontsize=18)
plt.legend(loc = 'best', fontsize=18)
plt.title(method, fontsize=18)
plt.grid(which = 'both')
fig = plt.gcf()
fig.set_size_inches(15, 15)
plt.savefig(os.path.join(save_path_fig_b,'plot_'+str(method)+".png"), dpi = 100)
plt.show()
######################################################
# LASSO
method = 'LASSO'
[theta_LASSO, f_LASSO, MSE_LASSO, MAE_LASSO] = reg.Run(method)
print(theta_LASSO)
print(MSE_LASSO)
print(MAE_LASSO)
# Plot
plt.scatter(f_LASSO,testy,c='g', s = 10, label = 'Result')
temp = np.linspace(min(testy),max(testy),100)
plt.plot(temp, temp, c = 'b', label = r'$y=x$')
plt.xlabel('Prediction Counts', fontsize=18)
plt.ylabel('True Counts', fontsize=18)
plt.legend(loc = 'best', fontsize=18)
plt.title(method, fontsize=18)
plt.grid(which = 'both')
fig = plt.gcf()
fig.set_size_inches(15, 15)
plt.savefig(os.path.join(save_path_fig_b,'plot_'+str(method)+".png"), dpi = 100)
plt.show()
######################################################
# RR
method = 'RR'
[theta_RR, f_RR, MSE_RR, MAE_RR] = reg.Run(method)
print(theta_RR)
print(MSE_RR)
print(MAE_RR)
# Plot
plt.scatter(f_RR,testy,c='g', s = 10, label = 'Result')
temp = np.linspace(min(testy),max(testy),100)
plt.plot(temp, temp, c = 'b', label = r'$y=x$')
plt.xlabel('Prediction Counts', fontsize=18)
plt.ylabel('True Counts', fontsize=18)
plt.legend(loc = 'best', fontsize=18)
plt.title(method, fontsize=18)
plt.grid(which = 'both')
fig = plt.gcf()
fig.set_size_inches(15, 15)
plt.savefig(os.path.join(save_path_fig_b,'plot_'+str(method)+".png"), dpi = 100)
plt.show()
######################################################
# BR
method = 'BR'
[theta_BR_mu, theta_BR_sigma, f_BR_mu, f_BR_sigma, MSE_BR,MAE_BR] = reg.Run(method)
print(theta_BR_mu)
print(MSE_BR)
print(MAE_BR)
# Plot
plt.scatter(f_BR_mu,testy,facecolors='none', edgecolors='g', s = np.diag(f_BR_sigma)*20, label = 'Result w/ size')

temp = np.linspace(min(testy),max(testy),100)
plt.plot(temp, temp, c = 'b', label = r'$y=x$')
plt.xlabel('Prediction Counts', fontsize=18)
plt.ylabel('True Counts', fontsize=18)
plt.legend(loc = 'best', fontsize=18)
plt.title(method, fontsize=18)
plt.grid(which = 'both')
fig = plt.gcf()
fig.set_size_inches(15, 15)
plt.savefig(os.path.join(save_path_fig_b,'plot_'+str(method)+".png"), dpi = 100)
plt.show()
######################################################
# Save dat

thetas = pd.DataFrame()
fs = pd.DataFrame()
errors = pd.DataFrame()

thetas = pd.concat([thetas, pd.DataFrame(theta_LS)],axis=1)
thetas = pd.concat([thetas, pd.DataFrame(theta_RLS)],axis=1)
thetas = pd.concat([thetas, pd.DataFrame(theta_LASSO)],axis=1)
thetas = pd.concat([thetas, pd.DataFrame(theta_RR)],axis=1)
thetas = pd.concat([thetas, pd.DataFrame(theta_BR_mu)],axis=1)
thetas = pd.concat([thetas, pd.DataFrame(np.diag(theta_BR_sigma))],axis=1)

thetas.columns = ['LS', 'RLS', 'LASSO', 'RR', 'BR-mu', 'BR-sigma']
thetas.to_csv(os.path.join(save_path_data_b,'thetas.csv'))

fs = pd.concat([fs, pd.DataFrame(f_LS)],axis=1)
fs = pd.concat([fs, pd.DataFrame(f_RLS)],axis=1)
fs = pd.concat([fs, pd.DataFrame(f_LASSO)],axis=1)
fs = pd.concat([fs, pd.DataFrame(f_RR)],axis=1)
fs = pd.concat([fs, pd.DataFrame(f_BR_mu)],axis=1)
fs = pd.concat([fs, pd.DataFrame(np.diag(f_BR_sigma))],axis=1)

fs.columns = ['LS', 'RLS', 'LASSO', 'RR', 'BR-mu', 'BR-sigma']
fs.to_csv(os.path.join(save_path_data_b,'fs.csv'))

MSEs = []
MSEs.append(MSE_LS)
MSEs.append(MSE_RLS)
MSEs.append(MSE_LASSO)
MSEs.append(MSE_RR)
MSEs.append(MSE_BR)

errors = pd.concat([errors, pd.DataFrame(MSEs)], axis = 1)

MAEs = []
MAEs.append(MAE_LS)
MAEs.append(MAE_RLS)
MAEs.append(MAE_LASSO)
MAEs.append(MAE_RR)
MAEs.append(MAE_BR)

errors = pd.concat([errors, pd.DataFrame(MAEs)], axis = 1)

errors.columns = ['MSE', 'MAE']
errors.index = ['LS', 'RLS', 'LASSO', 'RR', 'BR']
errors.to_csv(os.path.join(save_path_data_b,'errors.csv'))
print(errors)