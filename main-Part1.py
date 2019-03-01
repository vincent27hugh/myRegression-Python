#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 13:34:15 2018

@author: huwei
"""

from helper import Regression, SelectSubsamp

import numpy as np
# Only for quadratic programming & linear programming
import matplotlib.pyplot as plt
import pandas as pd
import os


##############################################################################
# Part 1
##############################################################################
# Load data 
# Sample input values
sampx = np.loadtxt('PA-1-data-text/polydata_data_sampx.txt')
# Input values for the true functions
polyx = np.loadtxt('PA-1-data-text/polydata_data_polyx.txt')
# Sample output values
sampy = np.loadtxt('PA-1-data-text/polydata_data_sampy.txt')
# Output values for the true functions
polyy = np.loadtxt('PA-1-data-text/polydata_data_polyy.txt')
# true value of theta
thtrue = np.loadtxt('PA-1-data-text/polydata_data_thtrue.txt')

plt.plot(polyx,polyy,'r')
plt.scatter(sampx,sampy)
plt.show()
######################################################
# Set parameters
# Known parameter: std of noise
sigma = np.sqrt(5)
# parameter for RLS
lamb_1 = 5
# parameter for LASSO
lamb_2 = 5
# parameters for BR 
# alpha = 0, strong belief 
# alpha = 1, uncertain about prior prob
alpha = 0.5
# Order of polynomial
degree = 5

save_path_data = os.path.join(os.getcwd(), 'Data','1')
save_path_fig = os.path.join(os.getcwd(), 'Plots','1')
######################################################
reg = Regression(sampx, sampy, polyx, polyy, lamb_1, lamb_2, alpha, sigma, degree)
######################################################
# (b)
save_path_fig_b = os.path.join(save_path_fig,'b')
save_path_data_b = os.path.join(save_path_data,'b')
# LS
method = 'LS'
[theta_LS, f_LS, MSE_LS] = reg.Run(method)
print(theta_LS)
print(MSE_LS)
reg.Plot(method, save_path_fig_b)
######################################################
# RLS
method = 'RLS'
[theta_RLS, f_RLS, MSE_RLS] = reg.Run(method)
print(theta_RLS)
print(MSE_RLS)
reg.Plot(method,save_path_fig_b)
######################################################
# LASSO
method = 'LASSO'
[theta_LASSO, f_LASSO, MSE_LASSO] = reg.Run(method)
print(theta_LASSO)
print(MSE_LASSO)
reg.Plot(method,save_path_fig_b)
######################################################
# RR
method = 'RR'
[theta_RR, f_RR, MSE_RR] = reg.Run(method)
print(theta_RR)
print(MSE_RR)
reg.Plot(method,save_path_fig_b)
######################################################
# BR
method = 'BR'
[theta_BR_mu, theta_BR_sigma, f_BR_mu, f_BR_sigma, MSE_BR] = reg.Run(method)
print(theta_BR_mu)
print(MSE_BR)
reg.Plot(method,save_path_fig_b)

# Save data
MSE = []
MSE.append(MSE_LS)
MSE.append(MSE_RLS)
MSE.append(MSE_LASSO)
MSE.append(MSE_RR)
MSE.append(MSE_BR)
np.savetxt(os.path.join(save_path_data_b,'MSE.csv'),MSE, delimiter =',')

theta = pd.DataFrame()
theta = pd.concat([theta,pd.DataFrame(theta_LS.T,columns=['LS'])],axis=1)
theta = pd.concat([theta,pd.DataFrame(theta_RLS.T,columns=['RLS'])],axis=1)
theta = pd.concat([theta,pd.DataFrame(theta_LASSO.T,columns=['LASSO'])],axis=1)
theta = pd.concat([theta,pd.DataFrame(theta_RR.T,columns=['RR'])],axis=1)
theta = pd.concat([theta,pd.DataFrame(theta_BR_mu.T,columns=['BR_mu'])],axis=1)
theta = pd.concat([theta,pd.DataFrame(np.diag(theta_BR_sigma).T,columns=['BR_sigma'])],axis=1)
theta.to_csv(os.path.join(save_path_data_b,'theta.csv'))

f = pd.DataFrame()
f = pd.concat([f,pd.DataFrame(f_LS.T,columns=['LS'])],axis=1)
f = pd.concat([f,pd.DataFrame(f_RLS.T,columns=['RLS'])],axis=1)
f = pd.concat([f,pd.DataFrame(f_LASSO.T,columns=['LASSO'])],axis=1)
f = pd.concat([f,pd.DataFrame(f_RR.T,columns=['RR'])],axis=1)
f = pd.concat([f,pd.DataFrame(f_BR_mu.T,columns=['BR_mu'])],axis=1)
f = pd.concat([f,pd.DataFrame(np.diag(f_BR_sigma).T,columns=['BR_sigma'])],axis=1)
f.to_csv(os.path.join(save_path_data_b,'f.csv'))
######################################################
# (c)
save_path_fig_c = os.path.join(save_path_fig,'c')
save_path_data_c = os.path.join(save_path_data,'c')
######################################################
# Plot
v_per = np.array(range(15,105,5))/100
methods = ['LS', 'RLS', 'LASSO', 'RR', 'BR']
for i in range(0,len(v_per)):
    per = v_per[i]
    print(str(per*100)+'% of subsample...')
    save_path_fig_per = os.path.join(save_path_fig_c, 
                                     'per='+str(int(np.rint(per*100))) + '/')
    if not os.path.exists(save_path_fig_per):
        os.makedirs(save_path_fig_per)
    for method in methods:
        print('Method is '+method)
        #np.random.seed(127)
        Sample = np.array(range(len(sampx)))
        p = int(np.rint(len(Sample)*per))
        subindex = np.random.choice(Sample, size = p, replace = False)
        subsampx = sampx[subindex]
        subsampy = sampy[subindex]
        reg = Regression(subsampx, subsampy, polyx, polyy, 
                         lamb_1, lamb_2, alpha, sigma, degree)
        reg.Plot(method,save_path_fig_per)
######################################################
# Generate Data
v_per = np.array(range(15,101,1))/100
methods = ['LS', 'RLS', 'LASSO', 'RR', 'BR']
thetas = []
fs = []
MSEs = pd.DataFrame()
# Number of trials 
N_trials = 10
for i in range(0,len(v_per)):
    per = v_per[i]
    print(str(per*100)+'% of subsample...')
    theta = pd.DataFrame()
    f = pd.DataFrame()
    MSE = []
    for method in methods:
        print('Method is '+method)
        f_trial = pd.DataFrame()
        MSE_trial = []
        for t in range(0,N_trials):
            Sample = np.array(range(len(sampx)))
            p = int(np.rint(len(Sample)*per))
            subindex = np.random.choice(Sample, size = p, replace = False)
            subsampx = sampx[subindex]
            subsampy = sampy[subindex]
            reg = Regression(subsampx, subsampy, polyx, polyy, 
                             lamb_1, lamb_2, alpha, sigma, degree)
            if method != 'BR':
                #[theta_m, f_m, MSE_m] = reg.Run(method)
                [_, f_t, MSE_t] = reg.Run(method)
            else:
                #[theta_m,_,f_m,_,MSE_m] = reg.Run(method)
                [_,_,f_t,_,MSE_t] = reg.Run(method)
                
            MSE_trial.append(MSE_t)
            f_trial = pd.concat([f_trial, pd.DataFrame(f_t)],axis = 1)
            MSE_m = sum(MSE_trial)/float(N_trials)
            f_m = f_trial.mean(axis = 1)
            
        #theta = pd.concat([theta, pd.DataFrame(theta_m)],axis = 1)
        f = pd.concat([f, pd.DataFrame(f_m)],axis = 1)
        MSE.append(MSE_m)
    
    #theta.columns = methods
    f.columns = methods
    #theta.to_csv(os.path.join(save_path_data,'theta_per='+str(int(np.rint(per*100)))+'.csv'))
    #thetas.append(theta)
    f.to_csv(os.path.join(save_path_data_c,'f.csv'))
    fs.append(f)
    temp = pd.DataFrame(MSE)
    temp.columns = ['Per='+str(per)]
    MSEs = pd.concat([MSEs, temp], axis =1)
    
MSEs.index = methods
MSEs.to_csv(os.path.join(save_path_data_c,'MSE.csv'))
print(MSEs)
######################################################
# Plot
for method in methods:
    plt.figure()
    plt.plot(polyx,polyy,c='r', label='true value')
    for i in range(0,len(v_per),15):
        
        plt.plot(polyx, fs[i][method] , 
                 marker = '.', 
                 linestyle = ':', 
                 label = 'estimated, per='+str(v_per[i]*100)+'%')   
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.title('Method='+str(method), fontsize=18)
    plt.legend(loc = 'best', fontsize=16)
    
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(os.path.join(save_path_fig_c,'plot_per_'+str(method)+".png"), dpi = 100)
    plt.show()
# Error vs Size
plt.figure() 
for i in range(0,len(methods)):
    plt.plot(v_per, MSEs.iloc[i] , 
                 label = methods[i])

plt.xlabel('Percentage', fontsize=18)
plt.ylabel('MSE', fontsize=18)
plt.title('Error vs training size', fontsize=18)
plt.legend(loc = 'best', fontsize=16)

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.savefig(os.path.join(save_path_fig_c,'plot_per-MSE.png'), dpi = 100)
plt.show()

plt.figure() 
for i in range(0,len(methods)):
    plt.plot(v_per, np.log(MSEs.iloc[i]) , 
                 label = methods[i])

plt.xlabel('Percentage', fontsize=18)
plt.ylabel('log(MSE)', fontsize=18)
plt.title('Error vs training size', fontsize=18)
plt.legend(loc = 'best', fontsize=16)

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.savefig(os.path.join(save_path_fig_c,'plot_per-logMSE.png'), dpi = 100)
plt.show()
    
######################################################         
# (d)
# Add outlier
Sample = np.array(range(len(sampy)))
#np.random.seed(127)
p = int(np.rint(len(Sample)*0.05))
index = np.random.choice(Sample, size = p, replace = False)
sampy2 = sampy
sampy2[index] = sampy2[index] * 10

reg = Regression(sampx, sampy2, polyx, polyy, lamb_1, lamb_2, alpha, sigma, degree)
######################################################
save_path_fig_d = os.path.join(save_path_fig,'d')
save_path_data_d = os.path.join(save_path_data,'d')
# LS
method = 'LS'
[theta_LS, f_LS, MSE_LS] = reg.Run(method)
print(theta_LS)
print(MSE_LS)
reg.Plot(method, save_path_fig_d)
######################################################
# RLS
method = 'RLS'
[theta_RLS, f_RLS, MSE_RLS] = reg.Run(method)
print(theta_RLS)
print(MSE_RLS)
reg.Plot(method, save_path_fig_d)
######################################################
# LASSO
method = 'LASSO'
[theta_LASSO, f_LASSO, MSE_LASSO] = reg.Run(method)
print(theta_LASSO)
print(MSE_LASSO)
reg.Plot(method, save_path_fig_d)
######################################################
# RR
method = 'RR'
[theta_RR, f_RR, MSE_RR] = reg.Run(method)
print(theta_RR)
print(MSE_RR)
reg.Plot(method, save_path_fig_d)
######################################################
# BR
method = 'BR'
[theta_BR_mu, theta_BR_sigma, f_BR_mu, f_BR_sigma, MSE_BR] = reg.Run(method)
print(theta_BR_mu)
print(MSE_BR)
reg.Plot(method, save_path_fig_d)
######################################################
# Save data
MSE = []
MSE.append(MSE_LS)
MSE.append(MSE_RLS)
MSE.append(MSE_LASSO)
MSE.append(MSE_RR)
MSE.append(MSE_BR)
np.savetxt(os.path.join(save_path_data_d,'MSE.csv'),MSE, delimiter =',')

theta = pd.DataFrame()
theta = pd.concat([theta,pd.DataFrame(theta_LS.T,columns=['LS'])],axis=1)
theta = pd.concat([theta,pd.DataFrame(theta_RLS.T,columns=['RLS'])],axis=1)
theta = pd.concat([theta,pd.DataFrame(theta_LASSO.T,columns=['LASSO'])],axis=1)
theta = pd.concat([theta,pd.DataFrame(theta_RR.T,columns=['RR'])],axis=1)
theta = pd.concat([theta,pd.DataFrame(theta_BR_mu.T,columns=['BR_mu'])],axis=1)
theta = pd.concat([theta,pd.DataFrame(np.diag(theta_BR_sigma).T,columns=['BR_sigma'])],axis=1)
theta.to_csv(os.path.join(save_path_data_d,'theta.csv'))

f = pd.DataFrame()
f = pd.concat([f,pd.DataFrame(f_LS.T,columns=['LS'])],axis=1)
f = pd.concat([f,pd.DataFrame(f_RLS.T,columns=['RLS'])],axis=1)
f = pd.concat([f,pd.DataFrame(f_LASSO.T,columns=['LASSO'])],axis=1)
f = pd.concat([f,pd.DataFrame(f_RR.T,columns=['RR'])],axis=1)
f = pd.concat([f,pd.DataFrame(f_BR_mu.T,columns=['BR_mu'])],axis=1)
f = pd.concat([f,pd.DataFrame(np.diag(f_BR_sigma).T,columns=['BR_sigma'])],axis=1)
f.to_csv(os.path.join(save_path_data_d,'f.csv'))
######################################################
# (e)
save_path_fig_e = os.path.join(save_path_fig,'e')
save_path_data_e = os.path.join(save_path_data,'e')

# Order of polynomial
degree = 10
######################################################
reg = Regression(sampx, sampy, polyx, polyy, lamb_1, lamb_2, alpha, sigma, degree)
######################################################
# LS
method = 'LS'
[theta_LS, f_LS, MSE_LS] = reg.Run(method)
print(theta_LS)
print(MSE_LS)
reg.Plot(method, save_path_fig_e)
######################################################
# RLS
method = 'RLS'
[theta_RLS, f_RLS, MSE_RLS] = reg.Run(method)
print(theta_RLS)
print(MSE_RLS)
reg.Plot(method,save_path_fig_e)
######################################################
# LASSO
method = 'LASSO'
[theta_LASSO, f_LASSO, MSE_LASSO] = reg.Run(method)
print(theta_LASSO)
print(MSE_LASSO)
reg.Plot(method,save_path_fig_e)
######################################################
# RR
method = 'RR'
[theta_RR, f_RR, MSE_RR] = reg.Run(method)
print(theta_RR)
print(MSE_RR)
reg.Plot(method,save_path_fig_e)
######################################################
# BR
method = 'BR'
[theta_BR_mu, theta_BR_sigma, f_BR_mu, f_BR_sigma, MSE_BR] = reg.Run(method)
print(theta_BR_mu)
print(MSE_BR)
reg.Plot(method,save_path_fig_e)
#####################################################
# Save data
MSE = []
MSE.append(MSE_LS)
MSE.append(MSE_RLS)
MSE.append(MSE_LASSO)
MSE.append(MSE_RR)
MSE.append(MSE_BR)
np.savetxt(os.path.join(save_path_data_e,'MSE.csv'),MSE, delimiter =',')

theta = pd.DataFrame()
theta = pd.concat([theta,pd.DataFrame(theta_LS.T,columns=['LS'])],axis=1)
theta = pd.concat([theta,pd.DataFrame(theta_RLS.T,columns=['RLS'])],axis=1)
theta = pd.concat([theta,pd.DataFrame(theta_LASSO.T,columns=['LASSO'])],axis=1)
theta = pd.concat([theta,pd.DataFrame(theta_RR.T,columns=['RR'])],axis=1)
theta = pd.concat([theta,pd.DataFrame(theta_BR_mu.T,columns=['BR_mu'])],axis=1)
theta = pd.concat([theta,pd.DataFrame(np.diag(theta_BR_sigma).T,columns=['BR_sigma'])],axis=1)
theta.to_csv(os.path.join(save_path_data_e,'theta.csv'))

f = pd.DataFrame()
f = pd.concat([f,pd.DataFrame(f_LS.T,columns=['LS'])],axis=1)
f = pd.concat([f,pd.DataFrame(f_RLS.T,columns=['RLS'])],axis=1)
f = pd.concat([f,pd.DataFrame(f_LASSO.T,columns=['LASSO'])],axis=1)
f = pd.concat([f,pd.DataFrame(f_RR.T,columns=['RR'])],axis=1)
f = pd.concat([f,pd.DataFrame(f_BR_mu.T,columns=['BR_mu'])],axis=1)
f = pd.concat([f,pd.DataFrame(np.diag(f_BR_sigma).T,columns=['BR_sigma'])],axis=1)
f.to_csv(os.path.join(save_path_data_e,'f.csv'))