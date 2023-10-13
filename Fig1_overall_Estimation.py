#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:04:04 2021

@author: macbjmu
"""
import numpy as np
from scipy.stats import nbinom
from scipy.optimize import fsolve
from scipy import special
import scipy.integrate as integrate
from scipy.stats import gamma
import pymc as pm
import arviz as az
import csv
import pandas as pd
from matplotlib import cm
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages

def p80_cal(tmpK,tmpR):
    t = nbinom.ppf(0.2,n=tmpK+1, p = tmpK/(tmpR+tmpK));
    p80 = 1 - nbinom.cdf(t+1,n=tmpK, p = tmpK/(tmpR+tmpK));
    return p80;

def gamma2discrete(mean_GT,sd_GT,MaxInfctPrd):
    
    '''
    Parameters
    ----------
    mean_GT : float
        The mean of generation time.
    sd_GT : float
        the sd of generation time.
    MaxInfctPrd : int
        the maximum length of infectious period.

    Returns
    w_s, the infectivity profile

    '''

    shape_para = (mean_GT/sd_GT)**2;
    rate_para  = shape_para/mean_GT
    
    wRatio = np.zeros(MaxInfctPrd+1)
    for tmps in range(1,len(wRatio)):
        wRatio[tmps] = gamma.cdf(tmps+0.5,shape_para,scale = 1/rate_para) - gamma.cdf(tmps-0.5,shape_para,scale = 1/rate_para)
    
    return wRatio/np.sum(wRatio);



def Inci2Epi_Para(outbreakData):
    outbreakName, IncDataPath, mean_GT,sd_GT,MaxInfctPrd = outbreakData
    IncData = pd.read_csv(IncDataPath)['IncData'].values    
    Wratios = gamma2discrete(mean_GT,sd_GT,MaxInfctPrd)
    smpLen = 40000
    
    '''
    1. preparing for the analysis
    '''    
    SimDays = len(IncData)
    tempFOI = signal.convolve(np.concatenate((np.zeros(len(Wratios)-1),IncData)),Wratios, mode='valid')     
    posterior_samples = pd.DataFrame(columns = ["outbreak","R_new","k_new"])    
    dta_Num = len(IncData)
    stDate = int(np.ceil(mean_GT));
    
    IncData_impt = IncData[stDate:]
    FOI_impt     = tempFOI[stDate:]
    
    
    # '''
    # 2. performing MCMC estimation
    # '''
    basic_model = pm.Model()
    with basic_model:
        k_para =  pm.Uniform("k_disp",1e-6,150)
        Rt_para = pm.Uniform("Rt", 0.1, 10);
        nb_r = FOI_impt*k_para;
        nb_p = k_para/(Rt_para+k_para);    
        Y_obs = pm.NegativeBinomial("Y_obs", n=nb_r, p=nb_p, observed=IncData_impt)
        trace = pm.sample(smpLen, return_inferencedata=False,cores=1,chains = 1);
    map_estimate = pm.find_MAP(model=basic_model)
    posterior_samples["k_new"] = pd.DataFrame(trace['k_disp'].squeeze().T)
    posterior_samples["R_new"] = pd.DataFrame(trace['Rt'].squeeze().T)
    
    k_hpd = az.hdi(trace["k_disp"],hdi_prob = 0.95)
    R_hpd = az.hdi(trace["Rt"],hdi_prob = 0.95)
    k_median = np.median(trace["k_disp"])
    R_median = np.median(trace["Rt"])
    
    posterior_samples['outbreak'] = outbreakName
    

    return [posterior_samples,map_estimate['k_disp'],k_hpd,map_estimate['Rt'],R_hpd];

'''

'''
# the csv file of the analyzed incidence data 
IncPath = '/Users/macbjmu/Documents/research/onGoing_project/mpx_in_China/mpx_data/'

Incfile = 'BJ-mpx.csv'
outbreakName = 'Mpx_BJ'
mean_GT, sd_GT= 5.6, 1.5; # multiple countries
# mean_GT, sd_GT= 8.5, 5.0; # USA data
MaxInfctPrd = int(mean_GT+3*sd_GT)+1
obkData = [outbreakName,IncPath+Incfile,mean_GT,sd_GT,MaxInfctPrd]
posterior_obk =  Inci2Epi_Para(obkData)
# the estimated k and R of Mers from literature
posterior_obk_all = posterior_obk[0]
# plt.style.use('ggplot') 

# calculation of p80
tmpK = posterior_obk[1];
tmpR_m = posterior_obk[3];
tmpR_lb = posterior_obk[4][0];
tmpR_ub = posterior_obk[4][1];
p80_md = p80_cal(tmpK,tmpR_m);
# p80_lb = p80_cal(tmpK,tmpR_lb);
# p80_ub = p80_cal(tmpK,tmpR_ub);


# R plot
plt.style.use('ggplot') 
fig,ax_r = plt.subplots(figsize=(16,9),dpi = 500)
sns.kdeplot(data=posterior_obk_all, x="R_new",fill=True,
            common_norm = False, alpha=.5,linewidth=0, color = '#348ABD',ax = ax_r)
R_CI_USA = [1.82,2.1];
R_CI_UK = [1.5,1.7];
R_CI_PG = [1.2,1.6];
R_CI_SP = [1.7,2.0];

# R_CI_all = [1.37,1.42]
R_CIs = [R_CI_SP,R_CI_PG,R_CI_UK]
ar_pos = [0.4,0.6,0.8]
for i in range(len(R_CIs)):
    tempCI = R_CIs[i]
    ax_r.hlines(ar_pos[i] , tempCI[0],tempCI[1],color = 'k',lw = 5)   
    # ax_r.plot((tempCI[0]+tempCI[1])/2,ar_pos[i],'r+',markersize=16)
ax_r.set_xlim(0.5,3.5)
ax_r.set_xticks([1,1.5,2,2.5,3])
ax_r.set_yticks([0,0.4,0.8,1.2])
plt.grid(axis = 'y');
plt.show();



# k plot
logKDE = rpackages.importr('logKDE')
k_sample = posterior_obk_all["k_new"];
k_input = robjects.FloatVector(k_sample)
k_fit = logKDE.logdensity(k_input)
support = np.asarray(k_fit[0])
dens = np.asarray(k_fit[1])
# plt.style.use('ggplot') 
fig,ax_k = plt.subplots(figsize=(16,9),dpi = 500)
sns.lineplot(x=support,y=dens,ax = ax_k,alpha=0)
ax_k.set_xscale('log')
ax_k.set_xlim(0.3,150)
ax_k.set_xticks([1,10,100])

ax_k.set_yticks([0,0.04,0.08])
ax_k.set_ylim(ymin = 0)
plt.fill_between(support, dens, alpha = 0.5,color = '#E24A33')
plt.minorticks_off()
plt.grid(axis = 'y');
plt.show();



