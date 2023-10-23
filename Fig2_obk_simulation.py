
"""
Created on Mon Sep 11 15:34:59 2023
09-28 added the inter-quartile range for each paramete setting

@author: macbjmu
"""
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from scipy import special
import scipy.integrate as integrate
from scipy.stats import gamma
import pymc as pm
import arviz as az
from scipy import signal
import matplotlib.pyplot as plt


def gamma2discrete(mean_GT,sd_GT,MaxInfctPrd):
    
    '''
    Parameters
    ----------
    mean_GT : float
        The mean of generation time.
    sd_GT : float
        the sd of generation time.
    MaxInfctPrd : TYPE
        DESCRIPTION.

    Returns
    -------
    np.array
        the ratio for each .

    '''
    shape_para = (mean_GT/sd_GT)**2;
    rate_para  = shape_para/mean_GT
    
    wRatio = np.zeros(MaxInfctPrd+1)
    for tmps in range(1,len(wRatio)):
        wRatio[tmps] = gamma.cdf(tmps+0.5,shape_para,scale = 1/rate_para) - gamma.cdf(tmps-0.5,shape_para,scale = 1/rate_para)
    
    return wRatio/np.sum(wRatio);

def ttlInfFun(IncData,Wratios):
    # return the total infectiousness at time t
    # truncate or expand the IncData 
    wm = len(Wratios)
    Im = len(IncData)
    if Im <= wm-1:
        FOI = signal.convolve(IncData,Wratios[1:Im+1], mode='valid')
    elif Im > wm-1:
        FOI = signal.convolve(IncData[Im-wm+1:],Wratios[1:], mode='valid')
    return FOI[0]   




def IncSimu_dailyNB_Rtsrs_IndSrs(SimDays,IndexSrs,mean_GT,sd_GT,MaxInfctPrd,tmpRt_srs,tmpKt): 
    Wratios = gamma2discrete(mean_GT,sd_GT,MaxInfctPrd)
    IndexSrs_len = IndexSrs.size;
    IncData = np.append(IndexSrs,np.zeros(SimDays-IndexSrs_len));
    for i in range(IndexSrs_len,SimDays):
        # get the ttlInf, kt and Rgt; added on 0920
        tmpRt = tmpRt_srs[i]; 
        ttl_Inf = ttlInfFun(IncData[:i],Wratios);
        if ttl_Inf>0:
            IncData[i] = nbinom.rvs(tmpKt*ttl_Inf,tmpKt/(tmpRt+tmpKt)); # generate the incidence     
    return IncData

def obk_simu(iniEPI_file,tmpKt,Rt_srs,mean_GT, sd_GT,simuK):
    initialsrs = pd.read_csv(iniEPI_file)['IncData'].values  
    Epi_size_July = np.zeros(simuK);
    Epi_size_Aug  = np.zeros(simuK);
    Epi_size_Sep = np.zeros(simuK);
    simLen = 122;

    
    MaxInfctPrd = int(mean_GT+3*sd_GT)+1

    July_idx = 62;
    Aug_idx = 92;
    Sep_idx = 122
    simu_result = np.zeros((len(Rt_srs)+1,10));

    for rt_id in range(len(Rt_srs)):
        simu_result[rt_id,0] = 37
        test_Rt = Rt_srs[rt_id]; # get the test Rt
        
        # simulation
        tmpRt_srs = np.ones(simLen)*test_Rt;
        
        for i in range(simuK):
            EpiSimuData  = IncSimu_dailyNB_Rtsrs_IndSrs(simLen,initialsrs,mean_GT,sd_GT,MaxInfctPrd,tmpRt_srs,tmpKt)
            
            Epi_size_July[i] = np.sum(EpiSimuData[:July_idx]);
            Epi_size_Aug[i]  = np.sum(EpiSimuData[:Aug_idx]);
            Epi_size_Sep[i]  = np.sum(EpiSimuData[:Sep_idx]);

        # get the results
        mb_size_July = np.median(Epi_size_July)
        mb_size_Aug = np.median(Epi_size_Aug)
        mb_size_Sep = np.median(Epi_size_Sep)
        
        # output the result
        simu_result[rt_id,1:4] = np.percentile(Epi_size_July,[50, 25, 75])
        simu_result[rt_id,4:7] = np.percentile(Epi_size_Aug,[50, 25, 75])
        simu_result[rt_id,7:] = np.percentile(Epi_size_Sep,[50, 25, 75])
    simu_result[-1,:] = np.array([21,62,62,62,92,92,92,122,122,122]);

    acc_case = np.cumsum(initialsrs);
    rpt_date = np.arange(0,len(initialsrs));

    late_case = np.array([45+81,45+81+54,45+81+54+42])
    late_date = np.array([62,92,122])

    acc_case = np.append(acc_case,late_case);
    rpt_date = np.append(rpt_date,late_date);
    
    return[acc_case,rpt_date,simu_result]

    



Rt_srs = np.arange(6)*0.05+0.8;
# plot the accumulated cases
plt.style.use('ggplot') 
lw = 2.5

k_srs = [1,2,4,0.5]
simuK = 50000  # times of simulation
mean_GT, sd_GT = 8.5, 5.0; # USA data
# mean_GT, sd_GT= 5.6, 1.5;  # multiple countries

# the csv file of the initial incidence data 
iniEPI_file = '/Users/BJ-mpx.csv'


# simulation with the k_srs
cmp_re = []
for k_id in range(len(k_srs)):
    tmpK = k_srs[k_id];
    cmp_re += [obk_simu(iniEPI_file,tmpK,Rt_srs,mean_GT, sd_GT,simuK)];


# plot the results
fig,axes = plt.subplots(2,2,figsize=(16,10),dpi = 500,sharex=True,sharey = False)
for k_id in range(len(k_srs)):
    tmpAx = axes[int((k_id-k_id%2)/2),k_id%2];
    tmpK = k_srs[k_id];
    acc_case,rpt_date,simu_result = cmp_re[k_id];    
    tmpAx.plot(rpt_date,acc_case,c='k',linewidth = 1.5*lw);
    for rt_id in range(len(Rt_srs)):
        tmpAx.plot(simu_result[-1,[0,1,4,7]],simu_result[rt_id,[0,1,4,7]],linewidth = lw);
    tmpAx.set_xticks([0,21,62,92,122])
    if k_id<3:
        tmpAx.set_yticks([100,200,300,400])
    else:
        tmpAx.set_yticks([100,200,300])
    if k_id%2 == 1:
        tmpAx.yaxis.set_label_position("right")
        tmpAx.yaxis.tick_right()
plt.subplots_adjust(hspace = 0.05,wspace = 0.03)
