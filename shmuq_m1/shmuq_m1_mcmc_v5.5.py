# MCMC evaluation of M1 coupling constant distributions
# Fox 2021
#

version = 5.5

# v1 : gaussian prior
# v2 : uniform prior
# v3 : use SVD reparameterization
# v4 : do UQ in pairs, first spin part (4.0), then orbit part (4.1)
# v5 : change to IS/IV basis and fix only the orbital IS part, do UQ for other 3
#           5.0: flat prior + free nucleon g_lis,  5.1: Gaussian based on Brown's table of values + free nucleon g_lis
#           5.2: flat prior + Brown's g_lis for USDB,  5.3: Gaussian based on Brown's table of values + Brown's g_lis for USDB
#           5.4: Gaussian based on Brown's table + Brown's g_liv fixed for USDB
#           5.5: Gaussian based on Brown's table + Brown's g_sis fixed for USDB
#           5.6: Gaussian based on Brown's table + Brown's g_siv fixed for USDB

#import pdb

### IMPORTANT:
###    the conventional theta for M1 is the vector [g_sp, g_sn, g_lp, g_ln]
###    the conventional omega for M1 is the vector [g_sis, g_siv, g_lis, g_liv]
###    this must be the same EVERYWHERE

import numpy as np
import os, sys
from glob import glob
from tqdm import tqdm
from time import time
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'serif',
        'serif'  : ['Palatino'],
        'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
import pandas as pd
import pickle as pkl
import emcee
from multiprocessing import Pool
from scipy.optimize import minimize, minimize_scalar

# IGNORE DIVIDE BY ZERO WARNINGS
np.seterr(divide='ignore')

def optimal_n_bins(y,max_bins=100):
    from scipy.stats import iqr
    n_bins = int((max(y) - min(y))/(2*iqr(y)*len(y)**(-1/3)))
    return min(n_bins,max_bins)

#def Bweisskopf(l,A):
#    # Weisskopf (single-particle) estimate in e^2 fm^2l
#    return (1/(4*np.pi)) * (3/(3+l))**2 * (1.2*A**(1/3))**(2*l)

def B_M1_Wu(B):
    return B / 1.79

print('Loading data...')
fn_data_for_model = 'm1_data_for_model_140422.pkl'
with open(fn_data_for_model,'rb') as fh:
    [nuc_dict_list, Mth_sp_usdb_list, Mth_sn_usdb_list, Mth_lp_usdb_list, Mth_ln_usdb_list, Mth_sp_vec_list, Mth_sn_vec_list, Mth_lp_vec_list, Mth_ln_vec_list, title_string_list, Bexp_Wu_list, Bexp_unc_Wu_list] = pkl.load(fh)

A_vec = np.array([nuc_dict['A'] for nuc_dict in nuc_dict_list])
Bexp_Wu_vec = np.array(Bexp_Wu_list)
Bexp_unc_Wu_vec = np.array(Bexp_unc_Wu_list) 

Mth_sp_usdb_vec = np.array(Mth_sp_usdb_list)
Mth_sn_usdb_vec = np.array(Mth_sn_usdb_list)
Mth_lp_usdb_vec = np.array(Mth_lp_usdb_list)
Mth_ln_usdb_vec = np.array(Mth_ln_usdb_list)
Mth_sp_array = np.array(Mth_sp_vec_list)
Mth_sn_array = np.array(Mth_sn_vec_list)
Mth_lp_array = np.array(Mth_lp_vec_list)
Mth_ln_array = np.array(Mth_ln_vec_list)
n_transitions, n_samples = Mth_sp_array.shape
g_sp_free = 5.5857
g_sn_free = -3.8263
g_lp_free = 1.
g_ln_free = 0.
#g_free_list = [g_sp_free, g_sn_free, g_lp_free, g_ln_free]
g_sis_free = g_sp_free + g_sn_free
g_siv_free = g_sp_free - g_sn_free
g_lis_free = g_lp_free + g_ln_free
g_liv_free = g_lp_free - g_ln_free

#conventions
# theta = g_sp, gsn, g_lp, g_ln
# omega = g_sis, g_siv, g_lis, g_liv

def pn2iso(theta):
    g_sis = theta[0] + theta[1]
    g_siv = theta[0] - theta[1]
    g_lis = theta[2] + theta[3]
    g_liv = theta[2] - theta[3]
    return [ g_sis, g_siv, g_lis, g_liv ]

def iso2pn(omega):
    g_sp = 0.5 * (omega[0] + omega[1])
    g_sn = 0.5 * (omega[0] - omega[1])
    g_lp = 0.5 * (omega[2] + omega[3])
    g_ln = 0.5 * (omega[2] - omega[3])
    return [ g_sp, g_sn, g_lp, g_ln ]

def Bth_model_Wu(theta,Mth_sp,Mth_sn,Mth_lp,Mth_ln):
    g_sp,g_sn,g_lp,g_ln = theta
    Bth_vec = (g_sp * Mth_sp + g_sn * Mth_sn + g_lp * Mth_lp + g_ln * Mth_ln)**2
    return B_M1_Wu(Bth_vec)

def chi_squared(theta,sigmaB_apriori):
    sqr_errors = (Bexp_Wu_vec - Bth_model_Wu(theta, Mth_sp_usdb_vec, Mth_sn_usdb_vec, Mth_lp_usdb_vec, Mth_ln_usdb_vec) )**2
    B_unc_sqr_vec = np.array( [(sigmaB_apriori**2 + sigmaB**2) for sigmaB in Bexp_unc_Wu_vec])
    R_sqr =  sqr_errors / B_unc_sqr_vec
    return  np.sum(R_sqr)

def objective(sigmaB_apriori):
    theta_0 = (g_sp_free, g_sn_free, g_lp_free, g_ln_free)
    X2 = chi_squared(theta_0,sigmaB_apriori)
    dof = n_transitions - 66
    return (X2/dof - 1)**2

print('Finding a priori B-value uncertainty...')
opt_result = minimize_scalar(objective)
sigmaB_apriori = opt_result.x
print(f'A priori B(M1) uncertainty = {sigmaB_apriori}')

B_unc_sqr_vec = np.array( [sigmaB_apriori**2 + sigmaB**2 for sigmaB in Bexp_unc_Wu_vec] )

def likelihood(theta, sample):
    sqr_errors = (Bexp_Wu_vec - Bth_model_Wu(theta, Mth_sp_array[:,sample], Mth_sn_array[:,sample], Mth_lp_array[:,sample], Mth_ln_array[:,sample] ) )**2
    R_sqr_vec =  sqr_errors / B_unc_sqr_vec
    chi_sq = np.sum(R_sqr_vec)
    return  np.exp(- 0.5 * chi_sq ) 

low_bound = -20
top_bound = 20

def normal_pdf(x,m,s):
    norm = (s * np.sqrt(2*np.pi))**-1
    return norm * np.exp( - 0.5 * (x-m)**2 / s**2)
    
#def prior(theta):
#    if all([(t>low_bound) and (t<top_bound) for t in theta]): 
##         return np.array([normal_pdf(t,g_free_list[i],g_sigma_list[i]) for i,t in enumerate(theta)])
#        return 1.0
#    else:
#        return 0.0



# based on Brown's table
g_sp_brown = 5.15
g_sn_brown = -3.55
g_lp_brown = 1.159
g_ln_brown = -0.09
g_brown_means = [ g_sp_brown, g_sn_brown, g_lp_brown, g_ln_brown ]

g_sis_brown = g_sp_brown + g_sn_brown 
g_siv_brown = g_sp_brown - g_sn_brown 
g_lis_brown = g_lp_brown + g_ln_brown 
g_liv_brown = g_lp_brown - g_ln_brown 

# std determined by 4*std(g) for g in Brown's table
g_sp_std_brown = 1.023
g_sn_std_brown = 0.661
g_lp_std_brown = 0.295
g_ln_std_brown = 0.175
g_brown_stds = [ g_sp_std_brown, g_sn_std_brown, g_lp_std_brown, g_ln_std_brown ]

def prior(theta):
    if all([(t>low_bound) and (t<top_bound) for t in theta]): 
        return np.prod([ normal_pdf(t,g_brown_means[i],g_brown_stds[i]) for i,t in enumerate(theta) ])
    else:
        return 0.0

def log_posterior(omega3):
    # omega3 = [g_siv, g_lis, g_liv]
    # CHECK THE ORDER!
    omega = np.array([g_sis_brown, omega3[0], omega3[1], omega3[2]])
    theta = iso2pn(omega)
    x = prior(theta) * np.mean(np.array( [likelihood(theta,k) for k in range(n_samples)] ))
    return np.log(x)

def starting_point(n_params):
    t0 = [np.random.uniform(low_bound,top_bound) for _ in range(n_params)]
    return t0

n_walkers = 32
n_steps = 16000
n_params = 3
use_pool = True

q0 = [starting_point(n_params) for _ in range(n_walkers)]

print('Beginning MCMC...')
print(f'N walkers: {n_walkers}')
print(f'N steps: {n_steps}')

ckpt_filename = f'checkpoint_v{version}.h5'
backend = emcee.backends.HDFBackend(ckpt_filename)
backend.reset(n_walkers, n_params)

if use_pool:
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_params, log_posterior, pool=pool, backend=backend)
        start = time()
        sampler.run_mcmc(q0, n_steps, progress=True,);
        end = time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))
else:
    sampler = emcee.EnsembleSampler(n_walkers, n_params, log_posterior)
    t1 = time()
    sampler.run_mcmc(q0, n_steps, progress=True,);
    t2 = time()
    print("Processing took {0:.1f} seconds".format(t2-t1))

with open(f'traces_v{version}.pkl','wb') as fh:
    pkl.dump(sampler.chain,fh)

