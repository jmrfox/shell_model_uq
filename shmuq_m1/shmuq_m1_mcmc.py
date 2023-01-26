# MCMC evaluation of M1 coupling constant distributions
# Fox 2021
#

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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-f','--fix_component',type=str,default='lis',help='choose component to fix: sis, siv, lis, liv' )
parser.add_argument('-ret','--relative_err_threshold',type=float,default=None,help='used to exclude transitions where (Bexp - Bth)/Bexp > R.E.T.  ' )
parser.add_argument('-p','--prior_type',type=str,default='flat',help='choose prior: flat, normal' )


args = parser.parse_args()
if args.fix_component in ['sis','siv','lis','liv']:
    fix_component = args.fix_component
else:
    print('Error: bad input option')
    exit()
    
if args.relative_err_threshold is None:
    print('No relative error threshold.')
elif (args.relative_err_threshold is not None) & (args.relative_err_threshold >= 0.0):
    relative_err_threshold = args.relative_err_threshold
    print(f'Relative error threshold = {relative_err_threshold}')
else:
    print('Error: bad input option')
    exit()

if args.prior_type in ['flat','normal']:
    prior_type = args.prior_type
else:
    print('Error: bad prior option')
    exit()


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

g_sp_free = 5.5857
g_sn_free = -3.8263
g_lp_free = 1.
g_ln_free = 0.
g_free_list = [g_sp_free, g_sn_free, g_lp_free, g_ln_free]
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

# load model data
print('Loading data...')
#fn_data_for_model = 'm1_data_for_model_140422.pkl'
fn_data_for_model = 'm1_data_for_model_truncF.pkl'
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

# drop transitions with large relative error
if args.relative_err_threshold is not None:
    Bth_usdb_prior = Bth_model_Wu(g_free_list,Mth_sp_usdb_vec, Mth_sn_usdb_vec, Mth_lp_usdb_vec, Mth_ln_usdb_vec)
    Berr_rel = (Bexp_Wu_vec - Bth_usdb_prior)/Bexp_Wu_vec
    mask = np.abs(Berr_rel) < relative_err_threshold

    A_vec = A_vec[mask]
    Bexp_Wu_vec = Bexp_Wu_vec[mask]
    Bexp_unc_Wu_vec = Bexp_unc_Wu_vec[mask]
    nuc_dict_list = np.array(nuc_dict_list)[mask]
    Mth_sp_usdb_vec = Mth_sp_usdb_vec[mask]
    Mth_sn_usdb_vec = Mth_sn_usdb_vec[mask]
    Mth_lp_usdb_vec = Mth_lp_usdb_vec[mask]
    Mth_ln_usdb_vec = Mth_ln_usdb_vec[mask]
    Mth_sp_array = Mth_sp_array[mask]
    Mth_sn_array = Mth_sn_array[mask]
    Mth_lp_array = Mth_lp_array[mask]
    Mth_ln_array = Mth_ln_array[mask]
    n_transitions, n_samples = Mth_sp_array.shape

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
print(f'N transitions included: {len(Bexp_Wu_vec)}')

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

def log_normal_pdf(x,m,s):
    norm = (s * np.sqrt(2*np.pi))**-1
    return np.log(norm) - 0.5 * ((x-m)/s)**2

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
# 6/11/22 I don't like what i did to get these (average tabulated values from Richter) 
# since it is skewed by how much free values and effective values change
#g_sp_std_brown = 1.023
#g_sn_std_brown = 0.661
#g_lp_std_brown = 0.295
#g_ln_std_brown = 0.175

# instead, just take uncertainties from Richter and multiply by a constant

g_sp_std_brown = 0.09
g_sn_std_brown = 0.1
g_lp_std_brown = 0.023
g_ln_std_brown = 0.26
c = 3
#g_brown_stds = [ g_sp_std_brown, g_sn_std_brown, g_lp_std_brown, g_ln_std_brown ]
g_brown_stds = [ c*g_sp_std_brown, c*g_sn_std_brown, c*g_lp_std_brown, c*g_ln_std_brown ]


# create big arrays for speed
Bexp_tiled = np.tile(Bexp_Wu_vec.reshape(n_transitions,1),n_samples)
B_unc_sqr_tiled = np.tile(B_unc_sqr_vec.reshape(n_transitions,1),n_samples)

#prior_type = 'normal'
if prior_type == 'normal':
    def log_prior(theta):
        if all([(t>low_bound) and (t<top_bound) for t in theta]): 
            p = np.sum( [ log_normal_pdf(t,g_brown_means[i],g_brown_stds[i]) for i,t in enumerate(theta) ] )
        else:
            p = -np.infty
        return p
elif prior_type == 'flat':
    def log_prior(theta):
        if all([(t>low_bound) and (t<top_bound) for t in theta]): 
            p = 1
        else:
            p = -np.infty
        return p

if fix_component == 'sis':
    def standardize_params(omega3):
        # omega3 = [g_siv, g_lis, g_liv]
        # CHECK THE ORDER!
        omega = np.array([g_sis_brown, omega3[0], omega3[1], omega3[2]])
        theta = iso2pn(omega)
        return theta

elif fix_component == 'siv':
    def standardize_params(omega3):
        # omega3 = [g_sis, g_lis, g_liv]
        # CHECK THE ORDER!
        omega = np.array([omega3[0], g_siv_brown, omega3[1], omega3[2]])
        theta = iso2pn(omega)
        return theta
elif fix_component == 'lis':
    def standardize_params(omega3):
        # omega3 = [g_sis, g_siv, g_liv]
        # CHECK THE ORDER!
        omega = np.array([omega3[0], omega3[1], g_lis_brown, omega3[2]])
        theta = iso2pn(omega)
        return theta
elif fix_component == 'liv':
    def standardize_params(omega3):
        # omega3 = [g_sis, g_siv, g_lis]
        # CHECK THE ORDER!
        omega = np.array([omega3[0], omega3[1], omega3[2], g_liv_brown])
        theta = iso2pn(omega)
        return theta

def log_posterior(omega3):
    theta = standardize_params(omega3)
    sqr_error_tiled = ( Bexp_tiled - B_M1_Wu( (theta[0]*Mth_sp_array + theta[1]*Mth_sn_array + theta[2]*Mth_lp_array + theta[3]*Mth_ln_array)**2 ) )**2
    log_likelihood =  - 0.5 * np.sum(sqr_error_tiled / B_unc_sqr_tiled) / n_samples
    return log_prior(theta) + log_likelihood

def starting_point(n_params):
    t0 = [np.random.uniform(low_bound,top_bound) for _ in range(n_params)]
    return t0

n_walkers = 32
n_steps = 5000
n_params = 3
use_pool = True

q0 = [starting_point(n_params) for _ in range(n_walkers)]

print('Beginning MCMC...')
print(f'N walkers: {n_walkers}')
print(f'N steps: {n_steps}')

if args.relative_err_threshold is None:
    nametag = f'fixed_{fix_component}_{prior_type}prior'
else:
    nametag = f'fixed_{fix_component}_RET{relative_err_threshold}_{prior_type}prior'

ckpt_filename = f'checkpoint_{nametag}.h5'
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

with open(f'traces_{nametag}.pkl','wb') as fh:
    pkl.dump(sampler.chain,fh)

