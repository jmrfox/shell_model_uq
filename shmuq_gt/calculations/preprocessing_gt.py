#
# preprocessing for GT data
# input ~ sd_GT_data.csv
# output ~ sd_GT_processed.csv
# Fox 2020

import numpy as np
import pandas as pd
from fractions import Fraction
import pickle as pkl

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('input_filename')
parser.add_argument('output_filename')

args = parser.parse_args()

fn_in = args.input_filename
fn_out = args.output_filename
#fn_in = 'sd_GT_data.csv'
#fn_out = 'sd_GT_data.pkl'

df = pd.read_csv(fn_in,header=1)

# get rid of empty last column
#df = df.drop(df.columns[[-1]], 1)

# drop any columns with NaNs (should only drop empty columns from conversion of excel format to csv)
df = df.dropna(axis=1, how='all')

df['twoJi'] = df['Ji'].apply(lambda x : int(2*Fraction(x)))
df['twoJf'] = df['Jf'].apply(lambda x : int(2*Fraction(x)))

#df = df.drop('B',axis=1)  # B is Bexp without accounting for intensity, get rid of it

df['Bth'] = 0.  # add column for theory values

#df['Tmirror'] = np.logical_and(df['Zi']==df['Nf'], df['Zf']==df['Ni'])
df['Tmirror'] = (df['Zi']==df['Nf']) & (df['Zf']==df['Ni'])   # note here & does element-wise logic

df['deltaJ'] = 0.5*(df['twoJf']-df['twoJi'])

df = df[(df['twoJi']!=0) | (df['deltaJ']!=0.0)] #remove 100% Fermi transitions

for edge in [8,20]:
    df['include'].loc[df['Zi']==edge] = False
    df['include'].loc[df['Ni']==edge] = False
    df['include'].loc[df['Zf']==edge] = False
    df['include'].loc[df['Nf']==edge] = False

#with open(fn_out,'wb') as fh:
#    pkl.dump(df,fh)


df.to_csv(fn_out,index=False)

