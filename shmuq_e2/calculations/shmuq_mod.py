#
#   module for SHMUQ
#   Fox 2021
#

import numpy as np
from usdbsa_mod import *
import os
import pandas as pd
import shutil
import sys


verbose = True

core_Z = 8
core_N = 8
core_mass = core_Z + core_N
max_Z = 20
max_N = 20

class nucleus:
    def __init__(self,element,Zv,Nv,twoJ):
        self.element = element
        self.Zv = Zv
        self.Nv = Nv
        self.A = core_mass + self.Zv + self.Nv
        self.name = self.element
        self.scaling = ['1',str(core_mass+2),str(self.A),'0.3']
        self.twoJ = twoJ
        self.twoTz = core_Z + Zv - core_N - Nv

def get_parent(t):
    return nucleus(t['parent'],t['Zi']-core_Z,t['Ni']-core_N,t['2Ji'])
def get_daughter(t):
    return nucleus(t['daughter'],t['Zf']-core_Z,t['Nf']-core_N,t['2Jf'])
def get_nucleus(t):
    return nucleus(str(t['A'])+t['Element'],t['Z']-core_Z,t['N']-core_N,int(t['A'])%2)


### SET PARAMETERS HERE

#n_samples = 4000
#use_usdb = True
#do_bigstick_runs = False
parallel = True
collect_only = False

int_file_name = 'usdb'
sps = "sd"
T2int = "T2sd"
opme_name_GT = "GTsd"
opme_name_E2_p = 'E2sd_p'
opme_name_E2_n = 'E2sd_n'

diag_opt = "ld"
nkeep_n = 16
nkeep_d = 50

####
nthreads=8    # THESE VALUES MAY NOT BE USED... CHECK BELOW
nranks = 2

####
index_digits = 5
int_name = 'usdb'
usdb_res_dir = '/p/lustre2/fox39/shmuq_e2/run_usdb/'
n_params = 66
milcoms_filename = 'approx.milcom'

gtstrength_cmd = "./gtstrength.x"
genstrength_cmd = "./genstrength_mod.x"
use_openmp = True
use_mpi = False
if use_openmp:
    include_frag = False
    #os.environ['OMP_NUM_THREADS']=str(nthreads)
    bigstick_cmd = './bigstick-openmp.x'
if use_mpi:
    include_frag = True
    bigstick_cmd = './bigstick-mpi-omp.x'
if not use_openmp and not use_openmp:
    include_frag = False
    bigstick_cmd = './bigstick.x'

bigstick_serial_cmd = './bigstick.x'


def make_bigstick_inputs(int_name,nuc,opt,twoJz=0,Tshift=0.):

    print('Writing bigstick inputs...')
    #int_files = []
    #for int_fn in glob.glob("usdb_rand*.int"):
    #    int_files.append(int_fn[:-4])

    #opt = "d"
    ZvNv = " ".join([str(nuc.Zv),str(nuc.Nv)])
    #twoJz = str(nuc.twoJz)
    frag = "0"
    if opt=='d':
        nkeep = nkeep_d
    elif opt=='n':
        nkeep = nkeep_n
    if verbose: print(f'nkeep = {nkeep}')

    # option for near shell boundaries -> small  dimension
    if (nuc.Zv<2) | (nuc.Nv<2) | ((max_Z-core_Z-nuc.Zv)<2) | ((max_N-core_N-nuc.Nv)<2):
        diag_opt = 'ex'
        #twoJz = twoJz%2
        #nkeep //=4
    else:
        diag_opt = "ld"

    lanit = str(20*nkeep)
    nkeep = str(nkeep)
    lanopt = " ".join([nkeep,lanit])
    #lanopt = nkeep

    run_name_list = [nuc.name,int_name]
    if Tshift>0.:
        run_name_list.append(f'Tsh{Tshift}')
    run_name_list.append(f'2M{twoJz}')
    run_name = '_'.join(run_name_list)

    scaling = nuc.scaling
    in_list = [opt,run_name,sps,ZvNv,str(twoJz)]
    if include_frag:
        in_list.append(frag)
    in_list = in_list + [int_name," ".join(scaling)]
    if Tshift>0. and opt=='d':
        tshift_scaling = [-1*Tshift,-1*Tshift,0,0]
        in_list = in_list + [T2int,' '.join([str(x) for x in tshift_scaling])]
    in_list = in_list + ["end",diag_opt,lanopt]

    outfn = "in."+run_name+".b"
    with open(outfn,'w') as outfh:
         outfh.write("\n".join(in_list)+"\n")
    if verbose: print(f'Bigstick input written to file : {outfn}')
    return run_name

def run_bigstick(run_name):
    print('Running bigstick for '+run_name)
    cmd = " ".join([bigstick_cmd,"<",'in.'+run_name+'.b'])
    subprocess.call(cmd,shell=True)

def compute_overlaps(wfn_initial,wfn_final,i_state):
    ### NOTE: this input to bigstick requires a modification to
    ### bdenslib1.f90 to prompt for a custom output name

    i_name = wfn_initial.split('/')[-1]
    f_name = wfn_final.split('/')[-1]
    fn_overlaps = f'{i_name}_st{i_state}_{f_name}.ovlp'

    bargs = ['v',wfn_initial,fn_overlaps,str(i_state),wfn_final]
    brun = subprocess.Popen(bigstick_serial_cmd,stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    brun.stdin.write("\n".join(bargs).encode())
    bout = brun.communicate()[0]
    bout = bout.decode('UTF-8')
    brun.stdin.close()
    if verbose:
        print('Computing overlaps:',wfn_initial,wfn_final,i_state)
        print('ovlp args:',bargs)
        print(bout)

# read overlaps
    with open(fn_overlaps,'r') as fh:
        overlaps = np.loadtxt(fh,skiprows=2)
    return overlaps

def overlap_index(wfn_initial,wfn_final,i_state):
# counting starts at zero!
    overlaps = compute_overlaps(wfn_initial,wfn_final,i_state)
    idx = np.argmax(overlaps[:,-1])    # get overlap magnitude |<f|i>|^2
    if overlaps[idx,-1]<0.25:
        sys.exit(f'Detected overlap<0.3: {wfn_initial}, {wfn_final}, i={i_state}')
    return idx

def state_index(res_file,J,n):
    spectrum = np.array(getspectrum(res_file),dtype=float)
    return int(spectrum[spectrum[:,3]==J][n-1,0])

def make_gtstrength_inputs(parent_name,daughter_name,density_name,parent,daughter,Tshift=0.):
    print('Writing gtstrength inputs...')

    scaling = 1.0

    par_ZN = ' '.join([str(x) for x in [parent.Zv,parent.Nv]])
    dau_ZN = ' '.join([str(x) for x in [daughter.Zv,daughter.Nv]])

    gt_run_name = '_'.join([parent_name,opme_name_GT])

    outfn = "in." + gt_run_name +".gts"
    in_list = [opme_name_GT,str(scaling),parent_name,"0",daughter_name,"0",\
            density_name,f'{Tshift}',"n",gt_run_name,par_ZN,dau_ZN]
    with open(outfn,'w') as outfh:
        outfh.write("\n".join(in_list)+"\n")
    return gt_run_name

def make_genstrength_inputs(run_name,int_name,nuc,opme_name):
    print('Writing genstrength inputs...')

    scaling = 1.0

    ZvNv = " ".join([str(nuc.Zv),str(nuc.Nv)])

    gs_run_name = '_'.join([nuc.name,int_name,f'2M{nuc.twoJ}',opme_name])

    outfn = "in." + gs_run_name +".gsin"
    in_list = [opme_name,str(scaling),run_name,"0",run_name,"0",run_name,'n',run_name+'_'+opme_name]
    with open(outfn,'w') as outfh:
        outfh.write("\n".join(in_list)+"\n")
    return gs_run_name

def run_genstrength(gs_run_name):
    input_fn = 'in.'+gs_run_name+'.gsin'
    print('Running genstrength_mod for '+input_fn)
    cmd = " ".join([genstrength_cmd,"<",input_fn])
    subprocess.call(cmd,shell=True)

def run_gtstrength(gt_run_name):
    input_fn = 'in.'+gt_run_name+'.gts'
    print('Running gtstrength for '+input_fn)
    cmd = " ".join([gtstrength_cmd,"<",input_fn])
    subprocess.call(cmd,shell=True)

def parse_strength_file(gs_run_name,trans_dataframe,key_name):
    fn = gs_run_name+'.mstr'  #filename convention
    print("Parsing "+fn+"...")
    for i_t,t in trans_dataframe.iterrows():
        if verbose: print(f'i_t = {i_t}')
        line_num = 0
        with open(fn,'r') as fh:
            print('transition:',t)
            counteri = t['ni'] - 1
            counterf = t['nf'] - 1
            found_parent = False
            for line in fh:
                line_num += 1
                if line_num < 3:
                    continue
                ls = line.split()
                if ('parent' in ls) and (float(ls[2])*2 == t['twoJi']):
                    if (counteri==0):
                        found_parent = True
                        print("found parent :",line)
                    elif (counteri>0):
                        counteri = counteri - 1
                if found_parent and ('daughter' in ls) and (float(ls[2])*2 == t['twoJf']):
                    if (counterf==0):
                        trans_dataframe.loc[i_t,key_name] = float(ls[4])
                        print("found daughter :",line)
                        break
                    elif (counterf>0):
                        counterf = counterf - 1
    return trans_dataframe

def parse_strength_file_by_idx(gs_run_name,state_index_list,trans_dataframe,key_name):
    # same as parse_strenth_file but uses a state idx list, e.g. from state overlaps
    fn = gs_run_name+'.mstr'  #filename convention
    if verbose: print("Parsing "+fn+"...")
    count=0
    for i_t,t in trans_dataframe.iterrows():
        idx_i = state_index_list[count][0]
        idx_f = state_index_list[count][1]
        count += 1
        with open(fn,'r') as fh:
            if verbose: print('looking for this transition:',t)
            if verbose: print(f'i_t = {i_t}')
            line_num = 0
            found_i = False
            found_f = False
            for line in fh:
                line_num += 1
                #if line_num < 3:
                #    continue
                ls = line.split()
                #if verbose: print(f'line {line_num}:',line)
                if found_i and ('parent' in ls):
                    # if next parent state is reached w/o finding daughter
                    if verbose: print(f'ERROR: initial state has no final state: {gs_run_name}')
                    sys.exit(f'ERROR: initial state has no final state: {gs_run_name}')
                if ('parent' in ls) and (int(ls[0])==int(idx_i+1)):
                    # found parent
                    found_i = True
                    if verbose: print("found initial state :",line)
                if found_i and ('daughter' in ls) and (int(ls[0])==int(idx_f+1)):
                    found_f = True
                    trans_dataframe.loc[i_t,key_name] = float(ls[4])
                    if verbose: print("found final state :",line)
                    break
            if not found_i:
                print(t)
                sys.exit(f'ERROR: initial state is missing: {gs_run_name}')
            if not found_f:
                print(t)
                sys.exit(f'ERROR: final state is missing: {gs_run_name}')
    return trans_dataframe

def parse_strength_file_gt(gt_run_name,state_index_list,dataframe):
    fn = gt_run_name+'.str'  #filename convention
    if verbose: print("Parsing "+fn+"...")
    count=0
    for i_t,t in dataframe.iterrows():
        i_parent = state_index_list[count][0]
        i_daughter = state_index_list[count][1]
        count+=1
        with open(fn,'r') as fh:
            if verbose: print(f'i_t = {i_t}')
            line_num = 0
            if verbose:print('transition:',t)
            found_parent = False
            found_daughter = False
            for line in fh:
                line_num += 1
                if line_num < 3:
                    continue
                ls = line.split()
                if found_parent and ('parent' in ls):
                    print(t)
                    sys.exit(f'ERROR: initial state has no final state: {gt_run_name}')
                if ('parent' in ls) and (int(ls[0])==int(i_parent+1)):
                    found_parent = True
                    if verbose: print("found initial state:",line)
                if found_parent and ('daughter' in ls) and (int(ls[0])==int(i_daughter+1)):
                    found_daughter = True
                    dataframe.loc[i_t,'Bth'] = float(ls[4])
                    if verbose: print("found final state:",line)
                    break
            if not found_parent:
                print(t)
                sys.exit(f'ERROR: initial state is missing: {gt_run_name}')
            if not found_daughter:
                print(t)
                sys.exit(f'ERROR: final state is missing: {gt_run_name}')
    return dataframe

def state_indices(transitions,sample_int_name,run_name):
    # state inidices within one nucleus
    index_pairs = []
    nuc_name_usdb = usdb_res_dir+run_name.replace(sample_int_name,'usdb')
    for i_t,t in transitions.iterrows():
        if verb: print(f'i_t = {i_t}')
        n_i = int(transitions.loc[i_t,'ni'])
        n_f = int(transitions.loc[i_t,'nf'])
        j_i = int(transitions.loc[i_t,'twoJi'])/2
        j_f = int(transitions.loc[i_t,'twoJf'])/2
        idx_usdb_i = state_index(nuc_name_usdb,j_i,n_i)
        idx_usdb_f = state_index(nuc_name_usdb,j_f,n_f)
        idx_i = overlap_index(nuc_name_usdb,run_name,idx_usdb_i)
        idx_f = overlap_index(nuc_name_usdb,run_name,idx_usdb_f)
        index_pairs.append([idx_i,idx_f])
    return index_pairs

def parent_daughter_indices(transitions,sample_int_name,run_name_par,run_name_dau):
    # state indices for a charge-changing transition (between 2 nuclei)
    index_pairs = []
    parent_name_usdb = usdb_res_dir+run_name_par.replace(sample_int_name,'usdb')
    daughter_name_usdb = usdb_res_dir+run_name_dau.replace(sample_int_name,'usdb')
    for i_t,t in transitions.iterrows():
        if verb: print(f'i_t = {i_t}')
        n_parent = int(transitions.loc[i_t,'ni'])
        n_daughter = int(transitions.loc[i_t,'nf'])
        j_parent = int(transitions.loc[i_t,'2Ji'])/2
        j_daughter = int(transitions.loc[i_t,'2Jf'])/2
        i_parent_usdb = state_index(parent_name_usdb,j_parent,n_parent)
        i_daughter_usdb = state_index(daughter_name_usdb,j_daughter,n_daughter)
        i_parent = overlap_index(parent_name_usdb,run_name_par,i_parent_usdb)
        i_daughter = overlap_index(daughter_name_usdb,run_name_dau,i_daughter_usdb)
        index_pairs.append([i_parent,i_daughter])
    return index_pairs

def make_sample_interaction(int_name,sample_number,milcoms_filename):
    hessian_eigenvalues = np.loadtxt(milcoms_filename,skiprows=1,delimiter="\t")[:66]
    param_variance = 1/hessian_eigenvalues
    sample_int_name_list = [int_name,'rand' + sample_number.zfill(index_digits)]
    sample_int_name = '_'.join(sample_int_name_list)
    pert = np.random.multivariate_normal(mean=np.zeros(n_params),cov=np.diag(param_variance))
    pert = np.stack((np.arange(1,n_params+1),pert),axis=-1)
    perturb_milcom(sample_int_name,pert)
    return sample_int_name

def move_files_to_folder(sample_int_name,sample_number):
    folder_name = 'run_' + sample_int_name
    codes = {}
    if os.path.isdir(folder_name):
        print(f'Directory {folder_name} already exists.')
    else:
        code_mkdir = subprocess.call(['mkdir',folder_name])
        file_list = glob.glob(f'*rand' + str(sample_number).zfill(index_digits) + '*')
    for file_name in file_list:
        shutil.move(file_name,folder_name)
    if code_mkdir==0:
        print(f'Files moved to {folder_name}')
    else:
        sys.exit(f'Error: nonzero exit code in mkdir: {sample_number}')


def match_states(run_name,trans_dataframe):
    spectrum = get_spectrum(run_name)
    matched_indices = []
    for t in trans_dataframe.iterrows():
        #match states
        pass
    return matched_indices
