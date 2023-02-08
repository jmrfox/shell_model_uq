
# point at batch_job directory
# for each folder run_usdb_rand????? inside, will copy out csv files and
# then tar the run directory
#
# IMPORTANT: check that the csv name below is correct (GT,E2,etc)

import argparse
import numpy as np
import os
from glob import glob
import shutil
import tarfile
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('batch_dir',type=str)
args = parser.parse_args()
remove_dirs_after_tar = True

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    return output_filename

def tar_run_dir(run_dir):
    run_number = run_dir.split('rand')[-1]
    csv_fn = f'sd_E2_usdb_rand{run_number}_complete.csv'
    int_vec_fn = f'usdb_rand{run_number}.vec'
    mil_vec_fn = f'usdbmil_rand{run_number}.vec'
    for fn in [csv_fn,int_vec_fn,mil_vec_fn]:
        if os.path.isfile(os.path.join(run_dir,fn)):
            shutil.copy(os.path.join(run_dir,fn),args.batch_dir)
            print(f'Found file : {fn}')
        else:
            print(f'ERROR: File does not exist: {fn}')
            print(f'Failed to make tarball: {run_dir}')
            return 1
    fn_out = make_tarfile(os.path.join(args.batch_dir,os.path.split(run_dir)[1]+'.tar.gz'),run_dir)
    print('Tarball written: ',fn_out)
    if remove_dirs_after_tar:
        shutil.rmtree(run_dir)
        print('Directory removed: ',run_dir)
        os.mkdir(run_dir)
        shutil.move(os.path.join(args.batch_dir,f'sd_E2_usdb_rand{run_number}_complete.csv'),run_dir)
        shutil.move(os.path.join(args.batch_dir,f'usdb_rand{run_number}.vec'),run_dir)
        shutil.move(os.path.join(args.batch_dir,f'usdbmil_rand{run_number}.vec'),run_dir)
    return 0

if __name__=='__main__':
    ti = time()
    tn = time()
    for run_dir in glob(os.path.join(args.batch_dir,f'run_usdb_rand?????')):
        print(f'Tarring {run_dir}')
        tn = time()
        exit_code = tar_run_dir(run_dir)
        if exit_code !=0:
            print('Failed. Continuing to tar run directories...')
            continue
        print(f'Time to tarball:   {int(time()-tn)} seconds')
    print(f'Time for tarballs in {args.batch_dir}:   {int(time()-ti)} seconds')
