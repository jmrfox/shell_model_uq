
# point at batch_job directory
# for each folder run_usdb_rand????? inside, will copy out csv files and
# then tar the run directory
#

import argparse
import numpy as np
import os
from glob import glob
import shutil
import tarfile
from time import time
import subprocess as sp

parser = argparse.ArgumentParser()
parser.add_argument('batch_dir',type=str)
parser.add_argument('--skip_rm',action='store_true',help='Skip deleting the directory after compression.')
args = parser.parse_args()
skip_rm = args.skip_rm
batch_dir = args.batch_dir

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    return output_filename

def make_tarfile_pigz(output_filename, source_dir):
    threads = 34
    cmd = f'tar cvf - {source_dir} | pigz -9 -p{threads} > {output_filename}'
    # cmd = f'tar -c --use-compress-program=pigz -f {output_filename} {source_dir} '
    print(f'Compressing {output_filename} with pigz on {threads} processes')
    sp.run(cmd,shell=True)
    return output_filename

ti = time()
tn = time()
os.chdir(batch_dir)
for run_dir in glob('run_usdb_rand*'):
    run_number = run_dir.split('rand')[-1]
    shutil.copy(os.path.join(run_dir,f'sd_GT_usdb_rand{run_number}.csv'),'.')
    shutil.copy(os.path.join(run_dir,f'usdb_rand{run_number}.vec'),'.')
    shutil.copy(os.path.join(run_dir,f'usdbmil_rand{run_number}.vec'),'.')
    if run_dir.endswith('/') | run_dir.endswith("\\"):
        run_dir = run_dir[:-1]
    fn_out = make_tarfile_pigz(run_dir+'.tar.pigz',run_dir)
    print('Tarball written: ',fn_out)
    if skip_rm:
        pass
    else:
        shutil.rmtree(run_dir)
        print('Directory removed: ',run_dir)
    print(f'Time to tarball:   {int(time()-tn)} seconds')
    tn = time()
os.chdir('..')
print(f'Time for tarballs in {args.batch_dir}:   {int(time()-ti)} seconds')
