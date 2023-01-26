
# point at batch_job directory
# will create a bash file that does...
# for each folder run_usdb_rand????? inside
#   copy out csv files and
#   then tar the run directory
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

def make_tar_cmd_pigz(output_filename, source_dir):
    threads = 34
    cmd = f'tar cvf - {source_dir} | pigz -9 -p{threads} > {output_filename}'
    # cmd = f'tar -c --use-compress-program=pigz -f {output_filename} {source_dir} '
    return cmd

contents = []
for run_dir in glob(os.path.join(batch_dir,'run_usdb_rand?????')):
    run_number = run_dir.split('rand')[-1]
    csv_path = os.path.join(run_dir,f'sd_M1_usdb_rand{run_number}.csv')
    int_vec_path = os.path.join(run_dir,f'usdb_rand{run_number}.vec') 
    mil_vec_path = os.path.join(run_dir,f'usdbmil_rand{run_number}.vec')
    contents.append(f'cp {csv_path} .')
    contents.append(f'cp {int_vec_path} .')
    contents.append(f'cp {mil_vec_path} .')
    if run_dir.endswith('/') | run_dir.endswith("\\"):
        run_dir = run_dir[:-1]
    contents.append(make_tar_cmd_pigz(run_dir+'.tar.pigz',run_dir))
    if skip_rm:
        pass
    else:
        contents.append(f'rm {run_dir}')

    contents.append("\n")

with open(f'tar_script_{batch_dir}.bash','w') as fh:
    fh.write("\n".join(contents))


