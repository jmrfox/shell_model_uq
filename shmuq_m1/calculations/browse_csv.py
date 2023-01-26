#
# tool for browing a csv file using pandas
# run interactively with -i
# Fox 12/20
#

import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()
df = pd.read_csv(args.filename)

