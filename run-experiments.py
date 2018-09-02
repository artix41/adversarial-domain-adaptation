import numpy as np
import yaml
import argparse
import glob
import os
import shutil
from ADA import ADA
from experiment import experiment

# ================== Arguments ==================
    
parser = argparse.ArgumentParser(
    description='Run all experiments described in the different config files'
)
parser.add_argument('configs_dir', type=str, help='Path to configs directory')
parser.add_argument('output_dir', type=str, help='Path to the output directory')
parser.add_argument('-v', choices=['0','1','2','3'], default=1, dest='verbose',
                    help='Level of verbosity, between 0 and 3')
args = parser.parse_args()

output_dir = os.path.abspath(args.output_dir)
configs_dir = os.path.abspath(args.configs_dir)
verbose = args.verbose
     
# ================ Load config files =============

config_files = glob.glob(os.path.join(configs_dir, '*.yaml'))
experiment_names = [os.path.splitext(os.path.basename(f))[0] for f in config_files]
    
def load_config(path):
    with open(path, 'r') as f:
        config = yaml.load(f)
    return config
    
for i_exp in range(len(config_files)):
    exp_name = experiment_names[i_exp]
    verbose = int(verbose)
    if verbose >= 1:
        print("\n===========================================================")
        print("============== Running Experiment {:02d}: {} ==============".format(i_exp, exp_name))
        print("===========================================================\n")
        print("- Loading config files and creating output directories...\n")

    config = load_config(config_files[i_exp])
    exp_output_dir = os.path.join(output_dir, exp_name)

    if os.path.exists(exp_output_dir):
        print("Output path already exists for this experiment. Do you want to delete it? [y/n]")
        answer = str(input())
        while answer != 'y' and answer != 'n':
            answer = input()
        if answer == 'y':
            shutil.rmtree(exp_output_dir)
        else:
            exit()
    os.makedirs(exp_output_dir)
    
    if verbose >= 1:
        print("- Starting experiment...\n")
        
    experiment(config, exp_output_dir, verbose=verbose)

if verbose >= 1:
    print("\n=================== Done ===================\n")
