import subprocess
from argparse import ArgumentParser
import os, re, glob
from scripts.utils import write_several_seasons, date_time_nn_to_netcdf
import time

def str2list(li):
    if type(li)==list:
        li2=li
        return li2
    elif type(li)==str:
        
        if ', ' in li :
            li2=li[1:-1].split(', ')
        else :
            
            li2=li[1:-1].split(',')
        return li2
    
    else:
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))


parser = ArgumentParser()

parser.add_argument("--seasons", type=str2list, help="input seasons, pass it as a list", default=[2016,2017,2018,2019,2020])
parser.add_argument("--models", type=str2list, help="input models, pass it as a list", default=['graphcast'])
parser.add_argument("--range", help="whether seasons should be interpreted as a range", action="store_true")
parser.add_argument("--all_tcs", help="whether to use all tcs of the season or one at random per basin", action="store_true")
parser.add_argument("--input", help="'cds' or 'local'. Note that fcnv can only be run with cds currently", default="local")

args = parser.parse_args()
cds = "" if args.input=="local" else "cds"

print("Input args:\n", args)

seasons = args.seasons
if args.range:
    seasons = list(range(int(seasons[0]), int(seasons[-1])+1))
models = args.models

inputs = write_several_seasons(output_path="/users/lpoulain/louis/TCBench_0.1/input_params/", 
                      seasons=seasons, step=6, max_lead=168, all_tcs=args.all_tcs,
                      ibtracs_path='/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv')

print(f"Number of tcs before exclusion: {len(inputs)}")
excluded = open("/users/lpoulain/louis/TCBench_0.1/input_params/excluded.txt", "r").readlines()
excluded = [excl.strip() for excl in excluded]

tbr = set()
d = 0
for input in inputs:
    if os.path.basename(input) in [os.path.basename(excl) for excl in excluded]:
        d += 1
        tbr.add(input)
        
inputs = list(set(inputs) - set(tbr))
inputs_len = [len(open(input).readlines())-1 for input in inputs]

c = {}
for model in models:
    c[model] = 0
    
print(f"Number of tcs after exclusion: {len(inputs)}")
for i, input in enumerate(inputs):
    l = inputs_len[i]
    for model in models:
        sid = input.split("_")[2]
        model_name = "pangu" if model=="panguweather" else model
        
        start_date1, start_time1, lead_time1 = open(input).readlines()[1].split()[1:]
        lead_time1 = int(lead_time1)
        start_nc1 = date_time_nn_to_netcdf(start_date1, start_time1, 0)
        end_nc1 = date_time_nn_to_netcdf(start_date1, start_time1, lead_time1)
        
        start_date2, start_time2, lead_time2 = open(input).readlines()[-1].split()[1:]
        lead_time2 = int(lead_time2)
        start_nc2 = date_time_nn_to_netcdf(start_date2, start_time2, 0)
        end_nc2 = date_time_nn_to_netcdf(start_date2, start_time2, lead_time2)
        
        fnames1 = [f"/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/{model}/{model_name}"+\
                f"_{start_nc1}_to_{end_nc1}_ldt_{lead_time1}.nc",
                  f"/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/{model}/{model_name}"+\
                f"_{start_nc2}_to_{end_nc2}_ldt_{lead_time2}.nc"]
        fnames2 = [f"/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/{model}/{model_name}"+\
                    f"_{start_nc1}_to_{end_nc1}_ldt_{lead_time1}_{sid}*.nc",
                f"/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/{model}/{model_name}"+\
                    f"_{start_nc2}_to_{end_nc2}_ldt_{lead_time2}_{sid}*.nc"]
        
        # original file may have been deleted already so check also for TC-specific file 
        if False in [os.path.isfile(fname) for fname in fnames1] and False in [os.path.isfile(fname) for fname in fnames2]:
            #subprocess.run(["bash", "/users/lpoulain/louis/TCBench_0.1/slurms/slurm_manager.sh", model, input, str(l), cds])
            c[model] += 1
    
    # once it is treated, add to the list of excluded inputs
    #with open("/users/lpoulain/louis/TCBench_0.1/input_params/excluded.txt", 'a') as f:
        #f.write(input+"\n")
        
print(f"Number of jobs submitted: {c}")