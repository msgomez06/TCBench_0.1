import subprocess
from argparse import ArgumentParser
import os, re, glob

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

args = parser.parse_args()

seasons = args.seasons
if args.range:
    seasons = list(range(int(seasons[0]), int(seasons[-1])+1))
models = args.models

inputs = []
excluded = open("/users/lpoulain/louis/TCBench_0.1/input_params/excluded.txt").readlines()

for season in seasons:
    tmp_inputs = glob.glob(f"/users/lpoulain/louis/TCBench_0.1/input_params/{season}/*")
    inputs.extend(tmp_inputs)
print("Seasons: {}".format(seasons))

tbr = []
for input in inputs:
    for inp in excluded:
        if re.match(os.path.dirname(input), inp):
            tbr.append(input)
            break
inputs = list(set(inputs) - set(tbr))
inputs_len = [len(open(input).readlines())-1 for input in inputs]
    
for i, input in enumerate(inputs):
    l = inputs_len[i]
    for model in models:
        subprocess.run(["bash", "/users/lpoulain/louis/TCBench_0.1/slurms/slurm_manager.sh", model, input, str(l)])
    
    # once it is treated, add to the list of excluded inputs
    with open("/users/lpoulain/louis/TCBench_0.1/input_params/excluded.txt", 'a') as f:
        f.write(input+"\n")