#!/bin/bash -l

# --mail-type ALL 
# --mail-user louis.poulain-auzeau@unil.ch

#SBATCH --chdir /work/FAC/FGSE/IDYST/tbeucler/default/louis/TCBench_0.1/
#SBATCH --job-name PanguMultiple
#SBATCH --output /work/FAC/FGSE/IDYST/tbeucler/default/louis/TCBench_0.1/out_files/pangu_multiple.out
# -- error /work/FAC/FGSE/IDYST/tbeucler/default/louis/repos/2021_TCGraphNet/dev/logs/script_12_err-%j.err

#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G 
#SBATCH --time 04:00:00 
#SBATCH --export NONE

# clearing modules and loading python
module purge
module load gcc cuda cudnn
source activate /work/FAC/FGSE/IDYST/tbeucler/default/louis/ecmwf_venv/


# Read parameters from the file
while  IFS=' ' read -r date time lead_time
do
  ONNXRUNTIME=onnxruntime-gpu ai-models --assets ../panguweather/ --input cds --date "$date" --time "$time" \
  --lead-time "$lead_time" --path "/scratch/lpoulain/panguweather/pangu_d_${date}_t_${time}_{step}h.grib" panguweather
done < input_params_test.txt

conda deactivate
