#!/bin/bash -l

# --mail-type ALL 
# --mail-user louis.poulain-auzeau@unil.ch

#SBATCH --chdir /work/FAC/FGSE/IDYST/tbeucler/default/louis/TCBench_0.1/
#SBATCH --job-name GraphcastMultiple
#SBATCH --output /work/FAC/FGSE/IDYST/tbeucler/default/louis/TCBench_0.1/out_files/graphcast_multiple%A_%a.out

#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G 
#SBATCH --time 02:00:00 
#SBATCH --export NONE
#SBATCH --array=0-50

# clearing modules and loading python
module purge
module load cuda
source /work/FAC/FGSE/IDYST/tbeucler/default/louis/graphcast_venv/bin/activate


# Specify the path to the config file
config=./input_params_2019326S08163.txt

# Extract the date, time and lead time for the current $SLURM_ARRAY_TASK_ID
date=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)

time=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

lead_time=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)

# Print to a file a message that includes the current $SLURM_ARRAY_TASK_ID, the same name, and the sex of the sample
# echo "This is array task ${SLURM_ARRAY_TASK_ID}, the date ${date} and the time is ${time}, the lt is ${lead_time}, file=${config}." >> "./output_gphcst.txt"

ai-models --download-assets --assets ../graphcast/ --input cds --expver 0001 --date "$date" --time "$time" --lead-time "$lead_time" \
    --path "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/graphcast/graphcast_d_${date}_t_${time}_{step}h.grib" graphcast

conda deactivate
