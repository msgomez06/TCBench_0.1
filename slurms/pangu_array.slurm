#!/bin/bash -l

# --mail-type ALL 
# --mail-user louis.poulain-auzeau@unil.ch

#SBATCH --chdir /work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/TCBench_0.1/
#SBATCH --job-name panguMultiple
#SBATCH --output /work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/TCBench_0.1/out_files/pangu_multiple_%A_%a.out

#SBATCH --partition cpu
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task 1
#SBATCH --mem 32G 
#SBATCH --time 03:00:00 
#SBATCH --export NONE

# clearing modules and loading python
module purge
module load gcc cuda cudnn proj cdo
eval "$(/dcsrsoft/spack/arolle/v1.0/spack/opt/spack/linux-rhel8-zen2/gcc-10.4.0/miniconda3-4.10.3-gpvric5au5ue2cp2qiiar6vijzx4ibnb/condabin/conda shell.bash hook)"
conda deactivate
conda activate /work/FAC/FGSE/IDYST/tbeucler/default/louis/ecmwf_venv/


# Specify the path to the config file
config=$1 #./input_params_2019326S08163.txt
ERA5_DIR="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5"

# Extract the date, time and lead time for the current $SLURM_ARRAY_TASK_ID
date=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)

time=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

lead_time=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)

year=${date:0:4}

month=${date:4:2}

SAVE_DIR="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/AI-milton/panguweather/$year/"

# if the save directory does not exist, create it
if [ ! -d "$SAVE_DIR" ]; then
    mkdir -p $SAVE_DIR
fi

out=$(python3 /work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/TCBench_0.1/slurms/dates.py --date "$date" --time "$time" --lead_time "$lead_time")
#echo "test this is out: $out" | cat >> ./out_files/pangu_multiple_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
if ! compgen -G "${SAVE_DIR}/pangu_${out}*.nc" > /dev/null; then
    echo "Save directory: $SAVE_DIR" | cat >> ./out_files/pangu_multiple_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
    echo "$date $time $lead_time $config $out" | cat >> ./out_files/pangu_multiple_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
    ONNXRUNTIME=onnxruntime ai-models --assets /work/FAC/FGSE/IDYST/tbeucler/default/louis/panguweather/ --date "$date" --time "$time" --lead-time "$lead_time" \
    --input file --file "${ERA5_DIR}/ERA5_${year}_${month}" \
    --path "$SAVE_DIR/pangu_d_${date}_t_${time}_{step}h.grib" panguweather
    wait $!
    echo "Compressing ..." | cat >> ./out_files/pangu_multiple_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
    python3 scripts/utils/filtering.py --params "$out" --model "pangu"
else
    echo "File already exists"
    # rm ./out_files/pangu_multiple_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
fi

conda deactivate
