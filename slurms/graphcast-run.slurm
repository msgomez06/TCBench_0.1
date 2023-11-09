#!/bin/bash -l

#SBATCH --chdir /work/FAC/FGSE/IDYST/tbeucler/default/louis/TCBench_0.1/
#SBATCH --output /work/FAC/FGSE/IDYST/tbeucler/default/louis/TCBench_0.1/out_files/graphcast.out
#SBATCH --job-name graphcast

#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 64G
#SBATCH --time 02:00:00
#SBATCH --export NONE

# clearing modules and loading python
module purge
module load cuda
source /work/FAC/FGSE/IDYST/tbeucler/default/louis/graphcast_venv/bin/activate

ai-models --download-assets --assets ../graphcast/ --input cds --expver 0001 --date 20180110 --time 0000 --lead-time 24 \
    --path '/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/graphcast/graphcast_{step}h.grib' graphcast