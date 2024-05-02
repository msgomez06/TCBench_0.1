#!/bin/bash -l

# --mail-type ALL 
# --mail-user milton.gomez@unil.ch

#SBATCH --chdir /work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/TCBench_0.1
# old directory: /work/FAC/FGSE/IDYST/tbeucler/default/louis/TCBench_0.1/
#SBATCH --job-name era5_sp_downloader
#SBATCH --output /work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/TCBench_0.1/download_logs/era5_download_sp_%a.txt
#SBATCH --error /work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/TCBench_0.1/download_logs/era5_download_sp_%a.err

#SBATCH --partition cpu
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G 
#SBATCH --time 72:00:00 
#SBATCH --export NONE
#SBATCH --array=0-14

# clearing modules and loading python
module purge
module load gcc cuda
eval "$(/dcsrsoft/spack/arolle/v1.0/spack/opt/spack/linux-rhel8-zen2/gcc-10.4.0/miniconda3-4.10.3-gpvric5au5ue2cp2qiiar6vijzx4ibnb/condabin/conda shell.bash hook)"
conda deactivate
conda activate /work/FAC/FGSE/IDYST/tbeucler/default/louis/e2mip_venv/

config=/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/TCBench_0.1/input_params/input_era5_downloader_SP.txt

# Extract the date, time and lead time for the current $SLURM_ARRAY_TASK_ID
min_year=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)

min_month=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

echo "Starting ERA5 download (surface) for year $min_year and month $min_month"
python3 scripts/ERA5_downloader_SP.py --min_year "$min_year" --min_month "$min_month" --max_years 1
echo "Finished"
conda deactivate
