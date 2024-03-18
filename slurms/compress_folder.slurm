#!/bin/bash -l

# --mail-type ALL 
# --mail-user milton.gomez@unil.ch

#SBATCH --chdir /work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/TCBench_0.1/
#SBATCH --job-name CompressFolder
#SBATCH --output /work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/TCBench_0.1/out_files/compress_%j.out
#SBATCH --error /work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/TCBench_0.1/out_files/compress_%j.err 

#SBATCH --partition cpu
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task 2
#SBATCH --mem 16G 
#SBATCH --time 24:00:00 
#SBATCH --export NONE

#clearing modules
module purge
module load gcc cuda proj cdo
eval "$(/dcsrsoft/spack/arolle/v1.0/spack/opt/spack/linux-rhel8-zen2/gcc-10.4.0/miniconda3-4.10.3-gpvric5au5ue2cp2qiiar6vijzx4ibnb/condabin/conda shell.bash hook)"
conda deactivate
source /work/FAC/FGSE/IDYST/tbeucler/default/louis/graphcast_venv/bin/activate

folder=$1

# Compressing the folder
echo "Compressing folder $folder" | cat >> /work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/TCBench_0.1/out_files/compress_%j.out
python3 scripts/utils/compressing.py --folder $folder