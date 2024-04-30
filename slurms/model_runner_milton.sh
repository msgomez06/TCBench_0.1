#!/bin/bash -l

# --mail-type ALL 
# --mail-user milton.gomez@unil.ch

#SBATCH --chdir /work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/TCBench_0.1/
#SBATCH --job-name ModelRunner
#SBATCH --output /work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/TCBench_0.1/out_files/model_runner-%j.out
#SBATCH --error /work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/TCBench_0.1/out_files/model_runner-%j.err


#SBATCH --partition cpu
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task 1
#SBATCH --mem 4G 
#SBATCH --time 72:00:00 
#SBATCH --export NONE

# clearing modules and loading python
module purge
module load gcc
source /work/FAC/FGSE/IDYST/tbeucler/default/louis/graphcast_venv/bin/activate

log_dir=/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/TCBench_0.1/out_files
model=$1
season=$2

script=/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/TCBench_0.1/scripts/slurm_manager.py
#script=/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/TCBench_0.1/scripts/dummy_manager.py

for index in {0..12}; do
    # if [[ $model == "graphcast" ]]; then
        echo This is $index for $model and year $season
        python  $script --seasons [$season] --models [$model] --all_tcs --index $index
        
        if [[ $model != "panguweather" ]]; then
            while true; do
                squeue -u mgomezd1 | grep ${model:0:5}
                if [ $(squeue -u mgomezd1 | grep ${model:0:5} -c) -eq 0 ]; then
                    echo All jobs finished for ${model:0:5} finished
                    break
                else
                    echo There are still jobs running
                    squeue -u mgomezd1 | grep ${model:0:5}
                    echo Waiting another 15 minutes for the jobs to finish
                    sleep 900
                fi
            done
        fi

    # else
    #     python  $script --seasons [$season] --models [$model] --all_tcs
    #     echo submitted $model
    #     break
    # fi
done

# fi

#     python  $script --seasons [$season] --models [$model] --all_tcs --index
#     echo This is $index
#     for job in {0..4}; do
#         sleep_time=$((2 + RANDOM % 8))
#         sbatch --chdir /work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/TCBench_0.1/ --job-name ${model:0:6} --output $log_dir/model_runner-%j.out --partition cpu --ntasks 1 --cpus-per-task 1 --mem 100M --time 00:02:00 --export NONE --wrap="sleep $sleep_time; echo $job"
#     done
#     while true; do
#         if [ $(squeue -u mgomezd1 | grep ${model:0:6} -c) -eq 0 ]; then
#             echo All jobs finished
#             break
#         else
#             echo There are still jobs running
#             squeue -u mgomezd1 | grep ${model:0:6}
#             echo Waiting another 4 seconds for the jobs to finish
#             sleep 4
#         fi
#     done