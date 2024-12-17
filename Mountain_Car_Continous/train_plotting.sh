#!/bin/sh
#SBATCH --job-name=FrozenLake_actor_critic # Job name
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --time=24:00:00 # Time limit hrs:min:sec
#SBATCH --output=test_job_frozenlake_plotting%j.out # Standard output and error log

 
 
python /data/home/milinbhade/Prashansa/Acrobot/plot_all.py
 
echo "FinishedTraining"