#!/bin/bash
#SBATCH --job-name=FrozenLake_actor_critic # Job name
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --time=24:00:00 # Time limit hrs:min:sec
#SBATCH --output=test_job_frozenlake_single_ac%j.out # Standard output and error log

 
 
python /data/home/milinbhade/Prashansa/frozen_lake/single_timescale_ac.py
 
echo "FinishedTraining"
