#!/bin/sh
#SBATCH --job-name=FrozenLake_ppo_actor_critic # Job name
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --time=24:00:00 # Time limit hrs:min:sec
#SBATCH --output=test_job_frozenlake_ppo_ac%j.out # Standard output and error log

 
 
python /data/home/milinbhade/Prashansa/frozen_lake/PPO_actor_critic.py
 
echo "FinishedTraining"