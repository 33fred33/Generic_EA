#!/bin/sh
#SBATCH -p ProdQ
#SBATCH -N 1
#SBATCH -t 72:00:00
# Charge job to my account 
#SBATCH -A nuim01
# Write stdout+stderr to file
#SBATCH -o output.txt
#SBATCH --mail-user=fredvaldezameneyro@gmail.com
#SBATCH --mail-type=BEGIN,END

module load taskfarm
taskfarm MOEA_CGP_tasks.txt
