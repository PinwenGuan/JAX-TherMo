#!/bin/bash                                                                                      
#SBATCH --exclude=c004,c011,c013                                                                                               
#SBATCH -J PB-UQ-train
#SBATCH -n 1 # Number of total cores
#SBATCH -N 1 # Number of nodes                                          
#SBATCH -A gpu
#SBATCH --mem-per-cpu=2000 # Memory pool for all cores in MB (see also --mem-per-cpu)                        
#SBATCH -o job.sh.o%j # File to which STDOUT will be written %j is the job #                                       
#SBATCH --mail-type=END # Type of email notification- BEGIN,END,FAIL,ALL                         
#SBATCH --time=5-00

#mkdir pb
export JAX_ENABLE_X64=True
python train.py
