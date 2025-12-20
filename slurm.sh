#!/bin/bash
#SBATCH --job-name=ml_train 
#SBATCH --output=res_%j.txt 
#SBATCH --error=err_%j.txt  
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=4   
#SBATCH --mem=16G           
#SBATCH --time=02:00:00     
#SBATCH --partition=msismall 

module load anaconda

source activate ml2025

python3 src/train.py