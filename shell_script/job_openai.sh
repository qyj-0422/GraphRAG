#!/bin/bash
#SBATCH --partition=M1
#SBATCH --qos=q_d8_norm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=30G
#SBATCH --job-name=GraphRAG-cs
#SBATCH --output=./logs/output_GraphRAG.out
#SBATCH --error=./logs/error_GraphRAG.err

module load anaconda
eval "$(conda shell.bash hook)"
conda activate digimon

python main.py -opt Option/Method/Dalk.yaml -dataset_name mix