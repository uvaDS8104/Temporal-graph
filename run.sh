#!/bin/bash

#SBATCH --job-name=roi_res
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=30GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --nodelist=sds02
#SBATCH --output=log9.txt


nvidia-smi

module load cuda-toolkit-11.7.0 python3

source activate
conda deactivate

conda activate /s/compvision/.conda/envs/action

# python utils/preprocess_data.py --data wikipedia --bipartite
# python3 utils/preprocess_data.py --data reddit --bipartite
# python3 utils/preprocess_data.py --data mooc --bipartite


# python3 train_self_supervised.py -d wikipedia --use_memory --prefix tgn-attn  --bs 10000
# python3 train_self_supervised.py -d reddit --use_memory --prefix tgn-attn  --bs 10000
# python3 train_self_supervised.py -d mooc --use_memory --prefix tgn-attn  --bs 10000


# python3 train_supervised.py -d wikipedia --use_memory --prefix tgn-attn  --bs 10000
# python3 train_supervised.py -d reddit --use_memory --prefix tgn-attn  --bs 10000
python3 train_supervised.py -d mooc --use_memory --prefix tgn-attn  --bs 10000