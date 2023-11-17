#!/bin/bash
#SBATCH --job-name=MoCoV3_official
#SBATCH --gpus=8
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH -p plgrid-gpu-a100 
#SBATCH -A plgplgccontrastive-gpu-a100
#SBATCH --time=48:00:00 

set -e

eval "$(conda shell.bash hook)"
conda activate uj

set -x

cd $HOME/uj/moco-v3


# override imagenet100 with imagenet

python main_moco.py -a vit_base --cassle --lr 1.5e-4 --weight-decay 0.1 --stop-grad-conv1 --moco-t 0.2 --moco-m-cos --moco-mlp-dim 4096 --moco-dim 256  --batch-size 1024  --warmup-epochs=40 --epochs 300 --dist-url 'tcp://localhost:8008' --multiprocessing-distributed --world-size 1 --rank 0 "$PLG_GROUPS_STORAGE/plgg_gmum_cc/datasets/ImageNet/ILSVRC/Data/CLS-LOC"

