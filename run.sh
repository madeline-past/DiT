#!/bin/sh
#SBATCH --output=./output/%j.out
#SBATCH --error=./output/%j.err
export  PYTHONUNBUFFERED=1
source activate DiT
# torchrun --nnodes=1 --nproc_per_node=4 train.py --model DiT-S/8 --ckpt /HOME/scw6fby/run/DiT/results/014-DiT-S-8/checkpoints/0050000.pt
# python sample_for_debug.py --seed 1 --model DiT-S/8 --ckpt /HOME/scw6fby/run/DiT/results/020-DiT-S-8/checkpoints/0012300.pt
python sample.py --seed 1 --model DiT-S/8 --ckpt /HOME/scw6fby/run/DiT/results/020-DiT-S-8/checkpoints/0012300.pt