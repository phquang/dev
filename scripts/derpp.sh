#!/bin/bash

#MY_PYTHON="CUDA_VISIBLE_DEVICES=4 python"
#beta 0.05 lr 0.01
gpu=$1
lr=0.03
n=1
path=results/

CIFAR_100i="--save_path $path --batch_size 16 --cuda yes --seed 0 --n_epochs 1 --lr 0.003 --n_runs 1"


CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model derpp --n_memories 20000

