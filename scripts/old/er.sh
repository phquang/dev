#!/bin/bash

#MY_PYTHON="CUDA_VISIBLE_DEVICES=4 python"
#beta 0.05 lr 0.01
gpu=$1
lr=0.03
n=2
path=results/

CIFAR_100i="--save_path $path --batch_size 64 --cuda yes --seed 0 --n_epochs 40 --lr 0.003 --n_runs 5"

CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model ta --n_memories 64 --train_csv s_minus
CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model ta --n_memories 64 --train_csv s_plus
CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model ta --n_memories 64 --train_csv s_in
CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model ta --n_memories 64 --train_csv s_out
CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model ta --n_memories 64 --train_csv s_pl

