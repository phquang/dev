#!/bin/bash

#MY_PYTHON="CUDA_VISIBLE_DEVICES=4 python"
#beta 0.05 lr 0.01
gpu=$1
n=2
path=results/

CIFAR_100i="--save_path $path --batch_size 64 --cuda yes --seed 0 --n_epochs 100 --lr 0.003 --n_runs 3 --inner_steps 1"

#CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model dualnet_ta --n_memories 100 --train_csv s_minus
#CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model dualnet_ta --n_memories 100 --train_csv s_plus
CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model dualnet_ta --n_memories 100 --train_csv s_in
CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model dualnet_ta --n_memories 100 --train_csv s_out
CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model dualnet_ta --n_memories 100 --train_csv s_pl

#CIFAR_100i="--save_path $path --batch_size 64 --cuda yes --seed 0 --n_epochs 100 --lr 0.001 --n_runs 1 --inner_steps 2"
#
#CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model dualnet_ta --n_memories 64 --train_csv s_minus
#CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model dualnet_ta --n_memories 64 --train_csv s_plus
#CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model dualnet_ta --n_memories 64 --train_csv s_in
#CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model dualnet_ta --n_memories 64 --train_csv s_out
#CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model dualnet_ta --n_memories 64 --train_csv s_plastic


