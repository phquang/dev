#!/bin/bash

#MY_PYTHON="CUDA_VISIBLE_DEVICES=4 python"
#beta 0.05 lr 0.01
gpu=$1
lr=0.03
n=2
path=results/

CIFAR_100i="--save_path $path --batch_size 64 --cuda yes --seed 0 --n_epochs 100 --lr 0.0001 --n_runs 5"

CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model er --n_memories 64 --train_csv CTRLplus
CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model er --n_memories 64 --train_csv CTRLminus
CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model er --n_memories 64 --train_csv CTRLplastic
CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model er --n_memories 64 --train_csv CTRLin
CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model er --n_memories 64 --train_csv CTRLout

