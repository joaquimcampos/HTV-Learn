#!/bin/bash

echo -n "Enter nb datapoints: "
read nb
echo -n "Enter cpu-list (x,y): "
read cpu

REPO=/home/jcampos/repos/HTV-Learn
seed=8

for lmbda in 8e-4 9e-4 1e-3 2e-3 3e-3 4e-3 5e-3 6e-3 7e-3 8e-3 9e-3 1e-2 3e-2 5e-2 8e-2;
do
    taskset --cpu-list "$cpu" python3 "$REPO"/htvlearn/main.py --method htv --lmbda "$lmbda" \
    --log_dir "$REPO"/output/face_seed_"$seed"_num_train_"$nb"/htv/ \
    --model_name constant_lmbda_"$lmbda" --dataset_name face --num_train "$nb" \
    --noise_ratio 0 --seed "$seed" --lsize 194 --admm_iter 200000 -v
done
