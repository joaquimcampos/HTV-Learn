#!/bin/bash

echo -n "Enter nb datapoints: "
read nb
echo -n "Enter cpu-list (x,y): "
read cpu

REPO=/home/jcampos/repos/HTV-Learn
seed=8

for lmbda in 1e-4 1e-3 1e-2
do
    for eps in 50
    do
	taskset --cpu-list "$cpu" python3 "$REPO"/htvlearn/main.py --method rbf --lmbda "$lmbda" \
	--eps "$eps" --log_dir "$REPO"/output/face_seed_"$seed"_num_train_"$nb"/rbf/ \
	--model_name lmbda_"$lmbda"_eps_"$eps" --dataset_name face --num_train "$nb" \
	--noise_ratio 0.0 --seed "$seed" --lsize 194 -v
    done
done
