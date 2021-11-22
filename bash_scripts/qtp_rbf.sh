#!/bin/bash

echo -n "Enter seed: "
read seed
echo -n "Enter nb datapoints: "
read nb
echo -n "Enter cpu-list (x,y): "
read cpu

REPO=/home/jcampos/repos/HTV-Learn

for lmbda in 0.02 0.05 0.08 0.1 0.2 0.5 0.8
do
    for eps in 1 3 5 7 9
    do
	taskset --cpu-list "$cpu" python3 "$REPO"/htvlearn/main.py --method rbf --lmbda "$lmbda" \
	--eps "$eps" --log_dir "$REPO"/output/qtp_seed_"$seed"_num_train_"$nb"/rbf/ \
	--model_name lmbda_"$lmbda"_eps_"$eps" --dataset_name quad_top_planes --num_train "$nb" \
	--noise_ratio 0.05 --seed "$seed" --lsize 64 -v
    done
done
