#!/bin/bash

echo -n "Enter lmbda: "
read lmbda
echo -n "Enter cpu-list (x,y): "
read cpu

REPO=/home/jcampos/repos/HTV-Learn
seed=19
admm_iter=400000


for nb in 500 1000 2000 5000 8000 10000;
do
    for sigma in 2e-3 5e-3 8e-3 2e-2 5e-2;
    do
	taskset --cpu-list "$cpu" python3 "$REPO"/htvlearn/main.py --method htv --lmbda "$lmbda" \
	--log_dir "$REPO"/output/non_uniform_face_seed_"$seed"_num_train_"$nb"/htv \
	--model_name lmbda_"$lmbda"_sigma_"$sigma" --dataset_name face --num_train "$nb" \
	--noise_ratio 0.0 --seed "$seed" --lsize 194 --admm_iter "$admm_iter" \
	--sigma "$sigma" --non_uniform -v
    done
done
