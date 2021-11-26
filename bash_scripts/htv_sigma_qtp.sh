#!/bin/bash

echo -n "Enter lmbda: "
read lmbda
echo -n "Enter cpu-list (x,y): "
read cpu

REPO=/home/jcampos/repos/HTV-Learn
seed=14
nb=150
admm_iter=600000

# for lmbda in 8e-4 9e-4 1e-3 2e-3 3e-3 4e-3 5e-3 6e-3 7e-3 8e-3 1e-2 10;

taskset --cpu-list "$cpu" python3 "$REPO"/htvlearn/main.py --method htv --lmbda "$lmbda" \
--log_dir "$REPO"/output/htv_sigma_qtp_seed_"$seed"_num_train_"$nb"_admm_iter_"$admm_iter"/ \
--model_name lmbda_"$lmbda"_sigma_same --dataset_name quad_top_planes --num_train "$nb" \
--noise_ratio 0.05 --seed "$seed" --lsize 64 --admm_iter "$admm_iter" \
--simplex -v

for sigma in 2e-3 2e-2 2e-1 2 20 200 2000;
do
    taskset --cpu-list "$cpu" python3 "$REPO"/htvlearn/main.py --method htv --lmbda "$lmbda" \
    --log_dir "$REPO"/output/htv_sigma_qtp_seed_"$seed"_num_train_"$nb"_admm_iter_"$admm_iter"/ \
    --model_name lmbda_"$lmbda"_sigma_"$sigma" --dataset_name quad_top_planes --num_train "$nb" \
    --noise_ratio 0.05 --seed "$seed" --lsize 64 --admm_iter "$admm_iter" \
    --simplex --sigma "$sigma" -v
done
