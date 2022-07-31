#!/bin/bash

# number of datapoints
nb=$1

echo -n "Enter cpu-list (x,y): "
read cpu

# from 0 to 4
echo -n "Enter lmbda index:"
read idx

REPO=/home/jogoncal/repos/HTV-Learn

lmbda_list=(8e-4 9e-4 1e-3 2e-3 3e-3 4e-3
            5e-3 6e-3 7e-3 8e-3 9e-3 1e-2 3e-2 5e-2 8e-2)

start_idx=$((lmbda_range*3))
end_idx=$((start_idx+3))

seed=10

for lmbda in ${lmbda_list[@]:$start_idx:$end_idx};
do
    taskset --cpu-list "$cpu" python3 "$REPO"/htvlearn/main.py --method htv \
    --lmbda "$lmbda" --log_dir "$REPO"/output/htv/ \
    --model_name htv_cut_face_gaps_seed_"$seed"_num_train_"$nb"_lmbda_"$lmbda" \
    --dataset_name cut_face_gaps --num_train "$nb" \
    --noise_ratio 0 --seed "$seed" --lsize 194 --admm_iter 400000 -v
done
