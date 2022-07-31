#!/bin/bash

# number of datapoints
nb=$1

echo -n "Enter cpu-list (x,y): "
read cpu

# from 0 to 9
echo -n "Enter lmbda index:"
read idx

REPO=/home/jogoncal/repos/HTV-Learn

# seq first step last
eps_list=($(seq 5 0.5 24.5))

start_idx=$(($idx*4))
len=4
seed=10

for eps in ${eps_list[@]:$start_idx:$len};
do
    taskset --cpu-list "$cpu" python3 "$REPO"/htvlearn/main.py --method rbf \
    --eps "$eps" --log_dir "$REPO"/output/rbf/ \
    --model_name rbf_cut_face_gaps_seed_"$seed"_num_train_"$nb"_eps_"$eps" \
    --dataset_name cut_face_gaps --num_train "$nb" \
    --noise_ratio 0 --seed "$seed" -v
done
