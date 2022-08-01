#!/bin/bash

# number of datapoints
nb=$1

echo -n "Enter cpu-list (x,y): "
read cpu

echo -n "Enter weight decay:"
read wd

REPO=/home/jogoncal/repos/HTV-Learn

hidden_list=(40 60 85 125 180 260 380 550 800 1150)
seed=10

for hidden in ${hidden_list[@]};
do
    taskset --cpu-list "$cpu" python3 "$REPO"/htvlearn/main.py \
    --method neural_net --num_hidden_neurons "$hidden" \
    --log_dir "$REPO"/output/nn/nn_"$nb"_wd_"$wd" \
    --model_name nn_cut_face_gaps_seed_"$seed"_num_train_"$nb"_hidden_"$hidden" \
    --dataset_name cut_face_gaps --num_train "$nb" \
     --noise_ratio 0 --seed "$seed" -v
done
