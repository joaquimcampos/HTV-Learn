#!/bin/bash

echo -n "Enter seed: "
read seed
echo -n "Enter nb datapoints: "
read nb
echo -n "Enter cpu-list (x,y): "
read cpu

REPO=/home/jcampos/repos/HTV-Learn

# milestones
m1=375
m2=450
# epochs
ep=500

for COMB in relu-40 leakyrelu-40 relu-256 leakyrelu-256;
do
    activ=${COMB%-*}
    nhidden=${COMB#*-}
    taskset --cpu-list "$cpu" python3 "$REPO"/htvlearn/main.py --method neural_net \
    --log_dir "$REPO"/output/qtp_seed_"$seed"_num_train_"$nb"/nn/ \
    --model_name "$activ"_nhidden_"$nhidden"_milestones_"$m1"_"$m2"_epochs_"$ep" \
    --dataset_name quad_top_planes --num_train "$nb" --noise_ratio 0.05 --seed "$seed" --lsize 64 \
    --net_model "$activ"fcnet2d --milestones "$m1" "$m2" --num_epochs "$ep" --weight_decay 1e-6 --batch_size 10 \
    --num_hidden_layers 4 --num_hidden_neurons "$nhidden" --device cpu -v
done
