#!/bin/bash

echo -n "Enter activation (relu, leakyrelu): "
read activ
echo -n "Enter cpu-list (x,y): "
read cpu

REPO=/home/jcampos/repos/HTV-Learn

# milestones
m1=1750
m2=1900
# epochs
ep=2000
# hidden neurons
nhidden=256
# seed
seed=19

for nb in 500 1000 2000 5000 8000 10000;
do
    taskset --cpu-list "$cpu" python3 "$REPO"/htvlearn/main.py --method neural_net \
    --log_dir "$REPO"/output/non_uniform_face_seed_"$seed"_num_train_"$nb"/nn/ \
    --model_name "$activ"_nhidden_"$nhidden"_milestones_"$m1"_"$m2"_epochs_"$ep" \
    --dataset_name face --num_train "$nb" --noise_ratio 0.0 --seed "$seed" \
    --lsize 194 --net_model "$activ"fcnet2d --milestones "$m1" "$m2" \
    --num_epochs "$ep" --weight_decay 1e-6 --batch_size 100 \
    --num_hidden_layers 5 --num_hidden_neurons "$nhidden" \
    --device cpu --non_uniform -v
done
