#!/bin/bash

echo -n "Enter num_hidden_neurons: "
read hidden

REPO=/home/jcampos/repos/HTV-Learn
plt_dir="${REPO}/output/nn/weight_decay_vs_htv"

num_array=(1e-6 5e-6 1e-5 5e-5 1e-4 5e-4)
arg_str=""

for i in ${num_array[@]};
do
    arg_str+="output/nn/nn_6000_wd_${i}/nn_cut_face_gaps_seed_10_"
    arg_str+="num_train_6000_hidden_${hidden}_wd_${i} "
done

python3 ${REPO}/scripts/plot_htv_vs_parameter.py weight_decay ${arg_str} \
--save_dir ${plt_dir}
