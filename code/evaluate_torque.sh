#!/bin/bash

task="torque"
model="roberta-large"
suffix="_end2end_final.json"
batchsizes=(12)
l=1e-5
seeds=(5 7 23)
eps=(3)
device="0"
root="output"
for s in "${batchsizes[@]}"
do
    for ep in "${eps[@]}"
    do
        for seed in "${seeds[@]}"
        do
          model_dir="contrastive_lm_roberta-large_batch_8_lr_1e-6_epochs_10_seed_7_perturb_0.5_cw1.0_${ep}_roberta-large_batch_${s}_lr_1e-5_epochs_10_seed_${seed}"
          python ./code/eval_end_to_end.py \
          --fp16 \
          --task_name ${task} \
          --do_lower_case \
          --model ${model} \
          --file_suffix ${suffix} \
          --learning_rate ${l} \
          --data_dir ../data/ \
          --model_dir ${root}/${model_dir}/  \
          --device_num ${device} \
          --max_seq_length 178 \
          --eval_batch_size 32
        done
    done
done
