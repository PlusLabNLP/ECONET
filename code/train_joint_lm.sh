#!/bin/bash
task="generator"
batchsizes=(16)
learningrates=(1e-6)
seeds=(7)
epoch=10
mlp_hid_size=64
model="roberta-large"
ga=1
device="0"
filename1="samples_temporal_100K_roberta.json"
filename2="samples_event_100K_roberta.json"
evt_w=1.0
for s in "${batchsizes[@]}"
  do
    for l in "${learningrates[@]}"
    do
        for seed in "${seeds[@]}"
        do
            python ./code/train_joint_lm.py \
            --task_name ${task} \
            --fp16 \
            --device_num ${device} \
            --do_train \
            --do_eval \
            --do_lower_case \
            --event_weight ${evt_w} \
            --mlp_hid_size ${mlp_hid_size} \
            --model ${model} \
            --data_dir ./data/ \
            --filename1 ${filename1} \
            --filename2 ${filename2} \
            --train_batch_size ${s} \
            --learning_rate ${l} \
            --num_train_epochs ${epoch}  \
            --gradient_accumulation_steps=${ga}  \
            --seed ${seed} \
            --event_weight ${evt_w} \
            --output_dir output/${task}_${model}_batch_${s}_lr_${l}_epochs_${epoch}_seed_${seed}
        done
    done
done
