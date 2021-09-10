#!/bin/bash
task="transfer"
ratio=(1.0)
epoch=10
batch=(2)
mlp_hid_size=64
seeds=(5)
learnrates=(5e-6)
model="roberta-large"
root="output"
te="tbd"
device="0"
model_dir="${task}_${te}_${model}_batch_${b}_lr_${lr}_epochs_${epoch}_seed_${seed}_${r}"
for seed in "${seeds[@]}"
do
  for r in "${ratio[@]}"
  do
    for lr in "${learnrates[@]}"
    do
      for b in "${batch[@]}"
      do
        python ./code/eval_singletask_te.py \
        --task_name ${task} \
        --eval_ratio ${r} \
        --do_lower_case \
        --te_type ${te} \
        --model ${model} \
        --device_num ${device} \
        --mlp_hid_size ${mlp_hid_size} \
        --data_dir ./data/ \
        --model_dir ${root}/${model_dir}/\
        --max_seq_length 200 \
        --eval_batch_size 32 \
        --seed ${seed}
      done
    done
  done
done