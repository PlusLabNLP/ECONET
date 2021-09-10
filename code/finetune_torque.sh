#!/bin/bash
task="torque"
batchsize=12
lr=1e-5
seeds=(5 7 23)
epochs=(10)
mlp_hid_size=64
model="roberta-large"
ga=1
device="0"
checkpoint=3

root="output"
pretrained_dir="contrastive_lm_roberta-large_batch_8_lr_1e-6_epochs_10_seed_7_perturb_0.5_cw1.0"
#pretrained_dir="original"

suffix="_end2end_final.json"

for ep in "${epochs[@]}"
do
  load_model="${root}/${pretrained_dir}/pytorch_model_e${checkpoint}.bin"
  for seed in "${seeds[@]}"
  do
    python ./code/run_end_to_end.py \
    --task_name ${task} \
    --fp16 \
    --device_num ${device} \
    --load_model ${load_model} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --mlp_hid_size ${mlp_hid_size} \
    --model ${model} \
    --data_dir ./data/ \
    --file_suffix ${suffix} \
    --max_seq_length 178 \
    --train_batch_size ${batchsize} \
    --learning_rate ${lr} \
    --num_train_epochs ${ep}  \
    --gradient_accumulation_steps=${ga}  \
    --seed ${seed} \
    --output_dir output/${pretrained_dir}_${ep}_${model}_batch_${batchsize}_lr_${lr}_epochs_${ep}_seed_${seed}
  done
done
