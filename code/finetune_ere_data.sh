#!/bin/bash
task="transfer"
batch=(2)
mlp_hid_size=64
seeds=(5 7 23)
model="roberta-large"
ga=1
data="tbd"
checkpoint=3
device="0"
lrs=(1e-5)

root="output"
pretrained_dir="contrastive_lm_roberta-large_batch_8_lr_1e-6_epochs_10_seed_7_perturb_0.5_cw1.0"
dir="${root}/${pretrained_dir}/pytorch_model_e${checkpoint}.bin"

for l in "${lrs[@]}"
do
  for s in "${batch[@]}"
  do
    epochs=( 10 )
    for e in "${epochs[@]}"
    do
	    for seed in "${seeds[@]}"
	    do
		    python ./code/run_singletask_te.py \
		    --task_name "${task}" \
		    --do_train \
		    --do_eval \
		    --device_num ${device} \
		    --do_lower_case \
		    --te_type ${data} \
		    --mlp_hid_size ${mlp_hid_size} \
		    --model ${model} \
		    --model_dir ${dir} \
		    --data_dir ./data/ \
		    --max_seq_length 220 \
		    --train_batch_size ${s} \
		    --learning_rate ${l} \
		    --num_train_epochs ${e}  \
		    --gradient_accumulation_steps=${ga}  \
		    --output_dir ${root}/${task}_${data}_${model}_batch_${s}_lr_${l}_epochs_${e}_seed_${seed} \
		    --seed ${seed}
	    done
	  done
  done
done
