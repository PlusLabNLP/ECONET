task="contrastive_lm"
batchsizes=(8)
learningrates=(2e-6)
seeds=(7)
epoch=10
mlp_hid_size=64
model="bert-large-uncased"
ga=1
device="0"
filename1="samples_temporal_100K_bert.json"
filename2="samples_event_100K_bert.json"
evt_w=1.0
cst_w=1.0
orig_r=0.5

for s in "${batchsizes[@]}"
  do
    for l in "${learningrates[@]}"
    do
        for seed in "${seeds[@]}"
        do
            python ./code/train_contrastive_lm.py \
            --task_name ${task} \
            --orig_ratio ${orig_r} \
            --device_num ${device} \
            --do_train \
            --fp16 \
            --do_lower_case \
            --event_weight ${evt_w} \
            --contrastive_weight ${cst_w} \
            --mlp_hid_size ${mlp_hid_size} \
            --model ${model} \
            --data_dir  ./data/ \
            --filename1 ${filename1} \
            --filename2 ${filename2} \
            --train_batch_size ${s} \
            --learning_rate ${l} \
            --num_train_epochs ${epoch}  \
            --gradient_accumulation_steps=${ga}  \
            --seed ${seed} \
            --output_dir output/${task}_${model}_batch_${s}_lr_${l}_epochs_${epoch}_seed_${seed}_perturb_${orig_r}
        done
    done
  done
