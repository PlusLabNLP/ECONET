This repo contains the instructions on how to reproduce results in the paper.

## Updates on March 10th, 2022. 
People are requesting data pre-processing code for MATRES and TBD pickle files. The original pickles files were produced by the internal NLP annotation tools used by the Information Sciences Institute at USC. Due to contract restriction, we are not able to make those tools public. However, we made effort to replicate those files using public tools: https://github.com/rujunhan/TEDataProcessing. There are some unavoidable but minor differences between the two versions.

## 0. Environment
Some key packages.
- pytorch=1.6.0
- transformer=3.1.0
- cudatoolkit=10.1.243
- apex=0.1
- A detailed lisf of packages used can be found in env.yml


## 1. Continual Pretraining
For replication purpose, we provide several checkpoints for our pretrained models here: https://drive.google.com/drive/folders/1otj3NjBfra9bzPNWGXOvMsB7SQL4WSDr?usp=sharing

Under the `pretrained_models/` folder,
- contrastive_lm_roberta-large*: final ECONET model based on RoBERTa-large. `pytorch_model_ep3.bin` is the one picked (as stated in the paper).
- contrastive_lm_bert-large-uncased*: final ECONET model based on BERT-large-uncased. `pytorch_model_ep1.bin` is the one picked.
- generator_lm_roberta-large*: generator-only model based on RoBERTa-large. `pytorch_model_ep1.bin` is the one picked.
- random_mask_roberta-large*: model pretrained with random mask objective based on RoBERTa-large. `pytorch_model_ep0.bin` is the one picked.
- random_mask_bert-large-uncased*: model pretrained with random mask objective based on BERT-large-uncased. `pytorch_model_ep0.bin` is the one picked.

If you are interested in re-training ECONET (and its variants), here are the instructions.
#### 1.1 Pretraining Data
We also released our pre-processed pretraining data using the same Google Drive link above.

Under the `data/` folder,
- Vocabulary for temporal indicators: `temporal_indicators.txt`; for events: `all_vocab.json`.
- Data for continual pretraining BERT-large, `samples_temporal_100K_bert.json` and `samples_event_100K_bert.json`
- Data for continual pretraining RoBERTa-large, `samples_temporal_100K_roberta.json` and `samples_event_100K_roberta.json`
- Data for continual pretraining with random masks, `random_mask_bert_tokens.json` and `random_mask_roberta_tokens.json`

#### 1.2 Generator + Discriminator (ECONET)
Save all of the above data objects in the local `.data/` folder.
- For BERT-large, run `./code/train_contrastive_lm_bert.sh`. Models for each epoch (25K steps) will be saved in `./output/`
- For RoBERTa-large, run `./code/train_contrastive_lm_roberta.sh`. 

#### 1.3 Generator Only
- We only experimented with RoBERTa-large. Run `./code/train_joint_lm.sh`.

#### 1.4 Random Masks
- For BERT-large, run `./code/train_random_lm_bert.sh`.
- For RoBERTa-large, run `./code/train_random_lm_roberta.sh`. 


## 2. Replicating Fine-tuning Results
- We provide finetuned model objects for the reported RoBERTa-large and RoBERTa-large + ECONET results.
- Download and save them in `./output/`
- Run the evaluation script below to replicate results.

#### 2.1 TORQUE.
- Data: `https://allennlp.org/torque.html`. Download, process and save data in `./data/`
- Run `bash ./code/evaluate_torque.sh`, set `model_dir` to the correct folder in `./output/`
- In the bash script, `pretrained=roberta-large_0` indicates finetuning original RoBERTa-large; otherwise, ECONET.

#### 2.2 TBD / MATRES / RED
- Data: provided in `./data/`
- Run `bash ./code/evaluate_ere_data.sh`, set `model_dir' to the correct folder in `./output/`
- In the bash script, `task=no_transfer` indicates finetuning RoBERTa-large; `task=transfer` means RoBERTa-large + ECONET.

#### 2.3 MCTACO
- Finetuning details can be found in 3.3 below.
- We did not save trained models, but simply save the outputs as the original code does. These outputs are names as `eval_outputs.txt` in the associated folders.
- As in 2.2, `no_transfer` indicates finetuning RoBERTa-large; `transfer` means finetuning RoBERTa-large + ECONET.


## 3. Re-run Fine-tuning
If you are interested in re-running our fine-tuing, we also provide instructions here.

#### 3.1 TORQUE.
Fine-tuning with ECONET: run `bash ./code/finetune_torque.sh` A couple parameters you will have to set,
- `model={roberta-large | bert-large-uncased}` (default=roberta-large)
- `pretrained_dir` models pretrained from Step 1
- `checkpoint`: number of epochs (0-9) in pretraining

Fine-tuning with RoBERTa-large or BERT-large: run `bash ./code/finetune_torque.sh` with parameters,
- `model={roberta-large | bert-large-uncased}`
- `pretrained_dir=${model}` 
- `checkpoint=0`

#### 3.2 TBD / MATRES / RED
Fine-tuning with ECONET: run `bash ./code/finetune_ere_data.sh` A couple parameters you will have to set,
- `model={roberta-large | bert-large-uncased}` (default=roberta-large)
- `pretrained_dir` models pretrained from Step 1
- `checkpoints`: number of epochs (0-9) in pretraining
- `data={tbd | matres | red}` (default=tbd)
- `task=transfer`

Fine-tuning with RoBERTa-large or BERT-large: run `bash ./code/finetune_ere_data.sh` with parameters,
- `model={roberta-large | bert-large-uncased}` (default=roberta-large)
- `checkpoints=0`
- `data={tbd | matres | red}` (default=tbd)
- `task=no_transfer`

#### 3.3 MCTACO
- Data and source code can be found here: https://github.com/CogComp/MCTACO
- To finetune with ECONET, simply modify the code here: https://github.com/CogComp/MCTACO/tree/master/experiments/bert by loading our pretrained models. 
- We do not repeat their provided instruction here.


## 4. TacoLM Baseline
- The recommended pretrained model used in our paper can be found here: https://drive.google.com/drive/folders/1kx5Vc8iFYorWHrxHndkUzOssdeOm8oYC.
- You can load it to the finetuning code above by setting `pretrained_dir` to it and then run the same scripts above.
