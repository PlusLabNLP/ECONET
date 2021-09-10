This file contains instructions on how to reproduce results in the paper.

### 0. Environment
Some key packages.
- pytorch=1.6.0
- transformer=3.1.0
- cudatoolkit=10.1.243
- apex=0.1
- A detailed lisf of packages used can be found in env.yml


### 1. Continual Pretraining

#### 1.1 Data
- You download pretraining data here: https://drive.google.com/drive/folders/1Ak6eEUDn0x-21ZdvRycnHayPAAvtXWv0?usp=sharing (anonymous account) and save them in ./data/
- Vocabulary for temporal indicators: `temporal_indicators.txt`; for events: `all_vocab.json`.
- Data for continual pretraining BERT-large, `samples_temporal_100K_bert.json` and `samples_event_100K_bert.json`
- Data for continual pretraining RoBERTa-large, `samples_temporal_100K_roberta.json` and `samples_event_100K_roberta.json`
- Data for continual pretraining with random masks, `random_mask_bert_tokens.json` and `random_mask_roberta_tokens.json`

#### 1.2 Generator + Discriminator (ECONET)
- For BERT-large, run `./code/train_contrastive_lm_bert.sh`. Models for each epoch (25K steps) will be saved in `./output/`
- For RoBERTa-large, run `./code/train_contrastive_lm_roberta.sh`. 

#### 1.3 Generator Only
- We only experimented with RoBERTa-large. Run `./code/train_joint_lm.sh`.

#### 1.4 Random Masks
- For BERT-large, run `./code/train_random_lm_bert.sh`.
- For RoBERTa-large, run `./code/train_random_lm_roberta.sh`. 


### 2. Fine-tuning
#### 2.1 TORQUE.
##### 2.1.1 Data: `https://allennlp.org/torque.html`. Download, process and save data in `./data/`
##### 2.1.2 Fine-tuning with ECONET: run `bash ./code/finetune_torque.sh` A couple parameters you will have to set,
- `model={roberta-large | bert-large-uncased}` (default=roberta-large)
- `pretrained_dir` models pretrained from Step 1
- `checkpoints`: number of epochs (0-9) in pretraining (default=3)
##### 2.1.2 Fine-tuning with RoBERTa-large or BERT-large: run `bash ./code/finetune_torque.sh` with parameters,
- `model={roberta-large | bert-large-uncased}`
- `pretrained_dir=original` 
- `checkpoints=0`
##### 2.1.3 Evaluate: run `bash ./code/evaluate_torque.sh`, set `model_dir` to the output from finetuning above.


#### 2.2 TBD / MATRES / RED
##### 2.2.1 Data: provided in this package
##### 2.2.2 Fine-tuning with ECONET: run `bash ./code/finetune_ere_data.sh` A couple parameters you will have to set,
- `model={roberta-large | bert-large-uncased}` (default=roberta-large)
- `pretrained_dir` models pretrained from Step 1
- `checkpoints`: number of epochs (0-9) in pretraining (default=3)
- `data={tbd | matres | red}` (default=tbd)
- `task=transfer`
##### 2.2.3 Fine-tuning with RoBERTa-large or BERT-large: run `bash ./code/finetune_ere_data.sh` with parameters,
- `model={roberta-large | bert-large-uncased}` (default=roberta-large)
- `checkpoints=0`
- `data={tbd | matres | red}` (default=tbd)
- `task=no_transfer`
##### 2.2.3 Evaluate: run `bash ./code/evaluate_ere_data.sh`, set `model_dir' to the output from finetuning above.


#### 2.3 MCTACO
- Data and source code can be found here: https://github.com/CogComp/MCTACO
- To finetune with ECONET, simply use our template above to modify the code here: https://github.com/CogComp/MCTACO/tree/master/experiments/bert by loading the pretrained model. We do not repeat.


### 3. Baseline -- TacoLM.
- The recommended pretrained model used in our paper can be found here: https://drive.google.com/drive/folders/1kx5Vc8iFYorWHrxHndkUzOssdeOm8oYC.
- You can load it to the finetuning code above by directing `pretrained_dir` to it and then run the same scripts above.



