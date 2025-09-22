# SE-DPO: Learning Preference Orders for In-Context Example Retrieval
This repository contains the code implementation, data of our paper.
Our code is largely borrowed from [$Se^2$](https://github.com/microsoft/LMOps/tree/main/se2). Thanks for their awesome codebases.


Rodin version 9.22

## Setup <a name="setup"></a>
```bash
conda create -n se_dpo python=3.11
conda activate se_dpo
bash install.sh 
```


# Quick Start
The pipeline contains following stages: Scoring, Training, Inference. 
You have following tasks to run: `paraphrase reading nli coreference`

**Skip for Reproducibility!**  Our default environment settings:
```bash
# by default
export NUM_OF_GPUS=8 # based on CUDA_VISIBLE_DEVICES
export RETRIEVER_BSZ=16 # 32 by default
export SCORER_BSZ=10
export SCORE_LLM='EleutherAI/gpt-neo-2.7B'
export INF_LLM='EleutherAI/gpt-neo-2.7B'
# config for hugging face biencoder: `${PWD}/DPR/conf/encoder/hf_bert.yaml`
# google-bert/bert-base-uncased
```



## Prepare scored datasets
Download [processed datasets](https://pan.baidu.com/s/15ML_hoYgvTSOxNE-z8tdew?pwd=w9p3) to `./cache/saved_datasets`, or download and parse raw datasets by running: 
```
python `./DPR/dpr/utils/tasks.py`.
```


Generate experiment script to `./my_data/experiment`
```bash
# You must rerun the following command once environment settings change
cd SE-DPO
sh getcmd.sh
```

Download [scored datasets](https://pan.baidu.com/s/1iFMebBD6UbjT2LBR2ze6xA?pwd=vngr) to `./my_data/`,
or generate scored datasets from scratch by running:
```bash
sh get_score.sh ${task}
# such as: 
sh get_score.sh paraphrase
```


## Training & Inference with SE-DPO
pref_beta: interval([1.0, 0.001])
```bash
sh run_sedpo.sh ${task} ${pref_beta}

# such as:
sh run_sedpo.sh paraphrase 0.02
# ckpt will be saved to my_data/experiment/paraphrase/saves/dp2-r1d0-b0d02
# inference result will be saved to my_data/experiment/paraphrase/eval_res_for_paraphrase.txt
# finding the best hparam for SE-DPO stage
# using `run_sedpo.sh` will be prefixed with `d`
```
You can check scoring log in `my_data/experiment/paraphrase/scorer.log`, 
training log in `my_data/experiment/paraphrase/train_dense_encoder.log`.

For a specific task cluster, check training log to ensure all subsets are included in "dpr files".
For example, the task cluster of "paraphrase" consist of 3 subsets, the log should be:
```log
[2024-12-15 10:59:25,453][dpr.data.biencoder_data][INFO] - cluster files: ['${PWD}/my_data/scored/paraphrase/*_scored_train.json']
[2024-12-15 10:59:25,455][dpr.data.biencoder_data][INFO] - Toal files num 3
[2024-12-15 10:59:25,455][dpr.data.biencoder_data][INFO] - dpr files: ['${PWD}/my_data/scored/paraphrase/qqp_scored_train.json', '${PWD}/my_data/scored/paraphrase/paws_scored_train.json', '${PWD}/my_data/scored/paraphrase/mrpc_scored_train.json']
```

## Training & Inference with $Se^2$ (Baseline)
```bash
sh run_se2.sh ${task}
# such as:
sh run_se2.sh paraphrase
# ckpt will be saved to my_data/experiment/paraphrase/saves/p0-r1d0
# inference result will be saved to my_data/experiment/paraphrase/eval_res_for_paraphrase.txt
```

## Ablation on Complementary Strength
Finetune pretrained SE-DPO model on $Se^2$, investigating their complementary strength
```bash
sh run_sedpo_se2.sh ${task} ${sedpo_model}
# such as:
sh run_sedpo_se2.sh paraphrase my_data/experiment/paraphrase/saves/dp2-r1d0-b0d02/dpr_biencoder.best_valid
# inference result will be saved to my_data/experiment/paraphrase/eval_res_for_paraphrase.txt
```

Or finetune pretrained $Se^2$ model on SE-DPO.
```bash
sh run_se2_sedpo.sh ${task} ${se2_model} ${pref_beta}
# such as:
sh run_se2_sedpo.sh paraphrase my_data/experiment/paraphrase/saves/dp2-r1d0-b0d02/dpr_biencoder.best_valid 0.02
# inference result will be saved to my_data/experiment/paraphrase/eval_res_for_paraphrase.txt
```

## Enhance SEDPO with RoBERTa
Change the setting in `./DPR/conf/encoder/hf_bert.yaml` as follows:

```bash
# encoder_model_type: hf_bert
# pretrained_model_cfg: google-bert/bert-base-uncased
# pretrained_file:

# download roberta from its offical sources
encoder_model_type: fairseq_roberta
pretrained_model_cfg: ${PWD}/cache/FacebookAI/roberta-base
pretrained_file: ${PWD}/cache/roberta.base
```

## Citation