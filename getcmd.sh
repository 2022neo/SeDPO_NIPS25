export OUTPUT_DIR=my_data # main experience folder
export CACHE_DIR=cache # folder caching the LLM checkpoints, task datasets, etc.
export HYDRA_FULL_ERROR=1
export SCORER_BSZ=${SCORER_BSZ:-10}
export SCORE_LLM=${SCORE_LLM:-'EleutherAI/gpt-neo-2.7B'} # LLM to score the data
export INF_LLM=${INF_LLM:-'EleutherAI/gpt-neo-2.7B'} # LLM for inference

retriever_bsz=${RETRIEVER_BSZ:-32}
num_of_gpus=${NUM_OF_GPUS:-8}
for task in paraphrase reading nli coreference;do
    export TASK=${task}
    export TRAIN_CLUSTERS=${task}
    export TEST_CLUSTERS=${task}
    python get_cmds.py \
        --output_dir ${OUTPUT_DIR} \
        --model_folder "model" \
        --train_clusters ${TRAIN_CLUSTERS} \
        --test_clusters ${TEST_CLUSTERS} \
        --scr_model ${SCORE_LLM} \
        --inf_model ${INF_LLM} \
        --retriever_bsz ${retriever_bsz} \
        --gpus ${num_of_gpus} \
        --cache_dir ${CACHE_DIR}
done