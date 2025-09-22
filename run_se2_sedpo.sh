set -e
task_name=$1
export PT_MODEL=$2
export IF_PREFER_LOSS=2
export PEFT_BETA=$3
export DIV_REJECT=1

rm -rf  my_data/experiment/${task_name}/dense_retriever_beam_score.log
rm -rf  my_data/experiment/${task_name}/generate_dense_embeddings.log
rm -rf  my_data/experiment/${task_name}/inference.log
rm -rf  my_data/experiment/${task_name}/train_dense_encoder.log
rm -rf  my_data/experiment/${task_name}/model
rm -rf  my_data/experiment/${task_name}/preds_for_${task_name}
rm -rf  my_data/experiment/${task_name}/se2_prompts_for_${task_name}
sh my_data/experiment/${task_name}/train.sh
sh my_data/experiment/${task_name}/infer.sh

