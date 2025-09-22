set -e
task_name=$1
export HYDRA_FULL_ERROR=1
sh my_data/experiment/${task_name}/score.sh

