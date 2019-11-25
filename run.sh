#!/usr/bin/env bash

source /home/LAB/meijj/Env/miniconda3/etc/profile.d/conda.sh
conda activate allennlp
export PATH="/home/LAB/meijj/Env/cudnn-10.1-v7.6.3:/usr/local/cuda-10.0/bin:/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/NsightCompute-2019.1${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="/home/LAB/meijj/Env/cudnn-10.1-v7.6.3/lib64:/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"


dataset="${1}"
mode="${2}"
[[ "${dataset}" == "" ]] && exit 1
[[ "${mode}" == "" ]] && exit 1


path_data=${HOME}/Datasets/semi_supervised
[[ ${dataset} == imdb ]] && path_data=${path_data}/imdb_2375_500/imdb_json
[[ ${dataset} == elec ]] && path_data=${path_data}/elec_uncased_2375_500/elec_json


export DATA_DIR=${path_data}
export VOCAB_SIZE=30000
export VAMPIRE_DIR="$(pwd)/model_logs/vampire"



if [[ ${mode} == preprocess ]]; then
  srun -p cpu python -m scripts.preprocess_data \
  --train-path ${path_data}/train_labeled_unlabeled.jsonl \
  --dev-path ${path_data}/dev.jsonl \
  --tokenize \
  --tokenizer-type spacy \
  --vocab-size ${VOCAB_SIZE} \
  --serialization-dir ${DATA_DIR} > log_data.txt 2>&1
elif [[ ${mode} == pretrain ]]; then
  srun -p sugon --gres=gpu:P100:1 python -m scripts.train \
  --config training_config/vampire.jsonnet \
  --serialization-dir ${VAMPIRE_DIR} \
  --environment VAMPIRE \
  --device 0 > log_pretrain.txt 2>&1
elif [[ ${mode} == train ]]; then
  export VAMPIRE_DIM=81
  #export THROTTLE=200
  export EVALUATE_ON_TEST=1

  srun -p sugon --gres=gpu:P100:1 python -m scripts.train \
  --config training_config/classifier.jsonnet \
  --serialization-dir model_logs/clf \
  --environment CLASSIFIER \
  --device 0 > log_class.txt 2>&1
fi
