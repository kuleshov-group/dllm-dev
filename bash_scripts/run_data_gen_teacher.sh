#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Example configuration for collecting distillation samples from GSM8K
# Modify these parameters as needed

# Model configuration
MODEL_PATH="Qwen/Qwen3-1.7B"  # Path to model checkpoint or HuggingFace model
# MODEL_PATH="${RUN_DIR}/<PATH_TO_SAVED_MODEL_DIR>"  # Alternative: local checkpoint path
REVISION=null

# Generation parameters
L=256  # max_new_tokens
DO_SAMPLE=false
MAX_LENGTH=1024

# Dataset configuration
DATASET_CONFIG="gsm8k_train"  # Options: gsm8k_train, gsm8k_eval, wmt_train, etc.

# Output configuration
OUTPUT_DIR="outputs/distillation/${MODEL_PATH##*/}/${DATASET_CONFIG}"
OUTPUT_PATH="${OUTPUT_DIR}/L-${L}-do_sample-${DO_SAMPLE}"
mkdir -p ${OUTPUT_PATH}

# Distributed training setup
NUM_VISIBLE_DEVICES=${NUM_VISIBLE_DEVICES:-1}  # Number of GPUs to use
PORT=29500

# Run the collection script
torchrun --nproc_per_node ${NUM_VISIBLE_DEVICES} --master_port=${PORT} scripts/data_gen/collect_samples.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  dataset=${DATASET_CONFIG} \
  dataset.max_length=${MAX_LENGTH} \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  tokenizer.pretrained_model_name_or_path="Qwen/Qwen3-0.6B-Base" \
  output_path=${OUTPUT_PATH} \
  max_length=${MAX_LENGTH} \
  max_new_tokens=${L} \
  batch_size=1 \
  dataloader.batch_size=1 \
  dataloader.num_workers=0 \
  generation_config.do_sample=${DO_SAMPLE} \
  generation/stopping_criteria@stopping_criteria_list='[max_length_criteria,eos_token_criteria]' \
  ~generation/logits_processor@logits_processor_list \
  gen_kwargs.logits_processor=null

