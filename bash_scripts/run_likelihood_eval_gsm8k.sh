#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

#MODEL_PATH="kuleshov-group/e2d2-gsm8k-finetune-Qwen3-2B"
MODEL_PATH="/share/kuleshov/yzs2/runs/dllm-dev/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec14_e2d2_debug_v19_tie-weights"
# MODEL_PATH="outputs/<PATH_TO_SAVED_MODEL_DIR>"
REVISION=null

EVAL_DATASET="gsm8k_eval"
BLOCK_SIZE=4  # TODO: Change as needed
BATCH_SIZE=8
PRETRAINED_MODEL_NAME_OR_PATH="Qwen/Qwen3-1.7B-Base"  # TODO: Change as needed
CKPT_FILE="best-rank0.pt"
USE_EMA=true

composer -n ${NUM_VISIBLE_DEVICES} scripts/eval/likelihood_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=none \
  hydra/hydra_logging=none \
  +eval@task=likelihood \
  +dataset@task.eval_dataset=${EVAL_DATASET} \
  task.load_ema_weights=${USE_EMA} \
  task.ckpt_file=${CKPT_FILE} \
  batch_size=${BATCH_SIZE} \
  block_size=${BLOCK_SIZE} \
  task.eval_dataloader.batch_size=${BATCH_SIZE} \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  tokenizer.pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  output_path=null \
  task.collator.block_size=${BLOCK_SIZE} \
  task.collator.global_batch_size=${BATCH_SIZE} \
  +metrics@task.metrics='[loss,nll,bpd,perplexity]' \
  +composer/trainer@task.trainer=eval_trainer \
  ~generation@generation_config \
  ~generation/logits_processor@logits_processor_list \
  ~generation/stopping_criteria@stopping_criteria_list \
  gen_kwargs=null
