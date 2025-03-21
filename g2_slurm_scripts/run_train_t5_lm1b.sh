#!/bin/bash
#SBATCH -J lm1b_dit                   # Job name
#SBATCH -o ../watch_folder/%x_%j.out  # Output file (%j expands to jobID)
#SBATCH --get-user-env                # Retrieve the users login environment
#SBATCH --partition=kuleshov               # Request partition
#SBATCH --constraint="[a100|a6000|a5000|3090]"
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --mem=64000                   # Server memory requested (per node)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:8                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption
#SBATCH --mail-user=yzs2@cornell.edu  # Email
#SBATCH --mail-type=END               # Request status by email


# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

composer -n ${SLURM_GPUS_ON_NODE} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=lm1b-t5-test_ao \
  tokenizer.pretrained_model_name_or_path=bert-base-uncased \
  dataset@train_dataset=lm1b_preprocessed_train \
  dataset@eval_dataset=lm1b_preprocessed_eval \
  model=aomdlm \
  model/backbone@model.config.backbone_config=t5 \
  model/backbone@model.config.backbone_config.encoder_net=dit \
  model/backbone@model.config.backbone_config.decoder_net=dit \
  model.config.backbone_config.encoder_net.n_blocks=8 \
  model.config.backbone_config.decoder_net.n_blocks=4 \
  model.config.backbone_config.encoder_hidden_dim='$\{model.config.backbone_config.encoder_net.hidden_size\}' \
  model.config.backbone_config.decoder_hidden_dim='$\{model.config.backbone_config.decoder_net.hidden_size\}' \
  model.config.length=128 \
  ~trainer.parallelism_config
