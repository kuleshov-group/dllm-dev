#!/bin/bash

# runs to copy. use * to copy all checkpoints
RUN_NAME="gsm8k-bs96-block32-keep8-lr1e-5-warmup5000ba-schedlinear_decay_with_warmup-gc1.0-wd1e-5-sweep3"

# ssh into remote machine
REMOTE_HOST="train_node" # specified by ~/.ssh/config
REMOTE_DIR=/home/ubuntu/runs/dllm-dev/$RUN_NAME
LOCAL_DEST=/share/kuleshov/ma2238/runs/dllm-dev/$RUN_NAME

# Step 1: ssh and zip files on the remote machine
ssh "${REMOTE_HOST}" << EOF
cd "${REMOTE_DIR}" || exit 1
tar --zstd -cvf "checkpoints.tar.zst" ${REMOTE_DIR}
EOF

# Step 2: scp the zipped file to local machine
scp "${REMOTE_HOST}:${REMOTE_DIR}/checkpoints.tar.zst" "${LOCAL_DEST}"

# Step 3: unzip the file on the local machine
tar --zstd -xvf "${LOCAL_DEST}/checkpoints.tar.zst" -C "${LOCAL_DEST}"
