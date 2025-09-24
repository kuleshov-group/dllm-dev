#!/bin/bash

# Setup TorchInductor cache for specific GPU types
# Usage: setup_torchinductor_cache.sh <gpu_type>

set -eo pipefail

GPU_TYPE=${1:-"unset"}

echo "Setting up TorchInductor cache for GPU type: $GPU_TYPE"

# Download and unzip TorchInductor cache, if one exists.
if [ $GPU_TYPE = "h100_sxm5" ] || [ $GPU_TYPE = "a100_sxm4" ]; then
  echo "GPU is $GPU_TYPE, checking for TorchInductor cache..."
  
  if [ ! -d .ti_cache ]; then
    echo "TorchInductor cache not found, downloading and extracting..."
    CACHE_URL="https://huggingface.co/eczech/dllm-dev/resolve/main/cache/torchinductor-$GPU_TYPE-ubuntu22.04x86-py311-cuda128-torch270-triton330.zip"
    TEMP_DIR=$(mktemp -d)
    CACHE_ZIP="$TEMP_DIR/torchinductor-cache.zip"
    
    echo "Downloading TorchInductor cache to temporary location ($CACHE_ZIP)"
    wget -q -O "$CACHE_ZIP" "$CACHE_URL"
    
    echo "Extracting TorchInductor cache ($CACHE_ZIP) to temporary location ($TEMP_DIR)"
    unzip -q "$CACHE_ZIP" -d "$TEMP_DIR"
    
    echo "Moving cache to local directory (.ti_cache)"
    mv "$TEMP_DIR/.ti_cache" .ti_cache
    
    echo "Cleaning up temporary files ($TEMP_DIR)"
    rm -rf "$TEMP_DIR"
    
    echo "TorchInductor cache setup complete."
  else
    echo "TorchInductor cache already exists, skipping download and extraction."
  fi
else
  echo "GPU is $GPU_TYPE (not h100_sxm5 or a100_sxm4), skipping TorchInductor cache download and extraction."
fi
