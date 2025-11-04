import datetime
import json
import os

import hydra
import torch
import torch.distributed as dist
from datasets import Dataset, load_from_disk
from omegaconf import DictConfig
from streaming import StreamingDataset
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, GenerationConfig
from transformers.generation import StopStringCriteria

from src.datasets.streaming_dataset_hf import StreamingHFDataset

from scripts.utils import (
    count_parameters,
    format_number,
    load_model_from_ckpt_dir_path,
    maybe_add_missing_special_tokens,
    register_useful_resolvers,
    set_seed,
)
from src.datasets.preprocessed_dataset import load_preprocessed_dataset
from src.utils import fsspec_exists, fsspec_mkdirs


def gather_results(results, world_size):
    """Gather results from all ranks."""
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, results)

    # gathered_results is now a list of lists (one per rank)
    all_results = []
    for partial in gathered_results:
        all_results.extend(partial)  # type: ignore

    return all_results


def load_progress(progress_path: str) -> set:
    """Load progress tracking dataset and return set of processed indices."""
    if fsspec_exists(progress_path):
        try:
            progress_dataset = load_preprocessed_dataset(progress_path)
            # Extract processed indices from dataset
            # The dataset is in torch format, so we need to access it properly
            processed_indices = set()
            for i in range(len(progress_dataset)):
                idx = progress_dataset[i]["dataset_idx"]
                # Handle both tensor and int types
                if isinstance(idx, torch.Tensor):
                    processed_indices.add(idx.item())
                else:
                    processed_indices.add(int(idx))
            return processed_indices
        except Exception as e:
            print(f"Warning: Could not load progress dataset: {e}. Starting fresh.")
            return set()
    return set()


def save_progress(progress_path: str, processed_indices: set, num_samples: int):
    """Save progress tracking as a HuggingFace dataset."""
    # Convert set to sorted list for consistent ordering
    processed_list = sorted(list(processed_indices))
    
    if not processed_list:
        # Don't save empty progress
        return
    
    # Convert set to list and create dataset
    progress_data = {
        "dataset_idx": processed_list,
        "num_samples": [num_samples] * len(processed_list),  # Store num_samples for reference
    }
    progress_dataset = Dataset.from_dict(progress_data)
    
    # Save to disk
    if not fsspec_exists(progress_path):
        fsspec_mkdirs(progress_path)
    progress_dataset.save_to_disk(progress_path)


def load_or_create_dataset(output_path: str) -> tuple[Dataset, set]:
    """Load existing dataset or create new one. Returns dataset and set of processed indices."""
    progress_path = os.path.join(output_path, "progress")
    processed_indices = set()
    
    # Load progress dataset
    processed_indices = load_progress(progress_path)
    
    if fsspec_exists(output_path):
        try:
            # Try to load existing dataset (excluding progress directory)
            # Check if main dataset exists (not just progress)
            dataset_files = ["dataset_info.json", "state.json"]
            has_main_dataset = any(fsspec_exists(os.path.join(output_path, f)) for f in dataset_files)
            
            if has_main_dataset:
                existing_dataset = load_from_disk(output_path)
                return existing_dataset, processed_indices
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"Warning: Could not load existing dataset: {e}. Starting fresh.")
    
    # Create new dataset
    return None, processed_indices


def save_dataset_incremental(
    output_path: str,
    new_samples: list[dict],
    existing_dataset: Dataset | None = None,
    save_interval: int = 100,
) -> Dataset:
    """Save dataset incrementally by appending new samples."""
    if existing_dataset is None:
        # Create new dataset
        distillation_dataset = Dataset.from_list(new_samples)
    else:
        # Append to existing dataset
        new_dataset = Dataset.from_list(new_samples)
        distillation_dataset = Dataset.concatenate_datasets([existing_dataset, new_dataset])
    
    # Save to disk
    if not fsspec_exists(output_path):
        fsspec_mkdirs(output_path)
    distillation_dataset.save_to_disk(output_path)
    
    return distillation_dataset


def setup_ddp() -> int:
    """Sets up torch.distributed and selects GPU.

    Returns:
        (int) local_rank
    """
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=120))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


@hydra.main(version_base=None, config_path="../../configs", config_name="data_gen_config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    local_rank = setup_ddp()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)

    # Load the dataset
    dataset = hydra.utils.instantiate(
        cfg.dataset,
        tokenizer=tokenizer,
    )
    
    # Create distributed sampler for distributed inference
    # Note: Streaming datasets don't use DistributedSampler
    sampler = None
    if not isinstance(dataset, StreamingDataset) and not isinstance(dataset, StreamingHFDataset):
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=getattr(cfg.dataloader, "shuffle", False),
            drop_last=getattr(cfg.dataloader, "drop_last", False),
        )
    
    # Instantiate dataloader from config
    dataloader = hydra.utils.instantiate(
        cfg.dataloader,
        _convert_="partial",
        dataset=dataset,
        sampler=sampler,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.pretrained_model_name_or_path,
        trust_remote_code=True,
        revision=getattr(cfg, "pretrained_model_revision", None),
    )
    model = model.to(device)
    if local_rank == 0:
        print(f"Num. params: {format_number(count_parameters(model, trainable=False))}")
        print(f"Num. trainable params: {format_number(count_parameters(model))}")
    model.eval()
    
    # Instantiate generation kwargs
    gen_kwargs = hydra.utils.instantiate(cfg.gen_kwargs)
    stop_tokens = None
    if getattr(gen_kwargs, "stopping_criteria", None) is not None:
        for sc in gen_kwargs["stopping_criteria"]:
            if isinstance(sc, StopStringCriteria):
                stop_tokens = list(sc.stop_strings)
                break
    
    # Merge model generation config with gen_kwargs generation config
    if getattr(model, "generation_config", None) is not None:
        model_gen_config = model.generation_config.to_dict()
        gen_kwargs["generation_config"] = GenerationConfig(**{**model_gen_config, **gen_kwargs["generation_config"].to_dict()})

    # Setup resumption: load existing dataset and progress
    save_interval = getattr(cfg, "save_interval", 100)  # Save every N samples
    progress_path = os.path.join(cfg.output_path, "progress")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    if local_rank == 0:
        if not fsspec_exists(cfg.output_path):
            fsspec_mkdirs(cfg.output_path)
        existing_dataset, global_processed_indices = load_or_create_dataset(cfg.output_path)
        if existing_dataset is not None:
            print(f"Resuming: Found {len(existing_dataset)} existing samples")
    else:
        existing_dataset = None
        global_processed_indices = set()
    
    # Broadcast processed indices to all ranks
    if dist.is_initialized():
        global_processed_indices_list = [None] * world_size
        dist.all_gather_object(
            global_processed_indices_list,
            list(global_processed_indices) if local_rank == 0 else [],
        )
        # Merge all processed indices from all ranks
        global_processed_indices = set()
        for indices_list in global_processed_indices_list:
            if indices_list:
                global_processed_indices.update(indices_list)

    # Subset dataset based on processed indices (only for non-streaming datasets)
    is_streaming = isinstance(dataset, StreamingDataset) or isinstance(dataset, StreamingHFDataset)
    subset_to_original_idx_map = None
    original_dataset_size = len(dataset) if not is_streaming else None
    
    if not is_streaming and global_processed_indices:
        # Calculate unprocessed indices
        all_indices = set(range(len(dataset)))
        unprocessed_indices = sorted(list(all_indices - global_processed_indices))
        
        if unprocessed_indices:
            if local_rank == 0:
                print(f"Subsetting dataset: {len(unprocessed_indices)} unprocessed samples out of {len(dataset)} total")
            # Create mapping from subset indices to original indices
            # subset index i -> original index unprocessed_indices[i]
            subset_to_original_idx_map = {i: orig_idx for i, orig_idx in enumerate(unprocessed_indices)}
            # Subset the dataset to only include unprocessed indices
            dataset = dataset.select(unprocessed_indices)
            # Need to recreate sampler and dataloader with filtered dataset
            sampler = DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=getattr(cfg.dataloader, "shuffle", False),
                drop_last=getattr(cfg.dataloader, "drop_last", False),
            )
            dataloader = hydra.utils.instantiate(
                cfg.dataloader,
                _convert_="partial",
                dataset=dataset,
                sampler=sampler,
            )
        else:
            if local_rank == 0:
                print("All samples have already been processed. Nothing to do.")
            if dist.is_initialized():
                dist.destroy_process_group()
            return

    # Get sampler indices to determine which samples this rank processes
    # For regular datasets after subsetting, indices are remapped, so we need to track original indices
    # For streaming datasets, we still need to check against global_processed_indices
    sampler_indices = list(sampler) if sampler is not None else (list(range(len(dataset))) if not is_streaming else None)
    
    # Iterate through the dataset and generate samples
    local_samples_buffer = []  # Buffer for this rank's samples
    local_indices_buffer = []  # Buffer for this rank's dataset indices
    total_processed = 0
    
    for batch_idx, elem in tqdm(
        enumerate(dataloader),
        desc=f"Generating samples (rank {local_rank})",
        total=len(dataloader),
        disable=(local_rank != 0),
    ):
        # Calculate the actual dataset indices for this batch
        if sampler_indices is not None:
            batch_start_idx = batch_idx * cfg.dataloader.batch_size
            batch_indices = sampler_indices[batch_start_idx:batch_start_idx + cfg.dataloader.batch_size]
        else:
            # For streaming datasets, we can't pre-compute indices
            batch_indices = None
        
        input_ids = elem["input_ids"].to(device)  # type: ignore
        batch_size = input_ids.shape[0]
        
        # Handle target_prompt_text if dataset has it
        if hasattr(dataset, "target_prompt_text") and dataset.target_prompt_text is not None:
            post_prompt_ids = (
                torch.tensor(tokenizer.encode(dataset.target_prompt_text.strip()))
                .to(input_ids.dtype)
                .to(input_ids.device)
                .unsqueeze(0)
            )
            # Add prompt to each item in the batch
            post_prompt_ids = post_prompt_ids.repeat(batch_size, 1)
            prompt_ids = torch.cat((input_ids[:, :torch.where(input_ids[0] == tokenizer.eos_token_id)[0][1] + 1], post_prompt_ids), dim=-1)
        # Generate samples
        with torch.no_grad():
            outputs = model.generate(
                inputs=prompt_ids,
                **gen_kwargs,
            )
        # Process each sample in the batch
        for i in range(batch_size):
            # Get dataset index (subset index for regular datasets after subsetting, original index otherwise)
            if batch_indices is not None:
                subset_idx = batch_indices[i]
            else:
                # For streaming datasets without pre-computed indices, we can't track original indices
                # This is a limitation of streaming datasets
                subset_idx = None
            
            # Map subset index to original index if we have a mapping
            if subset_to_original_idx_map is not None and subset_idx is not None:
                dataset_idx = subset_to_original_idx_map[subset_idx]
            elif subset_idx is not None:
                dataset_idx = subset_idx
            else:
                # For streaming datasets, we can't reliably track indices
                # Use a placeholder or skip tracking
                dataset_idx = -1  # Indicates we can't track the index
                # Note: For streaming datasets, we can't skip already-processed samples
                # because we don't have reliable index tracking. This is a limitation
                # of streaming datasets.

            # Add EOS token to output (truncated from stopping criteria)
            sample_output_ids = torch.cat((outputs[i], torch.tensor([tokenizer.eos_token_id]).to(input_ids.device)), dim=-1)
            
            # Add to buffer
            local_samples_buffer.append({
                "input_ids": sample_output_ids.cpu().tolist(),
                "original_input_ids": input_ids[i].cpu().tolist(),
                "dataset_idx": dataset_idx,  # Store original index for tracking
            })
            if dataset_idx != -1:  # Only track indices we can reliably map
                local_indices_buffer.append(dataset_idx)
            total_processed += 1
            
            # Save incrementally if buffer is large enough (gather from all ranks)
            if len(local_samples_buffer) >= save_interval:
                # Gather all ranks' samples
                all_samples = gather_results(local_samples_buffer, world_size)
                all_indices = gather_results(local_indices_buffer, world_size)
                
                # Only rank 0 saves
                if local_rank == 0:
                    # Update processed indices
                    global_processed_indices.update(all_indices)
                    
                    # Save dataset incrementally
                    existing_dataset = save_dataset_incremental(
                        cfg.output_path,
                        all_samples,
                        existing_dataset,
                        save_interval,
                    )
                    
                    # Save progress
                    save_progress(progress_path, global_processed_indices, len(existing_dataset))
                    
                    print(f"Saved {len(existing_dataset)} samples (processed {len(global_processed_indices)} indices)")
                
                # Broadcast updated processed indices to all ranks
                if dist.is_initialized():
                    indices_list = [None] * world_size
                    dist.all_gather_object(
                        indices_list,
                        list(global_processed_indices) if local_rank == 0 else [],
                    )
                    # Update all ranks' processed indices
                    global_processed_indices = set()
                    for idx_list in indices_list:
                        if idx_list:
                            global_processed_indices.update(idx_list)
                
                # Clear buffers
                local_samples_buffer = []
                local_indices_buffer = []

    # Gather and save remaining samples
    if local_samples_buffer:
        all_samples = gather_results(local_samples_buffer, world_size)
        all_indices = gather_results(local_indices_buffer, world_size)
        
        if local_rank == 0:
            global_processed_indices.update(all_indices)
            existing_dataset = save_dataset_incremental(
                cfg.output_path,
                all_samples,
                existing_dataset,
                save_interval,
            )
            save_progress(progress_path, global_processed_indices, len(existing_dataset))
    
    if local_rank == 0:
        if existing_dataset is None:
            print("No samples collected.")
        else:
            print(f"\nFinal dataset saved to {cfg.output_path}")
            print(f"Total samples: {len(existing_dataset)}")
            print(f"Total processed indices: {len(global_processed_indices)}")
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    register_useful_resolvers()
    main()

