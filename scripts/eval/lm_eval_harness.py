#!/usr/bin/env python3
"""
LM Eval Harness CLI wrapper with Hydra configuration support.
"""

import logging
import subprocess
import sys
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def build_cli_command(cfg: DictConfig) -> List[str]:
    """
    Build the lm_eval CLI command from Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        List of command arguments for subprocess
    """
    command = ["python", "-m", "lm_eval"]
    
    # Add value-based arguments (--arg value)
    if hasattr(cfg, 'args') and cfg.args:
        for arg_name, arg_value in cfg.args.items():
            if arg_value is not None:
                command.extend([f"--{arg_name}", str(arg_value)])
    
    # Add boolean flags (--flag)
    if hasattr(cfg, 'flags') and cfg.flags:
        for flag_name, flag_value in cfg.flags.items():
            if flag_value:
                command.append(f"--{flag_name}")
    
    return command


def validate_environment() -> bool:
    """
    Validate that lm_eval is available in the environment.
    
    Returns:
        True if lm_eval is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["python", "-m", "lm_eval", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def run_evaluation(command: List[str]) -> int:
    """
    Run the lm_eval command and return the exit code.
    
    Args:
        command: Command arguments to execute
        
    Returns:
        Exit code from the subprocess
    """
    logger.info(f"Executing LM Eval CLI command:")
    logger.info(f"   {' '.join(command)}\n")
    
    try:
        # Run the command and stream output in real-time
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            logger.info(line.rstrip())
        
        # Wait for completion
        process.wait()
        return process.returncode
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        process.terminate()
        return 130
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        return 1


@hydra.main(version_base=None, config_path="../../configs/eval/lm_eval_harness", config_name="cli")
def main(cfg: DictConfig) -> None:
    """
    Main function that loads Hydra config and runs lm_eval CLI.
    
    Args:
        cfg: Hydra configuration object
    """
    logger.info("Starting LM Eval Harness CLI wrapper")
    
    # Print configuration for debugging
    logger.info("Configuration:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    
    # Validate environment
    if not validate_environment():
        logger.error("lm_eval is not available in the current environment")
        sys.exit(1)
    
    # Build CLI command
    command = build_cli_command(cfg)
    
    # Run evaluation
    exit_code = run_evaluation(command)
    
    if exit_code == 0:
        logger.info("Evaluation completed successfully")
    else:
        logger.error(f"Evaluation failed with exit code: {exit_code}")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
