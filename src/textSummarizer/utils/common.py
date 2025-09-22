import os
from box.exceptions import BoxValueError
import yaml
from textSummarizer.logging import logger
from box import ConfigBox
from pathlib import Path
from typing import Any, Union, Iterable

# Patched version without @ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a yaml file and returns"""
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError as e:
        raise e
    except Exception as e:
        logger.error(f"Error occurred while reading yaml file: {e}")
        raise e

# Patched version without @ensure_annotations
def create_directories(path_to_directories: Iterable[Path], verbose: bool = True):
    """Create directories without ensure_annotations"""
    for path in path_to_directories:
        path = Path(path)  # Ensure it's a Path object
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

# Patched version without @ensure_annotations  
def get_size(path: Path) -> str:
    """get size in KB"""
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"