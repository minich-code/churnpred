import os
# Exception for handling Box value errors
from box.exceptions import BoxValueError
import sys
import yaml
from src.churn import logging
import json
import joblib
import numpy as np
# Decorator for runtime type checking
from ensure import ensure_annotations
# Enhanced dictionary that allows for dot notation access
from box import ConfigBox
from pathlib import Path
from typing import Any


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox object.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Returns:
        ConfigBox: The contents of the YAML file as a ConfigBox object.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            # Load the YAML file content
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            print(content)
            return ConfigBox(content)
    except BoxValueError:
        # Raise an error if the YAML file is empty
        raise ValueError("yaml file is empty")
    except Exception as e:
        # Raise any other exceptions that occur
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Creates directories specified in the list if they do not exist.

    Args:
        path_to_directories (list): List of directory paths to create.
        verbose (bool): If True, logs the directory creation.
    """
    for path in path_to_directories:
        # Create the directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"created directory at: {path}")

def save_object(file_path, obj):
    """
    Saves a Python object to a file using joblib.

    Args:
        file_path (str): Path where the object will be saved.
        obj (Any): The Python object to save.
    """
    # Get the directory path of the file
    dir_path = os.path.dirname(file_path)
    # Create the directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)
    # Save the object to the file
    with open(file_path, 'wb') as file_obj:
        joblib.dump(obj, file_obj)

def load_object(file_path):
    """
    Loads a Python object from a file using joblib.

    Args:
        file_path (str): Path of the file to load the object from.

    Returns:
        Any: The loaded Python object.
    """
    # Load the object from the file
    with open(file_path, 'rb') as file_obj:
        return joblib.load(file_obj)

@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Saves a dictionary to a JSON file.

    Args:
        path (Path): Path where the JSON file will be saved.
        data (dict): Dictionary to save as JSON.
    """
    with open(path, "w") as f:
        # Dump the dictionary to a JSON file
        json.dump(data, f, indent=4)
    logging.info(f"json file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Loads a JSON file and returns its contents as a ConfigBox object.

    Args:
        path (Path): Path of the JSON file to load.

    Returns:
        ConfigBox: The contents of the JSON file as a ConfigBox object.
    """
    with open(path) as f:
        # Load the JSON file content
        content = json.load(f)
    logging.info(f"json file loaded successfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Saves data to a binary file using joblib.

    Args:
        data (Any): Data to save.
        path (Path): Path where the binary file will be saved.
    """
    joblib.dump(value=data, filename=path)
    logging.info(f"binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Loads data from a binary file using joblib.

    Args:
        path (Path): Path of the binary file to load.

    Returns:
        Any: The loaded data.
    """
    data = joblib.load(path)
    logging.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Gets the size of the file at the given path in kilobytes.

    Args:
        path (Path): Path of the file.

    Returns:
        str: Size of the file in kilobytes, rounded to the nearest whole number.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"