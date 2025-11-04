import logging
import os
from datetime import datetime


def create_log_dir(log_dir_root="logs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_dir_root, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    return log_dir


def setup_logger(log_dir):
    logger = logging.getLogger("TrainingLog")
    logger.setLevel(logging.INFO)

    logger.propagate = False

    log_file_path = os.path.join(log_dir, "training.log")
    file_handler = logging.FileHandler(log_file_path)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
