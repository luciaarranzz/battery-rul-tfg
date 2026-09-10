"""Utilidades comunes: semillas reproducibles y logging sencillo."""
import logging
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fija las semillas de random y numpy (y de torch/pytorch si está instalado)
    para que los experimentos sean reproducibles."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def get_logger(name: str = "battery_rul") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger