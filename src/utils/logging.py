"""One logger format for the whole project."""
import logging

QUIET = ("httpx", "httpx2", "httpcore", "urllib3", "huggingface_hub")


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s", datefmt="%H:%M:%S")
    for noisy in QUIET:  # one line per HTTP request drowns the training log
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)
