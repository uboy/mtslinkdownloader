import logging
import os

LOGS_ROOT = 'logs'
LOG_FILENAME = 'mtslinkdownloader.log'
LOG_FILEPATH = os.path.join(LOGS_ROOT, LOG_FILENAME)


def initialize_logger():
    os.makedirs(LOGS_ROOT, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%d.%m.%Y %H:%M:%S',
        handlers=[logging.FileHandler(LOG_FILEPATH), logging.StreamHandler()],
        encoding='utf-8'
    )


def create_directory_if_not_exists(directory: str) -> str:
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory
