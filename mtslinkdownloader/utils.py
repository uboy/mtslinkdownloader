import logging
import os
import sys


def normalize_log_level(level: str) -> int:
    if isinstance(level, int):
        return level
    text = str(level or "INFO").strip().upper()
    value = getattr(logging, text, None)
    if not isinstance(value, int):
        raise ValueError(f"Unsupported log level: {level}")
    return value


def get_base_path():
    """Get the path where the application is running (as script or as binary)."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.getcwd()

def get_logs_path():
    base = get_base_path()
    logs_dir = os.path.join(base, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    return os.path.join(logs_dir, 'mtslinkdownloader.log')


def initialize_logger(force: bool = False, stream=None, level: str = "INFO"):
    root_logger = logging.getLogger()
    if root_logger.handlers and not force:
        return

    log_file = get_logs_path()
    logging.basicConfig(
        level=normalize_log_level(level),
        format='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(stream),
        ],
        force=force,
    )


def create_directory_if_not_exists(directory_name: str) -> str:
    base_path = get_base_path()
    full_path = os.path.join(base_path, directory_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path


def restore_terminal():
    """Ensure terminal echo and cursor are restored on Linux/macOS."""
    if os.name != 'nt': # Linux/macOS
        try:
            # Send escape sequence to show cursor
            sys.stdout.write('\033[?25h')
            sys.stdout.flush()
            # Restore echo
            os.system('stty echo')
        except Exception:
            pass
