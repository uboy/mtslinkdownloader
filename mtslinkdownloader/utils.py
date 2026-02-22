import logging
import os
import sys

def get_base_path():
    """Get the path where the application is running (as script or as binary)."""
    if getattr(sys, 'frozen', False):
        # Running as bundled executable (PyInstaller)
        return os.path.dirname(sys.executable)
    # Running as script
    # Look for the project root (where the main script usually resides)
    return os.getcwd()

def get_logs_path():
    base = get_base_path()
    logs_dir = os.path.join(base, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    return os.path.join(logs_dir, 'mtslinkdownloader.log')

def initialize_logger():
    log_file = get_logs_path()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%d.%m.%Y %H:%M:%S',
        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()],
    )

def create_directory_if_not_exists(directory_name: str) -> str:
    base_path = get_base_path()
    full_path = os.path.join(base_path, directory_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path
