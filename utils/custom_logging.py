import logging
import os

def setup_logging(config, log_file_key):
    """
    Set up logging configuration.
    """
    log_dir = config['logging']["log_dir"]
    log_file = os.path.join(log_dir, config['logging'][log_file_key])

    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=config['logging']['log_level'],
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )