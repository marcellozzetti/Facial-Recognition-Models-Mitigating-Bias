import logging

def setup_logging(config, log_file_key):
    """
    Set up logging configuration.
    """
    logging.basicConfig(
        level=config['logging']['log_level'],
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(config['logging'][log_file_key]),
            logging.StreamHandler()
        ]
    )