import logging

def setup_logging(config, log_file):
    """
    Set up logging configuration.
    """
    logging.basicConfig(
        level=config['logging']['log_level'],
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(config['logging'][log_file]),
            logging.StreamHandler()
        ]
    )