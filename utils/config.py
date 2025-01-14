import yaml

def load_config():
    """
    Load configuration from a YAML file.
    """
    config_path = load_config('configs/default.yaml')
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config