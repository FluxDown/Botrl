import yaml
import os


def load_config(config_path='config.yaml'):
    """Charge la configuration depuis un fichier YAML"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def save_config(config, save_path):
    """Sauvegarde la configuration dans un fichier YAML"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
