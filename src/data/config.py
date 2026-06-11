from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path = "config/data.yaml") -> dict:
    """
    Load the project configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file, relative to the project root
        or absolute.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config
