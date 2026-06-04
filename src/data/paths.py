from pathlib import Path

from src.data.config import PROJECT_ROOT


def project_path(*parts: str) -> Path:
    """
    Build an absolute path from the project root.
    """
    return PROJECT_ROOT.joinpath(*parts)


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not already exist.
    """
    path = Path(path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dirs(config: dict) -> dict:
    """
    Return absolute paths for raw, interim and processed data folders.
    """
    paths = config["paths"]

    return {
        "raw": project_path(paths["raw"]),
        "interim": project_path(paths["interim"]),
        "processed": project_path(paths["processed"]),
        "secrets": project_path(paths["secrets"]),
    }


def ensure_data_dirs(config: dict) -> dict:
    """
    Create raw, interim and processed folders if needed.
    """
    data_dirs = get_data_dirs(config)

    for path in data_dirs.values():
        ensure_dir(path)

    return data_dirs
