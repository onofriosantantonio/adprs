import os
import yaml
from pathlib import Path
from typing import Any, Dict

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Carica il file di configurazione YAML del progetto.
    """
    path = Path(config_path)
    if not path.is_absolute():
        # Calcola il percorso relativo alla radice del progetto
        root_dir = Path(__file__).resolve().parent.parent.parent
        path = root_dir / config_path

    if not path.exists():
        raise FileNotFoundError(f"File di configurazione non trovato in {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
