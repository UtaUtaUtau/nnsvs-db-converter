from pathlib import Path
from typing import Any
import json

def load_config(config_path: Path | str) -> dict[str, Any]:
    config: dict[str, Any] = None
    with open(config_path) as f:
        config = json.load(f)
    
    if base_config_path := config.get('base_config'):
        base_config = load_config(base_config_path)
        for k in base_config.keys():
            if new_v := config.get(k):
                base_config[k] = new_v
        return base_config
    else:
        return config