import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Resolve paths
CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent.parent

class ConfigNamespace:
    """Helper class to turn nested dictionaries into object attributes."""
    def __init__(self, dictionary: dict):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNamespace(value))
            else:
                setattr(self, key, value)
                
    def to_dict(self) -> dict:
        """Converts namespace back to a dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNamespace):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

class Settings:
    """Centralized settings manager reading from YAML files and env variables."""
    def __init__(self):
        # Default empty configurations
        self.training = ConfigNamespace({})
        self.datasets = ConfigNamespace({})
        self.model = ConfigNamespace({})
        self.tokenizer = ConfigNamespace({})
        self.logging = {}

        self.load_configs()
        self.override_from_env()

    def load_configs(self):
        """Loads yaml configs from the config directory."""
        # 1. Training Configuration
        training_path = CONFIG_DIR / "training.yaml"
        if training_path.exists():
            with open(training_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self.training = ConfigNamespace(data.get("training", {}))

        # 2. Datasets Configuration
        datasets_path = CONFIG_DIR / "datasets.yaml"
        if datasets_path.exists():
            with open(datasets_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self.datasets = ConfigNamespace(data.get("datasets", {}))

        # 3. Models Configuration
        models_path = CONFIG_DIR / "models.yaml"
        if models_path.exists():
            with open(models_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self.model = ConfigNamespace(data.get("model", {}))

        # 4. Tokenizer Configuration
        tokenizer_path = CONFIG_DIR / "tokenizer.yaml"
        if tokenizer_path.exists():
            with open(tokenizer_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self.tokenizer = ConfigNamespace(data.get("tokenizer", {}))

        # 5. Logging Configuration
        logging_path = CONFIG_DIR / "logging.yaml"
        if logging_path.exists():
            with open(logging_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self.logging = data.get("logging", {})

    def override_from_env(self):
        """Overrides settings using specific environment variables if defined."""
        # Override HF token
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            setattr(self.model, "hf_token", hf_token)

        # Override Cache Dir
        cache_dir = os.getenv("CACHE_DIR")
        if cache_dir:
            setattr(self.datasets, "cache_dir", cache_dir)

        # Override Log Level
        log_level = os.getenv("LOG_LEVEL")
        if log_level and "root" in self.logging:
            self.logging["root"]["level"] = log_level

# Instantiate global settings object
settings = Settings()
