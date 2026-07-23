import logging
import logging.config
from app.utils.cache import setup_cache_directories
from app.configs.settings import settings

# Apply logging configuration from settings
if settings.logging:
    try:
        logging.config.dictConfig(settings.logging)
    except Exception as e:
        # Fallback basic logging if dictConfig fails
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        logging.warning(f"Failed to apply logging config from settings: {e}. Using basicConfig fallback.")
else:
    # Default fallback setup
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# Expose a centralized logger
logger = logging.getLogger("manipurigpt")
