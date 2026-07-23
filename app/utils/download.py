import os
import urllib.request
from pathlib import Path
from typing import Any
from app.utils.logger import logger

def download_if_missing(url: str, dest_path: str) -> str:
    """
    Downloads a file from a URL to the destination path if it doesn't already exist.
    Args:
        url (str): The URL to download from.
        dest_path (str): The local path where the file should be saved.
    Returns:
        str: Path to the downloaded file.
    """
    path = Path(dest_path)
    if path.exists():
        logger.info(f"File already exists: {path}. Skipping download.")
        return str(path)

    # Ensure target directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {url} to {path}...")
    
    try:
        # Check if tqdm is installed for a progress bar
        try:
            from tqdm import tqdm
            
            class DownloadProgressBar(tqdm):
                def update_to(self, b=1, bsize=1, tsize=None):
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)
            
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=path.name) as t:
                urllib.request.urlretrieve(url, filename=str(path), reporthook=t.update_to)
                
        except ImportError:
            # Fallback to basic download without progress bar
            urllib.request.urlretrieve(url, filename=str(path))
            
        logger.info(f"Download complete: {path}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if path.exists():
            path.unlink()  # Remove partial download
        raise e

    return str(path)


def hf_load_dataset_with_backoff(repo: str, *args, max_retries: int = 5, backoff_factor: float = 2.0, **kwargs) -> Any:
    """
    Loads a dataset from HuggingFace Hub with automatic token injection and exponential backoff retry
    to handle transient 503 Service Unavailable, 429 Rate Limit, and ReadError/Connection timeouts.
    """
    import time
    import datasets
    from app.configs.settings import settings
    
    # Inject HF_TOKEN if not explicitly passed
    if "token" not in kwargs:
        token = os.getenv("HF_TOKEN", getattr(settings.model, "hf_token", None))
        if token:
            kwargs["token"] = token

    if "cache_dir" not in kwargs:
        from app.utils.cache import setup_cache_directories
        dirs = setup_cache_directories()
        kwargs["cache_dir"] = dirs["datasets"]

    retries = 0
    while True:
        try:
            return datasets.load_dataset(repo, *args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            is_transient = any(kw in err_str for kw in ["503", "service unavailable", "429", "too many requests", "readerror", "connection", "timeout", "not a socket", "10038", "retry"])
            if retries >= max_retries or not is_transient:
                logger.error(f"HuggingFace load_dataset failed for '{repo}' after {retries} retries: {e}")
                raise
            sleep_time = backoff_factor ** retries
            logger.warning(f"HuggingFace error ({e}) loading '{repo}'. Retrying in {sleep_time:.1f}s (attempt {retries+1}/{max_retries})...")
            time.sleep(sleep_time)
            retries += 1


def hf_stream_with_backoff(iterable_dataset: Any, max_retries: int = 5, backoff_factor: float = 2.0) -> Any:
    """
    Wraps an iterable dataset stream so that network hiccups during chunk yields retry cleanly.
    """
    import time
    it = iter(iterable_dataset)
    while True:
        retries = 0
        while True:
            try:
                item = next(it)
                break
            except StopIteration:
                return
            except Exception as e:
                err_str = str(e).lower()
                is_transient = any(kw in err_str for kw in ["503", "service unavailable", "429", "too many requests", "readerror", "connection", "timeout", "not a socket", "10038", "retry"])
                if retries >= max_retries or not is_transient:
                    raise
                sleep_time = backoff_factor ** retries
                logger.warning(f"Streaming error ({e}). Retrying chunk in {sleep_time:.1f}s (attempt {retries+1}/{max_retries})...")
                time.sleep(sleep_time)
                retries += 1
                try:
                    it = iter(iterable_dataset)
                except Exception:
                    pass
        yield item
