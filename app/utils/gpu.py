from app.utils.logger import logger

def get_gpu_info() -> dict:
    """
    Checks GPU availability and returns detailed information about installed GPUs.
    Returns:
        dict: GPU availability, count, and specifications.
    """
    info = {
        "is_available": False,
        "device_count": 0,
        "devices": [],
        "framework": "unknown"
    }

    try:
        import torch
        info["framework"] = f"PyTorch {torch.__version__}"
        if torch.cuda.is_available():
            info["is_available"] = True
            info["device_count"] = torch.cuda.device_count()
            for i in range(info["device_count"]):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "id": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}"
                })
            logger.info(f"GPU Info: Detected {info['device_count']} CUDA device(s).")
        else:
            # Check MPS for macOS
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                info["is_available"] = True
                info["device_count"] = 1
                info["devices"].append({
                    "id": 0,
                    "name": "Apple Silicon (MPS)",
                    "total_memory_gb": "shared",
                    "compute_capability": "N/A"
                })
                logger.info("GPU Info: Detected Apple Silicon (MPS).")
            else:
                logger.warning("GPU Info: No GPU available, falling back to CPU.")
    except ImportError:
        logger.warning("GPU Info: PyTorch is not installed. Unable to fetch GPU properties.")
        
    return info
