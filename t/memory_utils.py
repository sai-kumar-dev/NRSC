import psutil
import resource
import gc
import torch
from core import logger

# Add this at the very beginning, right after the imports
def set_memory_limits():
    """Set memory limits to prevent bus errors"""
    try:
        available_memory = psutil.virtual_memory().available
        memory_limit = int(available_memory * 0.8)  # Use 80% of available memory
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        print(f"Memory limit set to {memory_limit / 1024**3:.2f} GB")
    except Exception as e:
        print(f"Could not set memory limit: {e}")

def monitor_memory_usage():
    """Monitor current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    
    logger.info(f"Memory usage: {memory_info.rss / 1024**3:.2f} GB ({memory_percent:.1f}%)")
    
    if memory_percent > 75:
        logger.warning("High memory usage detected!")
        clear_memory()

# Function to clear memory
def clear_memory():
    """Clear memory to avoid OOM errors"""
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
