import time
from functools import wraps
from typing import Any, Callable
import logging

def retry_on_failure(max_retries: int = 3, delay: int = 60) -> Callable:
    """Decorator that retries a function on failure with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds (doubles each retry)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            retries = 0
            curr_delay = delay
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        raise e
                    
                    logging.warning(
                        f"Attempt {retries} failed with error: {str(e)}. "
                        f"Retrying in {curr_delay} seconds..."
                    )
                    time.sleep(curr_delay)
                    curr_delay *= 2
            
            return None  # Should never reach here
        return wrapper
    return decorator