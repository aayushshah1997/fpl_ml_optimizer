"""
Logging setup for FPL AI system.

Provides centralized logging configuration with file and console outputs,
proper formatting, and log rotation.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from .config import get_config


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    force: bool = False
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Custom log file path
        force: Force reconfiguration even if already setup
    """
    # Don't setup multiple times unless forced
    if logging.getLogger().handlers and not force:
        return
    
    config = get_config()
    
    # Get configuration
    log_level = level or config.get("logging.level", "INFO")
    log_format = config.get(
        "logging.format", 
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_enabled = config.get("logging.file_enabled", True)
    console_enabled = config.get("logging.console_enabled", True)
    max_bytes = config.get("logging.max_bytes", 10485760)  # 10MB
    backup_count = config.get("logging.backup_count", 5)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers if forcing
    if force:
        root_logger.handlers.clear()
    
    # Console handler
    if console_enabled and not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_enabled:
        if log_file is None:
            log_file = config.logs_dir / "fpl_ai.log"
        else:
            log_file = Path(log_file)
        
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file handler already exists
        if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root_logger.handlers):
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(getattr(logging, log_level.upper()))
            root_logger.addHandler(file_handler)
    
    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("plotly").setLevel(logging.WARNING)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {log_level}")


def get_performance_logger() -> logging.Logger:
    """Get logger specifically for performance metrics."""
    logger = logging.getLogger("fpl_ai.performance")
    
    # Add separate file handler for performance logs if not exists
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        config = get_config()
        perf_log_file = config.logs_dir / "performance.log"
        
        handler = logging.FileHandler(perf_log_file)
        formatter = logging.Formatter(
            "%(asctime)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


def log_function_call(func_name: str, args: dict, duration: float):
    """Log function call with performance data."""
    perf_logger = get_performance_logger()
    perf_logger.info(
        f"CALL {func_name} | "
        f"args={args} | "
        f"duration={duration:.3f}s"
    )


def log_api_call(endpoint: str, status_code: int, duration: float, cached: bool = False):
    """Log API call with performance data."""
    perf_logger = get_performance_logger()
    cache_status = "CACHED" if cached else "FRESH"
    perf_logger.info(
        f"API {endpoint} | "
        f"status={status_code} | "
        f"duration={duration:.3f}s | "
        f"cache={cache_status}"
    )


def log_model_performance(model_name: str, metrics: dict, data_shape: tuple):
    """Log model training/prediction performance."""
    perf_logger = get_performance_logger()
    metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    perf_logger.info(
        f"MODEL {model_name} | "
        f"shape={data_shape} | "
        f"{metrics_str}"
    )


class TimedLogger:
    """Context manager for timing code blocks."""
    
    def __init__(self, logger: logging.Logger, message: str, level: int = logging.INFO):
        self.logger = logger
        self.message = message
        self.level = level
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.log(self.level, f"Starting: {self.message}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.log(self.level, f"Completed: {self.message} ({duration:.3f}s)")
        else:
            self.logger.error(f"Failed: {self.message} ({duration:.3f}s) - {exc_val}")


def timed_log(message: str, logger: Optional[logging.Logger] = None, level: int = logging.INFO):
    """Decorator for timing function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            func_logger = logger or logging.getLogger(func.__module__)
            
            start_time = time.time()
            func_logger.log(level, f"Starting: {message}")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                func_logger.log(level, f"Completed: {message} ({duration:.3f}s)")
                return result
            except Exception as e:
                duration = time.time() - start_time
                func_logger.error(f"Failed: {message} ({duration:.3f}s) - {e}")
                raise
        
        return wrapper
    return decorator
