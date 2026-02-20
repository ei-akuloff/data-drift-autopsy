"""Logging configuration for drift autopsy."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    fmt: Optional[str] = None,
    use_structlog: bool = False,
) -> None:
    """
    Configure logging for drift autopsy.
    
    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        fmt: Custom format string (if None, use default)
        use_structlog: Use structlog for structured logging (default: False)
    """
    if fmt is None:
        fmt = "[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    
    # If structlog is requested and available, configure it
    if use_structlog:
        try:
            import structlog
            
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer(),
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )
            
            logging.info("Structured logging enabled")
        except ImportError:
            logging.warning("structlog not available, using standard logging")
    
    # Set library loggers to WARNING to reduce noise
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
