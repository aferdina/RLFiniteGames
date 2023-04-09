"""setup logging for a module"""
import logging
from logging.handlers import RotatingFileHandler
from typing import Union
import colorlog

LOG_COLORS = {
    'DEBUG': 'cyan',
    'INFO': 'white',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red,bg_white',
}


def setup_logger(logger_name: str = __name__,
                 logger_level: int = logging.DEBUG,
                 log_file: Union[str, None] = None,
                 file_handler_level: int = logging.DEBUG,
                 console_output: bool = False,
                 stream_handler_level: int = logging.DEBUG) -> logging.Logger:
    # Create a logger object
    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level)

    # Create a formatter for the log messages
    file_formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s')
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s %(name)s %(levelname)s:%(message)s',
        datefmt=None,
        reset=True,
        log_colors=LOG_COLORS,
        secondary_log_colors={},
        style='%')
    # Create a file handler if a file path is provided
    if log_file:
        file_handler = RotatingFileHandler(
            filename=log_file, maxBytes=1024 * 1024, backupCount=5)
        file_handler.setLevel(file_handler_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Create a stream handler if console output is requested
    if console_output:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(stream_handler_level)
        stream_handler.setFormatter(console_formatter)
        logger.addHandler(stream_handler)

    return logger
