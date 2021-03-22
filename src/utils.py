"""
File with utils functions for the engine.
"""
import logging
import os

LOG_TO_FILE = True
LOG_TO_CONSOLE = True

LOG_LEVEL = logging.DEBUG
LOG_FILE_LEVEL = logging.DEBUG

LOG_ONLY_LISTED_MODULES = False
LOG_LIST_MODULES = ['TOOLS']


def get_logger(log_level: int = LOG_LEVEL, log_file_level: int = LOG_FILE_LEVEL, module: str = 'GLOBAL',
               directory: str = f'{os.getcwd()}/logs') -> logging.Logger:
    """
    Get the logger of the application to use in the main program.
    Args:
        log_level: Level too show in the logs (logging.DEBUG, logging.INFO, ...)

    Returns: Logger to use to makes logs.

    """
    log = logging.getLogger(module)
    log.propagate = False
    log.setLevel(log_level)

    formatter = logging.Formatter(f"%(asctime)s - {module} - %(levelname)s: %(message)s",
                                  datefmt="%m/%d/%Y %I:%M:%S %p")

    handlers = []
    if LOG_TO_FILE:
        fh = logging.FileHandler(f'{directory}/{module}.log')
        fh.setLevel(log_file_level)
        fh.setFormatter(formatter)
        handlers.append(fh)

    if LOG_TO_CONSOLE:
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        handlers.append(ch)

    if LOG_ONLY_LISTED_MODULES:
        if module in LOG_LIST_MODULES:
            log.handlers = handlers
    else:
        log.handlers = handlers

    return log


def interpolate(value: float, value_min: float, value_max: float, target_min: float = -1,
                target_max: float = 1, convert: bool = True) -> float:
    """

    Args:
        convert: True to convert the value to float
        value: Value to interpolate.
        value_min: Minimum value of the values.
        value_max: Maximum value of the values.
        target_min: Minimum value of the interpolation interval.
        target_max: Maximum value of the interpolation interval.

    Returns: Interpolated value.

    """
    if convert:
        value = float(value)

    return (value - value_min) * (float(target_max) - target_min) / (float(value_max) - value_min) + target_min


def is_numeric(value: str) -> bool:
    """
    Check if a value can be converted to float.

    Args:
        value: value to check

    Returns: boolean indicating if can be converted
    """

    try:
        float(value)
        return True
    except ValueError:
        return False
