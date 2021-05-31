# BEGIN GPL LICENSE BLOCK
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# END GPL LICENSE BLOCK

"""
File with utils functions for the engine.
"""
import logging
import os

LOG_TO_FILE = False
LOG_TO_CONSOLE = False

LOG_LEVEL = logging.DEBUG
LOG_FILE_LEVEL = logging.DEBUG

LOG_ONLY_LISTED_MODULES = True
LOG_LIST_MODULES = [
    'CONTROLLER',
    'ENGINE',
    'PROGRAM',
    'SCENE',
    'TEXT_MODAL',
    'MAP3DMODEL'
]


def get_logger(log_level: int = LOG_LEVEL, log_file_level: int = LOG_FILE_LEVEL, module: str = 'GLOBAL',
               directory: str = f'resources/logs') -> logging.Logger:
    """
    Get the logger of the application to use in the main program.
    Args:
        log_level: Level too show in the logs (logging.DEBUG, logging.INFO, ...)

    Returns: Logger to use to makes logs.

    """
    log = logging.getLogger(module)
    log.propagate = False
    log.setLevel(log_level)

    formatter = logging.Formatter(f"%(asctime)s.%(msecs)03d - {module} - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d,%H:%M:%S')

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
    Interpolate the given value between the other specified values.

    To interpolate the value between the target values it is necessary to specify an initial interval in which
    the value exists (value_min and value_max) and the target interval to interpolate it (target_min and target_max).

    If value_min and value_max are equal, then average between the targets values will be returned.

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

    # case initial interval is just one value.
    if value_min == value_max:
        return (target_min + target_max)/2.0

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


def is_clockwise(points):
    """
    Check if a list of 2D points are in CW order or not.

    Args:
        points: List of 2D points [(1.x,1.y),(2.x,2.y),...]

    Returns: Boolean indicating if points are CW or not.
    """

    # points is your list (or array) of 2d points.
    assert len(points) > 0
    s = 0.0
    for p1, p2 in zip(points, points[1:] + [points[0]]):
        s += (p2[0] - p1[0]) * (p2[1] + p1[1])
    return s > 0.0
