# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys

# Example log message:
# DEBUG:2023-06-07 00:14:40,280:jax._src.xla_bridge:590: Initializing backend 'cpu'
logging_formatter = logging.Formatter(
    "{levelname}:{asctime}:{name}:{lineno}: {message}", style='{')

_logging_level_handler_set: dict[str, tuple[logging.Handler, int]] = {}
_default_TF_CPP_MIN_LOG_LEVEL = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "1")

_nameToLevel = {
    'CRITICAL': logging.CRITICAL,
    'FATAL': logging.FATAL,
    'ERROR': logging.ERROR,
    'WARN': logging.WARNING,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'NOTSET': logging.NOTSET,
}
def _getLevelNamesMapping():
  return _nameToLevel


def _logging_level_to_int(logging_level: str):
  # attempt to convert the logging level to integer
  try:
    # logging level is a string representation of an integer
    logging_level_num = int(logging_level)
  except ValueError:
    # logging level is a name string
    logging_level_num = _getLevelNamesMapping()[logging_level]
  return logging_level_num

_tf_cpp_map = {
    'CRITICAL': 3,
    'FATAL': 3,
    'ERROR': 2,
    'WARN': 1,
    'WARNING': 1,
    'INFO': 0,
    'DEBUG': 0,
}

def _set_TF_CPP_MIN_LOG_LEVEL(logging_level: str | None = None):
  # resetting to user-default TF_CPP_MIN_LOG_LEVEL
  # this is typically "1", but if the user overrode it, it can be != "1"
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = _default_TF_CPP_MIN_LOG_LEVEL

  # set cpp runtime logging level if the level is anything but NOTSET
  if logging_level is not None and logging_level != "NOTSET":
    if logging_level not in _tf_cpp_map:
      raise ValueError(f"Attempting to set log level \"{logging_level}\" which"
                       f" isn't one of the supported:"
                       f" {list(_tf_cpp_map.keys())}.")
    # config the CPP logging level 0 - debug, 1 - info, 2 - warning, 3 - error
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(_tf_cpp_map[logging_level])

def update_logging_level_global(logging_level: str | None) -> None:
  # remove previous handlers
  for logger_name, (handler, level) in _logging_level_handler_set.items():
    logger = logging.getLogger(logger_name)
    logger.removeHandler(handler)
    logger.setLevel(level)
  _logging_level_handler_set.clear()
  _set_TF_CPP_MIN_LOG_LEVEL(logging_level)

  if logging_level is None:
    return

  logging_level_num = _logging_level_to_int(logging_level)
  handler = logging.StreamHandler()
  handler.setLevel(logging_level_num)
  handler.setFormatter(logging_formatter)

  # update jax and jaxlib root loggers for propagation
  root_loggers = [logging.getLogger("jax"), logging.getLogger("jaxlib")]
  for logger in root_loggers:
    logger.setLevel(logging_level_num)
    logger.addHandler(handler)
    logger.propagate
    logger.parent
    _logging_level_handler_set[logger.name] = (handler, logger.level)


_debug_handler = logging.StreamHandler(sys.stderr)
_debug_handler.setLevel(logging.DEBUG)
_debug_handler.setFormatter(logging_formatter)

_debug_enabled_loggers = []


def _enable_debug_logging(logger_name):
  """Makes the specified logger log everything to stderr.

  Also adds more useful debug information to the log messages, e.g. the time.

  Args:
    logger_name: the name of the logger, e.g. "jax._src.xla_bridge".
  """
  logger = logging.getLogger(logger_name)
  _debug_enabled_loggers.append((logger, logger.level))
  logger.addHandler(_debug_handler)
  logger.setLevel(logging.DEBUG)


def _disable_all_debug_logging():
  """Disables all debug logging enabled via `enable_debug_logging`.

  The default logging behavior will still be in effect, i.e. WARNING and above
  will be logged to stderr without extra message formatting.
  """
  for logger, prev_level in _debug_enabled_loggers:
    logger: logging.Logger
    logger.removeHandler(_debug_handler)
    logger.setLevel(prev_level)
  _debug_enabled_loggers.clear()

def update_debug_log_modules(module_names_str: str | None):
  _disable_all_debug_logging()
  if not module_names_str:
    return
  module_names = module_names_str.split(',')
  for module_name in module_names:
    _enable_debug_logging(module_name)
