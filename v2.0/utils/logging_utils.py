### From FourCastNet repo

import os
import sys
import logging


_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class _TqdmStream:
    """Wraps sys.stderr so that tqdm always uses carriage-return in-place
    updates, even when stderr is redirected to a log file (non-TTY).

    ``isatty()`` always returns True, which tells tqdm to use ``\\r``-based
    overwrite mode instead of printing a new line for every update.
    All writes are forwarded to the real ``sys.stderr`` so the output still
    ends up in the log file.
    """
    def write(self, msg: str) -> int:
        return sys.stderr.write(msg)

    def flush(self) -> None:
        sys.stderr.flush()

    def isatty(self) -> bool:
        return True


# Singleton stream instance — import and pass as ``file=tqdm_stream`` to
# every tqdm call so they all share the same forced-TTY stream.
tqdm_stream = _TqdmStream()


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that emits records via ``tqdm.write()``.

    When tqdm bars are active, ``tqdm.write()`` temporarily clears the bar,
    prints the log message, then redraws the bar.  This prevents log output
    from fragmenting progress bars into a new line on each update.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
            self.flush()
        except Exception:
            self.handleError(record)


def config_logger(log_level: int = logging.INFO) -> None:
    """Configure the root logger to write through tqdm.

    Replaces the default StreamHandler (which writes directly to stderr and
    breaks tqdm bars) with a TqdmLoggingHandler that uses ``tqdm.write()``.
    """
    root = logging.getLogger()
    root.setLevel(log_level)

    # Remove any pre-existing StreamHandlers on the root logger so that
    # basicConfig-style stderr handlers don't conflict with tqdm.
    for handler in root.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            root.removeHandler(handler)

    tqdm_handler = TqdmLoggingHandler(log_level)
    tqdm_handler.setFormatter(logging.Formatter(_format))
    root.addHandler(tqdm_handler)


def log_to_file(logger_name=None, log_level=logging.INFO, log_filename='tensorflow.log'):

  if not os.path.exists(os.path.dirname(log_filename)):
    os.makedirs(os.path.dirname(log_filename))

  if logger_name is not None:
    log = logging.getLogger(logger_name)
  else:
    log = logging.getLogger()

  fh = logging.FileHandler(log_filename)
  fh.setLevel(log_level)
  fh.setFormatter(logging.Formatter(_format))
  log.addHandler(fh)

def log_versions():
  import torch
  import subprocess

  logging.info('--------------- Versions ---------------')
  #logging.info('git branch: ' + str(subprocess.check_output(['git', 'branch']).strip()))
  #logging.info('git hash: ' + str(subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()))
  logging.info('Torch: ' + str(torch.__version__))
  logging.info('----------------------------------------')
