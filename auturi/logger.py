import logging
import os

LOG_LEVEL = dict(
    DEBUG=logging.DEBUG, INFO=logging.INFO, WARNING=logging.WARNING, ERROR=logging.ERROR
)
_custom_logger = None


class SingletonType(type):
    def __call__(cls, *args, **kwargs):
        try:
            return cls.__instance
        except AttributeError:
            cls.__instance = super(SingletonType, cls).__call__(*args, **kwargs)
            return cls.__instance


class _CustomLogger(object):
    __metaclass__ = SingletonType
    _logger = None

    def __init__(self):
        self._logger = logging.getLogger("crumbs")
        self._logger.propagate = False
        level = LOG_LEVEL[os.getenv("AUTURI_LOG_LEVEL", "DEBUG")]
        self._logger.setLevel(level)
        formatter = logging.Formatter(
            "[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s"
        )

        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        self._logger.addHandler(streamHandler)

    def get_logger(self):
        return self._logger


def get_logger():
    global _custom_logger

    if _custom_logger is None:
        _custom_logger = _CustomLogger.__call__().get_logger()

    return _custom_logger
