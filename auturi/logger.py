import logging
import os

LOG_LEVEL = dict(
    DEBUG=logging.DEBUG, INFO=logging.INFO, WARNING=logging.WARNING, ERROR=logging.ERROR
)


def get_logger(proc_name: str):
    _logger = logging.getLogger("crumbs")
    _logger.propagate = False
    level = LOG_LEVEL[os.getenv("AUTURI_LOG_LEVEL", "DEBUG")]
    _logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(levelname)s|%(filename)s:%(lineno)s, %(proc_name)s] %(asctime)s > %(message)s"
    )

    if not _logger.hasHandlers():
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        _logger.addHandler(streamHandler)

    return logging.LoggerAdapter(_logger, {"proc_name": proc_name})
