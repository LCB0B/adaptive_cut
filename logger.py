import logging

logger = logging.getLogger(__name__)

# the handler determines where the logs go: stdout/file
handler = logging.StreamHandler()

# the formatter determines what our logs will look like
fmt = '%(message)s'
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)

logger.addHandler(handler)