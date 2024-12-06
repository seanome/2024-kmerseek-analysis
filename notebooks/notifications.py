import logging


# logger = logging.getLogger(__name__)

formatter = logging.Formatter(
    fmt="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)
# Write to stderr
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(formatter)


def notify(msg):
    logger.info(msg)


def notify_done():
    logger.info("Done.")
