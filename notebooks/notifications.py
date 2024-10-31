import logging


# logger = logging.getLogger(__name__)

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)

logger = logging.getLogger()
# Write to stderr
handler = logging.StreamHandler()
logger.addHandler(handler)


def notify(msg):
    logger.debug(msg)


def notify_done():
    logger.debug("Done.")
