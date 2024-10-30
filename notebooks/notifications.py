import logging


# logger = logging.getLogger(__name__)

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)


def notify(message):
    logging.debug(f"\n{message}")


def notify_done():
    logging.debug("\tDone.")
