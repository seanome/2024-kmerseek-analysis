import logging


# logger = logging.getLogger(__name__)

logging.basicConfig(
    format="{asctime} - {levelname}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)


def notify(msg):
    logging.debug(f"\n{msg}")


def notify_done():
    logging.debug("\tDone.")
