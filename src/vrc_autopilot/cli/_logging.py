import logging
import os


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.DEBUG if os.getenv("DEBUG") else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
