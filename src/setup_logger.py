import logging

LOGGER_NAME = 'googlifier'

# Create the Logger
logger = logging.getLogger(LOGGER_NAME)

# Create a handler and a Formatter for formatting the log messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

# Set logging level
handler.setLevel(logging.INFO)
logger.setLevel(logging.INFO)