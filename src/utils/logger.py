import logging

def get_logger(name="dw-ilp"):
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)
