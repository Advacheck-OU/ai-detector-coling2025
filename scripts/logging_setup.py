import logging

def setup_logger(name=__name__, level=logging.INFO):
    """
    Sets up a logger with console output only.

    Args:
        name (str): Name of the logger.
        level (int): Logging level.
    
    Returns:
        logging.Logger: Configured logger.
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(console_handler)
    
    logger.propagate = False
    
    return logger

logger = setup_logger()
