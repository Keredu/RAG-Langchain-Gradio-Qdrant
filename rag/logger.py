import logging


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Ensure logger is set to DEBUG level
    
    # Prevent adding multiple handlers to the logger
    if not logger.handlers:
        # Create a stream handler with DEBUG level
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        
        # Define a specific format for the handler
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s', 
                                      '%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(handler)

    # Prevent the logger from propagating messages to the root logger
    logger.propagate = False
    
    return logger
