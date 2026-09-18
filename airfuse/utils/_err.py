

def log_class_errors(cls):
    """
    Class decorator to bind error logging to all methods.
    """
    import functools
    import logging
    logger = logging.getLogger(cls.__name__)

    for attr_key, attr_val in cls.__dict__.items():
        # Check if it's a callable method
        if callable(attr_val) and not attr_key.startswith('__'):

            @functools.wraps(attr_val)
            def wrapper(self, *args, method=attr_val, name=attr_key, **kwargs):
                try:
                    return method(self, *args, **kwargs)
                except Exception as e:
                    msg = f'Uncaught error in {cls.__name__}.{name}: {e}'
                    logger.exception(msg, exc_info=True)
                    raise

            setattr(cls, attr_key, wrapper)

    return cls
