import logging
import functools
import termcolor
import sudonim

# add custom logging.SUCCESS level and logging.success() function
#logging.STATUS = 25  # https://docs.python.org/3/library/logging.html#logging-levels
logging.SUCCESS = 35 

# default message formats and color highlighting
DEFAULT_FORMAT='[%(asctime)s] sudonim | %(message)s' #'%(asctime)s | %(levelname)s | %(message)s'
DEFAULT_DATEFMT='%H:%M:%S'

DEFAULT_COLORS = {
    logging.DEBUG: ('light_grey', 'dark'),
    logging.INFO: (None, 'dark'),
    #logging.STATUS: (None, 'dark'),
    logging.WARNING: 'yellow',
    logging.SUCCESS: 'green',
    logging.ERROR: 'red',
    logging.CRITICAL: 'red'
}

class LogFormatter(logging.Formatter):
    """
    Colorized log formatter (inspired from https://stackoverflow.com/a/56944256)
    Use LogFormatter.config() to enable it with the desired logging level.
    """    
    def __init__(self, format=DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT, colors=DEFAULT_COLORS, **kwargs):
        """
        @internal it's recommended to use config()
        """
        self.formatters = {}
        
        for level in DEFAULT_COLORS:
            if colors is not None and level in colors and colors[level] is not None:
                color = colors[level]
                attrs = None
                
                if not isinstance(color, str):
                    attrs = color[1:]
                    color = color[0]

                fmt = termcolor.colored(format, color, attrs=attrs)
            else:
                fmt = format
                
            self.formatters[level] = logging.Formatter(fmt=fmt, datefmt=datefmt)

    def format(self, record):
        """
        Implementation of logging.Formatter record formatting function
        """
        return self.formatters[record.levelno].format(record)

def basicConfig(level='info', format=DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT, colors=DEFAULT_COLORS, **kwargs):
    """
    Configure the root logger with formatting and color settings.
    
    Parameters:
        level (str|int) -- Either the log level name 
        format (str) -- Message formatting attributes (https://docs.python.org/3/library/logging.html#logrecord-attributes)
        
        datefmt (str) -- Date/time formatting string (https://docs.python.org/3/library/logging.html#logging.Formatter.formatTime)
        
        colors (dict) -- A dict with keys for each logging level that specify the color name to use for those messages
                        You can also specify a tuple for each couple, where the first entry is the color name,
                        followed by style attributes (from https://github.com/termcolor/termcolor#text-properties)
                        If colors is None, then colorization will be disabled in the log.
                        
        kwargs (dict) -- Additional arguments passed to logging.basicConfig() (https://docs.python.org/3/library/logging.html#logging.basicConfig)
    """
    #logging.addLevelName(logging.STATUS, 'STATUS')
    logging.addLevelName(logging.SUCCESS, 'SUCCESS')

    logging.success = logSuccess
    
    if not level:
        level = 'info'

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    log_handler = logging.StreamHandler()
    log_handler.setFormatter(LogFormatter(format=format, datefmt=datefmt, colors=colors))

    #log_handler.setLevel(level)
    #if len(logging.getLogger().handlers) > 0:
    #    logging.getLogger().removeHandler(logging.getLogger().handlers[0])
    #logger.handlers.clear()
    #logging.getLogger(__name__).setLevel(level)
    
    logging.basicConfig(handlers=[log_handler], level=level, force=True, **kwargs)

def getLogger(name=__name__):
    """
    Shortcut for importing logging and `logger.getLogger(__name__)`
    """
    logger = logging.getLogger(name=name)

    logger.basicConfig = basicConfig

    #logger.status = functools.partial(logStatus, logger=logger)
    logger.success = functools.partial(logSuccess, logger=logger)

    return logger

#def logStatus(*args, **kwargs):
#    kwargs.pop('logger', logging).log(logging.STATUS, *args, **kwargs)

def logSuccess(*args, **kwargs):
    kwargs.pop('logger', logging).log(logging.SUCCESS, *args, **kwargs)
