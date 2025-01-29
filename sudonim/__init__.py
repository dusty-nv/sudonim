
try:
  import importlib.metadata
  __version__ = importlib.metadata.version('sudonim')
except Exception as error:
  import warnings
  __version__ = '0.0.0'
  warnings.warn(f"Missing package metadata for 'sudonim' (reporting default version of {__version__})\nTo fix the version and avoid this warning, install the project with pip.\nThe error message from importlib is:  '{error}'")

from .env import *

from .utils.log import *
from .utils.misc import *
from .utils.table import *
from .utils.cuda import *
from .utils.shell import *
from .utils.hub import *
from .utils.docker import *

from .backends import *
from .runners import *
from .args import *
