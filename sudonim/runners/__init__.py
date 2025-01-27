from .download import *
from .benchmark import *
from .server import *

RUNNERS = {
  'download': download_repo,
  'bench': run_benchmark,
  'serve': server_up,
  'stop': server_down,
}

from .command import *