from .download import *
from .upload import *
from .export import *
from .benchmark import *
from .server import *

RUNNERS = {
  'download': download_repo,
  'upload': upload_repo,
  'export': export_repo,
  'bench': run_benchmark,
  'serve': server_up,
  'stop': server_down,
}

from .command import *