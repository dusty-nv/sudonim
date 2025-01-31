
from .mlc import *
from .llama_cpp import *

RUNTIMES = {
  'mlc': MLC,
  'llama_cpp': LlamaCpp,
}

QUANTIZATIONS = {
  k:v.Quantizations for k,v in RUNTIMES.items()
}

from .quantization import *