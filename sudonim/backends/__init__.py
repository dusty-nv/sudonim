
from .mlc import *
from .llama_cpp import *

BACKENDS = {
  'mlc': MLC,
  'llama_cpp': LlamaCpp,
}

QUANTIZATIONS = {
  k:v.Quantizations for k,v in BACKENDS.items()
}

from .quantization import *