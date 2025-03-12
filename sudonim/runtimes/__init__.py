
from .mlc import *
from .llama_cpp import *

class vLLM:
  Quantizations = ['fp16']
  
  Link = {
    'name': 'vLLM',
    'url': 'https://github.com/vllm-project/vllm',
  }

  Name = "vLLM"
  

class ollama:
  Quantizations = ['q4_0']

  Link = {
    'name': 'ollama',
    'url': 'https://ollama.com/',
  }

  Name = "ollama"
  

RUNTIMES = {
  'mlc': MLC,
  'llama_cpp': LlamaCpp,
  'vllm': vLLM,
  'ollama': ollama,
}

QUANTIZATIONS = {
  k:v.Quantizations for k,v in RUNTIMES.items()
}

from .quantization import *