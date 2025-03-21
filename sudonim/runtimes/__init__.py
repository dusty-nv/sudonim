
from .mlc import *
from .llama_cpp import *

class vLLM:
  Quantizations = ['bnb4', 'fp8', 'fp16']
  
  Link = {
    'name': 'vLLM',
    'url': 'https://github.com/vllm-project/vllm',
  }

  Name = "vLLM"
  

class ollama:
  Quantizations = ['q4_k_m']

  Link = {
    'name': 'ollama',
    'url': 'https://ollama.com/',
  }

  Name = "ollama"
  

RUNTIMES = {
  'mlc': MLC,
  'llama_cpp': LlamaCpp,
  'ollama': ollama,
  'vllm': vLLM,
}

QUANTIZATIONS = {
  k:v.Quantizations for k,v in RUNTIMES.items()
}

from .quantization import *