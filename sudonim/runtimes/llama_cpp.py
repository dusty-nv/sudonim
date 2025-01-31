import os

from pathlib import Path

from sudonim import (
  download_model, hf_hub_exists, 
  model_has_file, split_model_name,
  shell, getenv
)

env, log = getenv()

class LlamaCpp:
    """
    llama.cpp deployment (GGUF)
    https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#openai-compatible-api-endpoints
    """
    Quantizations = ['q4_k_m', 'q4_k_l', 'q5_k_s', 'q5_k_m', 'q5_k_l', 'q6_k']

    Link = {
        'name': 'llama.cpp',
        'url': 'https://github.com/ggerganov/llama.cpp',
    }

    @staticmethod
    def deploy(model: str=None, quantization: str=None, 
                max_context_len: int=None, prefill_chunk: int=None, 
                chat_template: str=None,
                host: str='0.0.0.0', port: int=9000,
                log_level: str='info', **kwargs):
      
        if not env.HAS_LLAMA_CPP:
            raise RuntimeError(f"Could not find llama.cpp installed in this environment (missing llama-server in $PATH)")

        model_path = Path(model)

        if model_path.suffix.lower() != '.gguf': # TODO hf->conversion
            raise ValueError(f"Expected a file with .gguf extension for --api=llama_cpp")
        
        if not model_path.is_file():
            model_path = Path(LlamaCpp.download(model, quantization=quantization, **kwargs))

        cmd = [
            f'llama-server',
            f'--model {model_path}',
            f'--alias {model_path.name}',
            f'--flash-attn',
            f'--n-gpu-layers 999',
            f'--ctx-size {max_context_len}' if max_context_len else '',
            f'--batch-size {prefill_chunk}' if prefill_chunk else '',
            f'--chat-template {chat_template}' if chat_template else '',
            f'--host {host} --port {port}',
            '--verbose' if log_level == 'debug' else ''
        ]

        shell(cmd, echo='Running llama.cpp server')
        return model_path

    @staticmethod
    def download(model: str, quantization: str=None, **kwargs):
        is_quant = (Path(model).suffix.lower() == '.gguf')

        if not is_quant:
            quant_repo = LlamaCpp.find_quantized(model, quantization, **kwargs)
            if quant_repo:
                model, is_quant = quant_repo, True

        if not is_quant and not hf_hub_exists(model, warn=True, **kwargs):
            raise IOError(f"could not locate or access model {model}")
        
        return download_model(model, 
            cache=kwargs.get('cache_llama_cpp' if is_quant else 'cache_hf'),
            **kwargs
        )
    
    @staticmethod
    def find_quantized(model: str, quantization: str=None, **kwargs):
        model_org, model_name = split_model_name(model)

        quant_model = f'{model_name}-GGUF'
        quant_hosts = ['bartowski']

        for quant_host in quant_hosts:
            quant_file = f'{model_name}-{quantization.upper()}.gguf'
            quant_url = os.path.join(quant_host, quant_model)
            if model_has_file(quant_url, quant_file, **kwargs):
                return os.path.join(quant_url, quant_file)