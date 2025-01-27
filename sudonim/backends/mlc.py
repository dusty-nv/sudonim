import os

from pathlib import Path
from sudonim import download_model, hf_hub_exists, resolve_path, shell, getenv

env, log = getenv()

class MLC:
    """
    MLC/TVM deployment wrapper that automates. the steps to run a model in MLC.
    """
    Quantizations = [
        'q4f16_0', 'q4f16_1', 'q4f32_1', 'q4f16_2', 'q4f16_autoawq', 'q4f16_ft',
        'e5m2_e5m2_f16', 'e4m3_e4m3_f16', 'e4m3_e4m3_f16_max_calibrate',
    ]
    
    @staticmethod
    def deploy(model: str=None, quantization: str=None, **kwargs):
        """
        Download, quantizatize, compile, and serve model with MLC
        """
        if not env.HAS_MLC:
            raise RuntimeError(f"Could not find MLC installed in this environment (missing mlc_llm in $PATH)")

        model_path = Path(model)

        if model_path.is_dir():
            model_lib = MLC.find_model_lib(model_path)

            if model_lib:
                return MLC.serve(model_lib, quantization=quantization, **kwargs)

            quant_path = [x for x in Path(model_path).glob('**/params_*.bin')]
            quant_path = os.path.dirname(quant_path[0]) if quant_path else None
            quantized = bool(quant_path)

            if not quantized:
                quant_path = MLC.quantize(path, quantization=quantization, **kwargs)
            
            has_mlc_config = model_path.joinpath('mlc-chat-config.json').exists()

            if not quantized or not has_mlc_config:
                MLC.config(model_path, quant_path, quantization=quantization, **kwargs)

            model_lib = MLC.compile(quant_path, quantization=quantization, **kwargs)        
        else:
            if len(model_path.parts) != 2:
                raise ValueError(f"Invalid local path or remote URL, or resource not found ({model_path})")
            
            model_path = MLC.download(model=model, quantization=quantization, **kwargs)
            return MLC.deploy(model=model_path, quantization=quantization, **kwargs)

        return MLC.serve(model_lib, quantization=quantization, **kwargs)

    @staticmethod
    def download(model: str, quantization: str=None, **kwargs):
        model_org, model_name = Path(model).parts
        is_quant = quantization in model_name and '-mlc' in model_name.lower()

        if not is_quant:
            quant_model = f'{model_name}-{quantization}-MLC'
            quant_hosts = ['dusty-nv', 'mlc-ai']
            for quant_host in quant_hosts:
                quant_repo = os.path.join(quant_host, quant_model)
                if hf_hub_exists(quant_repo, **kwargs):
                    model, is_quant = quant_repo, True
                    break

        if not is_quant and not hf_hub_exists(model, **kwargs):
            raise IOError(f"could not locate or access model {model}")
        
        return download_model(model, 
            cache=kwargs.get('cache_mlc' if is_quant else 'cache_hf'),
            **kwargs
        )

    @staticmethod
    def quantize(model_path : str, quantization: str=None, cache_mlc: str=None, **kwargs):
        quant_path = os.path.join(cache_mlc, os.path.basename(model_path)) + f"-{quantization}-MLC"
        cmd = [
            f'mlc_llm convert_weight --quantization {quantization}',
            f"{model_path}",
            f"--output {quant_path}"
        ]
        shell(cmd, echo='Running MLC quantization')
        return quant_path
    
    @staticmethod
    def config(model_path : str, quant_path : str, quantization: str=None, **kwargs):
        kwargs.setdefault('chat_template', MLC.get_chat_template(model_path))
        cmd = [f'mlc_llm gen_config --quantization {quantization}']
        cmd += MLC.overrides(packed=False, **kwargs)
        cmd += [f'--output {quant_path}', f'{model_path}']
        shell(cmd, echo='Generating MLC configuration')
        return quant_path

    @staticmethod
    def compile(quant_path : str, **kwargs):
        model_lib = os.path.join(quant_path, f"model.so")
        cmd = [f"mlc_llm compile --device cuda --opt O3"]
        cmd += MLC.overrides(**kwargs)
        cmd += [f"{quant_path}", f"--output {model_lib}"]
        shell(cmd, echo='Compiling MLC model')
        return model_lib

    @staticmethod
    def serve(model_lib : str, quantization: str=None, host: str='0.0.0.0', port: int=9000, max_batch_size: int=1, cache_mlc: str=None, **kwargs):
        model_lib = Path(model_lib)
        model_lib = model_lib.relative_to(resolve_path(cache_mlc))
        mode = 'local' if max_batch_size > 1 else 'interactive'
        cmd = [f"mlc_llm serve --mode {mode} --device cuda",
               f"--host {host} --port {port}"]
        cmd += MLC.overrides(exclude=['max_batch_size'], **kwargs)
        cmd += [f"--model-lib {model_lib}", f"{model_lib.parent}"]
        return shell(cmd, cwd=resolve_path(cache_mlc), echo='Loading model')

    @staticmethod
    def overrides(packed=True, exclude=[], **kwargs):
        overrides = {'tensor_parallel_shards': env.NUM_GPU}

        for k,v in kwargs.items():
            if v and k in MLC.CONFIG_MAP and k not in exclude:
                overrides[MLC.CONFIG_MAP[k]] = v

        if not overrides:
            return []
               
        if packed:
            overrides = ';'.join([f'{k}={v}' for k,v in overrides.items()])
            return [f"--overrides='{overrides}'"]
        else:
            return [f"--{k.replace('_', '-')} {v}" for k,v in overrides.items()]

    @staticmethod
    def find_model_lib(model):
        path = Path(model)
        if path.is_dir():
            so = [x for x in path.glob('**/*.so')]
            return so[0] if so else None

    @staticmethod
    def get_chat_template(model):
      """ Fallback to best estimate the model's conversation or tokenization template. """
      name = model.lower()
              
      if 'llama' in name:   
          if '-2-' in name:
              return "llama-2"
          return "llama-3_1"
      elif 'qwen' in name:
          return 'qwen2'
      elif 'phi-3' in name:
          return 'phi-3'
      elif 'smol' in name:
          return 'chatml'

      log.warning(f"{model} | a default chat template wasn't found, please set it with --chat-template")    
      return None

    # argument name mapping
    CONFIG_MAP = {
        'chat_template': 'conv_template',
        'max_batch_size': 'max_batch_size',
        'max_context_len': 'context_window_size',
        'prefill_chunk': 'prefill_chunk_size'
    }