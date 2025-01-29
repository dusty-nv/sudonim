import os
import json

from pathlib import Path

from sudonim import (
  download_model, hf_hub_exists, push_to_hub, 
  resolve_path, valid_model_repo, split_model_name,
  shell, getenv, cudaShortVersion, NamedDict
)

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

        kwargs.setdefault('source_model', model)

        if os.path.isdir(model):
            model_path = model
        else: # as repo/name instead of local path
            model_path = MLC.download(model=model, quantization=quantization, **kwargs)

        quant_path = MLC.quantize(model_path, quantization=quantization, **kwargs)
        config_path = MLC.config(model_path, quant_path=quant_path, quantization=quantization, **kwargs)
        model_lib = MLC.compile(quant_path, quantization=quantization, **kwargs) 

        return MLC.serve(model_lib, quantization=quantization, config_path=config_path, **kwargs)

    @staticmethod
    def download(model: str, quantization: str=None, cache_mode=env.CACHE_MODE, **kwargs):
        if not valid_model_repo(model):
            raise ValueError(f"Invalid local path or remote URL - this path could not be found locally, and was not a well-formed model repo/name:  {model}")
        
        model_org, model_name = split_model_name(model)
        is_quant = quantization in model_name and '-mlc' in model_name.lower()

        if not is_quant:
            quant_model = f'{model_name}-{quantization}-MLC'
            quant_hosts = ['dusty-nv', 'mlc-ai']
            for quant_host in quant_hosts:
                quant_repo = os.path.join(quant_host, quant_model)
                if hf_hub_exists(quant_repo, **kwargs):
                    model, is_quant = quant_repo, True
                    break

        if not is_quant and not hf_hub_exists(model, warn=True, **kwargs):
            raise IOError(f"could not locate or access model {model}")
        
        return download_model(model, 
            cache=kwargs.get('cache_mlc' if is_quant else 'cache_hf'),
            **kwargs
        )

    @staticmethod
    def quantize(model_path : str, quantization: str=None, cache_mlc: str=None, cache_mode=env.CACHE_MODE, **kwargs):
        cache_mlc = resolve_path(cache_mlc)
        quant_path = os.path.join(cache_mlc, os.path.basename(model_path)) + f"-{quantization}-MLC"
        weights = [x for x in Path(quant_path).glob('**/params_*.bin')]

        if weights and cache_mode.quantization:
            #log.warning(f"Found existing quantized weights ({quant_path}), skipping quantization (set the CACHE_MODE='quantization:off' environment variable or '--cache-mode quantization:off' command-line argument to re-quantize the model)")
            return os.path.dirname(weights[0])

        cmd = [
            f'mlc_llm convert_weight --quantization {quantization}',
            f"{model_path}",
            f"--output {quant_path}"
        ]

        shell(cmd, echo='Running MLC quantization')
        return quant_path
    
    @staticmethod
    def config(model_path : str, quant_path : str, quantization: str=None, cache_mode=env.CACHE_MODE, **kwargs):
        config_path = os.path.join(quant_path, 'mlc-chat-config.json')
        if os.path.isfile(config_path) and cache_mode.engine:
            return config_path
        kwargs.setdefault('chat_template', MLC.get_chat_template(model_path))
        cmd = [f'mlc_llm gen_config --quantization {quantization}']
        cmd += MLC.overrides(packed=False, **kwargs)
        cmd += [f'--output {quant_path}', f'{model_path}']
        shell(cmd, echo='Generating MLC configuration')
        return config_path

    @staticmethod
    def compile(quant_path : str, cache_mode=env.CACHE_MODE, **kwargs):
        model_lib = MLC.find_model_lib(quant_path)

        if model_lib and cache_mode.engine:
            #log.warning(f"Found existing model library ({model_lib}), skipping model builder (set the CACHE_MODE='engine:off' environment variable or '--cache-mode engine:off' command-line argument to rebuild the model)")
            return model_lib
    
        model_lib = os.path.join(quant_path, MLC.get_model_lib())

        cmd = [f"mlc_llm compile --device cuda --opt O3"]
        cmd += MLC.overrides(**kwargs, exclude=['max_batch_size', 'chat_template'])
        cmd += [f"{quant_path}", f"--output {model_lib}"]

        shell(cmd, echo='Compiling MLC model')
        return model_lib

    @staticmethod
    def serve(model_lib : str, quantization: str=None, 
              host: str='0.0.0.0', port: int=9000, 
              max_batch_size: int=1, cache_mlc: str=None, 
              config_path: str=None, push: str=None, **kwargs):
        """ Start inference server after the model has been quantized & built """
        model_dir = os.path.dirname(model_lib)
        model_lib = Path(model_lib).relative_to(resolve_path(cache_mlc))

        if push:
            metadata = MLC.metadata(config_path, **kwargs)
            push_to_hub(model_dir, readme=metadata, **kwargs)
        
        mode = 'local' if max_batch_size > 1 else 'interactive'

        cmd = [f"mlc_llm serve --mode {mode} --device cuda",
               f"--host {host} --port {port}"]
        cmd += MLC.overrides(exclude=['max_batch_size', 'chat_template'], **kwargs)
        cmd += [f"--model-lib {model_lib}", f"{model_dir}"]
        
        return shell(cmd, cwd=resolve_path(cache_mlc), echo='Loading model')

    @staticmethod
    def metadata(config_path: str, source_model: str=None, **kwargs):
        with open(config_path) as file:
            cfg = json.load(file)

        keys = [
            'quantization', 'model_type', 'vocab_size', 
            'context_window_size', 'prefill_chunk_size',
            'temperature', 'repetition_penalty', 'top_p',
            'pad_token_id', 'bos_token_id', 'eos_token_id',
        ]

        out = NamedDict()

        if source_model:
            out.source_model = source_model

        out.api = 'MLC_LLM'

        for key in keys:
            val = cfg.get(key)
            if val is not None:
                out[key] = val

        return out
    
    @staticmethod
    def overrides(packed=True, exclude=[], **kwargs):
        overrides = {'tensor_parallel_shards': env.NUM_GPU}

        if isinstance(exclude, str):
            exclude = [exclude]

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
        model_lib = os.path.join(model, MLC.get_model_lib(model))
        if os.path.exists(model_lib):
            return model_lib
        #if path.is_dir():
            #so = [x for x in path.glob(f'**/{MLC.get_model_lib(model)}')]   #'**/*.so'
            #return so[0] if so else None

    @staticmethod
    def get_model_lib(model=None):
        return f'{env.CPU_ARCH}-{cudaShortVersion()}-{env.GPU_ARCH}.so'
    
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