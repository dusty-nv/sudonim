
import os
import re
import json
import pprint

from pathlib import Path
from datetime import datetime

from sudonim import (
  download_model, download_dataset, 
  get_model_name, get_model_repo,
  get_model_info, model_has_file,
  getenv, merge_dicts, 
  QUANTIZATIONS, RUNTIMES
)

env, log = getenv()

OS_VERSIONS = {
    "jp6": "JetPack 6.1+"
}

def export_repo( model: str=None, dataset: str=None, **kwargs ):
    """
    Export a model into graphDB template for sharing on jetson-ai-lab

    TODO import unknown model - find working combinations - export those
         for now the "find working combinations" part is manual

         have this export the entire DB structure as opposed to one model
    """
    if dataset:
        location = export_dataset(dataset, **kwargs)
    elif model:
        location = export_model(model, **kwargs)
    else:
        raise ValueError(f"Either --model or --dataset is required")

    return location

def export_model( model: str=None, quantization: str=None, 
                  cache_export: str=env.CACHES.export, 
                  title_sep = '❯', **kwargs ):
    if not cache_export:
        raise ValueError(f"An --cache_export path is required in export mode")

    if Path(model).suffix != '.json':
        raise ValueError(f"Expected a json file for the --model path in export mode ({model})")
    
    if not Path(cache_export).suffix:
        output = os.path.join(cache_export, os.path.basename(model))
    else:
        output = cache_export

    with open(model) as file:
        cfg = json.load(file)

    wildcards = {}
    temp_cfg = {}

    for k,v in cfg.items():
        if '*' in k:
            wildcards[k] = v
        else:
            temp_cfg[k] = v
    
    cfg = temp_cfg

    mod_keys = list(cfg.keys())
    mod_root = cfg[mod_keys[0]]

    def should_inherit(x, selectors=['*', '-', '_', ' ']):
        if not isinstance(x, str):
            return True
        return any([x.startswith(s) or x.endswith(s) for s in selectors])

    def inherit_key(src, dst, key):
        if key not in src:
            log.warning("Missing key '{key}' from:\n{src}\nduring the inheritance of:\n{dst}\n")
            return
        if key not in dst:
            dst[key] = src[key]
        if should_inherit(dst[key]):
            if isinstance(src[key], (str, list, tuple)):
                dst[key] = src[key] + dst[key]
            elif isinstance(src[key], dict):
                merge_dicts(src[key], dst[key], replace=False)
            else:
                log.warning("Skipping inheritance of key '{key}' with type {type(x)}")

        return dst[key]

    for mod_idx, mod_key in enumerate(mod_keys):
        mod = cfg[mod_key]
        url = mod.get('url')
        name = mod.get('name')

        if not name and url:
            name = get_model_name(url).replace('-', ' ').replace('_', ' ')
        elif name and not url:
            url = name.replace(' ', '-')
        #else:
        #    name = url = mod_key
        
        if name:
            mod['name'] = name

        if url:
            mod['url'] = url

        #if url:
        #    name = get_model_name(url).replace('-', ' ').replace('_', ' ')
        #else:
        #    name = mod_key
        #    url = mod_key
        #    mod['url'] = url

        #mod.setdefault('title', name)
        
        inherit = should_inherit(mod_key)

        if inherit:
            log.debug(f"Inheriting {mod_key} {mod} from {mod_keys[0]} {mod_root}")
            name = inherit_key(mod_root, mod, 'name')
            url = inherit_key(mod_root, mod, 'url')
            inherit_key(mod_root, mod, 'links')
            new_key = mod_keys[0] + mod_key
            cfg[new_key] = mod
            del cfg[mod_key]
            mod_key = new_key

        title = mod.get('title', name)

        if 'links' not in mod:
            mod['links'] = {}

        links = mod.setdefault('links', {})

        if 'hf' not in links:
            links['hf'] = {
                'name': 'Hugging Face',
                'color': 'yellow',
            }

        #if 'url' not in links['hf']:
        links['hf']['url'] = url

        model_repo = get_model_repo(url)
        model_info = get_model_info(model_repo, warn=True, **kwargs)

        if model_info: 
            mod.setdefault('created_at', str(model_info.created_at))
            mod.setdefault('last_modified', str(model_info.last_modified))

        if not inherit:
            continue

        for api, api_cls in RUNTIMES.items():
            api_alt = 'gguf' if api == 'llama_cpp' else api
            api_key = f"{mod_key}-{api_alt}"

            tags = [] # additional tags to add

            if api_key in cfg:
                tags.append(api_key)
                api_tags = cfg[api_key].setdefault('tags', [])
                if mod_key not in api_tags:
                    api_tags.insert(0,mod_key)
                api_links = cfg[api_key].setdefault('links', {})
                api_links.setdefault(api_key, api_cls.Link)
            else:
                tags.insert(0,mod_key)

            for quant_type in api_cls.Quantizations:
                quant_key = f"{mod_key}-{quant_type}-{api_alt}"
                #quant = cfg.setdefault(quant_key, {})
                #quant_name = quant.setdefault('name', url.replace('hf.co/', '') + f'-{quant_type}-{api_alt.upper()}')
                api_name = api_cls.Name if hasattr(api_cls, 'Name') else api_cls.Link['name']
                quant_title = f"{title} {title_sep} {api_name} {quant_type}" #quant.setdefault('title', f"{name} {title_sep} {api} {quant_type}")

                for os_version, os_name in OS_VERSIONS.items():
                    os_key = f"{mod_key}-{quant_type}-{api_alt}-{os_version}"
                    os_title = f"{quant_title} {title_sep} {os_name}"

                    nim = cfg.setdefault(os_key, {})

                    nim.setdefault('title', os_title)
                    nim.setdefault('quantization', quant_type)

                    #nim_links = nim.setdefault('links', {})
                    #nim_links.setdefault(api_key, api_cls.Link)
                    
                    nim_tags = nim.setdefault('tags', [])

                    for tag in tags + [quant_type, f"{api}:{os_version}"]:
                        if tag not in nim_tags:
                            nim_tags.append(tag)

                    if not model_info:
                        raise ValueError(f"Could not locate model {model_repo}")
                    
                    if 'url' not in nim:
                        quant_url = api_cls.find_quantized(
                            model_repo, 
                            quantization=quant_type, 
                            warn=True, **kwargs
                        )

                        if quant_url:
                            quant_info = get_model_info(quant_url)
                            nim['url'] = quant_url
                            nim['created_at'] = str(quant_info.created_at)
                            nim['last_modified'] = str(quant_info.last_modified)

    for wild_key, wild_val in wildcards.items():
        regex = re.compile(wild_key)
        for cfg_key, cfg_val in cfg.items():
            if not re.match(wild_key, cfg_key):
                continue
            log.debug(f"Updating wildcard match ({wild_key} => {cfg_key}) with:\n{wild_val}")
            cfg_val.update(wild_val)
            log.debug(f"Result:\n{cfg_val}")

    log.info(f"Generated model metadata:\n\n")
    
    for k,v in cfg.items():
        print(f"{k}\n{pprint.pformat(v, indent=2)}\n")  

    log.info(f"Saving exported model configurations to {output}")

    with open(output, 'w') as file:
        json.dump(cfg, file, indent=2)
    
    return output
    
def export_dataset( dataset: str=None, **kwargs ):
    raise NotImplementedError(f"Exporting of datasets is not yet implemented")