
import os
import json
import pprint

from pathlib import Path

from sudonim import (
  download_model, download_dataset, 
  get_model_name, getenv, merge_dicts, 
  QUANTIZATIONS, BACKENDS
)

env, log = getenv()

OS_VERSIONS = {
    "r36.4.0": "JetPack 6.1+"
}

def export_repo( model: str=None, dataset: str=None, **kwargs ):
    """
    Export a model into graphDB template for sharing on jetson-ai-lab

    TODO import unknown model - find working combinations - export those
         for now the "find working combinations" part is manual
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

    mod_keys = list(cfg.keys())
    mod_root = cfg[mod_keys[0]]

    def should_inherit(x, selectors=['-', '_', ' ']):
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
                print('setting ', key, dst[key], ' to ', src[key] + dst[key])
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

        if not inherit:
            continue

        for api, api_cls in BACKENDS.items():
            api_alt = 'gguf' if api == 'llama_cpp' else api
            api_key = f"{mod_key}-{api_alt}"

            tags = [] # additional tags to add

            if api_key in cfg:
                tags.append(api_key)
                api_tags = cfg[api_key].setdefault('tags', [])
                if mod_key not in api_tags:
                    api_tags.insert(0,mod_key)
            else:
                tags.insert(0,mod_key)

            for quant_type in api_cls.Quantizations:
                quant_key = f"{mod_key}-{quant_type}-{api_alt}"
                #quant = cfg.setdefault(quant_key, {})
                #quant_name = quant.setdefault('name', url.replace('hf.co/', '') + f'-{quant_type}-{api_alt.upper()}')
                quant_title = f"{name} {title_sep} {api} {quant_type}" #quant.setdefault('title', f"{name} {title_sep} {api} {quant_type}")

                for os_version, os_name in OS_VERSIONS.items():
                    os_key = f"{mod_key}-{quant_type}-{api_alt}-{os_version}"
                    obj = cfg.setdefault(os_key, {})
                    print('OBJ OS KEY', os_key, 'TITLE', obj.setdefault('title', f"{quant_title} {title_sep} {os_name}"))
                    obj.setdefault('quantization', quant_type)
                    obj_tags = obj.setdefault('tags', [])
                    for tag in tags + [quant_type, f"{api}:{os_version}"]:
                        if tag not in obj_tags:
                            obj_tags.append(tag)
                    pprint.pprint(obj, indent=2)

    log.info(f"Generated model metadata:\n\n{pprint.pformat(cfg, indent=2, sort_dicts=False)}")          
    log.info(f"Saving exported model configurations to {output}")

    with open(output, 'w') as file:
        json.dump(cfg, file, indent=2)
    
    return output
    
def export_dataset( dataset: str=None, **kwargs ):
    raise NotImplementedError(f"Exporting of datasets is not yet implemented")