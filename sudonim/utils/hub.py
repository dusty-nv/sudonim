import os
import pprint

from pathlib import Path
from sudonim import getenv, property_table

env, log = getenv()

def hf_hub_exists(model: str, api_key: str=None, warn=False, **kwargs):
    """
    Check if a model repo exists / is accessible on HF Hub or not.
    """
    if not env.HAS_HF_HUB:
        raise ImportError(f"Attempted to use huggingface_hub without it being installed first")
    
    from huggingface_hub import model_info

    try:
        info = model_info(model, token=api_key)
        log.debug(f"Downloaded model info for: {model}\n\n{pprint.pformat(info, indent=2)}")
    except Exception as error:
        if warn:
            log.warning(f"Could not find or access {model} on HF Hub ({error})")
        return False
    
    return True

def download_model(model: str, cache: str=None, api_key: str=None, flatten=False, download_kwargs={}, **kwargs):
    """
    Download a model repo or file from HuggingFace Hub

    For now we are assuming HF API is available,
    but this should move to launching downloader in docker.
    """
    if not env.HAS_HF_HUB:
        raise ImportError(f"Attempted to use huggingface_hub without it being installed first")
    
    from huggingface_hub import hf_hub_download, snapshot_download

    model = model.replace('hf.co/', '')
    
    repo_path = Path(model)
    repo_path = str(repo_path.parent if repo_path.suffix.lower() == '.gguf' else repo_path)

    if not cache:
        cache = 'hf'
        if 'gguf' in model.lower(): cache = 'llama_cpp' 
        if 'mlc' in model.lower(): cache = 'mlc'
        for k in env.CACHES:
            if k in model:
                cache = k
        cache = kwargs.get(f'cache_{cache}')

    cache = os.path.join(
        resolve_path(cache), 
        repo_path.replace('/', '--') if flatten else repo_path
    )

    download_kwargs.setdefault('resume_download', True)
    download_kwargs.setdefault('repo_type', 'model')

    # Handle either "org/repo" or individual "org/repo/file"
    # the former has 0-1 slashes, while the later has 2.
    num_slashes = 0
    
    for c in model:
        if c == '/':
            num_slashes += 1
            
    if num_slashes >= 2:  
        slash_count = 0
        
        for idx, i in enumerate(model):
            if i == '/':
                slash_count += 1
                if slash_count == 2:
                    break
                    
        repo_id = model[:idx]
        filename = model[idx+1:]

        log.info(f"Downloading file {filename} from HF Hub:  {model} -> {cache}")
        repo_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=cache, token=api_key, **download_kwargs)
    else:
        log.info(f"Downloading model from HF Hub:  {model} -> {cache}")
        download_kwargs.setdefault('ignore_patterns', 'consolidated*.pth')
        repo_path = snapshot_download(repo_id=model, local_dir=cache, token=api_key, **download_kwargs)

    log.success(f"Downloaded {download_kwargs['repo_type']} {model} to:  {repo_path}")
    return repo_path

def download_dataset(dataset: str=None, cache: str=None, api_key: str=None, **kwargs):
    """
    Download a dataset from HF Hub, NGC, TFDS, ect.
    """
    return download_model(dataset, api_key=api_key, repo_type='dataset', 
                          cache=cache if cache else kwargs.get('cache_datasets', 
                          env.CACHES['datasets']), download_kwargs={'repo_type': 'dataset'}, 
                          **kwargs)

def push_to_hub(path, username: str=None, api_key: str=None, retry=10, readme=None, **kwargs):
    """
    Upload a model or dataset to HuggingFace Hub, creating the repo first if needed.
    """
    log.debug(f"push_to_hub('{path}') =>\n\n{pprint.pformat({'username': username, 'api_key': api_key, 'retry': retry, **kwargs}, indent=2)}\n")
              
    path = str(Path(path).resolve())

    if not os.path.isdir(path):
        raise ValueError(f"The path provided does not exist or is not a valid folder to upload ({path})")
    
    if not username:
        raise ValueError(f"It is required provide a valid HF username with --push to upload repositories")
    
    if not api_key:
        raise ValueError(f"It is required to set $HF_TOKEN or --api-key to upload repositories")
    
    if not env.HAS_HF_HUB:
        raise ImportError(f"Attempted to use huggingface_hub without it being installed first")
    
    from huggingface_hub import HfApi, utils as hf_hub_utils

    api = HfApi(token=api_key)
    model = os.path.basename(path)
    repo = f"{username}/{model}"
    url = f"https://hf.co/{repo}"

    log.info(f"Uploading {model} to {url}")

    try:
        api.create_repo(repo_id=repo, private=False)
    except hf_hub_utils.HfHubHTTPError as error:
        if error.response.status_code != 409:
            raise
        log.info("[HF] Repo already exists. Skipping creation.")

    create_readme(path, contents=readme, **kwargs)

    if env.DRY_RUN:
        return url
    
    for _retry in range(retry):
        try:
            api.upload_folder(
                folder_path=path,
                repo_id=repo,
                ignore_patterns=["logs.txt"],
            )
        except Exception as exc:  # pylint: disable=broad-except
            log.error("%s. Retrying...", exc)
        else:
            break
    else:
        raise RuntimeError(f"Failed to upload to HuggingFace Hub with {retry} retries")
    
    log.success(f"Uploaded {path} to {url}")
    return url

def create_readme(path: str=None, filename: str='README.md', contents=None, overwrite=True, **kwargs):
    """
    Generate a default readme / model card
    """
    filename = os.path.join(path, filename)
    model = os.path.basename(path)

    if os.path.exists(filename) and not overwrite:
        log.info(f"Skipping creation of {filename} (already exists and overwrite={overwrite} was set)")
        return filename
    
    kwargs.pop('api_key', None)

    txt = f"# {model}\n\n"

    if contents:
        if isinstance(contents, (dict, list, tuple)):
            def filter_model_properties(obj, key, value):
                if key == 'source_model':
                    value = f"[`{value}`](https://huggingface.co/{value})"
                else:
                    value = f"`{value}`"

                if key == 'api':
                    key = 'Inference API'
                elif 'top_p' in key or 'token_id' in key:
                    key = f'{key}'
                else:
                    key = key.replace('_', ' ').title()
                    
                return key, value
                
            txt += property_table(
                contents, headers=[' ', 'Model Configuration'],
                colalign=['left', 'center'], tablefmt='github',
                filter=filter_model_properties, color=None, max_widths=None
            ).replace('-|--', '-|:-').replace('--|\n', '-:|\n') + '\n\n'

            #txt += f"```json\n"
            #txt += f"{pprint.pformat(kwargs, indent=2)}\n"
            #txt += f"```\n\n"
    
    txt += f"See [`jetson-ai-lab.com/models.html`](https://jetson-ai-lab.com/models.html) for benchmarks, examples, and containers to deploy local serving and inference for these quantized models.\n"

    log.info(f"Saving readme to {filename}\n\n{txt}\n\n")
    
    with open(filename, mode='w') as file:
        file.write(txt)

    return filename

def split_model_name(path):
    """
    Return a (repo, name) tuple from the model's path or URL.
    """
    path = Path(path).parts
    user = path[-2] if len(path) > 1 else None
    return user, path[-1]

def get_model_name(path):
    """
    Extract the model name from path or URL, not including the org/username.
    For example, ``Llama-3.2-3B-Instruct`` is just the model name and not the full repo ID.
    """
    return split_model_name(path)[1]

def get_model_repo(path):
    """
    Extract the model `repo/name` from path or URL, including the org/username.
    For example, ``meta-llama/Llama-3.2-3B-Instruct`` includes the full ID.
    """
    user, model = split_model_name(path)
    return f'{user}/{model}' if user else model

def valid_model_repo(path):
    """
    Check if the path is a valid model ID in the canonical `repo/name` form.
    This also returns true if the path is in `repo/name/file.ext` form.

    This gets used to check if the path is referring to a registry model,
    and if the path is local or a URL, it should return False.

    Note that this does not confirm that the repo actually exists online,
    only that the path is well-formed - for that see ``hf_hub_exists()``
    """
    return len(Path(path).parts) in [2,3]
   
def resolve_path(path, makedirs=True):
    """
    Perform substitutions and checks to resolve local paths
    """
    path = path.replace('$CACHE_ROOT', env.CACHE_ROOT)

    if makedirs:
        p = Path(path)
        p = str(p.parent) if (p.suffix.lower() == '.gguf') else path
        os.makedirs(p, mode=0o755, exist_ok=True)
    
    return path