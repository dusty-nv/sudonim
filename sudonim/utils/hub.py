import os

from pathlib import Path
from sudonim import getenv

env, log = getenv()

def hf_hub_exists(model: str, api_key: str=None, **kwargs):
    """
    Check if a model repo exists / is accessible on HF Hub or not.
    """
    if not env.HAS_HF_HUB:
        raise ImportError(f"Attempted to use huggingface_hub without it being installed first")
    
    from huggingface_hub import model_info

    try:
        info = model_info(model, token=api_key)
        log.debug(f"Downloaded model info for: {model}\n{pprint.pformat(info, indent=2)}")
    except Exception as error:
        log.error(f"Could not find or access {model} on HF Hub ({error})")
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