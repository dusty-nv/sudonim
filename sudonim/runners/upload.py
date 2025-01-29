
from sudonim import push_to_hub

def upload_repo(model: str=None, dataset: str=None, **kwargs):
    """
    Download a model, dataset, or repo from HF Hub, GitHub, NGC, ect.
    """
    path = dataset if dataset else model

    if not path:
        raise ValueError(f"Either --model or --dataset is required")
    
    return push_to_hub(path, **kwargs)