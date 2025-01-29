
from sudonim import download_model, download_dataset

def download_repo(model: str=None, dataset: str=None, **kwargs ):
    """
    Download a model, dataset, or repo from HF Hub, GitHub, NGC, ect.
    """
    if dataset:
        location = download_dataset(dataset, **kwargs)
    elif model:
        location = download_model(model, **kwargs)
    else:
        raise ValueError(f"Either --model or --dataset is required")

    return location
