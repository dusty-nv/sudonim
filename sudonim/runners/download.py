
from sudonim import download_model, download_dataset

def download_repo(model: str=None, dataset: str=None, **kwargs ):
    """
    Invoke different commands like 'download', 'serve', 'bench'
    """
    if dataset:
        location = download_dataset(dataset, **kwargs)
    elif model:
        location = download_model(model, **kwargs)
    else:
        raise ValueError(f"Either --model or --dataset is required")

    return location
