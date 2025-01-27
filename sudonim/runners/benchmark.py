import os
import json

from pathlib import Path
from sudonim import download_model, download_dataset, resolve_path, cudaShortVersion, getenv, shell

env, log = getenv()

def run_benchmark( model: str=None, dataset: str=None, tokenizer: str=None, host: str=None, port: int=None, **kwargs ):
    """
    Launch endpoint benchmark client (assumes server is already running)
    """
    if not env.HAS_MLC:
        raise RuntimeError(f"The benchmark client is installed in MLC, and could not find MLC installed in this environment (missing mlc_llm in $PATH)")

    if not model:
        raise ValueError(f"Missing required argument:  --model")
     
    if not dataset:
        dataset = 'anon8231489123/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json'

    dataset_path = download_dataset(dataset, **kwargs)

    tokenizer_path = download_model(tokenizer if tokenizer else model, 
                                    download_kwargs={} if tokenizer else {'local_files_only': True}, 
                                    **kwargs) 

    output_path = resolve_path(kwargs.get('cache_benchmarks'))
    output_file = os.path.join(output_path, str(Path(model).name).replace('.', '_').lower() + f'_{env.BOARD_ID}_{cudaShortVersion()}')

    with open(output_file + '.json', 'w') as file:
        json.dump(env, file, indent=2)

    cmd = ['python3 -m mlc_llm.bench']

    cmd += [f'--dataset sharegpt']
    cmd += [f'--dataset-path {dataset_path}']
    cmd += [f'--tokenizer {tokenizer_path}']
    cmd += [f'--api-endpoint openai']
    cmd += [f'--num-requests 25']
    cmd += [f'--num-warmup-requests 3']
    cmd += [f'--num-concurrent-requests 2']
    cmd += [f'--num-gpus {env.NUM_GPU}']
    cmd += [f'--host {host}']
    cmd += [f'--port {port}']
    cmd += [f'--output {output_file}.csv']

    shell(cmd, echo='Running benchmark client')