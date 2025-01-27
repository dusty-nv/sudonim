
from sudonim import MLC, LlamaCpp, Docker, find_quantization_api

def server_up( model: str=None, api: str=None, quantization: str=None, **kwargs ):
    """
    Launch model endpoint servers for the different APIs.
    """
    if not model:
        raise ValueError(f"Missing required argument:  --model")

    if not api:
        api = find_quantization_api(api, quantization)

    if api == 'mlc':
        MLC.deploy(model=model, quantization=quantization, **kwargs)
    elif api == 'llama_cpp':
        LlamaCpp.deploy(model=model, quantization=quantization, **kwargs)
    else:
        raise RuntimeError(f"Unsupported model, API, or quantization selected (api={api} quantization={quantization})")
    
def server_down( container: str='llm_server', **kwargs ):
    """
    Shutdown the model endpoint server
    """
    return Docker.stop(container)