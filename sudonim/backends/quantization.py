
import sudonim as nim

log = nim.getLogger()

def find_quantization_api(api: str=None, quantization: str=None, required=True, **kwargs):
    """ 
    Deduce the model API from its quantization type if needed.
    """
    if api:
        return api
    
    if not quantization:
        raise ValueError(f"Missing required argument:  --api or --quantization")
    
    for quant_api, quantizations in nim.QUANTIZATIONS.items():
        if quantization in quantizations:
            return quant_api

    error = f"Could not find API for quantization type {quantization}"

    if required:
        raise ValueError(error)
    else:   
        log.warning(error)