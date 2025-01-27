
import os
import shutil
import platform
import sudonim as nim

from pathlib import Path

ENV = None  # globals inherited from OS

def setup_env():
    """
    Populate the environment from OS defaults, scanned packages, and discovered devices.
    This function does not need called manually, it is automatically called the first time
    the environment gets retrieved from getenv() along with the system variables below.
    """
    env = nim.NamedDict()
    smi = nim.nvidia_smi_query()

    CUDA_DEVICES = nim.cudaDeviceQuery()

    if CUDA_DEVICES:
        env.BOARD_ID = nim.cudaShortName(CUDA_DEVICES[0].name);
    elif smi.get('gpu'): 
        env.BOARD_ID = smi['gpu'].lower().replace(' ', '-')
    else: 
        env.BOARD_ID = "unknown"

    env.CPU_ARCH = platform.machine()
    env.GPU_ARCH = f"sm{CUDA_DEVICES[0].cc}" if CUDA_DEVICES else None #  ({CUDA_DEVICES[0].family.lower()})
    
    env.NUM_CPU = os.cpu_count()
    env.NUM_GPU = smi.get('attached_gpus', 0)

    env.CUDA_VERSION = smi.get('cuda_version')
    env.NVIDIA_DRIVER = smi.get('driver_version')

    env.GPU = CUDA_DEVICES if CUDA_DEVICES else smi.get('gpu', [])

    env.CACHE_ROOT = str(Path(
        default_env(
            ['HF_HUB_CACHE', 'TRANSFORMERS_CACHE', 'HF_HOME'], 
            '~/.cache/huggingface'
        )
    ).expanduser().parent)

    env.CACHES = nim.NamedDict( 
        hf = '$CACHE_ROOT/huggingface',
        mlc = '$CACHE_ROOT/mlc_llm',
        trt_llm = '$CACHE_ROOT/trt_llm',
        llama_cpp = '$CACHE_ROOT/llama_cpp',
        datasets = '$CACHE_ROOT/datasets',
        benchmarks = '$CACHE_ROOT/benchmarks'
    )

    env.HF_TOKEN = default_env(['HF_TOKEN', 'HUGGINGFACE_TOKEN'])

    env.HAS_MLC = has_command('mlc_llm')
    env.HAS_HF_HUB = try_import('huggingface_hub')
    env.HAS_NVIDIA_SMI = has_command('nvidia-smi')
    env.HAS_LLAMA_CPP = has_command('llama-server')

    env.HAS_DOCKER_API=try_import('docker')
    env.HAS_DOCKER_CLI=has_command('docker')

    env.HAS_TEGRASTATS=has_command('tegrastats')
    
    #HAS_TRT=has_import('tensorrt')
    #HAS_TRT_LLM=has_import('tensorrt_llm')
    #HAS_TRANSFORMERS=has_import('transformers')

    return env

def getenv(keys=('env', 'log')):
    """
    Return a set of globals, by default a (env, log) tuple
    """
    global ENV

    if isinstance(keys, str):
        keys = (keys)

    vars = []

    for key in keys:
        if key == 'env':
            if ENV is None:
                ENV = setup_env()
            vars.append(ENV)
        elif key == 'log':
            vars.append(nim.getLogger())
        else:
            raise ValueError(f"invalid key in getenv() - '{key}'")

    return vars           

def default_env(keys, default=None):
    """ 
    Get environment variable from the OS with fallback options.
    """
    if isinstance(keys, str):
        keys = [keys]

    while len(keys) > 0:
        env = keys.pop(0)
        if env in os.environ:
            return os.environ[env] 
        
    return default

def filter_env(env, key):
    """
    Callback to merge or remove some keys for presentation in tables/ect.
    """
    if key in ['CACHES', 'cc', 'mp', 'threads', 'mem_total']:
        return
    if key == 'mem_free':
        return f"[{env.mem_free/1000.0:.1f} / {env.mem_total/1000.0:.1f} GB]"
    return env[key]

def has_command(exe):
    """
    Return true if there's an executable found in the PATH by this name. 
    """
    return shutil.which(exe) is not None

def try_import(module):
    """ 
    Return true if import succeeds, false otherwise 
    """
    try:
        __import__(module)
        return True
    except ImportError as error:
        nim.getLogger().debug(f"{module} not found ({error})")
        return False

__all__ = ['getenv', 'default_env', 'filter_env', 'has_command', 'try_import']