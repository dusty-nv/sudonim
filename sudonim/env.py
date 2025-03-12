
import os
import re
import shutil
import platform
import sudonim as nim

from pathlib import Path

env = None  # globals inherited from OS

def global_env(key=None, default=None):
    """
    Retrieve the global environment dict, or a specific environment variable by key.
    First this will populate the environment from OS defaults and scanned packages.
    """
    global env

    if env:
        return env.get(key, default) if key else env

    env = nim.NamedDict()

    env.DEBUG = default_env('DEBUG', False)
    env.DRY_RUN = default_env('DRY_RUN', False)

    smi = nim.nvidia_smi_query()

    env.CUDA_VERSION = smi.get('cuda_version')
    env.NVIDIA_DRIVER = smi.get('driver_version')

    CUDA_DEVICES = nim.cudaDeviceQuery()

    if CUDA_DEVICES:
        env.SYSTEM_ID = nim.cudaShortName(CUDA_DEVICES[0].name);
    elif smi.get('gpu'): 
        env.SYSTEM_ID = smi['gpu'].lower().replace(' ', '-')
    else: 
        env.SYSTEM_ID = "unknown"

    env.CPU_ARCH = platform.machine()
    env.GPU_ARCH = f"sm{CUDA_DEVICES[0].cc}" if CUDA_DEVICES else None #  ({CUDA_DEVICES[0].family.lower()})
    
    env.NUM_CPU = os.cpu_count()
    env.NUM_GPU = smi.get('attached_gpus', 0)

    env.GPU = CUDA_DEVICES if CUDA_DEVICES else smi.get('gpu', [])

    env.CACHE_MODE = parse_kwargs(
        default_env('CACHE_MODE', 'registry,quantization,engine')
    )

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
        benchmarks = '$CACHE_ROOT/benchmarks',
        export = None,
    )

    env.HF_TOKEN = default_env(['HF_TOKEN', 'HUGGINGFACE_TOKEN'])

    env.HAS_AWQ = has_command('vila-eval')
    env.HAS_MLC = has_command('mlc_llm')
    
    env.HAS_HF_HUB = try_import('huggingface_hub')
    env.HAS_NVIDIA_SMI = has_command('nvidia-smi')
    env.HAS_LLAMA_CPP = has_command('llama-server')

    env.HAS_DOCKER_API=try_import('docker')
    env.HAS_DOCKER_CLI=has_command('docker')

    #HAS_TRT=has_import('tensorrt')
    #HAS_TRT_LLM=has_import('tensorrt_llm')
    #HAS_TRANSFORMERS=has_import('transformers')
    
    return env.get(key, default) if key else env

def getenv(keys=('env', 'log')):
    """
    Return a list of globals from the given set of keys.
    """
    if isinstance(keys, str):
        keys = [keys]

    vars = []

    for key in keys:
        if key == 'env':
            vars.append(global_env())
        elif key == 'log':
            vars.append(nim.getLogger())
        elif key in global_env():
            vars.append(global_env(key))
        else:
            raise ValueError(f"invalid key in getenv() - '{key}'")
         
    if len(vars) == 1:
        vars = vars[0]

    return vars           

def default_env(keys, default=None, parse=True):
    """ 
    Get environment variable from the OS with fallback options.
    """
    if isinstance(keys, str):
        keys = [keys]

    while len(keys) > 0:
        key = keys.pop(0)
        if key in os.environ:
            val = os.environ[key]
            if val is None or (isinstance(val, str) and len(val.strip()) == 0):
                return True  # set but no value - return as flag
            return parse_value(val) if parse else val

    return default

def filter_env(env, key, value, blacklist=[]):
    """
    Callback to merge or remove some keys for presentation in tables/ect.
    """
    if blacklist is not None and len(blacklist) == 0:
        blacklist = [
            'CACHES', 'CACHE_MODE', 
            'cc', 'mp', 'threads', 'mem_total', 
            'NUM_CPU', 'NUM_GPU',
            'HAS_DOCKER_API'
        ]

    if blacklist and key in blacklist:
        return
        
    if key == 'mem_free':
        return f"[{env.mem_free/1000.0:.1f} / {env.mem_total/1000.0:.1f} GB]"
    
    if key == 'HF_TOKEN' and value:
        return f"{value[0:3]}*********"  # redact API key
    
    if isinstance(key, int):
        max_gpus_shown=2
        if key > max_gpus_shown:
            return
        elif len(env) > 1:
            return f"{key}/{len(env)}", value
    
    return value

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

def parse_kwargs(args, defaults=None, key_caps=False):
    """
    Parse k:v style arguments, like ``--option foo:bar,abc=ON,def``
    It returns a dict of the variables in addition to the defaults.
    If ``key_caps=True`` then the variable names get uppercased in the dict.
    """
    def split_keys(x):
        return re.split(',|;', x)
    def split_vars(x):
        return re.split('=|:', x)

    if defaults is None:
        defaults = nim.NamedDict()

    if isinstance(args, str):
        args = [args]

    for x in args:
        for y in split_keys(x):
            kv = split_vars(y)

            defaults[kv[0].upper() if key_caps else kv[0]] = \
                parse_value(kv[1]) if len(kv) > 1 else True

    return defaults

def parse_value(arg, type=None, required=False):
    """
    Convert a string into a boolean or numerical value
    """
    if not isinstance(arg, str):
        return arg
    
    if arg.lower() in ['0', 'false', 'off', 'no', 'n', 'disable', 'disabled']:
        return False if type != bool else 0
    elif arg.upper() in ['1', 'true', 'on', 'yes', 'y', 'enable', 'enabled']:
        return True if type != bool else 1
    
    try:
        return eval(arg)
    except Exception as error:
        if required:
            raise
        return arg

__all__ = ['getenv', 'default_env', 'filter_env', 'has_command', 'try_import', 'parse_kwargs', 'parse_value']