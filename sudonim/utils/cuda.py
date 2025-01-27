import ctypes

from sudonim import NamedDict, subshell, getLogger, xmlToJson

log = getLogger()

def cudaDeviceQuery():
    """
    Get GPU device info by loading/calling libcuda directly.
    """
    try:
        return _cudaDeviceQuery()
    except Exception as error:
        log.warning(f'cudaDeviceQuery() failed:  {error}')
        raise error
    
def _cudaDeviceQuery():
    """
    https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549
    """
    CUDA_SUCCESS = 0

    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36

    cuda = ctypes.CDLL('libcuda.so')

    nGpus = ctypes.c_int()
    name = b' ' * 100
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()
    cores = ctypes.c_int()
    threads_per_core = ctypes.c_int()
    clockrate = ctypes.c_int()
    freeMem = ctypes.c_size_t()
    totalMem = ctypes.c_size_t()

    result = ctypes.c_int()
    device = ctypes.c_int()
    context = ctypes.c_void_p()
    error_str = ctypes.c_char_p()

    result = cuda.cuInit(0)
    output = []

    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        raise RuntimeError("cuInit failed with error code %d: %s" % (result, error_str.value.decode()))
        return output
    result = cuda.cuDeviceGetCount(ctypes.byref(nGpus))
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        raise RuntimeError("cuDeviceGetCount failed with error code %d: %s" % (result, error_str.value.decode()))
    log.debug("Found %d CUDA devices" % nGpus.value)
    for i in range(nGpus.value):
        result = cuda.cuDeviceGet(ctypes.byref(device), i)
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            raise RuntimeError("cuDeviceGet failed with error code %d: %s" % (result, error_str.value.decode()))
        info = NamedDict()
        if cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device) == CUDA_SUCCESS:
            info.name = name.split(b'\0', 1)[0].decode()
        if cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device) == CUDA_SUCCESS:
            cc = (cc_major.value, cc_minor.value)
            info.family = cudaDeviceFamily(*cc)
            info.cc = cc_major.value * 10 + cc_minor.value
        if cuda.cuDeviceGetAttribute(ctypes.byref(cores), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device) == CUDA_SUCCESS:
            info.mp = cores.value
            info.cores = (cores.value * cudaCoresPerSM(cc_major.value, cc_minor.value) or 0)
            if cuda.cuDeviceGetAttribute(ctypes.byref(threads_per_core), CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device) == CUDA_SUCCESS:
                info.threads = (cores.value * threads_per_core.value)
        #if cuda.cuDeviceGetAttribute(ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device) == CUDA_SUCCESS:
        #    info.gpu_clock = clockrate.value / 1000.
        #if cuda.cuDeviceGetAttribute(ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device) == CUDA_SUCCESS:
        #    info.mem_clock = clockrate.value / 1000.
        try:
            result = cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device)
        except AttributeError:
            result = cuda.cuCtxCreate(ctypes.byref(context), 0, device)
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            raise RuntimeError("cuCtxCreate failed with error code %d: %s" % (result, error_str.value.decode()))
        else:
            try:
                result = cuda.cuMemGetInfo_v2(ctypes.byref(freeMem), ctypes.byref(totalMem))
            except AttributeError:
                result = cuda.cuMemGetInfo(ctypes.byref(freeMem), ctypes.byref(totalMem))
            if result == CUDA_SUCCESS:
                info.mem_total = int(totalMem.value / 1024**2)
                info.mem_free = int(freeMem.value / 1024**2)
            else:
                cuda.cuGetErrorString(result, ctypes.byref(error_str))
                raise RuntimeError("cuMemGetInfo failed with error code %d: %s" % (result, error_str.value.decode()))
            cuda.cuCtxDetach(context)
        if not info:
            continue

        # extended board name for Jetson's
        if info.name and info.mem_total:
            if info.name.lower() == 'orin':
                if info.mem_total > 50000:
                    info.name = 'AGX Orin 64GB'
                elif info.mem_total > 24000:
                    info.name = 'AGX Orin 32GB'
                elif info.mem_total > 12000:
                    info.name = 'Orin NX 16GB' 
                elif info.mem_total > 6000:
                    info.name = 'Orin Nano 8GB'  
                elif info.mem_total > 2500:
                    info.name = 'Orin Nano 4GB'

        output.append(info)
        
    return output

def cudaShortName(name):
    """
    Get board identifier and name
    """
    if name == 'AGX Orin 64GB': return 'agx-orin'
    elif name == 'AGX Orin 32GB': return 'agx-orin-32gb'
    elif name == 'Orin NX 16GB': return 'orin-nx'
    elif name == 'Orin NX 8GB': return 'orin-nx-8gb'
    elif name == 'Orin Nano 8GB': return 'orin-nano'
    elif name == 'Orin Nano 4GB': return 'orin-nano-4gb'
    return name.lower().replace(' ', '-')

def cudaShortVersion(version: str=None):
    """
    Return CUDA version tag (like cu126 for CUDA 12.6)
    """
    if not version:
        version = Env.CUDA_VERSION
    return f"cu{version.replace('.','')}"

def cudaCoresPerSM(major, minor):
    # Returns the number of CUDA cores per multiprocessor for a given
    # Compute Capability version. There is no way to retrieve that via
    # the API, so it needs to be hard-coded.
    # See _ConvertSMVer2Cores in helper_cuda.h in NVIDIA's CUDA Samples.
    return {(1, 0): 8,    # Tesla
            (1, 1): 8,
            (1, 2): 8,
            (1, 3): 8,
            (2, 0): 32,   # Fermi
            (2, 1): 48,
            (3, 0): 192,  # Kepler
            (3, 2): 192,
            (3, 5): 192,
            (3, 7): 192,
            (5, 0): 128,  # Maxwell
            (5, 2): 128,
            (5, 3): 128,
            (6, 0): 64,   # Pascal
            (6, 1): 128,
            (6, 2): 128,
            (7, 0): 64,   # Volta
            (7, 2): 64,
            (7, 5): 64,   # Turing
            (8, 0): 64,   # Ampere
            (8, 6): 128,
            (8, 7): 128,
            (8, 9): 128,  # Ada
            (9, 0): 128,  # Hopper
            }.get((major, minor), 0)

def cudaDeviceFamily(major, minor):
    """
    Map CUDA compute capability to GPU family names.
    """
    if major == 1:      return "Tesla"
    elif major == 2:    return "Fermi"
    elif major == 3:    return "Kepler"
    elif major == 5:    return "Maxwell"
    elif major == 6:    return "Pascal"
    elif major == 7:
        if minor < 5:   return "Volta"
        else:           return "Turing"
    elif major == 8: 
        if minor < 9:   return "Ampere"
        else:           return "Ada"
    elif major == 9:    return "Hopper"
    elif major == 10:   return "Blackwell"
    else:               return "Unknown"

def nvidia_smi_query():
    """ 
    Get GPU device info from nvidia-smi 
    """
    try:
        return xmlToJson(subshell('/usr/sbin/nvidia-smi -q -x', echo=False))
    except Exception as error:
        log.warning(f'Failed to query GPU devices from nvidia-smi ({error})')